from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import base64
import boto3
from io import BytesIO

# ---------------------------
# 기본 설정
# ---------------------------
IMAGE_SIZE = 200
CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled-in_Scale",
    "Scratches"
]

# ---------------------------
# AWS S3 설정
# ---------------------------
s3 = boto3.client('s3')
S3_BUCKET_NAME = 'anomaly-detection-bucket1'
MODEL_S3_KEY = '242392964/best_cnn_model.keras'

# ---------------------------
# FastAPI 초기화
# ---------------------------
app = FastAPI(title="Surface Defect Detection API")

# ---------------------------
# 모델 로드 함수 (S3에서 다운로드 후 로드)
# ---------------------------
def load_model_from_s3():
    try:
        model_file = BytesIO()
        s3.download_fileobj(S3_BUCKET_NAME, MODEL_S3_KEY, model_file)
        model_file.seek(0)
        model = tf.keras.models.load_model(model_file)
        return model
    except Exception as e:
        raise ValueError(f"모델 로드 실패: {str(e)}")

# ---------------------------
# 모델 로드 (서버 시작 시 1회)
# ---------------------------
model = load_model_from_s3()

# ---------------------------
# 이미지 전처리 함수
# ---------------------------
def preprocess_image(image_bytes):
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # 컬러 이미지로 변환할 경우
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 필요 시 변경

    img = img / 255.0
    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)  # RGB로 변경된 경우 3 채널로 변경

    return img

# ---------------------------
# 예측 API
# ---------------------------
@app.post("/predict/defect")
async def predict_defect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # 🔥 입력 이미지 그대로 base64 인코딩
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 전처리 및 예측
        img = preprocess_image(image_bytes)

        preds = model.predict(img, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])

        return JSONResponse({
            "predicted_class": CLASS_NAMES[class_idx],
            "confidence": confidence,
            "all_scores": {
                CLASS_NAMES[i]: float(preds[i])
                for i in range(len(CLASS_NAMES))
            },
            "image_base64": image_base64
        })

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
