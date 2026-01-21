from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import os
import base64  # 🔥 추가
from pathlib import Path



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

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_cnn_model.keras"

# ---------------------------
# FastAPI 초기화
# ---------------------------
app = FastAPI(title="Surface Defect Detection API")

# ---------------------------
# 모델 로드 (서버 시작 시 1회)
# ---------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------
# 이미지 전처리 함수
# ---------------------------
def preprocess_image(image_bytes):
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    return img

# ---------------------------
# 예측 API
# ---------------------------
@app.post("/predict/defect")
async def predict_defect(file: UploadFile = File(...)):
    try:
        # ---------------------------
        # 1. 이미지 읽기
        # ---------------------------
        image_bytes = await file.read()

        # 🔥 입력 이미지 그대로 base64 인코딩
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # ---------------------------
        # 2. 전처리 + 예측
        # ---------------------------
        img = preprocess_image(image_bytes)

        preds = model.predict(img, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])

        # ---------------------------
        # 3. 응답
        # ---------------------------
        return JSONResponse({
            "predicted_class": CLASS_NAMES[class_idx],
            "confidence": confidence,
            "all_scores": {
                CLASS_NAMES[i]: float(preds[i])
                for i in range(len(CLASS_NAMES))
            },
            "image_base64": image_base64  # ✅ 입력 이미지 그대로 반환
        })

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
