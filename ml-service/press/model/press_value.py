from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import json
import joblib
import os

app = FastAPI()

# ---------------------------
# 기본 설정
# ---------------------------
SEQUENCE = 20
FEATURE_DIM = 3
FEATURE_ORDER = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "", "best_lstm_ae.h5")
SCALER_PATH = os.path.join(BASE_DIR, "", "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "", "threshold.json")

# ---------------------------
# 모델 / 스케일러 / threshold 로드
# ---------------------------
lstm_ae = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH) as f:
    THRESHOLD = json.load(f)["threshold"]

# ---------------------------
# 요청 스키마
# ---------------------------
class PressData(BaseModel):
    data: list[list[float]]  # (sequence, feature)

# ---------------------------
# MSE 계산
# ---------------------------
def compute_mse(model, X):
    X_pred = model.predict(X, verbose=0)
    X_flat = X.reshape(X.shape[0], -1)
    X_pred_flat = X_pred.reshape(X_pred.shape[0], -1)
    return float(np.mean((X_flat - X_pred_flat) ** 2, axis=1)[0])

# ---------------------------
# 예측 API
# ---------------------------
@app.post("/predict/press")
def predict_press(req: PressData):
    # 1️⃣ numpy 변환
    X = np.array(req.data, dtype=np.float32)

    # 2️⃣ shape 검증
    if X.ndim != 2:
        return {"error": "data must be 2D (sequence, feature)"}

    if X.shape[0] != SEQUENCE:
        return {"error": f"sequence length must be {SEQUENCE}"}

    if X.shape[1] != FEATURE_DIM:
        return {"error": f"feature dimension must be {FEATURE_DIM}"}

    # 3️⃣ scaler 적용 (⚠️ reshape 전에)
    X_scaled = scaler.transform(X)

    # 4️⃣ LSTM 입력 shape로 변환
    X_scaled = X_scaled.reshape(1, SEQUENCE, FEATURE_DIM)

    # 5️⃣ MSE 계산
    mse = compute_mse(lstm_ae, X_scaled)

    # 6️⃣ 이상 여부 판단
    is_anomaly = int(mse > THRESHOLD)

    return {
        "mse_score": mse,
        "threshold": THRESHOLD,
        "is_anomaly": is_anomaly
    }
