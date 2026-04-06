from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# تحميل الموديل + scaler
data = joblib.load("model.pkl")  # dict {"model": model, "scaler": scaler}
model = data["model"]
scaler = data["scaler"]

# نموذج البيانات اللي تستقبله API
class InputData(BaseModel):
    features: list  # عدد العناصر = عدد أعمدة X بدون 'Class'

@app.get("/")
def home():
    return {"message": "Fraud Detection API 🚀"}

@app.post("/predict")
def predict(data: InputData):
    # تحويل list إلى numpy array
    features = np.array(data.features).reshape(1, -1)
    
    # scaling للعمود Amount فقط (نفترض هو آخر عمود)
    features[0][-1] = scaler.transform([[features[0][-1]]])[0][0]

    # توقع الاحتمالية
    prob = model.predict_proba(features)[0][1]

    # threshold لتحديد Fraud
    threshold = 0.9
    prediction = int(prob > threshold)

    return {
        "probability": float(prob),
        "prediction": prediction
    }