import os
import uvicorn
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# === Load model from MLflow ===
MODEL_URI = os.getenv("MODEL_URI", "models:/EkoPower Churn - XGBoost/Production")

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model from {MODEL_URI}: {e}")

# === FastAPI app ===
app = FastAPI(title="EkoPower Churn Predictor API")

# === Input schema ===
class ClientFeatures(BaseModel):
    features: List[float]  # Single row of features
    feature_names: List[str]

@app.get("/")
def root():
    return {"message": "EkoPower Churn Prediction API is running."}

@app.post("/predict")
def predict_churn(input_data: ClientFeatures):
    try:
        df = pd.DataFrame([input_data.features], columns=input_data.feature_names)
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1]
        return {
            "prediction": int(prediction[0]),
            "churn_probability": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

