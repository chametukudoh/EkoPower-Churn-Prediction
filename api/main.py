
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import joblib, json, os
import pandas as pd

from src.preprocess import basic_clean
from src.build_features import engineer_dates

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "models")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")

preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)

app = FastAPI(title="EkoPower Churn API", version="1.1.0")

class ClientRecord(BaseModel):
    __root__: Dict[str, Any]

class PredictRequest(BaseModel):
    records: List[ClientRecord] = Field(..., description="List of client records (key-value maps)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    rows = [r.__root__ for r in req.records]
    df = pd.DataFrame(rows)
    df = basic_clean(df)
    X = preprocessor.transform(engineer_dates(df))
    proba = model.predict_proba(X)[:, 1]
    return {"predictions": [{"churn_probability": float(p)} for p in proba]}
