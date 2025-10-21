# tests/test_pipeline_end_to_end.py
"""
End-to-end smoke test of the ML pipeline.

Covers:
- Data load
- Preprocessing
- Feature engineering
- Training
- Inference

Goal:
- Ensure the pipeline runs successfully on a sample slice
- Does not guarantee performance — just sanity

Assumptions:
- Data files exist in `data/raw`
- Code uses `train_xgboost`, `evaluate_model`, `build_features`, etc.
"""

import os
import pandas as pd
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_xgboost import train_model
from src.serving.inference import load_model, predict_churn


def test_pipeline_e2e():
    # Load raw data
    raw_path = "data/raw"
    client_df, price_df = load_data(raw_path, "train_client.csv", "train_price.csv")

    # Sample small subset for quick testing
    client_df = client_df.sample(n=200, random_state=42)
    price_df = price_df[price_df["id"].isin(client_df["id"])]

    # Preprocess
    pre_df = preprocess_data(client_df)

    # Build features
    final_df = build_features(pre_df, price_df)

    # Train and persist model
    model_path = train_model(final_df, target_col="has_churned")

    assert os.path.exists(model_path), "Trained model not saved"

    # Inference
    model = load_model(model_path)
    sample_input = final_df.drop(columns=["has_churned"]).iloc[:5]
    result_df = predict_churn(model, sample_input)

    assert "churn_probability" in result_df.columns
    assert "churn_predicted" in result_df.columns
