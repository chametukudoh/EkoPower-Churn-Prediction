# tests/test_model_inference.py
"""
Smoke test for model inference using the inference module.

Context:
- Assumes a trained model has been logged with MLflow and can be loaded locally.
- Uses a minimal synthetic input row with feature names matching engineered dataset.
- Uses `inference.py` to load model and predict.

Goal:
- Ensure pipeline can run inference on new data without crashing.
"""

import os
import pandas as pd
import numpy as np

from src.serving.inference import load_model, predict_churn

# Dummy sample based on features expected from build_features
def make_sample_input() -> pd.DataFrame:
    return pd.DataFrame([{
        "tenure_days": 400,
        "days_to_renewal": 30,
        "contract_active": 1,
        "contracted_tenure_years": 1.0,
        "has_gas": 1,
        "channel_sales_online": 1,
        "channel_sales_retail": 0,
        "origin_up_us_market": 1,
        "origin_up_eu_market": 0,
        "price_off_peak_var_mean": 0.08,
        "price_off_peak_var_std": 0.01,
        "price_peak_var_mean": 0.20,
        "price_peak_var_std": 0.02,
        "price_mid_peak_var_mean": 0.15,
        "price_mid_peak_var_std": 0.015,
        "price_off_peak_fix_mean": 0.03,
        "price_off_peak_fix_std": 0.005,
        "price_peak_fix_mean": 0.06,
        "price_peak_fix_std": 0.008,
        "price_mid_peak_fix_mean": 0.045,
        "price_mid_peak_fix_std": 0.007
    }])

def test_model_prediction_runs():
    # Load model
    model = load_model()

    # Build input
    sample_input = make_sample_input()

    # Predict
    result_df = predict_churn(model, sample_input)

    # Check structure
    assert "churn_probability" in result_df.columns, "Missing churn_probability output"
    assert "churn_predicted" in result_df.columns, "Missing churn_predicted output"
    assert 0 <= result_df["churn_probability"].iloc[0] <= 1, "Probability out of bounds"
    assert result_df["churn_predicted"].iloc[0] in [0, 1], "Invalid predicted label"
