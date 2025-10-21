# src/serving/inference.py
"""
Batch and programmatic inference for the EkoPower churn model.

- Loads a trained XGBoost model from MLflow OR a local artifact path.
- Preprocesses + builds features exactly like training (uses project modules).
- Applies the project default decision threshold (0.3) unless overridden.
- Supports DataFrame-to-DataFrame scoring, as well as CSV -> CSV batch scoring.

Context:
- Training/logging uses mlflow.xgboost.log_model(model, artifact_path="model")
- Features are created by src/features/build_features.build_features
- Preprocessing is handled by src/data/preprocess.preprocess_data
"""

from __future__ import annotations
from typing import Optional, Tuple

import os
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

# Project modules
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.utils import (
    DEFAULT_THRESHOLD,
    ensure_dir,
    safe_predict_proba,
    apply_threshold,
    snapshot_dataframe,
)

# MLflow is optional; only required if loading via model URI
try:
    import mlflow
    import mlflow.xgboost
except Exception:
    mlflow = None  # type: ignore


# -----------------------------
# Model loading
# -----------------------------

def load_model_mlflow(model_uri: str) -> XGBClassifier:
    """
    Load model from MLflow model registry or local MLflow artifacts.
    Example URIs:
      - "runs:/<run_id>/model"
      - "models:/EkoPower_XGB/Production"
      - "models:/EkoPower_XGB/1"
    """
    if mlflow is None:
        raise ImportError("mlflow is required to load model from MLflow URIs.")
    model = mlflow.xgboost.load_model(model_uri)
    if not hasattr(model, "predict_proba"):
        # Ensure sklearn API wrapper is present (mlflow returns a compatible estimator)
        raise TypeError("Loaded model does not expose predict_proba().")
    return model  # type: ignore[return-value]


def load_model_local(artifact_path: str) -> XGBClassifier:
    """
    Load a model saved by MLflow to a local path (e.g., ./src/serving/artifacts/model).
    Expectation: path contains an MLmodel file (MLflow format).
    """
    if mlflow is None:
        raise ImportError("mlflow is required to load local MLflow artifacts.")
    model = mlflow.xgboost.load_model(artifact_path)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not expose predict_proba().")
    return model  # type: ignore[return-value]


# -----------------------------
# Feature preparation
# -----------------------------

def prepare_features_for_inference(
    client_df: pd.DataFrame,
    price_df: pd.DataFrame,
    target_col: str = "has_churned"
) -> pd.DataFrame:
    """
    Apply the exact preprocessing + feature building steps used in training.
    Drops target column if accidentally present in the provided client_df.
    """
    client_df = preprocess_data(client_df)
    feat_df = build_features(client_df, price_df)

    # Drop target if present (inference only needs features)
    if target_col in feat_df.columns:
        feat_df = feat_df.drop(columns=[target_col])

    return feat_df


# -----------------------------
# Core prediction helpers
# -----------------------------

def predict_proba(
    model: XGBClassifier,
    features_df: pd.DataFrame
) -> np.ndarray:
    """
    Return P(churn=1) for each row.
    """
    return safe_predict_proba(model, features_df)


def predict_labels(
    model: XGBClassifier,
    features_df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """
    Return binary labels using the project default threshold (0.3) unless overridden.
    """
    proba = predict_proba(model, features_df)
    return apply_threshold(proba, threshold=threshold)


# -----------------------------
# Public scoring APIs
# -----------------------------

def score_from_dataframes(
    model: XGBClassifier,
    client_df: pd.DataFrame,
    price_df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """
    End-to-end: transforms raw inputs -> features -> predictions.
    Returns a DataFrame with `churn_proba` and `churn_pred` columns.
    """
    X = prepare_features_for_inference(client_df, price_df)
    proba = predict_proba(model, X)
    pred = (proba >= threshold).astype(int)

    out = X.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = pred
    return out


def score_from_csv(
    client_csv_path: str,
    price_csv_path: str,
    model: Optional[XGBClassifier] = None,
    model_uri: Optional[str] = None,
    local_model_path: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Batch scoring: read CSVs, load model (if not provided), produce predictions,
    and optionally save a snapshot under data/processed/ (or a custom save_path).

    One of `model`, `model_uri`, or `local_model_path` must be provided.
    """
    if model is None:
        if model_uri:
            model = load_model_mlflow(model_uri)
        elif local_model_path:
            model = load_model_local(local_model_path)
        else:
            raise ValueError("Provide either a fitted `model`, `model_uri`, or `local_model_path`.")

    client_df = pd.read_csv(client_csv_path)
    price_df = pd.read_csv(price_csv_path)

    scored = score_from_dataframes(model, client_df, price_df, threshold=threshold)

    # Persist if requested
    if save_path is not None:
        ensure_dir(os.path.dirname(save_path) or ".")
        scored.to_csv(save_path, index=False)
    else:
        # Default to processed snapshot if no explicit path provided
        save_path = snapshot_dataframe(scored, name="ekopower_scored")

    print(f"✅ Scoring complete. Output saved to: {save_path}")
    return scored
