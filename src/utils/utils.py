# src/utils/utils.py
"""
Small, reusable helpers for the EkoPower churn pipeline.

Keep this file lightweight and focused on utilities that are reused across:
- tuning (Optuna)
- training (XGBoost + MLflow)
- evaluation (thresholding, metrics formatting)
- pipeline orchestration (folders, snapshots)

NOTE:
- The project uses a fixed default decision threshold of 0.3 (from the base notebook).
- Threshold sweep utilities are provided for analysis, but the pipeline keeps 0.3 unless you change it.
"""

from __future__ import annotations

import os
import math
import time
import json
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import mlflow
except Exception:
    mlflow = None  # utils should not hard-depend on mlflow at import time


# ------------------------
# Reproducibility & paths
# ------------------------

def set_random_seeds(seed: int = 42) -> None:
    """Set seeds for python, numpy (extend if you add torch etc.)."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> str:
    """Create directory if missing; return the absolute path."""
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def timestamp() -> str:
    """Compact UTC timestamp for filenames."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def get_project_paths() -> Dict[str, str]:
    """
    Common project-relative folders used in this repo structure.
    """
    return {
        "data_raw": os.path.join("data", "raw"),
        "data_processed": os.path.join("data", "processed"),
        "artifacts": os.path.join("src", "serving", "artifacts"),
        "mlruns": "mlruns",
        "configs": "configs",
    }


# ------------------------
# Data snapshots
# ------------------------

def snapshot_dataframe(df: pd.DataFrame, name: str, folder: Optional[str] = None) -> str:
    """
    Save a CSV snapshot of a dataframe under data/processed (or custom folder).
    Returns the saved path.
    """
    paths = get_project_paths()
    out_dir = folder or paths["data_processed"]
    ensure_dir(out_dir)
    fname = f"{name}_{timestamp()}.csv"
    fpath = os.path.join(out_dir, fname)
    df.to_csv(fpath, index=False)
    return fpath


# ------------------------
# Class imbalance helpers
# ------------------------

def compute_scale_pos_weight(y: Iterable[int]) -> float:
    """
    XGBoost's recommended scale_pos_weight = (# negatives / # positives).
    Protects against division by zero.
    """
    y = np.asarray(y)
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    return float(neg / max(pos, 1))


# ------------------------
# Probability & thresholding
# ------------------------

DEFAULT_THRESHOLD = 0.3  # from the base notebook

def safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return class-1 probabilities if available, else fall back to decision_function
    with a sigmoid squashing.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # sigmoid
        return 1 / (1 + np.exp(-scores))
    else:
        # last resort: predictions as {0,1}
        preds = model.predict(X)
        return preds.astype(float)


def apply_threshold(y_proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Convert probabilities to binary labels using a fixed threshold."""
    return (y_proba >= threshold).astype(int)


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[Iterable[float]] = None,
    metric: str = "f1"
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """
    Sweep thresholds to locate a best cutoff for a chosen metric ('f1', 'recall', 'precision', 'youden').
    Returns (best_threshold, metrics_by_threshold). Does not mutate pipeline default (0.3).
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    results: Dict[float, Dict[str, float]] = {}
    best_t = DEFAULT_THRESHOLD
    best_v = -math.inf

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # Youden's J: sensitivity + specificity - 1
        # specificity:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        spec = tn / max(tn + fp, 1)
        youden = rec + spec - 1

        results[t] = {"precision": prec, "recall": rec, "f1": f1, "youden": youden}

        value = {"f1": f1, "recall": rec, "precision": prec, "youden": youden}.get(metric, f1)
        if value > best_v:
            best_v, best_t = value, t

    return best_t, results


# ------------------------
# Metrics formatting & CM
# ------------------------

def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return a dict with TP, FP, TN, FN counts."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def format_metrics(metrics: Dict[str, float], decimals: int = 4) -> Dict[str, float]:
    """Round metric floats consistently for printing/logging."""
    out = {}
    for k, v in metrics.items():
        try:
            out[k] = round(float(v), decimals)
        except Exception:
            out[k] = v
    return out


# ------------------------
# MLflow helpers (safe)
# ------------------------

def set_mlflow_experiment(name: str = "EkoPower Churn - XGBoost", tracking_uri: Optional[str] = None) -> None:
    """
    Initialize MLflow tracking for this project.
    If tracking_uri is provided, sets it; otherwise default MLflow behavior applies.
    """
    if mlflow is None:
        return
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(name)


def mlflow_log_params_safely(params: Dict) -> None:
    """Log params to MLflow if available (avoid crashing when mlflow missing)."""
    if mlflow is None:
        return
    try:
        mlflow.log_params(params)
    except Exception:
        pass


def mlflow_log_metrics_safely(metrics: Dict[str, float]) -> None:
    """Log metrics to MLflow if available."""
    if mlflow is None:
        return
    try:
        mlflow.log_metrics(format_metrics(metrics))
    except Exception:
        pass


def mlflow_log_dict_safely(payload: Dict, artifact_file: str = "details.json") -> None:
    """Log a small JSON artifact (e.g., threshold sweep results) if MLflow available."""
    if mlflow is None:
        return
    try:
        tmp_dir = ensure_dir(os.path.join("artifacts_tmp"))
        fpath = os.path.join(tmp_dir, artifact_file)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        mlflow.log_artifact(fpath)
    except Exception:
        pass
