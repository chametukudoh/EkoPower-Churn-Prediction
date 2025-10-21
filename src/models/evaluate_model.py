import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series,threshold: float = 0.30, verbose: bool = True) -> dict:
    """
    Evaluates an XGBoost classification model using standard classification metrics.

    Args:
        model (XGBClassifier): Trained model.
        X_test (pd.DataFrame): Features for testing.
        y_test (pd.Series): True labels for test data.
        verbose (bool): Whether to print detailed classification report.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Predict
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)


    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    if verbose:
        print("\n=== MODEL EVALUATION ===")
        for key, val in metrics.items():
            if key != "confusion_matrix":
                print(f"{key}: {val:.4f}")
        print("\nConfusion Matrix:")
        print(np.array(metrics["confusion_matrix"]))
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred))

    return metrics
