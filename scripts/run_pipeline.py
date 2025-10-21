"""
Run the full EkoPower churn prediction pipeline:
1) Load raw data
2) Preprocess + build features
3) Tune XGBoost with Optuna (maximize recall)
4) Retrain with best params
5) Evaluate and log to MLflow
"""

from __future__ import annotations

import os
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Project modules
from src.data.preprocess import preprocess_data
from src.models.tune_xgboost import tune_xgboost
from src.models.evaluate_model import evaluate_model


def run_pipeline(
    client_path: str = "data/client_data.csv",
    price_path: str = "data/price_data.csv",
    target_col: str = "churn",
    threshold: float = 0.3,
    n_trials: int = 35,
):
    """
    End-to-end execution of the EkoPower churn training workflow.
    """

    # === 1) Load and preprocess ===
    client_df = pd.read_csv(client_path)
    price_df = pd.read_csv(price_path)

    # Preprocess + feature build combined
    df = preprocess_data(client_df, price_df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # === 2) Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # === 3) Tune XGBoost using Optuna ===
    print("\nRunning Optuna hyperparameter optimization...")
    study = tune_xgboost(
        X_train, X_test, y_train, y_test, threshold=threshold, n_trials=n_trials
    )
    best_params = dict(study.best_params)
    print("\nBest parameters found:\n", best_params)
    print(f"Best recall score: {study.best_value:.4f}")

    # Add fixed params
    best_params.update(
        {
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
    )

    # === 4) Train final model with best params and log to MLflow ===
    mlruns_path = os.path.join(os.getcwd(), "mlruns").replace("\\", "/")
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")

    experiment_name = "EkoPower Churn - XGBoost"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        

    with mlflow.start_run(run_name="xgb_best_model", experiment_id=experiment_id):
        print("\nTraining final model with tuned parameters...")
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        # Evaluate on test set using probability threshold
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, proba)

        mlflow.log_params(best_params)
        mlflow.log_metrics(
            {"precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}
        )

        mlflow.xgboost.log_model(model, "model")

        print("\nFinal model performance:")
        print(classification_report(y_test, y_pred, digits=3))

        # Evaluate in more detail using project utility
        evaluate_model(model, X_test, y_test)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()

