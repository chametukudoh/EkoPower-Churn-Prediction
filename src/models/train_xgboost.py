import os
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    f1_score,
    classification_report
)

from src.data.preprocess import preprocess_data


def train_xgboost_mlflow(
    client_path: str = "data/client_data.csv",
    price_path: str = "data/price_data.csv",
    target_col: str = "churn"
):
    """
    Trains an XGBoost model for churn prediction and logs everything with MLflow.

    Args:
        client_path (str): Path to the client data CSV.
        price_path (str): Path to the price data CSV.
        target_col (str): Name of the target column in the final feature set.
    """

    # === Load and preprocess ===
    client_df = pd.read_csv(client_path)
    price_df = pd.read_csv(price_path)

    df = preprocess_data(client_df, price_df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in feature set.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col]

    # Convert categorical and boolean features into numeric values for XGBoost
    cat_cols = X.select_dtypes(include=["category"]).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            X[col] = X[col].cat.codes.astype("float64")

    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)


    # ===  Train/Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")

    # Class imbalance handling
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # === XGBoost model definition ===
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )

    # ===  MLflow tracking ===
    mlflow.set_experiment("EkoPower Churn Prediction")

    with mlflow.start_run(run_name="xgb-churn-model"):
        model.fit(X_train, y_train)
        threshold = 0.30
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)


        # Evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        # Log model details
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "accuracy": acc,
            "recall": rec,
            "roc_auc": auc,
            "f1_score": f1
        })

        mlflow.xgboost.log_model(model, name="model")

        # Log input dataset
        input_data = mlflow.data.from_pandas(X_train, source=client_path)
        mlflow.log_input(input_data, context="training")

        print("\nModel training complete.")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("\n" + classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_xgboost_mlflow()
