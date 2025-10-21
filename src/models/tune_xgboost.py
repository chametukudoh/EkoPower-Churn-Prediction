import optuna
from xgboost import XGBClassifier
from sklearn.metrics import recall_score


def tune_xgboost(X_train, X_test, y_train, y_test, threshold=0.30, n_trials=35):
    """
    Tune XGBoost hyperparameters using Optuna, optimizing recall score.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.
        threshold (float): Decision threshold for converting probabilities to class labels.
        n_trials (int): Number of Optuna trials.

    Returns:
        optuna.study.Study: Completed Optuna study object.
    """

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        return recall_score(y_test, y_pred, pos_label=1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study
