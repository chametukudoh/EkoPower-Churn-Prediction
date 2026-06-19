# EkoPower Churn Prediction

Portfolio project for building and evaluating an imbalanced customer-churn classifier with reproducible data preparation, experiment tracking, validation, tests and stakeholder-facing interfaces.

## What the repository demonstrates

- XGBoost classification and Optuna tuning
- explicit recall-focused model selection for the cost of missed churners
- preprocessing and feature-engineering modules
- MLflow experiment artifacts
- Great Expectations data validation
- pytest coverage for loading, features, inference, validation and pipeline flow
- Streamlit and FastAPI prototypes
- Docker packaging

## Evidence boundaries

This is a portfolio project, not a deployed EkoPower customer system. It does not establish production usage, churn reduction, revenue impact or live model monitoring.

Model performance must be reproduced from the current code before quoting exact metrics. Recall is the stated tuning priority, but a deployment decision should also consider precision, PR-AUC, calibration, intervention cost and capacity.

The FastAPI and Streamlit surfaces are prototypes. They require compatible generated model artifacts and should not be described as hosted production services.

## Repository structure

```text
api/                 FastAPI prototype
app/                 Streamlit prototype
data/                source and processed data used by the project
scripts/run_pipeline.py
src/data/            loading and preprocessing
src/features/        feature engineering
src/models/          training, tuning and evaluation
src/serving/         programmatic/batch inference
src/utils/           validation and shared helpers
test/                automated tests
mlruns/              checked-in reference experiment artifacts
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the pipeline from the repository root:

```powershell
python scripts/run_pipeline.py
```

Run tests:

```powershell
python -m pytest test -q
```

Inspect tracked MLflow runs:

```powershell
python -m mlflow ui --backend-store-uri ./mlruns
```

The UI and API require model artifacts compatible with their loaders. Confirm paths and schemas before starting either surface.

## Model review checklist

Before presenting a metric or deploying a new artifact:

1. Use a leakage-safe split that reflects the intended prediction time.
2. Report class balance and confusion matrix.
3. Report precision, recall, F1, ROC-AUC and PR-AUC.
4. Select the threshold using intervention cost/capacity, not recall alone.
5. Check probability calibration and subgroup behavior.
6. Record dataset, code and parameter versions in MLflow.
7. Run `python -m pytest test -q`.

## Next improvements

- Consolidate the API and Streamlit model-loading contracts.
- Generate a small versioned release artifact rather than relying on ad hoc local paths.
- Add calibration and intervention-cost analysis.
- Reduce checked-in MLflow output after publishing a reproducible benchmark report.
- Add a recorded demo once the current artifact is reproduced end to end.
