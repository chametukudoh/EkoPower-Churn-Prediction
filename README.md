# EkoPower Churn Prediction Pipeline 🔌📉


I developed an end-to-end churn prediction pipeline for EkoPower to proactively identify customers likely to leave the service. This empowers the team to target high-risk segments with tailored interventions, reduce churn, and improve customer lifetime value.

By combining historical customer data and pricing signals, we trained an XGBoost model optimized for **recall**, ensuring we catch as many potential churners as possible. All steps—from preprocessing and feature engineering to hyperparameter tuning, evaluation, and deployment—are tracked using MLflow, and the final product is delivered via an interactive Streamlit dashboard.

---

## Situation

EkoPower is a power distribution startup that tracks customer usage, billing, and consumption patterns. Churn—customers terminating their service—is a critical KPI.

EkoPower has access to customer profile data (`client_data.csv`) and pricing/usage metadata (`price_data.csv`). However, this raw data needs significant cleaning and transformation before it can be used for machine learning purposes.

---

## Complication

Despite having rich data, EkoPower faced multiple challenges:

- The churn class was **highly imbalanced**
- There was no standardized process for:
  - Loading and preprocessing data
  - Engineering features
  - Tuning and evaluating models
  - Logging experiments and comparing models
- Manual interventions were frequent, making reproducibility difficult
- No live interface existed to inspect churn drivers or simulate churn predictions

---

## Question

**How might we create a reliable, reproducible, and recall-optimized churn prediction pipeline for EkoPower—with full ML lifecycle tracking and an interactive dashboard for decision-makers?**

---

# What I Built

I implemented a full MLOps-compliant churn prediction solution using:

- `XGBoost` for classification
- `Optuna` for hyperparameter tuning (recall-focused)py
- `MLflow` for experiment tracking
- `Streamlit` for the deployment interface
- `Great Expectations` for data validation
- `pytest` for testing all pipeline components

**Pipeline Overview:**

1. **Preprocessing:**
   - Drop redundant IDs
   - Handle NAs, strip headers, map target variable
2. **Feature Engineering:**
   - Merge client and pricing data
   - Generate derived features (e.g., margin)
3. **Modeling:**
   - Train/test split
   - Tune with Optuna (recall score)
   - Final training with best parameters
4. **Evaluation:**
   - Precision, Recall, AUC, F1-score
   - Visual inspection via SHAP, correlation
5. **Deployment:**
   - `serve_model.py` for backend inference
   - `app_streamlit.py` for business-friendly frontend
6. **Validation:**
   - Run `validate_data.py` with Great Expectations
   - 5+ pytest test scripts for data loading, inference, and end-to-end flows

---

# 🚀 Deployment Flow

```bash
# Clone repo and setup
git clone https://github.com/<your-org>/ekopower-churn.git
cd ekopower-churn
pip install -r requirements.txt

# Run ML pipeline (data prep → tuning → training → evaluation)
python run_pipeline.py

# Launch dashboard
streamlit run app_streamlit.py

# Run tests
pytest tests/

# Launch MLflow UI
mlflow ui
