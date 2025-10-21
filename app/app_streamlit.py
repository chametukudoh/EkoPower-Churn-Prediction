import streamlit as st
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features


@st.cache_data
def load_and_prepare_data(client_path, price_path):
    client_df = pd.read_csv(client_path)
    price_df = pd.read_csv(price_path)

    client_df = preprocess_data(client_df)
    df = build_features(client_df, price_df)

    return df


def load_model(threshold=0.3):
    model_uri = "mlruns/0/latest/model"
    model = mlflow.xgboost.load_model(model_uri)
    return model


def predict_churn(df, model, threshold=0.3):
    if "has_churned" in df.columns:
        df = df.drop(columns=["has_churned"])

    proba = model.predict_proba(df)[:, 1]
    df["churn_risk"] = proba
    df["predicted_churn"] = (proba >= threshold).astype(int)
    return df


# ================== Streamlit App ==================
st.set_page_config(page_title="EkoPower Churn App", layout="wide")

st.title("📊 EkoPower Churn Prediction App")
st.markdown("Use the latest trained XGBoost model to predict customer churn.")

with st.sidebar:
    st.header("📂 Input Paths")
    client_path = st.text_input("Client Data", value="data/raw/client_data.csv")
    price_path = st.text_input("Price Data", value="data/raw/price_data.csv")
    threshold = st.slider("Churn Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    predict_btn = st.button("🚀 Predict")

if predict_btn:
    st.info("⏳ Running churn prediction pipeline...")
    try:
        df = load_and_prepare_data(client_path, price_path)
        model = load_model(threshold)
        result = predict_churn(df, model, threshold)

        st.success("✅ Prediction complete!")
        st.dataframe(result.head(50), use_container_width=True)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", csv, "churn_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
