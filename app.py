import joblib
import streamlit as st
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")

st.title("Customer Churn Prediction")
st.write("add customer informations then press **Predict** to see churn rate")

# load model and metadata
MODEL_PATH = os.path.join("models", "churn_model.pkl")
META_PATH = os.path.join("models", "model_meta.json")

# numeric col
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

if not os.path.exists(MODEL_PATH):
    st.error("can not find model at models/churn_model.pkl")
    st.stop()

model = joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
features = meta.get("features", [])

# Create forms
with st.form("churn_form"):
    st.subheader("Customer Profile")

    gender = st.selectbox("gender", ["Female", "Male"], index=0)
    SeniorCitizen = st.selectbox("SeniorCitizen (ผู้สูงอายุ)", ["0", "1"], index=0)
    Partner = st.selectbox("Partner", ["No", "Yes"], index=0)
    Dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)

    tenure = st.number_input("tenure", min_value=0, max_value=120, value=12, step=1)

    InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"], index=0)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
    PaperlessBilling = st.selectbox("PaperlessBilling", ["No", "Yes"], index=1)
    PaymentMethod = st.selectbox("PaymentMethod", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], index=0)

    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)
    TotalCharges = st.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=800.0, step=10.0)

    with st.expander("Advanced options"):
        PhoneService = st.selectbox("PhoneService", ["No", "Yes"], index=1)
        MultipleLines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"], index=0)

        OnlineSecurity = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"], index=0)
        OnlineBackup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"], index=0)
        DeviceProtection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"], index=0)
        TechSupport = st.selectbox("TechSupport", ["No", "Yes", "No internet service"], index=0)
        StreamingTV = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"], index=0)
        StreamingMovies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"], index=0)

    submitted = st.form_submit_button("Predict")

def make_input_df():

    data = {col: np.nan for col in features}

    # update data
    data.update({
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    })

    if "customerID" in features and "customerID" not in data:
        data["customerID"] = "APP-USER-0001"

    df_input = pd.DataFrame([data], columns=features)

    for col in df_input.columns:
        if col in NUMERIC_COLS:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0)
        else:
            df_input[col] = df_input[col].astype("string").fillna("Unknown")

    return df_input

if submitted:
    inp = make_input_df()
    proba = model.predict_proba(inp)[:, 1][0]
    pred = int(proba >= 0.5)

    st.subheader("Predicted result")
    st.metric(label="Churn Probability", value=f"{proba*100:.1f}%")
    st.write(f"**Predict** {'Churn' if pred == 1 else 'not Churn'} (threshold 0.5)")
    st.progress(min(max(proba, 0.0), 1.0))
