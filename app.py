import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction App")
st.subheader("Telecom Customer Retention Machine Learning Project")

st.markdown("""
This app predicts whether a telecom customer is likely to churn based on account, billing, and service information.

**Project value:**
- Business analytics
- Predictive modelling
- Customer retention insights
- Interactive machine learning deployment
""")

# ---------------------------
# Load and Prepare Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop("customerID", axis=1)
    return df

df = load_data()

# ---------------------------
# Train Model
# ---------------------------
@st.cache_resource
def train_model(dataframe):
    df_model = dataframe.copy()
    df_model["Churn"] = df_model["Churn"].map({"Yes": 1, "No": 0})

    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]

    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X_encoded.columns

model, feature_columns = train_model(df)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.sidebar.slider("Total Charges", 0.0, 10000.0, 1000.0)

# ---------------------------
# Input DataFrame
# ---------------------------
input_dict = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_dict])

# Encode input like training data
input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)[0]
    prediction_proba = model.predict_proba(input_encoded)[0][1]

    st.markdown("## Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn.")
    else:
        st.success(f"✅ This customer is likely to stay.")

    st.metric("Churn Probability", f"{prediction_proba:.2%}")

    st.markdown("### Business Interpretation")
    if contract == "Month-to-month":
        st.write("- Month-to-month contracts are typically associated with higher churn risk.")
    if monthly_charges > 80:
        st.write("- Higher monthly charges may increase churn probability.")
    if tenure < 12:
        st.write("- Newer customers are generally more likely to churn.")
    if tech_support == "No":
        st.write("- Lack of tech support may reduce retention.")

# ---------------------------
# Data Preview
# ---------------------------
with st.expander("Show Dataset Preview"):
    st.dataframe(df.head())

with st.expander("Show Input Data"):
    st.dataframe(input_df)