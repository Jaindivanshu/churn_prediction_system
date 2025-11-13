import streamlit as st
import pandas as pd
import pickle

st.markdown("""
    <style>
    button, .stButton>button, .st-selectbox, .stNumberInput, label, div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
feature_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
st.title("📊 Customer Churn Prediction System")

st.markdown("Enter customer details below and click **Predict** to see if the customer is likely to churn.")

# Collect user input fields (same as your training features)
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, step=1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

with col2:
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
# CENTERED TotalCharges field
# ------------------------------
st.write("")
center1, center2, center3 = st.columns([1, 2, 1])
with center2:
    TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.1)

# ------------------------------
# CENTERED Predict button
# ------------------------------
st.write("")
b1, b2, b3 = st.columns([1, 0.5, 1])
with b2:
    predict_btn = st.button("🔍 Predict")

# When user clicks Predict
if predict_btn:
    # Create DataFrame for model input
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    # Encode categorical features
    for column, encoder in encoders.items():
        if column in input_data.columns:
            input_data[column] = encoder.transform(input_data[column])

    # Predict
    prediction = model.predict(input_data)[0]
    pred_prob = model.predict_proba(input_data)[0][1]

    # Display results
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to churn** (Probability: {pred_prob:.2f})")
    else:
        st.success(f"✅ The customer is **not likely to churn** (Probability: {pred_prob:.2f})")