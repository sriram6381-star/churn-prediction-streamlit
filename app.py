import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.title("📊 Customer Churn Predictor")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
monthly_charges = st.number_input("Monthly Charges", 10.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

if st.button("Predict Churn"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'InternetService': internet_service,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    # Encode categorical values
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService']:
        le = label_encoders[col]
        input_data[col] = le.transform(input_data[col])

    # Scale numerical columns
    input_data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        input_data[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.info(f"Churn Probability: {probability:.2%}")
