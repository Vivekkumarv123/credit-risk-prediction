import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("💳 Credit Risk Prediction System")
st.write("Predict if a loan applicant is High Risk or Low Risk.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", value=50000)
employment = st.selectbox("Employment Type", ["Salaried", "Business"])
loan_amount = st.number_input("Loan Amount", value=20000)
loan_duration = st.number_input("Loan Duration (months)", value=36)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
prev_defaults = st.number_input("Number of Previous Defaults", min_value=0, value=0)
collateral = st.selectbox("Collateral", ["Yes", "No"])
dependents = st.number_input("Number of Dependents", min_value=0, value=0)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# Encode categorical inputs
employment_encoded = 1 if employment == "Salaried" else 0
collateral_encoded = 1 if collateral == "Yes" else 0
marital_single = 1 if marital_status == "Single" else 0
marital_divorced = 1 if marital_status == "Divorced" else 0

# Age group features
age_30_40 = 1 if 30 <= age < 40 else 0
age_40_50 = 1 if 40 <= age < 50 else 0
age_50_60 = 1 if 50 <= age < 60 else 0
age_60_70 = 1 if 60 <= age < 70 else 0

if st.button("Predict"):
    # Prepare input array in correct order
    input_data = np.array([[age, income, employment_encoded, loan_amount, loan_duration,
                            credit_score, prev_defaults, collateral_encoded, dependents,
                            marital_single, marital_divorced, age_30_40, age_40_50, age_50_60, age_60_70]])
    
    # Feature engineering (ratios)
    debt_income_ratio = loan_amount / income
    total_obligation_ratio = (loan_amount + prev_defaults*10000) / income
    # Append engineered features
    input_data = np.append(input_data, [[debt_income_ratio, total_obligation_ratio]], axis=1)
    
    # Scale
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.error("⚠️ High Risk Applicant")
    else:
        st.success("✅ Low Risk Applicant")
