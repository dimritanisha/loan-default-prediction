import streamlit as st
import numpy as np
import joblib

# Load trained model and imputer
model = joblib.load('rf_model_10features.pkl')
imputer = joblib.load('imputer_10features.pkl')

st.set_page_config(page_title="Loan Repayment Prediction", layout="centered")

st.title(" Microfinance Loan Repayment Predictor")
st.write("Enter the details below to predict whether a customer will repay the mobile micro-loan.")

# Input fields for the 10 features
aon = st.number_input("AON")
daily_decr90 = st.number_input("Daily Decrease 90")
payback30 = st.number_input("Payback 30")
amnt_loans90 = st.number_input("Amount Loans 90")
last_rech_amt_ma = st.number_input("Last Recharge Amount MA")
avg_loan_amount_30 = st.number_input("Average Loan Amount 30")
call_success_ratio = st.number_input("Call Success Ratio")
rech_amt_per_count_ma_30 = st.number_input("Recharge Amount Per Count MA")
loan_to_total_rech_ratio = st.number_input("Loan to Total Recharge Ratio")
daily_to_rental_ratio = st.number_input("Daily to Rental Ratio")

# Predict button
if st.button("Predict Repayment"):
    # Prepare feature array
    input_data = np.array([[aon, daily_decr90, payback30, amnt_loans90, last_rech_amt_ma,
                            avg_loan_amount_30, call_success_ratio, rech_amt_per_count_ma_30,
                            loan_to_total_rech_ratio, daily_to_rental_ratio]])
    
    # Apply imputation
    input_data_imputed = imputer.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_imputed)[0]
    prob = model.predict_proba(input_data_imputed)[0][1]  # probability of label 1

    # Show result
    if prediction == 1:
        st.success(f"✅ Likely to Repay Loan (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Likely to Default on Loan (Probability: {prob:.2f})")

