import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import shap

# -----------------------------
# PATH FIX (for utils import)
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
sys.path.append(src_path)

from utils import load_model, predict, preprocess_input

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
model_path = os.path.join(current_dir, "..", "models", "xgboost_model.pkl")
data_path = os.path.join(current_dir, "..", "data", "ml_ready_lending_club.pkl")

model = load_model(model_path)
df = pd.read_pickle(data_path)
training_columns = df.drop(columns=['target']).columns

# SHAP explainer
explainer = shap.Explainer(model)

# -----------------------------
# EMI CALCULATOR
# -----------------------------
def calculate_emi(P, annual_rate, months):
    r = annual_rate / (12 * 100)
    emi = (P * r * (1 + r)**months) / ((1 + r)**months - 1)
    return emi

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Loan Risk Predictor", layout="wide")
st.title("🏦 Loan Default Risk Predictor")

st.sidebar.header("📋 Applicant Details")

# Loan amount
loan_amnt = st.sidebar.number_input(
    "Loan Amount (₹)",
    min_value=1000,
    max_value=5000000,
    value=100000,
    step=1000
)

# Term
term_months = st.sidebar.selectbox(
    "Loan Term (Months)",
    list(range(3, 61, 3))
)

# Interest
int_rate = st.sidebar.slider(
    "Interest Rate (%)",
    5.0, 30.0, 12.0
)

# EMI AUTO CALCULATION
installment = calculate_emi(loan_amnt, int_rate, term_months)
st.sidebar.write(f"💳 Estimated EMI: ₹{int(installment):,}")

# Employment
emp_length = st.sidebar.selectbox(
    "Employment Length",
    ["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years"]
)

home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

# Income
annual_inc = st.sidebar.number_input(
    "Annual Income (₹)",
    min_value=10000,
    max_value=500000000,
    value=500000,
    step=10000
)

# Purpose
purpose = st.sidebar.selectbox(
    "Loan Purpose",
    ["credit_card", "debt_consolidation", "home_improvement", "major_purchase"]
)

# Financial
dti = st.sidebar.slider("DTI", 0.0, 50.0, 15.0)
revol_util = st.sidebar.slider("Revolving Utilization (%)", 0.0, 100.0, 40.0)
total_acc = st.sidebar.slider("Total Accounts", 1, 100, 20)

# Dummy dates
issue_d = "2020-01-01"
earliest_cr_line = "2010-01-01"

# -----------------------------
# VALIDATION
# -----------------------------
if loan_amnt > annual_inc * 2:
    st.sidebar.warning("⚠️ Loan is too high compared to income")

if dti > 40:
    st.sidebar.warning("⚠️ High DTI risk")

# -----------------------------
# PREDICTION
# -----------------------------
if st.sidebar.button("🔍 Predict"):

    input_data = {
        "loan_amnt": loan_amnt,
        "term": f"{term_months} months",
        "int_rate": int_rate,
        "installment": installment,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "purpose": purpose,
        "dti": dti,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "issue_d": issue_d,
        "earliest_cr_line": earliest_cr_line
    }

    processed = preprocess_input(input_data, training_columns)

    st.write(processed.head())
    st.write(processed.sum())

    prob = model.predict_proba(processed)[:, 1][0]

    st.subheader("📊 Prediction Result")

    # Decision
    if prob < 0.2:
        st.success("🟢 Approved")
    elif prob < 0.4:
        st.warning("🟡 Needs Review")
    else:
        st.error("🔴 Rejected")

    st.metric("Default Probability", f"{prob:.2%}")

    # SHAP
    shap_values = explainer(processed)
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches="tight")
    st.caption("Feature contributions to prediction")