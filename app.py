import streamlit as st
import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
sys.path.append(src_path)

from utils import load_model, predict

# -----------------------------
# LOAD MODEL + COLUMNS
# -----------------------------

model_path = os.path.join(current_dir, "..", "models", "xgboost_model.pkl")
data_path = os.path.join(current_dir, "..", "data", "ml_ready_lending_club.pkl")

model = load_model(model_path)

# Load training columns
df = pd.read_pickle(data_path)
training_columns = df.drop(columns=['target']).columns

# -----------------------------
# UI DESIGN
# -----------------------------
st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("🏦 Loan Default Risk Predictor")
st.markdown("Predict whether a loan should be approved or rejected.")

# Sidebar inputs
st.sidebar.header("📋 Enter Applicant Details")

# -----------------------------
# LOAN AMOUNT (Perfect Sync)
# -----------------------------
st.sidebar.subheader("💰 Loan Amount")

# Initialize session state
if "loan_amnt" not in st.session_state:
    st.session_state.loan_amnt = 1000

# Slider
loan_slider = st.sidebar.slider(
    "Select Loan Amount (₹)",
    min_value=1000,
    max_value=5000000,
    step=1000,
    value=st.session_state.loan_amnt,
    key="loan_slider"
)

# Number input
loan_input = st.sidebar.number_input(
    "Or Enter Loan Amount Manually (₹)",
    min_value=1000,
    max_value=5000000,
    step=1000,
    value=st.session_state.loan_amnt,
    key="loan_input"
)

# Detect which changed
if loan_slider != st.session_state.loan_amnt:
    st.session_state.loan_amnt = loan_slider

elif loan_input != st.session_state.loan_amnt:
    st.session_state.loan_amnt = loan_input

loan_amnt = st.session_state.loan_amnt

# -----------------------------
# LOAN TERM (Flexible Options)
# -----------------------------
term_months = st.sidebar.selectbox(
    "Loan Term (Months)",
    options=list(range(3, 61, 3))  # 3,6,9,...60
)

term = f"{term_months} months"

# -----------------------------
# INTEREST RATE
# -----------------------------
int_rate = st.sidebar.slider(
    "Interest Rate (%)",
    min_value=5.0,
    max_value=30.0,
    value=13.0
)

# -----------------------------
# INSTALLMENT
# -----------------------------
installment = st.sidebar.number_input(
    "Monthly Installment",
    min_value=50,
    max_value=1000000,
    value=300
)

# -----------------------------
# EMPLOYMENT
# -----------------------------
emp_length = st.sidebar.selectbox(
    "Employment Length",
    ["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years"]
)

home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

# -----------------------------
# ANNUAL INCOME (Perfect Sync)
# -----------------------------
if "annual_inc" not in st.session_state:
    st.session_state.annual_inc = 500000

income_slider = st.sidebar.slider(
    "Select Annual Income (₹)",
    min_value=10000,
    max_value=500000000,
    step=10000,
    value=st.session_state.annual_inc,
    key="income_slider"
)

income_input = st.sidebar.number_input(
    "Or Enter Annual Income Manually (₹)",
    min_value=10000,
    max_value=500000000,
    step=10000,
    value=st.session_state.annual_inc,
    key="income_input"
)

if income_slider != st.session_state.annual_inc:
    st.session_state.annual_inc = income_slider

elif income_input != st.session_state.annual_inc:
    st.session_state.annual_inc = income_input

annual_inc = st.session_state.annual_inc

# -----------------------------
# PURPOSE
# -----------------------------
purpose = st.sidebar.selectbox(
    "Loan Purpose",
    [
        "credit_card", "debt_consolidation", "home_improvement",
        "major_purchase", "medical", "small_business", "vacation"
    ]
)

# -----------------------------
# FINANCIAL DETAILS
# -----------------------------
dti = st.sidebar.slider(
    "Debt-to-Income Ratio",
    min_value=0.0,
    max_value=50.0,
    value=15.0
)

revol_util = st.sidebar.slider(
    "Revolving Utilization (%)",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

total_acc = st.sidebar.slider(
    "Total Accounts",
    min_value=1,
    max_value=200,
    value=20
)

# Dummy dates (for feature engineering)
issue_d = "2020-01-01"
earliest_cr_line = "2010-01-01"

# -----------------------------
# PREDICTION
# -----------------------------
if st.sidebar.button("🔍 Predict"):

    input_data = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": f"{int_rate}%",
        "installment": installment,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "purpose": purpose,
        "dti": dti,
        "revol_util": f"{revol_util}%",
        "total_acc": total_acc,
        "issue_d": issue_d,
        "earliest_cr_line": earliest_cr_line
    }

    prob, decision = predict(input_data, model, training_columns)

    # -----------------------------
    # OUTPUT DISPLAY
    # -----------------------------
    st.subheader("📊 Prediction Result")

    if "Rejected" in decision:
        st.error(f"{decision}")
    else:
        st.success(f"{decision}")

    st.metric(label="Default Probability", value=f"{prob:.2%}")

    # Risk interpretation
    st.markdown("### 🧠 Risk Interpretation")

    if prob < 0.2:
        st.write("🟢 Low Risk")
    elif prob < 0.4:
        st.write("🟡 Medium Risk")
    else:
        st.write("🔴 High Risk")