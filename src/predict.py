import os
import pickle
import pandas as pd
import numpy as np


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# -----------------------------
# PREPROCESS SINGLE INPUT
# -----------------------------
def preprocess_input(input_dict, training_columns):
    df = pd.DataFrame([input_dict])

    # -----------------------------
    # SAME CLEANING LOGIC
    # -----------------------------
    # Convert percentages
    for col in ['int_rate', 'revol_util']:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('%', '', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Term
    if 'term' in df.columns:
        df['term'] = pd.to_numeric(
            df['term'].astype(str).str.extract(r'(\d+)')[0],
            errors='coerce'
        )

    # Emp length
    if 'emp_length' in df.columns:
        df['emp_length'] = pd.to_numeric(
            df['emp_length'].astype(str).str.extract(r'(\d+)')[0],
            errors='coerce'
        ).fillna(0)

    # Dates → credit history
    if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'])
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])

        df['credit_hist_months'] = (
            (df['issue_d'] - df['earliest_cr_line']).dt.days // 30
        )

        df = df.drop(columns=['issue_d', 'earliest_cr_line'])

    # Feature engineering
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['installment_to_income'] = df['installment'] / (df['annual_inc'] + 1)

    # Missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    # One-hot encoding
    df = pd.get_dummies(df)

    # -----------------------------
    # ALIGN WITH TRAINING COLUMNS
    # -----------------------------
    df = df.reindex(columns=training_columns, fill_value=0)

    return df


# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict(input_dict, model, training_columns, threshold=0.22):
    processed = preprocess_input(input_dict, training_columns)

    prob = model.predict_proba(processed)[0][1]
    decision = "Rejected ❌" if prob > threshold else "Approved ✅"

    return prob, decision


# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, "..", "models", "xgboost_model.pkl")
    data_path = os.path.join(current_dir, "..", "data", "ml_ready_lending_club.pkl")

    model = load_model(model_path)

    # Load training columns
    df = pd.read_pickle(data_path)
    training_columns = df.drop(columns=['target']).columns

    # Example input (simulate user input)
    sample_input = {
        "loan_amnt": 10000,
        "term": "36 months",
        "int_rate": "13.5%",
        "installment": 300,
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "annual_inc": 50000,
        "purpose": "credit_card",
        "dti": 15,
        "revol_util": "45%",
        "total_acc": 20,
        "issue_d": "2020-01-01",
        "earliest_cr_line": "2010-01-01"
    }

    prob, decision = predict(sample_input, model, training_columns)

    print(f"Default Probability: {prob:.4f}")
    print(f"Decision: {decision}")