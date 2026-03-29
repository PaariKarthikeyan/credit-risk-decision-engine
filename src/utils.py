import pickle
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np

def preprocess_input(input_data, training_columns):
    df = pd.DataFrame([input_data])

    # -----------------------------
    # CLEAN NUMERIC FIELDS
    # -----------------------------
    # Remove % if present
    if isinstance(df.loc[0, 'int_rate'], str):
        df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float)

    if isinstance(df.loc[0, 'revol_util'], str):
        df['revol_util'] = df['revol_util'].str.rstrip('%').astype(float)

    # Term: "36 months" → 36
    if isinstance(df.loc[0, 'term'], str):
        df['term'] = df['term'].str.extract(r'(\d+)').astype(int)

    # Employment length: extract number
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)

    # -----------------------------
    # DATE FEATURES
    # -----------------------------
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])

    df['credit_hist_months'] = (
        (df['issue_d'] - df['earliest_cr_line']).dt.days // 30
    )

    df = df.drop(columns=['issue_d', 'earliest_cr_line'])

    # -----------------------------
    # ENCODING
    # -----------------------------
    df = pd.get_dummies(df)

    # -----------------------------
    # ALIGN WITH TRAINING DATA
    # -----------------------------
    df = df.reindex(columns=training_columns, fill_value=0)

    return df

# -----------------------------
# LOAD & SAVE MODEL
# -----------------------------
def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    return pd.read_pickle(path)


# -----------------------------
# GET FEATURES & TARGET
# -----------------------------
def split_features_target(df, target_col='target'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# -----------------------------
# CALCULATE CLASS IMBALANCE
# -----------------------------
def calculate_scale_pos_weight(y):
    return (y == 0).sum() / (y == 1).sum()


# -----------------------------
# PREPROCESS INPUT (REUSED)
# -----------------------------
def preprocess_input(input_dict, training_columns):
    df = pd.DataFrame([input_dict])

    # Convert percentage columns
    for col in ['int_rate', 'revol_util']:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('%', '', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Term conversion
    if 'term' in df.columns:
        df['term'] = pd.to_numeric(
            df['term'].astype(str).str.extract(r'(\d+)')[0],
            errors='coerce'
        )

    # Employment length
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

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align with training columns
    df = df.reindex(columns=training_columns, fill_value=0)

    return df


# -----------------------------
# MAKE PREDICTION
# -----------------------------
def predict(input_data, model, training_columns):
    processed = preprocess_input(input_data, training_columns)
    prob = model.predict_proba(processed)[:, 1][0]
    return prob, "Rejected" if prob > 0.22 else "Approved"

import pickle
import pandas as pd
import numpy as np


# -----------------------------
# LOAD & SAVE MODEL
# -----------------------------
def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    return pd.read_pickle(path)


# -----------------------------
# GET FEATURES & TARGET
# -----------------------------
def split_features_target(df, target_col='target'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# -----------------------------
# CALCULATE CLASS IMBALANCE
# -----------------------------
def calculate_scale_pos_weight(y):
    return (y == 0).sum() / (y == 1).sum()


# -----------------------------
# PREPROCESS INPUT (REUSED)
# -----------------------------
def preprocess_input(input_dict, training_columns):
    df = pd.DataFrame([input_dict])

    # Convert percentage columns
    for col in ['int_rate', 'revol_util']:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('%', '', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Term conversion
    if 'term' in df.columns:
        df['term'] = pd.to_numeric(
            df['term'].astype(str).str.extract(r'(\d+)')[0],
            errors='coerce'
        )

    # Employment length
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

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align with training columns
    df = df.reindex(columns=training_columns, fill_value=0)

    return df


# -----------------------------
# MAKE PREDICTION
# -----------------------------
def predict(input_dict, model, training_columns, threshold=0.22):
    processed = preprocess_input(input_dict, training_columns)

    prob = model.predict_proba(processed)[0][1]
    decision = "Rejected ❌" if prob > threshold else "Approved ✅"

    return prob, decision