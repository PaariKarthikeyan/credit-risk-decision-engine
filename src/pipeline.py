# src/pipeline.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Column selection ──────────────────────────────────────────────────────────
USECOLS = [
    'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment', 'grade',
    'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
    'issue_d', 'loan_status', 'purpose', 'dti', 'delinq_2yrs', 'earliest_cr_line',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'last_pymnt_amnt', 'fico_range_low', 'fico_range_high',
    'last_fico_range_low', 'last_fico_range_high',
    'mths_since_last_delinq', 'mths_since_last_record',
    'collections_12_mths_ex_med', 'acc_now_delinq',
    'tot_coll_amt', 'tot_cur_bal',
]

# Loans that are still open have no outcome yet — exclude them
CLOSED_STATUSES = {
    'Fully Paid', 'Charged Off', 'Default',
    'Does not meet the credit policy. Status:Fully Paid',
    'Does not meet the credit policy. Status:Charged Off',
}

DEFAULT_STATUSES = {'Charged Off', 'Default',
                    'Does not meet the credit policy. Status:Charged Off'}

GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

CAT_COLS = ['purpose', 'home_ownership', 'verification_status']


def load_and_filter(path: str) -> pd.DataFrame:
    print("[1/6] Loading CSV (this may take ~60 s for 1 GB+) ...")
    df = pd.read_csv(path, usecols=USECOLS, low_memory=False)
    print(f"      Raw rows: {len(df):,}  |  cols: {df.shape[1]}")

    df = df[df['loan_status'].isin(CLOSED_STATUSES)].copy()
    df['target'] = df['loan_status'].isin(DEFAULT_STATUSES).astype(np.int8)
    df.drop(columns=['loan_status'], inplace=True)

    default_rate = df['target'].mean() * 100
    print(f"      After filter: {len(df):,} rows  |  Default rate: {default_rate:.2f}%")
    return df


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Cleaning string columns ...")

    # Only strip % if the column is still a string dtype
    for col in ['int_rate', 'revol_util']:
        if df[col].dtype == object:
            df[col] = df[col].str.replace('%', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # term: "36 months" → 36 (only if string)
    if df['term'].dtype == object:
        df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
    else:
        df['term'] = pd.to_numeric(df['term'], errors='coerce')

    # emp_length: "10+ years" → 10, "< 1 year" → 0
    if df['emp_length'].dtype == object:
        df['emp_length'] = (
            df['emp_length']
            .str.replace(r'\+', '', regex=True)
            .str.replace('< 1 year', '0', regex=False)
            .str.extract(r'(\d+)')
            .astype(float)
        )
    else:
        df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Engineering features ...")
    df['issue_d']          = pd.to_datetime(df['issue_d'],          format='%b-%Y')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')

    # Credit history length in months (days ÷ 30.44)
    diff_days = (df['issue_d'] - df['earliest_cr_line']).dt.days
    df['credit_history_length'] = (diff_days / 30.44).astype(float)

    # FICO features
    df['fico_avg']      = (df['fico_range_low']      + df['fico_range_high'])      / 2
    df['last_fico_avg'] = (df['last_fico_range_low']  + df['last_fico_range_high']) / 2
    df['fico_drop']     = df['fico_avg'] - df['last_fico_avg']

    # Leverage ratios
    df['loan_to_income']        = df['loan_amnt']   / (df['annual_inc'] + 1)
    df['installment_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)

    # Interaction term
    df['revol_util_x_dti'] = df['revol_util'] * df['dti']

    # Temporal features
    df['issue_year']  = df['issue_d'].dt.year
    df['issue_month'] = df['issue_d'].dt.month

    df.drop(columns=['issue_d', 'earliest_cr_line',
                     'fico_range_low', 'fico_range_high',
                     'last_fico_range_low', 'last_fico_range_high'], inplace=True)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/6] Imputing missing values ...")
    num_cols = df.select_dtypes(include=[np.number]).columns.drop('target')
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CAT_COLS:
        if df[col].isna().any():
            mode_val = df[col].mode()
            fill = mode_val[0] if not mode_val.empty else 'Unknown'
            df[col] = df[col].fillna(fill)

    # emp_length and sub_grade may still have NaN
    df['emp_length'] = df['emp_length'].fillna(df['emp_length'].median())
    df['sub_grade']  = df['sub_grade'].fillna('Unknown')

    remaining = df.isna().sum().sum()
    print(f"      Remaining nulls: {remaining}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/6] Encoding categoricals ...")
    df['grade'] = df['grade'].map(GRADE_MAP).fillna(4)

    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True, dtype=np.int8)

    # sub_grade is already ordinal-ish — label encode alphabetically
    df['sub_grade'] = pd.Categorical(df['sub_grade']).codes

    print(f"      Final shape: {df.shape}")
    return df


def run(csv_path: str):
    out_path = Path('data/processed/lending_clean.parquet')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(csv_path)
    df = clean_strings(df)
    df = engineer_features(df)
    df = impute_missing(df)
    df = encode_categoricals(df)

    print(f"[6/6] Saving parquet → {out_path} ...")
    df.to_parquet(out_path, index=False)
    print(f"      Done! File size: {out_path.stat().st_size // 1_000_000} MB")
    print("\n✅ Pipeline complete.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py data/raw/accepted_2007_to_2018Q4.csv")
        sys.exit(1)
    run(sys.argv[1])