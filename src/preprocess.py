import os
import pandas as pd
import numpy as np

def clean_and_prepare_data(raw_file_path, output_file_path):
    print(f"Loading raw data from {raw_file_path}...")
    
    columns_to_keep = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'emp_length',
        'home_ownership', 'annual_inc', 'purpose', 'dti', 
        'revol_util', 'total_acc', 'issue_d', 'earliest_cr_line', 'loan_status'
    ]
    
    df = pd.read_csv(raw_file_path, usecols=columns_to_keep, low_memory=False)
    
    # -----------------------------
    # TARGET CREATION
    # -----------------------------
    print("Filtering for finished loans...")
    valid_statuses = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(valid_statuses)].copy()
    
    df['target'] = df['loan_status'].map({
        'Fully Paid': 0,
        'Charged Off': 1
    })
    
    df = df.drop(columns=['loan_status'])

    # -----------------------------
    # FEATURE CLEANING
    # -----------------------------
    print("Cleaning and converting features...")
    
    # Convert percentage strings → float
    for col in ['int_rate', 'revol_util']:
        df[col] = (
            df[col]
            .astype(str)                # convert everything to string
            .str.replace('%', '', regex=False)
        )
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert term ("36 months" → 36)
    df['term'] = pd.to_numeric(
        df['term'].astype(str).str.extract(r'(\d+)')[0],
        errors='coerce'
    )
    
    # Employment length ("10+ years" → 10)
    df['emp_length'] = pd.to_numeric(
        df['emp_length'].astype(str).str.extract(r'(\d+)')[0],
        errors='coerce'
    )
    df['emp_length'] = df['emp_length'].fillna(0)

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    print("Engineering features...")
    
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    
    df['credit_hist_months'] = (
        (df['issue_d'] - df['earliest_cr_line']).dt.days // 30
    )
    
    df = df.drop(columns=['issue_d', 'earliest_cr_line'])

    # Additional powerful features
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['installment_to_income'] = df['installment'] / (df['annual_inc'] + 1)

    # -----------------------------
    # OUTLIER HANDLING
    # -----------------------------
    print("Handling outliers...")
    
    df['annual_inc'] = np.clip(df['annual_inc'], 0, 500000)
    df['dti'] = np.clip(df['dti'], 0, 50)
    df['revol_util'] = np.clip(df['revol_util'], 0, 150)

    # -----------------------------
    # MISSING VALUES
    # -----------------------------
    print("Imputing missing values...")
    
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    # -----------------------------
    # ENCODING
    # -----------------------------
    print("Encoding categorical variables...")
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # -----------------------------
    # MEMORY OPTIMIZATION
    # -----------------------------
    print("Optimizing memory footprint...")
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
        
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int32)
        
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(np.uint8)

    # -----------------------------
    # SAVE
    # -----------------------------
    print(f"Saving final ML-ready dataset to {output_file_path}")
    df.to_pickle(output_file_path)
    
    print(f"✅ Done! Final shape: {df.shape}")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "..", "data")
    
    raw_filename = "accepted_2007_to_2018Q4.csv.gz"
    
    raw_file = os.path.join(data_folder, raw_filename)
    clean_file = os.path.join(data_folder, "ml_ready_lending_club.pkl")
    
    clean_and_prepare_data(raw_file, clean_file)