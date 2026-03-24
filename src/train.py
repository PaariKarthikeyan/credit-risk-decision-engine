import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


def train_model(data_path, model_path):
    print(f"Loading data from {data_path}...")
    
    df = pd.read_pickle(data_path)

    # -----------------------------
    # SPLIT FEATURES & TARGET
    # -----------------------------
    X = df.drop(columns=['target'])
    y = df['target']

    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {y.mean():.4f}")

    # -----------------------------
    # TRAIN-TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # HANDLE IMBALANCE
    # -----------------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    print("Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # EVALUATION
    # -----------------------------
    print("Evaluating model...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    print(f"Saving model to {model_path}...")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("✅ Training complete!")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(current_dir, "..", "data", "ml_ready_lending_club.pkl")
    model_path = os.path.join(current_dir, "..", "models", "xgboost_model.pkl")

    # Create models folder if not exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_model(data_path, model_path)