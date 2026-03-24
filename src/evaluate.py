import os
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)


def evaluate_model(data_path, model_path):
    print("Loading data and model...")

    df = pd.read_pickle(data_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # -----------------------------
    # SPLIT FEATURES & TARGET
    # -----------------------------
    X = df.drop(columns=['target'])
    y = df['target']

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # -----------------------------
    # BASIC METRICS
    # -----------------------------
    print("\n📊 Classification Report:")
    print(classification_report(y, y_pred))

    roc_auc = roc_auc_score(y, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y, y_pred)
    print("\n📉 Confusion Matrix:")
    print(cm)

    # -----------------------------
    # THRESHOLD TUNING
    # -----------------------------
    print("\n🔍 Finding best threshold...")

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = -np.inf

    for t in thresholds:
        preds = (y_prob > t).astype(int)

        # Example business logic:
        # +100 for correct non-default
        # -500 for missed default
        profit = (
            ((preds == 0) & (y == 0)).sum() * 100
            - ((preds == 0) & (y == 1)).sum() * 500
        )

        if profit > best_score:
            best_score = profit
            best_threshold = t

    print(f"💰 Best Threshold: {best_threshold:.2f}")
    print(f"💵 Estimated Profit: {best_score}")

    # -----------------------------
    # PRECISION-RECALL ANALYSIS
    # -----------------------------
    precision, recall, _ = precision_recall_curve(y, y_prob)

    print("\n📈 Precision-Recall Snapshot:")
    for i in range(0, len(precision), max(1, len(precision)//5)):
        print(f"Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(current_dir, "..", "data", "ml_ready_lending_club.pkl")
    model_path = os.path.join(current_dir, "..", "models", "xgboost_model.pkl")

    evaluate_model(data_path, model_path)