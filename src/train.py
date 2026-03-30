# src/train.py
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report)
from xgboost import XGBClassifier

PARQUET   = 'data/processed/lending_clean.parquet'
MODEL_DIR = Path('models')
OUT_DIR   = Path('outputs')
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# ── 1. Load ───────────────────────────────────────────────────────────────────
print("[1/5] Loading parquet ...")
df = pd.read_parquet(PARQUET)
print(f"      Shape: {df.shape}  |  Default rate: {df['target'].mean()*100:.2f}%")

X = df.drop(columns=['target'])
y = df['target']

# ── 2. Split ──────────────────────────────────────────────────────────────────
print("[2/5] Train/test split (80/20, stratified) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 3. Train ──────────────────────────────────────────────────────────────────
neg   = int((y_train == 0).sum())
pos   = int((y_train == 1).sum())
spw   = round(neg / pos, 2)
print(f"[3/5] Training XGBoost  (scale_pos_weight={spw}) ...")
print("      This may take 5–20 minutes depending on your CPU ...")

model = XGBClassifier(
    n_estimators       = 500,
    learning_rate      = 0.05,
    max_depth          = 6,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    scale_pos_weight   = spw,
    eval_metric        = 'aucpr',
    early_stopping_rounds = 20,
    tree_method        = 'hist',
    random_state       = 42,
    n_jobs             = -1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
print("\n[4/5] Evaluating ...")
proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, proba)
pr_auc  = average_precision_score(y_test, proba)
print(f"      ROC-AUC : {roc_auc:.4f}")
print(f"      PR-AUC  : {pr_auc:.4f}")

# Threshold tuning — find best F-beta (beta=0.5, precision-weighted)
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, proba)
beta = 0.5
fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-9)
best_idx       = np.argmax(fbeta[:-1])
best_threshold = float(thresholds[best_idx])
best_fbeta     = float(fbeta[best_idx])
print(f"      Optimal threshold (F-{beta}): {best_threshold:.4f}  →  F-score: {best_fbeta:.4f}")

preds = (proba >= best_threshold).astype(int)
print("\n" + classification_report(y_test, preds, target_names=['Paid','Default']))

# Save metrics
metrics = {
    'roc_auc':          round(roc_auc, 4),
    'pr_auc':           round(pr_auc, 4),
    'optimal_threshold': round(best_threshold, 4),
    'scale_pos_weight': spw,
    'feature_names':    list(X.columns),
}
with open(MODEL_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ── 5. SHAP ───────────────────────────────────────────────────────────────────
print("[5/5] Computing SHAP values (sample of 2000 rows) ...")
sample      = X_test.sample(2000, random_state=42)
explainer   = shap.TreeExplainer(model)
shap_values = explainer(sample)

# Beeswarm summary plot
plt.figure()
shap.summary_plot(shap_values, sample, show=False, max_display=20)
plt.tight_layout()
plt.savefig(OUT_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"      Saved → {OUT_DIR}/shap_summary.png")

# Save model and explainer
joblib.dump(model,    MODEL_DIR / 'xgb_model.pkl')
joblib.dump(explainer, MODEL_DIR / 'shap_explainer.pkl')
print(f"      Saved → {MODEL_DIR}/xgb_model.pkl")
print(f"      Saved → {MODEL_DIR}/shap_explainer.pkl")

print("\n✅ Training complete.")