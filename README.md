Here’s a **clean, polished, recruiter-ready `README.md`** — properly formatted, consistent, and aligned with your actual project.

---

# 🏦 Loan Default Risk Predictor

A full end-to-end **Machine Learning system** that predicts whether a loan should be **approved or rejected** based on applicant financial data.

This project simulates a **real-world credit risk model used by banks and fintech companies**.

---

## 🚀 Live Demo

> *(Add your Streamlit link here after deployment)*

---

## 📌 Problem Statement

Financial institutions must assess whether a borrower is likely to **default on a loan** before approving it.

This project builds a predictive system using historical Lending Club data to:

* Estimate **default probability**
* Make **approval/rejection decisions**
* Provide **risk-based interpretation**

---

## 📊 Dataset

* **Source:** Lending Club Dataset (Kaggle)
* **Size:** ~1.6 million records
* **Type:** Real-world financial data

### Key Features:

* Loan amount, interest rate, installment
* Annual income, employment length
* Debt-to-income ratio (DTI)
* Credit history

---

## ⚙️ Tech Stack

* **Python**
* **Pandas, NumPy** (data processing)
* **Scikit-learn** (evaluation & utilities)
* **XGBoost** (model)
* **Streamlit** (web app)
* **SHAP** *(optional upgrade)*

---

## 🧠 ML Pipeline

### 1️⃣ Data Preprocessing

* Missing value imputation (median / "Unknown")
* Feature engineering:

  * Credit history length
  * Loan-to-income ratio
* One-hot encoding for categorical variables
* Memory optimization for large dataset

---

### 2️⃣ Model Training

* Algorithm: **XGBoost Classifier**
* Stratified train-test split
* Handling class imbalance using:

  ```
  scale_pos_weight
  ```

---

### 3️⃣ Evaluation

* Precision, Recall, F1-score
* ROC-AUC Score
* Confusion Matrix
* Threshold tuning based on **business profit logic**

---

### 4️⃣ Prediction System

* Accepts user input via UI
* Outputs:

  * Default probability
  * Decision (Approve / Reject)
  * Risk category

---

## 📊 Model Performance

* **ROC-AUC:** ~0.75–0.80
* Handles imbalanced data effectively
* Optimized for **financial risk decisions**

---

## 💡 Decision Logic

Instead of using a default threshold (0.5), the model uses a **custom threshold (~0.22)**:

* `< 0.22` → ✅ Approved
* `0.22 – 0.40` → ⚠️ Medium Risk
* `> 0.40` → ❌ Rejected

This reflects **real-world banking strategy**.

---

## 🌐 Streamlit Dashboard

Features:

* Interactive sliders & inputs
* Real-time prediction
* Risk interpretation
* Clean and intuitive UI

---

## 📁 Project Structure

```
lending-club-ml/
│
├── data/
│   ├── accepted_2007_to_2018Q4.csv.gz
│   └── ml_ready_lending_club.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── app/
│   └── app.py
│
├── models/
│   └── xgboost_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Run Preprocessing

```bash
python src/preprocess.py
```

---

### 3️⃣ Train Model

```bash
python src/train.py
```

---

### 4️⃣ Launch App

```bash
python -m streamlit run app/app.py
```

---

## 🧠 Key Learnings

* Handling **large-scale real-world datasets**
* Managing **imbalanced classification problems**
* Building **end-to-end ML pipelines**
* Designing **user-facing ML applications**
* Applying **business-driven decision thresholds**

---

## 🚀 Future Improvements

* SHAP explainability integration
* Hyperparameter tuning
* API deployment (FastAPI)
* Cloud deployment (Streamlit / AWS)

---

## 👨‍💻 Authors

**PaariKarthikeyan**
**Mithunsurya-Kumarasamy**

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub — it helps!
