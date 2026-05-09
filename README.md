# 📉 Customer Churn Prediction

> **End-to-end ML project:** EDA → Feature Engineering → XGBoost → SHAP Explainability → FastAPI + Streamlit

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)

---

## 🎯 Project Overview

Predicts which telecom customers are likely to churn using machine learning, with full **explainability via SHAP** — answering not just *who* will churn, but *why*.

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features

---

## 📁 Project Structure

```
telco-churn-project/
│
├── data/
│   ├── raw/                  # data asli dari Kaggle
│   └── processed/            # data hasil cleaning
│
├── notebooks/
│   ├── 01-eda.ipynb
│   └── 02-modeling-shap.ipynb
│
├── plots/                    # semua visualisasi auto-save ke sini
│
├── src/                      # (optional, untuk fase modeling nanti)
│
├── outputs/                  # hasil model / metrics nanti
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔑 Key Results

| Model               | ROC-AUC | F1         | Precision  | Recall     |
| ------------------- | ------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.8484  | 0.6309     | 0.5259     | 0.7886     |
| Random Forest       | 0.8475  | 0.6309     | 0.5213     | **0.7993** |
| XGBoost             | 0.8482  | **0.6330** | **0.5300** | 0.7860     |

> **Key insight:** All three models perform nearly identically on this dataset — a common and realistic outcome on small, structured tabular data (~7k rows). XGBoost edges ahead on F1 and Precision, making it the best choice when **false positive cost matters** (e.g. avoid sending unnecessary retention offers). This result validates that the dataset signal is well-captured even by a linear model — and that more data or richer features would be the next lever to pull, not model complexity.

---

## 💡 Key Findings (EDA + SHAP)

- **Contract type** is the strongest churn predictor — month-to-month customers churn at ~42% vs <5% for 2-year contracts
- **Low tenure** (< 12 months) is the highest-risk segment
- **Fiber optic** internet users churn more than DSL users
- **High monthly charges** + low tenure = highest churn probability
- Having **TechSupport or OnlineSecurity** significantly reduces churn risk

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
```bash
# From Kaggle CLI:
kaggle datasets download blastchar/telco-customer-churn
unzip telco-customer-churn.zip
```

### 3. Run notebooks (in order)
```bash
jupyter notebook 01-eda.ipynb
jupyter notebook 02-modeling-shap.ipynb
```

### 4. Start the API
```bash
uvicorn api:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs
```

### 5. Launch Streamlit app
```bash
python -m streamlit run streamlit_app.py
```

---

## 🖥️ App Features

### Single Customer Mode
- Input customer details via form
- Get churn probability + risk level (Low / Medium / High)
- Animated gauge chart
- **SHAP waterfall** — explains which features drove the prediction

### Bulk Analysis Mode
- Upload CSV of customers
- Batch predictions with summary dashboard
- Distribution charts (histogram + pie)
- Export results as CSV

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | pandas, numpy, matplotlib, seaborn |
| Modeling | scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit, Plotly |
| Persistence | joblib |

---

## 👤 Author

**Muhammad Danu Firjatullah Rachman**  
[GitHub](https://github.com/) · [LinkedIn]([https://linkedin.com/](https://www.linkedin.com/in/muhammad-danu-firjatullah-rachman-740102261/))
