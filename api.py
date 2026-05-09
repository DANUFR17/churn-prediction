"""
api.py — FastAPI backend for Customer Churn Prediction
Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal, List
import joblib
import numpy as np
import pandas as pd
import shap
import os

# ── App init ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using XGBoost + SHAP explainability.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ─────────────────────────────────────────────────────────────
MODEL_PATH = "models/xgb_churn_pipeline.pkl"

@app.on_event("startup")
def load_model():
    global pipeline, explainer
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run 02-modeling-shap.ipynb first.")
    pipeline = joblib.load(MODEL_PATH)
    # Build SHAP explainer once at startup
    preprocessor = pipeline.named_steps['preprocessor']
    classifier   = pipeline.named_steps['classifier']
    # Dummy data to init explainer
    explainer = shap.TreeExplainer(classifier)
    print("Model & SHAP explainer loaded ✓")

# ── Schema ─────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    # Demographics
    gender:           Literal["Male", "Female"]
    SeniorCitizen:    Literal[0, 1]                  = Field(..., description="1 if senior citizen")
    Partner:          Literal["Yes", "No"]
    Dependents:       Literal["Yes", "No"]

    # Service
    tenure:           int   = Field(..., ge=0, le=72,  description="Months with company")
    PhoneService:     Literal["Yes", "No"]
    MultipleLines:    Literal["Yes", "No", "No phone service"]
    InternetService:  Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity:   Literal["Yes", "No", "No internet service"]
    OnlineBackup:     Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport:      Literal["Yes", "No", "No internet service"]
    StreamingTV:      Literal["Yes", "No", "No internet service"]
    StreamingMovies:  Literal["Yes", "No", "No internet service"]

    # Billing
    Contract:         Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod:    Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges:   float = Field(..., ge=0, le=200)
    TotalCharges:     float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.7, "TotalCharges": 151.65
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  bool
    risk_level:        Literal["Low", "Medium", "High"]
    shap_top_features: List[dict]   # [{feature, shap_value, direction}]


class BulkInput(BaseModel):
    customers: List[CustomerInput]


class BulkResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: dict


# ── Helpers ────────────────────────────────────────────────────────────────
def build_df(customer: CustomerInput) -> pd.DataFrame:
    data = customer.dict()
    df = pd.DataFrame([data])

    # Feature engineering — must match notebook
    df['has_support_services'] = (
        (df['OnlineSecurity'] == 'Yes') | (df['TechSupport'] == 'Yes')
    ).astype(int)
    df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['charge_per_tenure']  = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['num_streaming']      = (
        (df['StreamingTV'] == 'Yes').astype(int) +
        (df['StreamingMovies'] == 'Yes').astype(int)
    )
    return df


def risk_label(prob: float) -> str:
    if prob >= 0.6:  return "High"
    if prob >= 0.35: return "Medium"
    return "Low"


def get_shap_top(df_input: pd.DataFrame, n: int = 8) -> List[dict]:
    preprocessor = pipeline.named_steps['preprocessor']
    classifier   = pipeline.named_steps['classifier']

    num_cols = preprocessor.named_transformers_['num'].feature_names_in_.tolist()
    cat_cols = preprocessor.named_transformers_['cat'].feature_names_in_.tolist()
    ohe_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    all_features = num_cols + ohe_names

    X_transformed = preprocessor.transform(df_input)
    X_df = pd.DataFrame(X_transformed, columns=all_features)

    sv = explainer(X_df)
    vals = sv.values[0]

    top_idx = np.argsort(np.abs(vals))[::-1][:n]
    return [
        {
            "feature":    all_features[i],
            "shap_value": round(float(vals[i]), 4),
            "direction":  "increases churn risk" if vals[i] > 0 else "decreases churn risk"
        }
        for i in top_idx
    ]


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        df = build_df(customer)
        prob = float(pipeline.predict_proba(df)[0][1])
        shap_features = get_shap_top(df)
        return PredictionResponse(
            churn_probability = round(prob, 4),
            churn_prediction  = prob >= 0.5,
            risk_level        = risk_label(prob),
            shap_top_features = shap_features
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/bulk", response_model=BulkResponse)
def predict_bulk(payload: BulkInput):
    if len(payload.customers) > 500:
        raise HTTPException(status_code=400, detail="Max 500 customers per request.")
    try:
        results = []
        for customer in payload.customers:
            df   = build_df(customer)
            prob = float(pipeline.predict_proba(df)[0][1])
            results.append(PredictionResponse(
                churn_probability = round(prob, 4),
                churn_prediction  = prob >= 0.5,
                risk_level        = risk_label(prob),
                shap_top_features = get_shap_top(df, n=5)
            ))

        probs = [r.churn_probability for r in results]
        summary = {
            "total":          len(results),
            "predicted_churn": sum(r.churn_prediction for r in results),
            "high_risk":      sum(r.risk_level == "High" for r in results),
            "avg_probability": round(float(np.mean(probs)), 4),
        }
        return BulkResponse(predictions=results, summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
