"""FastAPI service for the TrustCheck research prototype."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.getenv("TRUSTCHECK_ARTIFACT_DIR", ROOT / "artifacts"))
REPORT_PATH = ROOT / "data" / "DATA_DEBT_REPORT.md"
summary = json.loads((ARTIFACT_DIR / "training_summary.json").read_text(encoding="utf-8"))
MODEL_A = joblib.load(ARTIFACT_DIR / "model_a.joblib")
MODEL_B = joblib.load(ARTIFACT_DIR / "model_b.joblib")
FEATURES = summary["features"]

app = FastAPI(title="TrustCheck API", version="0.1.0", description="Research-only endometriosis risk-pattern prototype.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class Patient(BaseModel):
    age: int = Field(ge=18, le=49)
    bmi: float = Field(ge=10, le=80)
    menstrual_irregularity: int = Field(ge=0, le=1)
    chronic_pain_level: float = Field(ge=0, le=10)
    hormone_level_abnormality: int = Field(ge=0, le=1)
    infertility: int = Field(ge=0, le=1)

    def row(self) -> pd.DataFrame:
        return pd.DataFrame([[self.age, self.bmi, self.menstrual_irregularity, self.chronic_pain_level, self.hormone_level_abnormality, self.infertility]], columns=FEATURES)

class CounterfactualRequest(BaseModel):
    patient: Patient
    feature: str
    delta: float

def explain(row: pd.DataFrame, base_risk: float) -> list[dict]:
    """Transparent one-at-a-time perturbation explanation, ordered by impact."""
    baseline = {"Age": 33.65, "BMI": 27.0, "Menstrual_Irregularity": 0, "Chronic_Pain_Level": 5.0, "Hormone_Level_Abnormality": 0, "Infertility": 0}
    factors = []
    for feature in FEATURES:
        altered = row.astype(float).copy(); altered.loc[0, feature] = baseline[feature]
        impact = base_risk - float(MODEL_B.predict_proba(altered)[0, 1])
        factors.append({"feature": feature, "contribution": round(impact, 3), "direction": "raises" if impact >= 0 else "lowers"})
    return sorted(factors, key=lambda item: abs(item["contribution"]), reverse=True)[:3]

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": True}

@app.post("/v1/assess")
def assess(patient: Patient) -> dict:
    row = patient.row()
    risk_a = float(MODEL_A.predict_proba(row)[0, 1])
    risk_b = float(MODEL_B.predict_proba(row)[0, 1])
    divergence = abs(risk_a - risk_b)
    return {"risk_score": round(risk_b, 3), "raw_risk_score": round(risk_a, 3), "trust_score": round(max(0.0, 1 - divergence / 0.5), 2),
            "confidence_range": [round(max(0, risk_b - summary["interval_radius"]), 3), round(min(1, risk_b + summary["interval_radius"]), 3)],
            "confidence_note": "A research calibration range, not an individual guarantee.", "top_factors": explain(row, risk_b), "disclaimer": summary["disclaimer"]}

@app.post("/v1/counterfactual")
def counterfactual(request: CounterfactualRequest) -> dict:
    mapping = {"age": "Age", "bmi": "BMI", "menstrual_irregularity": "Menstrual_Irregularity", "chronic_pain_level": "Chronic_Pain_Level", "hormone_level_abnormality": "Hormone_Level_Abnormality", "infertility": "Infertility"}
    if request.feature not in mapping:
        raise HTTPException(422, detail=f"feature must be one of: {', '.join(mapping)}")
    row = request.patient.row(); column = mapping[request.feature]
    row.loc[0, column] += request.delta
    if column == "Chronic_Pain_Level": row.loc[0, column] = np.clip(row.loc[0, column], 0, 10)
    risk = float(MODEL_B.predict_proba(row)[0, 1])
    return {"feature": request.feature, "delta": request.delta, "adjusted_risk_score": round(risk, 3), "disclaimer": summary["disclaimer"]}

@app.get("/v1/data-debt-report")
def data_debt_report() -> dict:
    return {"report_markdown": REPORT_PATH.read_text(encoding="utf-8"), "disclaimer": summary["disclaimer"]}
