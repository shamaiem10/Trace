"""FastAPI service for the Trace research prototype."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
try:
    import shap
except ImportError:  # Allows the API to start with an explicit fallback message.
    shap = None
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]


def load_local_env() -> None:
    """Load simple project-local environment settings without overriding deployment secrets."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()
ARTIFACT_DIR = Path(os.getenv("TRACE_ARTIFACT_DIR", ROOT / "artifacts"))
REPORT_PATH = ROOT / "data" / "DATA_DEBT_REPORT.md"
DEMO_PATH = ROOT / "app" / "index.html"
summary = json.loads((ARTIFACT_DIR / "training_summary.json").read_text(encoding="utf-8"))
required_sklearn = summary.get("reproducibility", {}).get("scikit_learn")
if required_sklearn and sklearn.__version__ != required_sklearn:
    raise RuntimeError(f"Trace artifacts require scikit-learn {required_sklearn}; installed {sklearn.__version__}. Install the pinned requirements and regenerate artifacts together.")
required_numpy = summary.get("reproducibility", {}).get("numpy")
if required_numpy and np.__version__ != required_numpy:
    raise RuntimeError(f"Trace artifacts require NumPy {required_numpy}; installed {np.__version__}. Install the pinned requirements and regenerate artifacts together.")
MODEL_A = joblib.load(ARTIFACT_DIR / "model_a.joblib")
MODEL_B = joblib.load(ARTIFACT_DIR / "model_b.joblib")
FEATURES = summary["features"]
SHAP_BACKGROUND = pd.read_csv(ARTIFACT_DIR / summary.get("explainability", {}).get("background_artifact", "shap_background.csv"))

app = FastAPI(title="Trace API", version="0.1.0", description="Research-only endometriosis risk-pattern prototype; not clinically validated.")
app.mount("/app", StaticFiles(directory=ROOT / "app"), name="app")
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


TRAJECTORY_MEASURES = ("bleeding_flow", "pain_level", "fatigue_level", "migraine_level", "brain_fog_level", "sleep_hours", "stress_level")


class DailyObservation(BaseModel):
    """A de-identified daily observation for the open Trace trajectory schema."""
    model_config = ConfigDict(extra="forbid")

    observed_on: date
    cycle_day: int | None = Field(default=None, ge=1, le=60)
    bleeding_flow: int | None = Field(default=None, ge=0, le=4)
    pain_level: float | None = Field(default=None, ge=0, le=10)
    fatigue_level: float | None = Field(default=None, ge=0, le=10)
    migraine_level: float | None = Field(default=None, ge=0, le=10)
    brain_fog_level: float | None = Field(default=None, ge=0, le=10)
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    stress_level: float | None = Field(default=None, ge=0, le=10)

    @model_validator(mode="after")
    def requires_a_measurement(self):
        if all(getattr(self, measure) is None for measure in TRAJECTORY_MEASURES):
            raise ValueError("each observation must include at least one symptom, flow, sleep, or stress measurement")
        return self


class TrajectoryRequest(BaseModel):
    """Identifier-free longitudinal data intended for research, not care decisions."""
    model_config = ConfigDict(extra="forbid")
    observations: list[DailyObservation] = Field(min_length=2, max_length=120)

    @model_validator(mode="after")
    def dates_are_unique(self):
        dates = [observation.observed_on for observation in self.observations]
        if len(dates) != len(set(dates)):
            raise ValueError("observations must have unique dates")
        return self


class TrajectoryExplanationRequest(BaseModel):
    """Opt-in request containing only an already-aggregated trajectory summary."""
    model_config = ConfigDict(extra="forbid")
    summary: dict

    @model_validator(mode="after")
    def rejects_raw_records_and_identifiers(self):
        serialized = json.dumps(self.summary).lower()
        forbidden = ("record", "observed_on", "name", "email", "phone", "address", "medical_record")
        if any(f'"{field}"' in serialized for field in forbidden):
            raise ValueError("send only the aggregated summary; raw observations and identifiers are not accepted")
        return self

def fallback_explain(row: pd.DataFrame, base_risk: float) -> list[dict]:
    """Temporary dependency-missing fallback; SHAP is used when installed."""
    baseline = summary["explanation_reference"]
    factors = []
    for feature in FEATURES:
        altered = row.astype(float).copy(); altered.loc[0, feature] = baseline[feature]
        impact = base_risk - float(MODEL_B.predict_proba(altered)[0, 1])
        factors.append({"feature": feature, "contribution": round(impact, 3), "direction": "above_reference" if impact >= 0 else "below_reference", "reference_value": baseline[feature]})
    return sorted(factors, key=lambda item: abs(item["contribution"]), reverse=True)[:3]


def predict_weighted_probability(rows: np.ndarray) -> np.ndarray:
    """Prediction callable used by SHAP, returning the positive-class probability."""
    return MODEL_B.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]


def summarize_measure(values: pd.Series, days: pd.Series) -> dict:
    """Describe a longitudinal signal without asserting a clinical interpretation."""
    available = values.notna()
    result = {"observed_count": int(available.sum()), "missing_rate": round(float(1 - available.mean()), 3)}
    if not available.any():
        return result
    observed_values = values[available].astype(float)
    result["mean"] = round(float(observed_values.mean()), 2)
    result["min"] = round(float(observed_values.min()), 2)
    result["max"] = round(float(observed_values.max()), 2)
    if len(observed_values) >= 2:
        slope_per_day = np.polyfit(days[available], observed_values, 1)[0]
        result["change_per_week"] = round(float(slope_per_day * 7), 2)
    return result


def generate_research_narrative(summary_payload: dict) -> str:
    """Request a bounded, non-clinical explanation from the OpenAI Responses API."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("TRACE_OPENAI_MODEL") or "gpt-4.1-mini"
    if not api_key:
        raise HTTPException(503, detail="AI explanation is not configured. Add OPENAI_API_KEY to the private .env file.")
    instructions = (
        "You explain an aggregated, de-identified menstrual-health research trajectory. "
        "Use plain, calm language. Describe only patterns explicitly present in the supplied summary, including missingness. "
        "Do not diagnose, assess risk, recommend treatment, suggest medication, infer a condition, or make claims about ovulation. "
        "State that this is a research summary and encourage professional care only for urgent, severe, or persistent symptoms. "
        "Return 2-3 short paragraphs with no markdown headings, no bullet points, and no numerical claims beyond supplied values."
    )
    payload = json.dumps({"model": model, "instructions": instructions,
                          "input": f"Aggregated Trace trajectory summary:\n{json.dumps(summary_payload, sort_keys=True)}"}).encode("utf-8")
    request = urllib.request.Request("https://api.openai.com/v1/responses", data=payload, method="POST",
                                     headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise HTTPException(502, detail=f"AI explanation request failed with status {error.code}.") from error
    except urllib.error.URLError as error:
        raise HTTPException(502, detail="AI explanation service could not be reached.") from error
    narrative = body.get("output_text")
    if not narrative:
        text_parts = []
        for item in body.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    text_parts.append(content["text"])
        narrative = "\n\n".join(text_parts)
    if not narrative:
        raise HTTPException(502, detail="AI explanation service returned no readable text.")
    return narrative.strip()


@lru_cache(maxsize=1)
def shap_explainer():
    if shap is None:
        return None
    return shap.Explainer(predict_weighted_probability, SHAP_BACKGROUND, feature_names=FEATURES, algorithm="permutation")


def shap_factors(row: pd.DataFrame) -> tuple[list[dict], float | None, str]:
    """Return local SHAP values in probability-score units for one assessment."""
    explainer = shap_explainer()
    if explainer is None:
        risk = float(MODEL_B.predict_proba(row)[0, 1])
        return fallback_explain(row, risk), None, "baseline_sensitivity_fallback"
    explanation = explainer(row, max_evals=2 * len(FEATURES) + 1)
    values = np.asarray(explanation.values)[0]
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    factors = [{"feature": feature, "value": float(row.iloc[0][feature]), "shap_value": round(float(value), 4),
                "direction": "raises_model_score" if value >= 0 else "lowers_model_score"}
               for feature, value in zip(FEATURES, values, strict=True)]
    return sorted(factors, key=lambda item: abs(item["shap_value"]), reverse=True)[:3], base_value, "shap_permutation"


@lru_cache(maxsize=1)
def global_shap_importance() -> list[dict]:
    explainer = shap_explainer()
    if explainer is None:
        raise HTTPException(503, detail="SHAP is unavailable. Install the pinned project requirements to enable SHAP explanations.")
    explanation = explainer(SHAP_BACKGROUND.iloc[:32], max_evals=2 * len(FEATURES) + 1)
    mean_abs = np.abs(np.asarray(explanation.values)).mean(axis=0)
    return sorted([{"feature": feature, "mean_absolute_shap_value": round(float(value), 4)} for feature, value in zip(FEATURES, mean_abs, strict=True)], key=lambda item: item["mean_absolute_shap_value"], reverse=True)

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": True, "validation_status": summary["validation_status"]}

@app.get("/", include_in_schema=False)
def demo() -> FileResponse:
    """Serve the lightweight demonstration interface with the API."""
    return FileResponse(DEMO_PATH)

@app.post("/v1/assess")
def assess(patient: Patient) -> dict:
    row = patient.row()
    risk_a = float(MODEL_A.predict_proba(row)[0, 1])
    risk_b = float(MODEL_B.predict_proba(row)[0, 1])
    divergence = abs(risk_a - risk_b)
    factors, shap_base_value, explanation_method = shap_factors(row)
    explanation_note = "Local SHAP values show how each input moves this model's probability score relative to its background dataset. They are not causal or medical explanations."
    if explanation_method != "shap_permutation":
        explanation_note = "SHAP is not installed in this runtime; these are baseline-sensitivity fallback values. Install the pinned requirements to enable SHAP. They are not causal or medical explanations."
    return {"risk_score": round(risk_b, 3), "raw_model_score": round(risk_a, 3),
            "model_agreement_score": round(max(0.0, 1 - divergence / 0.5), 2),
            "agreement_note": "Agreement between the raw and calibration-weighted research models; it is not clinical certainty or validation.",
            "research_prediction_range": [round(max(0, risk_b - summary["research_interval_radius"]), 3), round(min(1, risk_b + summary["research_interval_radius"]), 3)],
            "research_prediction_range_note": f"A held-out {int(summary['research_interval_coverage'] * 100)}% residual-error band. It may be broad and is not an individual prognosis or guarantee.",
            "top_factors": factors, "explanation_method": explanation_method, "shap_base_value": None if shap_base_value is None else round(shap_base_value, 4), "explanation_note": explanation_note,
            "model_metadata": {"schema_version": summary["schema_version"], "scikit_learn": summary["reproducibility"]["scikit_learn"], "validation_status": summary["validation_status"]}, "disclaimer": summary["disclaimer"]}


@app.get("/v1/explain/global")
def explain_global() -> dict:
    return {"method": "SHAP permutation explainer", "background_rows": int(len(SHAP_BACKGROUND)), "global_feature_importance": global_shap_importance(),
            "note": "Mean absolute SHAP values summarize model behavior across the saved background sample. They are not clinical importance or causal effects.", "disclaimer": summary["disclaimer"]}


@app.get("/v1/trajectory/schema")
def trajectory_schema() -> dict:
    """Expose the reusable, identifier-free daily-observation contract."""
    return {"schema_name": "Trace trajectory record", "schema_version": "1.0.0", "unit": "one de-identified person-day",
            "fields": {"observed_on": "ISO-8601 calendar date", "cycle_day": "optional self-reported cycle day (1-60)",
                       "bleeding_flow": "optional ordinal flow level (0-4)", "pain_level": "optional self-reported level (0-10)",
                       "fatigue_level": "optional self-reported level (0-10)", "migraine_level": "optional self-reported level (0-10)",
                       "brain_fog_level": "optional self-reported level (0-10)", "sleep_hours": "optional hours (0-24)",
                       "stress_level": "optional self-reported level (0-10)"},
            "privacy_note": "Do not submit names, contact details, precise location, or medical-record identifiers.",
            "use_note": "This schema supports research and dataset interoperability; it does not diagnose or recommend treatment."}


@app.post("/v1/trajectory")
def trajectory(request: TrajectoryRequest) -> dict:
    """Create a transparent, non-diagnostic summary of a daily trajectory record."""
    rows = sorted((observation.model_dump(mode="json") for observation in request.observations), key=lambda row: row["observed_on"])
    frame = pd.DataFrame(rows)
    start = pd.Timestamp(frame.loc[0, "observed_on"])
    elapsed_days = (pd.to_datetime(frame["observed_on"]) - start).dt.days
    measures = {measure: summarize_measure(frame[measure], elapsed_days) for measure in TRAJECTORY_MEASURES}
    cycle_day_count = int(frame["cycle_day"].notna().sum())
    return {"schema_name": "Trace trajectory record", "schema_version": "1.0.0", "record": rows,
            "summary": {"observation_window": {"start": rows[0]["observed_on"], "end": rows[-1]["observed_on"],
                                                   "calendar_days": int(elapsed_days.iloc[-1] + 1), "daily_records": len(rows)},
                        "cycle_context": {"cycle_day_observed_count": cycle_day_count,
                                          "cycle_start_markers": int((frame["cycle_day"] == 1).sum()),
                                          "note": "Cycle fields are self-reported context, not confirmation of ovulation or a medical condition."},
                        "measurements": measures,
                        "data_quality_note": "Missingness is reported per measure. Trend estimates need repeated observations and are descriptive only."},
            "disclaimer": "Research trajectory only. This summary is not a diagnosis, prognosis, or medical advice."}


@app.post("/v1/trajectory/explain")
def explain_trajectory(request: TrajectoryExplanationRequest) -> dict:
    """Generate an opt-in, bounded explanation of an aggregated trajectory summary."""
    return {"narrative": generate_research_narrative(request.summary), "method": "openai_responses_api",
            "privacy_note": "Only the aggregated trajectory summary was sent for this explanation; raw daily observations were excluded.",
            "disclaimer": "AI-generated research explanation only. It is not a diagnosis, prognosis, treatment recommendation, or medical advice."}

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
