from fastapi.testclient import TestClient

from api.main import app, shap


def test_assessment_returns_safety_fields():
    response = TestClient(app).post("/v1/assess", json={"age": 32, "bmi": 28, "menstrual_irregularity": 1, "chronic_pain_level": 7, "hormone_level_abnormality": 0, "infertility": 0})
    assert response.status_code == 200
    body = response.json()
    assert "disclaimer" in body
    assert "trust_score" not in body
    assert 0 <= body["model_agreement_score"] <= 1
    assert len(body["research_prediction_range"]) == 2
    assert "not clinical certainty" in body["agreement_note"]
    assert len(body["top_factors"]) == 3
    assert body["explanation_method"] in {"shap_permutation", "baseline_sensitivity_fallback"}


def test_assessment_rejects_out_of_range_binary_input():
    response = TestClient(app).post("/v1/assess", json={"age": 32, "bmi": 28, "menstrual_irregularity": 2, "chronic_pain_level": 7, "hormone_level_abnormality": 0, "infertility": 0})
    assert response.status_code == 422


def test_health_exposes_non_clinical_validation_status():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    status = response.json()["validation_status"]
    assert status["label_provenance_verified"] is False
    assert status["external_clinical_validation"] is False
    assert status["deployment_status"] == "research_only_not_clinically_validated"


def test_counterfactual_rejects_unknown_feature():
    patient = {"age": 32, "bmi": 28, "menstrual_irregularity": 1, "chronic_pain_level": 7, "hormone_level_abnormality": 0, "infertility": 0}
    response = TestClient(app).post("/v1/counterfactual", json={"patient": patient, "feature": "unknown", "delta": 1})
    assert response.status_code == 422


def test_trajectory_returns_identifier_free_research_record():
    payload = {"observations": [
        {"observed_on": "2026-07-01", "cycle_day": 1, "bleeding_flow": 3, "pain_level": 7, "fatigue_level": 6, "sleep_hours": 6.5},
        {"observed_on": "2026-07-02", "cycle_day": 2, "bleeding_flow": 2, "pain_level": 5, "fatigue_level": 4, "sleep_hours": 7.5, "stress_level": 5},
    ]}
    response = TestClient(app).post("/v1/trajectory", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.0.0"
    assert body["summary"]["observation_window"]["daily_records"] == 2
    assert body["summary"]["measurements"]["pain_level"]["change_per_week"] == -14.0
    assert "not a diagnosis" in body["disclaimer"]


def test_trajectory_rejects_identifiers_and_duplicate_dates():
    payload = {"observations": [
        {"observed_on": "2026-07-01", "pain_level": 3, "name": "not allowed"},
        {"observed_on": "2026-07-01", "pain_level": 4},
    ]}
    assert TestClient(app).post("/v1/trajectory", json=payload).status_code == 422


def test_global_explanation_reports_dependency_status():
    response = TestClient(app).get("/v1/explain/global")
    if shap is None:
        assert response.status_code == 503
        assert "SHAP is unavailable" in response.json()["detail"]
    else:
        assert response.status_code == 200
        assert len(response.json()["global_feature_importance"]) == 6
