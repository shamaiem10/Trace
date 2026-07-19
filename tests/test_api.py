from fastapi.testclient import TestClient

from api.main import app, shap


def test_assessment_returns_safety_fields():
    response = TestClient(app).post("/v1/assess", json={"age": 32, "bmi": 28, "menstrual_irregularity": 1, "chronic_pain_level": 7, "hormone_level_abnormality": 0, "infertility": 0})
    assert response.status_code == 200
    body = response.json()
    assert "disclaimer" in body
    assert "risk_score" not in body
    assert 0 <= body["calibrated_research_score"] <= 1
    assert 0 <= body["calibration_shift"] <= 1
    assert len(body["research_score_range"]) == 2
    assert "not clinical certainty" in body["calibration_shift_note"]
    assert len(body["top_factors"]) == 3
    assert body["explanation_method"] in {"shap_permutation", "baseline_sensitivity_fallback"}


def test_assessment_rejects_out_of_range_binary_input():
    response = TestClient(app).post("/v1/assess", json={"age": 32, "bmi": 28, "menstrual_irregularity": 2, "chronic_pain_level": 7, "hormone_level_abnormality": 0, "infertility": 0})
    assert response.status_code == 422


def test_assessment_rejects_unapproved_identifiers():
    response = TestClient(app).post("/v1/assess", json={"age": 32, "bmi": 28, "menstrual_irregularity": 1, "chronic_pain_level": 7, "hormone_level_abnormality": 0, "infertility": 0, "name": "not allowed"})
    assert response.status_code == 422


def test_health_exposes_non_clinical_validation_status():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    status = response.json()["validation_status"]
    assert status["label_provenance_verified"] is False
    assert status["external_clinical_validation"] is False
    assert status["deployment_status"] == "research_only_not_clinically_validated"


def test_model_card_and_representation_audit_expose_generated_artifacts():
    client = TestClient(app)
    card = client.get("/v1/model-card")
    audit = client.get("/v1/representation-audit")
    assert card.status_code == 200
    assert card.json()["source_data"]["rows"] == 10_000
    assert card.json()["reference_population"]["rows"] == 1206
    assert audit.status_code == 200
    assert len(audit.json()["strata"]) == 8


def test_demo_is_wired_to_live_research_model_backend():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "fetch('/v1/assess'" in response.text
    assert "fetch('/v1/explain/global'" in response.text
    assert "Model result, with context." in response.text


def test_demo_serves_the_local_hero_artwork():
    response = TestClient(app).get("/app/assets/traceimage.jpeg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_counterfactual_rejects_unknown_feature():
    profile = {"age": 32, "bmi": 28, "menstrual_irregularity": 1, "chronic_pain_level": 7, "hormone_level_abnormality": 0, "infertility": 0}
    response = TestClient(app).post("/v1/counterfactual", json={"profile": profile, "feature": "unknown", "delta": 1})
    assert response.status_code == 422


def test_counterfactual_rejects_values_outside_feature_domain():
    profile = {"age": 32, "bmi": 28, "menstrual_irregularity": 1, "chronic_pain_level": 7, "hormone_level_abnormality": 0, "infertility": 0}
    response = TestClient(app).post("/v1/counterfactual", json={"profile": profile, "feature": "menstrual_irregularity", "delta": 0.5})
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


def test_ai_explanation_rejects_raw_daily_records_before_any_provider_call():
    response = TestClient(app).post("/v1/trajectory/explain", json={"summary": {"record": [{"observed_on": "2026-07-01"}]}})
    assert response.status_code == 422


def test_ai_explanation_requires_explicit_server_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TRACE_OPENAI_MODEL", raising=False)
    response = TestClient(app).post("/v1/trajectory/explain", json={"summary": {"observation_window": {"daily_records": 2}}})
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_ai_explanation_uses_default_model_when_override_is_blank(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRACE_OPENAI_MODEL", "")
    from api.main import generate_research_narrative
    import urllib.request

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"output_text": "Research-only narrative."}'

    def fake_urlopen(request, timeout):
        captured.update(__import__("json").loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert generate_research_narrative({"measurements": {}}) == "Research-only narrative."
    assert captured["model"] == "gpt-4.1-mini"


def test_ai_explanation_reads_nested_responses_api_output(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRACE_OPENAI_MODEL", "test-model")
    from api.main import generate_research_narrative
    import urllib.request

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"output": [{"type": "message", "content": [{"type": "output_text", "text": "First paragraph."}, {"type": "output_text", "text": "Second paragraph."}]}]}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    assert generate_research_narrative({"measurements": {}}) == "First paragraph.\n\nSecond paragraph."


def test_global_explanation_reports_dependency_status():
    response = TestClient(app).get("/v1/explain/global")
    if shap is None:
        assert response.status_code == 503
        assert "SHAP is unavailable" in response.json()["detail"]
    else:
        assert response.status_code == 200
        assert len(response.json()["global_feature_importance"]) == 6
