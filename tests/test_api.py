from fastapi.testclient import TestClient

from api.main import app


def test_assessment_returns_safety_fields():
    response = TestClient(app).post("/v1/assess", json={"age": 32, "bmi": 28, "menstrual_irregularity": 1, "chronic_pain_level": 7, "hormone_level_abnormality": 0, "infertility": 0})
    assert response.status_code == 200
    body = response.json()
    assert "disclaimer" in body
    assert 0 <= body["trust_score"] <= 1
    assert len(body["top_factors"]) == 3
