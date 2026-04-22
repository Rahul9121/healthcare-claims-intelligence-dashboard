import numpy as np
from fastapi.testclient import TestClient

from src.api import main


class DummyModel:
    def predict_proba(self, frame):
        return np.array([[0.15, 0.85] for _ in range(len(frame))])


def _payload():
    return {
        "age": 68,
        "chronic_conditions_count": 3,
        "risk_score_baseline": 0.71,
        "length_of_stay": 5,
        "prior_ed_visits_6m": 2,
        "medication_adherence": 0.62,
        "allowed_amount": 9800,
        "paid_amount": 7450,
        "followup_within_7d": 0,
        "admission_type": "Emergency",
        "diagnosis_group": "Cardiology",
        "provider_specialty": "Hospitalist",
        "plan_type": "Silver",
    }


def test_predict_endpoint_returns_probability(monkeypatch):
    monkeypatch.setattr(main, "_model", DummyModel())
    client = TestClient(main.app)
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200

    data = response.json()
    assert data["risk_probability"] == 0.85
    assert data["risk_bucket"] == "high"


def test_batch_predict_endpoint(monkeypatch):
    monkeypatch.setattr(main, "_model", DummyModel())
    client = TestClient(main.app)
    response = client.post("/predict/batch", json={"records": [_payload(), _payload()]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["risk_bucket"] == "high"
