import pytest
from fastapi.testclient import TestClient
from arabic_sentiment_mlops.api import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_single_prediction_endpoint(client):
    payload = {"text": "خدمة العملاء ممتازة وسريعة الاستجابة"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert data["label"] in ["Positive", "Negative", "Neutral"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_empty_text_validation(client):
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    # Pydantic validation error expected for min_length=1
    assert response.status_code == 422


def test_batch_prediction_endpoint(client):
    payload = {
        "texts": [
            "المنتج ممتاز جداً",
            "تجربة سيئة للغاية ولن أكررها"
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["label"] in ["Positive", "Negative", "Neutral"]
        assert 0.0 <= pred["confidence"] <= 1.0