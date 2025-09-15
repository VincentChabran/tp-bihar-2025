from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_existing_date():
    response = client.get("/predict?date=2024-01-01")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_missing_date():
    response = client.get("/predict?date=2030-01-01")
    assert response.status_code == 404
    assert "message" in response.json()["detail"]
