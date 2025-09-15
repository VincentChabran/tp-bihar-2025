from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_combined_valid_range():
    payload = {
        "start_date": "2024-01-01",
        "end_date": "2024-01-02"
    }
    response = client.post("/combined", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json(), list)



def test_combined_empty_range():
    payload = {
        "start_date": "2030-01-01",
        "end_date": "2030-01-02"
    }
    response = client.post("/combined", json=payload)
    assert response.status_code == 404
