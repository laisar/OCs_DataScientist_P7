from fastapi.testclient import TestClient
from fastapi_predict import app

client = TestClient(app)

def test_root():
    response = client.get("/api/test")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}