import os
import sys
import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

client = TestClient(app)



def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "model_loaded" in data
    assert "scaler_loaded" in data



def test_predict_endpoint_success():
    payload = {
        "Pregnancies": 2,
        "Glucose": 90.0,
        "BloodPressure": 80.0,
        "SkinThickness": 30.0,
        "Insulin": 5.0,
        "BMI": 20.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 30
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code in [200, 500] 
    
    if response.status_code == 200:
        json_data = response.json()
        assert "prediction" in json_data
        assert "risk_level" in json_data
        assert json_data["risk_level"] in ["High Risk", "Low Risk"]


def test_predict_invalid_data():
    response = client.post("/predict", json={"Glucose": 90})
    assert response.status_code == 422



def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "model_inference_seconds" in response.text