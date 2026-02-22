"""API tests for DeepAMR backend."""

import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def _unique_email():
    return f"test-{uuid.uuid4().hex[:8]}@example.com"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_api_info():
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "DeepAMR API"


def test_drug_classes():
    response = client.get("/drug-classes")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 11
    assert "aminoglycoside" in data["drug_classes"]


def test_model_performance():
    response = client.get("/models/performance")
    assert response.status_code == 200
    data = response.json()
    assert "overall" in data
    assert data["overall"]["micro_f1"] > 0
    assert "model_version" in data


def test_bangladesh_guidelines():
    response = client.get("/guidelines/bangladesh")
    assert response.status_code == 200
    data = response.json()
    assert "resistance_prevalence" in data
    assert "referral_centers" in data


def test_register_and_login():
    email = _unique_email()
    # Register
    response = client.post("/auth/register", json={
        "email": email,
        "name": "Test User",
        "password": "testpassword123",
    })
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert data["user"]["email"] == email
    token = data["token"]

    # Login
    response = client.post("/auth/login", json={
        "email": email,
        "password": "testpassword123",
    })
    assert response.status_code == 200
    assert "token" in response.json()

    # Auth me
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["user"]["email"] == email

    # Logout
    response = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_register_duplicate():
    email = _unique_email()
    client.post("/auth/register", json={
        "email": email,
        "name": "Dup",
        "password": "pass123",
    })
    response = client.post("/auth/register", json={
        "email": email,
        "name": "Dup2",
        "password": "pass456",
    })
    assert response.status_code == 400


def test_login_wrong_password():
    response = client.post("/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "wrong",
    })
    assert response.status_code == 401


def test_prediction_not_found():
    response = client.get("/predictions/nonexistent")
    assert response.status_code == 404


def test_prediction_report_not_found():
    response = client.get("/predictions/nonexistent/report")
    assert response.status_code == 404


def test_predict_fasta_no_file():
    response = client.post("/predict/fasta")
    assert response.status_code == 422


def test_predictions_list():
    response = client.get("/predictions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_dashboard_stats():
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "totalPredictions" in data
