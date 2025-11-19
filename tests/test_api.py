"""
Unit tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime
from app.api.main import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_db():
    """Mock database."""
    with patch('app.api.main.Database') as mock:
        yield mock.return_value


@pytest.fixture
def mock_supervisor():
    """Mock supervisor workflow."""
    with patch('app.api.main.SupervisorWorkflow') as mock:
        yield mock.return_value


def test_root_endpoint(client):
    """Test root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_log_data_success(client, mock_db):
    """Test successful data logging."""
    mock_db.insert_user_log.return_value = True
    
    payload = {
        "user_id": "test_user",
        "timestamp": datetime.now().isoformat(),
        "activity_minutes": 30,
        "sleep_hours": 7.5,
        "water_intake_ml": 2000,
        "steps": 8000,
        "heart_rate": 70,
        "calories": 2000,
        "mood": "Happy"
    }
    
    response = client.post("/log_data", json=payload)
    assert response.status_code == 201
    assert response.json()["status"] == "success"


def test_log_data_invalid(client):
    """Test data logging with invalid data."""
    payload = {
        "user_id": "test_user",
        "activity_minutes": "invalid"
    }
    
    response = client.post("/log_data", json=payload)
    assert response.status_code == 422


def test_create_profile_success(client, mock_db):
    """Test profile creation."""
    mock_db.upsert_user_profile.return_value = True
    
    payload = {
        "user_id": "test_user",
        "age": 30,
        "weight_kg": 70.0,
        "height_cm": 175.0,
        "health_goals": ["weight_loss"],
        "medical_conditions": []
    }
    
    response = client.post("/profile", json=payload)
    assert response.status_code == 201
    assert response.json()["status"] == "success"


def test_get_profile_success(client, mock_db):
    """Test retrieving user profile."""
    mock_db.get_user_profile.return_value = {
        "user_id": "test_user",
        "age": 30,
        "weight_kg": 70.0,
        "height_cm": 175.0,
        "health_goals": ["weight_loss"],
        "medical_conditions": []
    }
    
    response = client.get("/profile/test_user")
    assert response.status_code == 200
    assert response.json()["user_id"] == "test_user"


def test_get_profile_not_found(client, mock_db):
    """Test retrieving non-existent profile."""
    mock_db.get_user_profile.return_value = None
    
    response = client.get("/profile/nonexistent")
    assert response.status_code == 404


def test_get_suggestions_success(client, mock_db, mock_supervisor):
    """Test getting suggestions."""
    mock_db.get_user_logs.return_value = [
        {"activity_minutes": 30, "sleep_hours": 7.0}
    ]
    mock_db.get_user_profile.return_value = {"user_id": "test_user", "age": 30}
    
    mock_supervisor.execute.return_value = {
        "recommendations": [
            {
                "suggestion_id": "123",
                "user_id": "test_user",
                "timestamp": datetime.now().isoformat(),
                "category": "sleep",
                "text": "Get more sleep",
                "reasoning": "Sleep is important",
                "confidence_score": 0.85,
                "source": "system"
            }
        ],
        "reasoning_trace": ["Step 1", "Step 2"]
    }
    
    payload = {
        "user_id": "test_user",
        "query": "Give me recommendations"
    }
    
    response = client.post("/get_suggestion", json=payload)
    assert response.status_code == 200
    assert len(response.json()["suggestions"]) > 0
    assert len(response.json()["reasoning"]) > 0


def test_get_suggestions_empty_recommendations(client, mock_db, mock_supervisor):
    """Test suggestions with no recommendations."""
    mock_db.get_user_logs.return_value = []
    mock_db.get_user_profile.return_value = None
    
    mock_supervisor.execute.return_value = {
        "recommendations": [],
        "reasoning_trace": []
    }
    
    payload = {
        "user_id": "test_user",
        "query": "Give me recommendations"
    }
    
    response = client.post("/get_suggestion", json=payload)
    assert response.status_code == 200
    assert len(response.json()["suggestions"]) == 0


def test_get_logs_success(client, mock_db):
    """Test retrieving user logs."""
    mock_db.get_user_logs.return_value = [
        {
            "id": 1,
            "user_id": "test_user",
            "timestamp": datetime.now().isoformat(),
            "activity_minutes": 30,
            "sleep_hours": 7.0
        }
    ]
    
    response = client.get("/logs/test_user?limit=30")
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_get_logs_empty(client, mock_db):
    """Test retrieving logs with no data."""
    mock_db.get_user_logs.return_value = []
    
    response = client.get("/logs/test_user")
    assert response.status_code == 200
    assert response.json()["count"] == 0