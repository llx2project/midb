"""
Test suite for API endpoints in Medical Image Search Platform.
"""
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set database URL to SQLite for testing to avoid PostgreSQL connection issues
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

# SQLAlchemy's StaticPool may not be available in all environments,
# so we import it with a fallback to avoid NameError.
try:
    from sqlalchemy.pool import StaticPool
except ImportError:
    # Fallback: create a minimal StaticPool-like class for testing
    class StaticPool:
        pass

from src.main import app
from src.config import settings
from src import models

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create test database engine
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables in the test database
models.Base.metadata.create_all(bind=engine)

# Create test client
client = TestClient(app)

class TestSearchEndpoint:
    """Test cases for search API endpoint."""
    
    def test_search_with_valid_query(self):
        """Test search endpoint with valid query parameters."""
        response = client.get("/api/v1/search?q=lung&modality=X-ray")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0
    
    def test_search_with_invalid_modality(self):
        """Test search endpoint with invalid modality."""
        response = client.get("/api/v1/search?q=lung&modality=invalid")
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_search_without_query(self):
        """Test search endpoint without query parameter."""
        response = client.get("/api/v1/search")
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

class TestAnalysisEndpoint:
    """Test cases for dataset analysis API endpoint."""
    
    def test_analysis_endpoint_success(self):
        """Test analysis endpoint returns dataset insights."""
        response = client.get("/api/v1/datasets/1/insights")
        assert response.status_code == 200
        data = response.json()
        assert "dataset_id" in data
        assert "total_images" in data
    
    def test_analysis_endpoint_invalid_dataset(self):
        """Test analysis endpoint with invalid dataset ID."""
        response = client.get("/api/v1/datasets/999/insights")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])