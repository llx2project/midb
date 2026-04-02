"""
Test configuration for Medical Image Search Platform.
"""
import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set database URL to SQLite for testing BEFORE importing src modules
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

# SQLAlchemy's StaticPool may not be available in all environments,
# so we import it with a fallback to avoid NameError.
try:
    from sqlalchemy.pool import StaticPool
except ImportError:
    # Fallback: create a minimal StaticPool-like class for testing
    class StaticPool:
        pass

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

# Create all tables in the test database
models.Base.metadata.create_all(bind=engine)

# Create test session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db_session():
    """Provide a database session for tests."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()