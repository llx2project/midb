"""Database setup for Medical Image Search Platform"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from .models import Base

# Database configuration - use SQLite for development, PostgreSQL for production
default_db_url = "sqlite:///./medical_images.db"
DATABASE_URL = os.getenv("DATABASE_URL", default_db_url)

# Create engine and session
if DATABASE_URL.startswith("sqlite"):
    # SQLite specific configuration
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

