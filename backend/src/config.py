"""
Configuration settings for Medical Image Search Platform.
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./medical_images.db")
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    APP_NAME: str = "Medical Image Search Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # AI Model
    AI_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    AI_EMBEDDING_DIM: int = 512
    
    # Ingestion
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    
    class Config:
        env_file = ".env"

settings = Settings()