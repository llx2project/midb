"""
Authentication utilities for Medical Image Search Platform.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from typing import Optional

# Security scheme
security = HTTPBearer(auto_error=False)

# Simple API key authentication (for demo purposes)
# In production, use proper JWT or OAuth2
API_KEY = os.getenv("API_KEY", "dev-api-key-change-in-production")

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Simple API key authentication.
    For production, replace with proper JWT validation or OAuth2.
    """
    # For development, allow requests without authentication
    if not credentials:
        return None
    
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def require_auth(user: Optional[str] = Depends(get_current_user)):
    """
    Dependency to require authentication for endpoints.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user