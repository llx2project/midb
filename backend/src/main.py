from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging
import time
from typing import Callable

from src.api.v1.endpoints import router as api_v1_router
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Image Search API",
    description="AI-powered search across medical imaging datasets with modality filtering and dataset insights",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS or ["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": exc.errors(), "status_code": 422}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

# Include API routers
app.include_router(api_v1_router, prefix="/api/v1")

@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Image Search API",
        "version": "0.1.0",
        "description": "AI-powered search across medical imaging datasets",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "medical-image-search-api"
    }

@app.get("/info", tags=["info"])
async def info():
    """API information endpoint"""
    return {
        "name": "Medical Image Search API",
        "version": "0.1.0",
        "description": "AI-powered search across medical imaging datasets with modality filtering",
        "features": [
            "Semantic search using CLIP embeddings",
            "Modality-based filtering (CT, MRI, XRAY, etc.)",
            "Dataset insights and analytics",
            "Multi-dataset support",
            "RESTful API with OpenAPI documentation"
        ],
        "endpoints": {
            "search": "/api/v1/search",
            "datasets": "/api/v1/datasets",
            "modalities": "/api/v1/modalities",
            "tags": "/api/v1/tags",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=settings.PORT,
        log_level="info"
    )