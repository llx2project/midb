"""API endpoints for Medical Image Search Platform"""

import os
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from datetime import datetime

from src.database import SessionLocal
from src.models import Image, Dataset, Tag, ImageMetadata
from src.ai_service import AIService
from src.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize AI service
ai_service = AIService()

@router.get("/search")
async def search_images(
    query: Optional[str] = Query(None, description="Text query for semantic search"),
    modality: Optional[str] = Query(None, description="Filter by modality (CT, MRI, XRAY, etc.)"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    dataset_id: Optional[int] = Query(None, description="Filter by dataset ID"),
    limit: int = Query(20, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: Session = Depends(get_db),
    user: Optional[str] = Depends(get_current_user)
):
    """
    Search medical images with optional filters.
    Supports text-based semantic search and modality filtering.
    """
    try:
        # Build query
        q = db.query(Image)
        
        # Apply filters
        if modality:
            q = q.filter(Image.modality == modality.upper())
        if dataset_id:
            q = q.filter(Image.dataset_id == dataset_id)
        if tags:
            for tag in tags:
                q = q.join(Image.tags).filter(Tag.name == tag)
        
        # Get total count before pagination
        total = q.count()
        
        # Get images with pagination
        images = q.offset(offset).limit(limit).all()
        
        # If text query provided, use AI to rank results
        if query and images:
            # Filter out images with missing file paths
            valid_images = [img for img in images if img.file_path and os.path.exists(img.file_path)]
            if valid_images:
                # Get embeddings for query and images
                query_embedding = await ai_service.get_text_embedding(query)
                image_paths = [img.file_path for img in valid_images]
                image_embeddings = await ai_service.get_image_embeddings(image_paths)
                
                # Compute similarities and reorder
                similarities = ai_service.compute_similarities(query_embedding, image_embeddings)
                ranked_images = sorted(zip(valid_images, similarities), key=lambda x: x[1], reverse=True)
                images = [img for img, _ in ranked_images]
            else:
                # If no valid images, return empty results
                images = []
        
        # Prepare response
        results = []
        for img in images:
            img_dict = img.to_dict()
            img_dict["dataset_name"] = img.dataset.name if img.dataset else None
            img_dict["preview_url"] = f"/api/v1/images/{img.id}/preview"
            results.append(img_dict)
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "count": len(results),
            "results": results,
            "has_next": offset + limit < total,
            "has_previous": offset > 0
        }
        
    except Exception as e:
        logger.error(f"Error in search_images: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during search")

@router.get("/images/{image_id}/preview")
async def get_image_preview(image_id: int, db: Session = Depends(get_db)):
    """Get image preview (thumbnail or full size)"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Check if file exists
    if image.file_path and os.path.exists(image.file_path):
        return {
            "image_id": image.id,
            "file_path": image.file_path,
            "modality": image.modality,
            "dimensions": {"width": image.width, "height": image.height},
            "size_bytes": image.size_bytes,
            "available": True
        }
    else:
        return {
            "image_id": image.id,
            "file_path": image.file_path,
            "modality": image.modality,
            "dimensions": {"width": image.width, "height": image.height},
            "available": False,
            "message": "Image file not found on server"
        }

@router.get("/images/{image_id}/metadata")
async def get_image_metadata(image_id: int, db: Session = Depends(get_db)):
    """Get detailed metadata for a specific image"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    result = image.to_dict()
    result["dataset_name"] = image.dataset.name if image.dataset else None
    
    # Include additional metadata if available
    if image.image_metadata:
        result["detailed_metadata"] = image.image_metadata.metadata_json
    
    # Include tags
    result["tags"] = [tag.to_dict() for tag in image.tags] if image.tags else []
    
    return result

@router.get("/datasets/{dataset_id}/insights")
async def get_dataset_insights(dataset_id: int, db: Session = Depends(get_db)):
    """
    Generate AI-powered insights for a dataset.
    Returns statistics and analysis of the dataset.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    images = db.query(Image).filter(Image.dataset_id == dataset_id).all()
    
    if not images:
        return {"dataset_id": dataset_id, "message": "No images found in dataset"}
    
    # Basic statistics
    modalities = {}
    tag_counts = {}
    total_size = 0
    dimensions = []
    
    for img in images:
        # Modality distribution
        modalities[img.modality] = modalities.get(img.modality, 0) + 1
        
        # Tag distribution
        for tag in img.tags:
            tag_counts[tag.name] = tag_counts.get(tag.name, 0) + 1
        
        # Size
        total_size += img.size_bytes or 0
        
        # Dimensions
        if img.width and img.height:
            dimensions.append((img.width, img.height))
    
    # AI-powered insights
    ai_insights = []
    try:
        # Sample a few images for AI analysis
        sample_images = images[:min(5, len(images))]
        sample_paths = [img.file_path for img in sample_images if img.file_path]
        
        if sample_paths:
            # Get image analysis from AI service
            analysis = await ai_service.analyze_images(sample_paths)
            ai_insights = analysis.get("insights", [])
    except Exception as e:
        ai_insights = [f"AI analysis unavailable: {str(e)}"]
    
    insights = {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "total_images": len(images),
        "modalities": modalities,
        "most_common_modality": max(modalities.items(), key=lambda x: x[1])[0] if modalities else None,
        "tag_distribution": tag_counts,
        "total_size_bytes": total_size,
        "average_dimensions": {
            "width": sum(d[0] for d in dimensions) // len(dimensions) if dimensions else None,
            "height": sum(d[1] for d in dimensions) // len(dimensions) if dimensions else None
        } if dimensions else None,
        "ai_insights": ai_insights,
        "generated_at": datetime.utcnow().isoformat()
    }
    
    return insights

@router.get("/modalities")
async def list_modalities(db: Session = Depends(get_db)):
    """List all available modalities in the database"""
    modalities = db.query(Image.modality).distinct().all()
    return {"modalities": [m[0] for m in modalities if m[0]]}

@router.get("/tags")
async def list_tags(category: Optional[str] = None, db: Session = Depends(get_db)):
    """List all tags, optionally filtered by category"""
    q = db.query(Tag)
    if category:
        q = q.filter(Tag.category == category)
    tags = q.all()
    return {"tags": [tag.to_dict() for tag in tags]}

@router.get("/datasets")
async def list_datasets(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List all datasets"""
    datasets = db.query(Dataset).offset(offset).limit(limit).all()
    total = db.query(Dataset).count()
    
    result = []
    for ds in datasets:
        ds_dict = ds.to_dict()
        ds_dict["image_count"] = len(ds.images)
        result.append(ds_dict)
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "datasets": result
    }