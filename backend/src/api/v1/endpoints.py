"""API endpoints for Medical Image Search Platform"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from datetime import datetime

from src.database import SessionLocal, Tag
from src.models import Image, Dataset, ImageMetadata
from src.ai_service import AIService

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
    db: Session = Depends(get_db)
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
        
        # Get images
        images = q.offset(offset).limit(limit).all()
        
        # If text query provided, use AI to rank results
        if query and images:
            # Get embeddings for query and images
            query_embedding = await ai_service.get_text_embedding(query)
            image_embeddings = await ai_service.get_image_embeddings([img.file_path for img in images])
            
            # Compute similarities and reorder
            similarities = ai_service.compute_similarities(query_embedding, image_embeddings)
            ranked_images = sorted(zip(images, similarities), key=lambda x: x[1], reverse=True)
            images = [img for img, _ in ranked_images]
        
        # Prepare response
        results = []
        for img in images:
            img_dict = img.to_dict()
            img_dict["dataset_name"] = img.dataset.name if img.dataset else None
            img_dict["preview_url"] = f"/api/v1/images/{img.id}/preview"
            results.append(img_dict)
        
        return {
            "total": q.count(),
            "limit": limit,
            "offset": offset,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/images/{image_id}/preview")
async def get_image_preview(image_id: int, db: Session = Depends(get_db)):
    """Get image preview (thumbnail or full size)"""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # In a real implementation, you would serve the actual image file
    # For now, return metadata and a placeholder
    return {
        "image_id": image.id,
        "file_path": image.file_path,
        "modality": image.modality,
        "dimensions": {"width": image.width, "height": image.height}
    }

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