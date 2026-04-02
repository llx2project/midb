"""
NIH ChestX-ray Ingestion Pipeline
Processes NIH ChestX-ray14 dataset and populates database
"""
import os
import csv
import json
import asyncio
from sqlalchemy.orm import Session
from src.database import SessionLocal
from src.models import Dataset, Image, ImageMetadata
from src.ai_service import ai_service
from datetime import datetime

def ingest_nih_chestxray(dataset_name="NIH ChestX-ray14", 
                        csv_path="nih_chestxray14_labels.csv",
                        image_dir="nih_images/"):
    """
    Ingest NIH ChestX-ray dataset into database
    
    Args:
        dataset_name: Name of the dataset
        csv_path: Path to CSV with labels and metadata
        image_dir: Directory containing images
    """
    db: Session = SessionLocal()
    
    try:
        # Create dataset record
        dataset = Dataset(
            name=dataset_name,
            source="nih",
            source_url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
            license="CC BY 4.0",
            description="Large chest X-ray dataset with 14 disease labels"
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        # Process each image entry
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract metadata
                file_name = row['Image Index']
                file_path = os.path.join(image_dir, file_name)
                
                # Skip if image file doesn't exist
                if not os.path.exists(file_path):
                    print(f"Warning: Image file not found: {file_path}")
                    continue
                
                # Get actual image dimensions and suggest modality
                try:
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(file_path)
                    width, height = pil_img.size
                    pil_img.close()
                    
                    # Use AI service to suggest modality based on image characteristics
                    suggested_modality = ai_service.suggest_modality(file_path)
                    # For NIH ChestX-ray, we know it's XRAY but we can use the suggestion
                    modality = "XRAY" if "xray" in suggested_modality.lower() or "chest" in suggested_modality.lower() else suggested_modality
                except Exception as e:
                    print(f"Warning: Could not process image {file_path}: {e}")
                    width, height = 1024, 1024  # Fallback
                    modality = "XRAY"
                
                # Create image record
                image = Image(
                    dataset_id=dataset.id,
                    file_path=file_path,
                    size_bytes=os.path.getsize(file_path),
                    width=width,
                    height=height,
                    modality=modality,
                    title=row.get('Caption', ''),
                    description=row.get('Finding', '')
                )
                db.add(image)
                db.commit()
                db.refresh(image)
                
                # Create metadata record
                metadata = ImageMetadata(
                    image_id=image.id,
                    metadata_json={
                        "patient_age": int(row.get('Age', 0)) if row.get('Age', '').isdigit() else 0,
                        "patient_sex": row.get('Sex', ''),
                        "view_position": row.get('View Position', ''),
                        "findings": row.get('Finding', ''),
                        "labels": json.loads(row.get('Labels', '[]')) if row.get('Labels', '[]') != '[]' else []
                    }
                )
                db.add(metadata)
                db.commit()
                
        print(f"Successfully ingested {dataset_name} with {reader.line_num - 1} images")
        return dataset.id
        
    except Exception as e:
        db.rollback()
        print(f"Error during ingestion: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    # Example usage (would need actual dataset files)
    ingest_nih_chestxray(
        csv_path="sample_nih_labels.csv",  # Sample CSV for testing
        image_dir="sample_nih_images/"    # Sample image directory
    )