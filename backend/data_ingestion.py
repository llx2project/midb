"""Data ingestion pipeline for medical image datasets"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Any
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NIH ChestX-ray14 dataset configuration
NIH_CHESTXRAY_URL = "https://nihcc.app.box.com/v/chestxray14?dl=zip"
NIH_CHESTXRAY_DIR = "nih_chestxray14"
EXTRACTED_DIR = "extracted"

# Create directories
Path(NIH_CHESTXRAY_DIR).mkdir(exist_ok=True)
Path(EXTRACTED_DIR).mkdir(exist_ok=True)

class DatasetIngestor:
    """Handles ingestion of medical image datasets"""
    
    def __init__(self):
        self.downloaded = False
        self.extracted = False
    
    def download_chestxray14(self):
        """Download NIH ChestX-ray14 dataset"""
        if self.downloaded:
            logger.info("NIH ChestX-ray14 already downloaded")
            return
        
        logger.info("Downloading NIH ChestX-ray14 dataset...")
        response = requests.get(NIH_CHESTXRAY_URL)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download dataset: {response.status_code}")
        
        # Save zip file temporarily
        zip_path = os.path.join(NIH_CHESTXRAY_DIR, "dataset.zip")
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(NIH_CHESTXRAY_DIR)
        
        self.downloaded = True
        logger.info(f"Downloaded and extracted to {NIH_CHESTXRAY_DIR}")
    
    def extract_images_and_metadata(self):
        """Extract images and parse XML annotations for metadata"""
        if self.extracted:
            logger.info("NIH ChestX-ray14 already extracted")
            return
        
        if not self.downloaded:
            self.download_chestxray14()
        
        logger.info("Parsing XML annotations...")
        
        # Find all XML files
        xml_files = list(Path(NIH_CHESTXRAY_DIR).glob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML annotation files")
        
        # Process each XML file
        image_records = []
        for xml_file in tqdm(xml_files, desc="Processing XML files"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get image path
                filename = root.find('.//filename')
                if filename is None:
                    continue
                    
                image_path = os.path.join(NIH_CHESTXRAY_DIR, filename.text)
                if not os.path.exists(image_path):
                    continue
                
                # Get width and height
                size = root.find('.//size')
                width = int(size.find('width').text) if size is not None and size.find('width') is not None else None
                height = int(size.find('height').text) if size is not None and size.find('height') is not None else None
                
                # Get modalities (NIH ChestX-ray has PA and AP frontal views)
                # We'll infer modality from the image filename or use 'XRAY' as default
                modality = 'XRAY'
                
                # Get tags (diseases/conditions)
                findings = root.find('.//findings')
                if findings is not None and findings.text:
                    # Simple mapping of findings to tags
                    tags = self._extract_tags_from_findings(findings.text)
                else:
                    tags = []
                
                # Get dataset info
                dataset_name = "NIH ChestX-ray14"
                source_url = "https://www.nih.gov/chestxray14"
                license = "Public Domain"
                
                # Create dataset record if it doesn't exist
                dataset_id = self._get_or_create_dataset(dataset_name, source_url, license)
                
                # Create image record
                image_record = {
                    'dataset_id': dataset_id,
                    'title': root.find('.//image').text if root.find('.//image') is not None else 'Unknown',
                    'description': root.find('.//description').text if root.find('.//description') is not None else '',
                    'file_path': image_path,
                    'modality': modality,
                    'width': width,
                    'height': height,
                    'size_bytes': os.path.getsize(image_path),
                    'tags': tags
                }
                
                image_records.append(image_record)
                
            except Exception as e:
                logger.error(f"Error processing {xml_file}: {e}")
                continue
        
        # Insert records into database
        self._insert_records_to_db(image_records)
        
        self.extracted = True
        logger.info(f"Processed {len(image_records)} images")
    
    def _extract_tags_from_findings(self, findings_text: str) -> List[str]:
        """Extract tags from findings text"""
        # Simplified tag extraction - in reality this would be more sophisticated
        if not findings_text:
            return []
        
        # Common findings in ChestX-ray
        common_findings = [
            'Atelectasis', 'Cardiomegaly', 'Edema', 'Enlarged Cardiomediastinum',
            'Lung Lesion', 'Lung Opacity', 'Pneumonia', 'Pneumothorax',
            'Respiratory', 'Support Devices'
        ]
        
        tags = []
        for finding in common_findings:
            if finding.lower() in findings_text.lower():
                tags.append(finding)
        
        return tags
    
    def _get_or_create_dataset(self, name: str, source_url: str, license: str) -> int:
        """Get or create dataset record in database"""
        from ..database import SessionLocal, Dataset
        
        db = SessionLocal()
        try:
            dataset = db.query(Dataset).filter(Dataset.name == name).first()
            if not dataset:
                dataset = Dataset(
                    name=name,
                    description=f"NIH {name} dataset",
                    source_url=source_url,
                    license=license
                )
                db.add(dataset)
                db.commit()
                db.refresh(dataset)
            return dataset.id
        finally:
            db.close()
    
    def _insert_records_to_db(self, records: List[Dict[str, Any]]):
        """Insert image records into database"""
        from ..database import SessionLocal, Image
        from ..database import Base, engine
        import sqlalchemy
        
        db = SessionLocal()
        try:
            # Ensure tables exist
            Base.metadata.create_all(bind=engine)
            
            for record in records:
                image = Image(
                    dataset_id=record['dataset_id'],
                    title=record['title'],
                    description=record['description'],
                    file_path=record['file_path'],
                    modality=record['modality'],
                    width=record['width'],
                    height=record['height'],
                    size_bytes=record['size_bytes']
                )
                
                # Add tags
                from ..database import Tag, image_tags
                for tag_name in record['tags']:
                    tag = db.query(Tag).filter(Tag.name == tag_name).first()
                    if not tag:
                        tag = Tag(name=tag_name, category='condition')
                        db.add(tag)
                        db.commit()
                    image.tags.append(tag)
                
                db.add(image)
            
            db.commit()
            logger.info(f"Inserted {len(records)} image records into database")
            
        except Exception as e:
            logger.error(f"Error inserting records: {e}")
            db.rollback()
        finally:
            db.close()

# Example usage
if __name__ == "__main__":
    ingestor = DatasetIngestor()
    ingestor.extract_images_and_metadata()