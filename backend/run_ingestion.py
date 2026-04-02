#!/usr/bin/env python3
"""Script to run data ingestion for medical image datasets"""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the ingestion pipeline"""
    try:
        from backend.data_ingestion import DatasetIngestor
        
        logger.info("Starting data ingestion...")
        ingestor = DatasetIngestor()
        
        # Extract images and metadata
        ingestor.extract_images_and_metadata()
        
        logger.info("Ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()