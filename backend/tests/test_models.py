"""
Test suite for database models in Medical Image Search Platform.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.models import Base, Dataset, Image, ImageMetadata, VectorEmbedding
from src.config import settings

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture
def engine():
    """Create a test database engine."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine

@pytest.fixture
def SessionLocal(engine):
    """Create a test session factory."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db_session(SessionLocal):
    """Provide a database session for tests."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

class TestDatasetModel:
    """Test cases for Dataset model."""
    
    def test_create_dataset(self, db_session):
        """Test creating a new dataset."""
        dataset = Dataset(
            name="NIH ChestX-ray14",
            source="nih",
            source_url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
            license="CC BY 4.0",
            description="Large chest X-ray dataset with 14 disease labels"
        )
        db_session.add(dataset)
        db_session.commit()
        
        assert dataset.id is not None
        assert dataset.name == "NIH ChestX-ray14"
        assert dataset.source == "nih"
        assert dataset.created_at is not None
    
    def test_dataset_relationships(self, db_session):
        """Test dataset relationships with images."""
        dataset = Dataset(name="Test Dataset", source="test")
        db_session.add(dataset)
        db_session.commit()
        
        image = Image(
            dataset_id=dataset.id,
            file_path="/data/images/test.png",
            size_bytes=1024,
            width=512,
            height=512
        )
        db_session.add(image)
        db_session.commit()
        
        assert len(dataset.images) == 1
        assert dataset.images[0].file_path == "/data/images/test.png"

class TestImageModel:
    """Test cases for Image model."""
    
    def test_create_image(self, db_session):
        """Test creating an image record."""
        dataset = Dataset(name="Test Dataset", source="test")
        db_session.add(dataset)
        db_session.commit()
        
        image = Image(
            dataset_id=dataset.id,
            file_path="/data/images/sample1.jpg",
            size_bytes=2048,
            width=1024,
            height=768,
            modality="X-ray"
        )
        db_session.add(image)
        db_session.commit()
        
        assert image.id is not None
        assert image.dataset_id == dataset.id
        assert image.modality == "X-ray"
        assert image.created_at is not None
    
    def test_image_metadata_relationship(self, db_session):
        """Test image relationship with metadata."""
        dataset = Dataset(name="Test Dataset", source="test")
        db_session.add(dataset)
        db_session.commit()
        
        image = Image(
            dataset_id=dataset.id,
            file_path="/data/images/sample2.png"
        )
        db_session.add(image)
        db_session.commit()
        
        metadata = ImageMetadata(
            image_id=image.id,
            metadata_json={
                "patient_age": 45,
                "patient_sex": "M",
                "view_position": "PA",
                "findings": "Normal"
            }
        )
        db_session.add(metadata)
        db_session.commit()
        
        assert image.image_metadata is not None
        assert image.image_metadata.metadata_json["patient_age"] == 45
        assert image.image_metadata.metadata_json["patient_sex"] == "M"

class TestImageMetadataModel:
    """Test cases for ImageMetadata model."""
    
    def test_create_metadata(self, db_session):
        """Test creating metadata record."""
        dataset = Dataset(name="Test Dataset", source="test")
        db_session.add(dataset)
        db_session.commit()
        
        image = Image(
            dataset_id=dataset.id,
            file_path="/data/images/sample3.jpg"
        )
        db_session.add(image)
        db_session.commit()
        
        metadata = ImageMetadata(
            image_id=image.id,
            metadata_json={
                "patient_age": 30,
                "patient_sex": "F",
                "view_position": "LAT",
                "findings": "Pneumonia",
                "labels": ["pneumonia", "infection"]
            }
        )
        db_session.add(metadata)
        db_session.commit()
        
        assert metadata.id is not None
        assert metadata.metadata_json["patient_age"] == 30
        assert metadata.metadata_json["patient_sex"] == "F"
        assert metadata.metadata_json["labels"] == ["pneumonia", "infection"]
        assert metadata.created_at is not None

class TestVectorEmbeddingModel:
    """Test cases for VectorEmbedding model."""
    
    def test_create_embedding(self, db_session):
        """Test creating vector embedding record."""
        dataset = Dataset(name="Test Dataset", source="test")
        db_session.add(dataset)
        db_session.commit()
        
        image = Image(
            dataset_id=dataset.id,
            file_path="/data/images/sample4.jpg"
        )
        db_session.add(image)
        db_session.commit()
        
        embedding = VectorEmbedding(
            image_id=image.id,
            embedding_type="clip",
            dimension=512,
            model_version="openai/clip-vit-base-patch32"
        )
        db_session.add(embedding)
        db_session.commit()
        
        assert embedding.id is not None
        assert embedding.embedding_type == "clip"
        assert embedding.dimension == 512
        assert embedding.image_id == image.id

if __name__ == "__main__":
    pytest.main([__file__, "-v"])