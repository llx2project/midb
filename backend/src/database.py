"""Database setup and models for Medical Image Search Platform"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from datetime import datetime

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

# Base class for models
Base = declarative_base()

# Association table for tags
image_tags = Table(
    'image_tags',
    Base.metadata,
    Column('image_id', Integer, ForeignKey('images.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

class Dataset(Base):
    """Dataset represents a collection of images from a specific source"""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    source = Column(String(100))  # e.g., 'nih', 'tcga', 'openneuro'
    source_url = Column(String(500))
    license = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    images = relationship("Image", back_populates="dataset")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "source_url": self.source_url,
            "license": self.license,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class Image(Base):
    """Image represents a single medical image with metadata"""
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    title = Column(String(200))
    description = Column(Text)
    file_path = Column(String(500))  # Path to stored image
    modality = Column(String(50))  # e.g., 'CT', 'MRI', 'XRAY', 'ULTRASOUND', 'PET'
    width = Column(Integer)  # Image width in pixels
    height = Column(Integer)  # Image height in pixels
    size_bytes = Column(Integer)  # File size in bytes
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="images")
    tags = relationship("Tag", secondary=image_tags, back_populates="images")
    image_metadata = relationship("ImageMetadata", uselist=False, back_populates="image")
    embedding = relationship("VectorEmbedding", uselist=False, back_populates="image")
    
    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "title": self.title,
            "description": self.description,
            "modality": self.modality,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": [tag.to_dict() for tag in self.tags] if self.tags else [],
            "metadata": self.image_metadata.to_dict() if self.image_metadata else None
        }

class Tag(Base):
    """Tag represents metadata tags for images"""
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    category = Column(String(100))  # e.g., 'pathology', 'organ', 'anatomical_region'
    
    # Relationships
    images = relationship("Image", secondary=image_tags, back_populates="tags")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category
        }

class ImageMetadata(Base):
    """Additional metadata for images"""
    __tablename__ = 'image_metadata'
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey('images.id'), unique=True, nullable=False)
    
    # JSON field for flexible metadata storage
    metadata_json = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    image = relationship("Image", back_populates="image_metadata")
    
    def to_dict(self):
        return {
            "id": self.id,
            "image_id": self.image_id,
            "metadata": self.metadata_json
        }

class VectorEmbedding(Base):
    """Vector embedding for image semantic search"""
    __tablename__ = 'vector_embeddings'
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    embedding_type = Column(String(50))
    dimension = Column(Integer)
    model_version = Column(String(100))
    embedding_data = Column(JSON)  # JSON array of embedding vector
    
    # Relationship
    image = relationship("Image", back_populates="embedding")
    
    def to_dict(self):
        return {
            "id": self.id,
            "image_id": self.image_id,
            "embedding_type": self.embedding_type,
            "dimension": self.dimension,
            "model_version": self.model_version
        }

