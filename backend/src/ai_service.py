"""AI service for Medical Image Search Platform"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from PIL import Image as PILImage
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import asyncio
import httpx
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class AIService:
    """Service for AI-powered image analysis and search"""
    
    def __init__(self, use_openrouter: bool = False, openrouter_api_key: str = None):
        self.use_openrouter = use_openrouter
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        if not self.use_openrouter:
            # Initialize local CLIP model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = None
            self.processor = None
            self._initialize_local_model()
            
            # Image preprocessing transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.model = None
            self.processor = None
            logger.info("Using OpenRouter API for AI services")
    
    def _initialize_local_model(self):
        """Initialize the CLIP model for image-text understanding"""
        try:
            model_name = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            logger.info(f"Loaded CLIP model on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            # Fallback to a simpler model or raise exception
            raise
    
    async def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text query"""
        if self.use_openrouter:
            return await self.get_text_embedding_openrouter(text)
        
        try:
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_outputs = self.model.get_text_features(**inputs)
                # Handle different output formats from CLIP model
                if hasattr(text_outputs, 'pooler_output'):
                    text_features = text_outputs.pooler_output
                elif hasattr(text_outputs, 'last_hidden_state'):
                    text_features = text_outputs.last_hidden_state.mean(dim=1)
                else:
                    text_features = text_outputs
                
                # Normalize properly
                text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(512)
    
    async def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Get embedding for a single image"""
        if self.use_openrouter:
            return await self.get_image_embedding_openrouter(image_path)
        
        try:
            # Load and preprocess image
            image = PILImage.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Use the dedicated method for image features
                image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                # Extract from BaseModelOutputWithPooling
                if hasattr(image_features, 'pooler_output'):
                    image_features = image_features.pooler_output
                else:
                    image_features = image_features.last_hidden_state.mean(dim=1)
                # Normalize
                image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error getting image embedding for {image_path}: {e}")
            # Return zero vector as fallback
            return np.zeros(512)
    
    async def get_image_embeddings(self, image_paths: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple images"""
        embeddings = []
        for path in image_paths:
            embedding = await self.get_image_embedding(path)
            embeddings.append(embedding)
        return embeddings
    
    def compute_similarities(self, query_embedding: np.ndarray, image_embeddings: List[np.ndarray]) -> List[float]:
        """Compute cosine similarities between query and image embeddings"""
        similarities = []
        query_norm = np.linalg.norm(query_embedding)
        
        for img_emb in image_embeddings:
            img_norm = np.linalg.norm(img_emb)
            if query_norm > 0 and img_norm > 0:
                similarity = np.dot(query_embedding, img_emb) / (query_norm * img_norm)
            else:
                similarity = 0.0
            similarities.append(float(similarity))
        
        return similarities
    
    async def analyze_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """Analyze images and generate insights"""
        if self.use_openrouter:
            return await self.analyze_images_openrouter(image_paths)
        
        try:
            insights = []
            
            # For each image, get some basic analysis
            for i, path in enumerate(image_paths):
                try:
                    # Load image
                    image = PILImage.open(path).convert("RGB")
                    
                    # Get image embedding
                    embedding = await self.get_image_embedding(path)
                    
                    # Generate some mock insights based on image properties
                    width, height = image.size
                    aspect_ratio = width / height
                    
                    insight = {
                        "image_index": i,
                        "file_path": path,
                        "dimensions": {"width": width, "height": height},
                        "aspect_ratio": round(aspect_ratio, 2),
                        "description": f"Medical image with dimensions {width}x{height}",
                        "suggested_modality": self._suggest_modality_from_aspect(aspect_ratio),
                        "quality_score": min(1.0, max(0.0, (width * height) / (1024 * 1024)))  # Normalized by 1MP
                    }
                    insights.append(insight)
                except Exception as e:
                    logger.warning(f"Could not analyze image {path}: {e}")
                    insights.append({
                        "image_index": i,
                        "file_path": path,
                        "error": str(e)
                    })
            
            return {"insights": insights}
        except Exception as e:
            logger.error(f"Error analyzing images: {e}")
            return {"insights": [], "error": str(e)}
    
    def _suggest_modality_from_aspect(self, aspect_ratio: float) -> str:
        """Suggest modality based on aspect ratio (simplified heuristic)"""
        if aspect_ratio > 1.5:
            return "Likely chest X-ray or panoramic view"
        elif aspect_ratio < 0.7:
            return "Likely sagittal MRI or ultrasound"
        else:
            return "Likely axial MRI/CT or standard X-ray"
    
    def suggest_modality(self, image_path: str) -> str:
        """Suggest modality based on image characteristics"""
        try:
            # Load image to get dimensions
            image = PILImage.open(image_path).convert("RGB")
            width, height = image.size
            aspect_ratio = width / height if height > 0 else 1.0
            return self._suggest_modality_from_aspect(aspect_ratio)
        except Exception as e:
            logger.warning(f"Could not suggest modality for {image_path}: {e}")
            return "UNKNOWN"
    
    async def get_text_embedding_openrouter(self, text: str) -> np.ndarray:
        """Get embedding using OpenRouter API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.openrouter_base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "nomic-ai/nomic-embed-text-v1.5:free",
                        "input": text
                    }
                )
                response.raise_for_status()
                result = response.json()
                return np.array(result["data"][0]["embedding"])
        except Exception as e:
            logger.error(f"Error getting text embedding from OpenRouter: {e}")
            return np.zeros(768)  # Nomic embed has 768 dimensions
    
    async def get_image_embedding_openrouter(self, image_path: str) -> np.ndarray:
        """Get image embedding using OpenRouter API (via base64 encoding)"""
        try:
            # Load and encode image as base64
            image = PILImage.open(image_path).convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.openrouter_base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "openai/clip-vit-base-patch32",  # Use a compatible model
                        "input": f"data:image/png;base64,{base64_image}"
                    }
                )
                response.raise_for_status()
                result = response.json()
                return np.array(result["data"][0]["embedding"])
        except Exception as e:
            logger.error(f"Error getting image embedding from OpenRouter: {e}")
            return np.zeros(512)
    
    async def analyze_images_openrouter(self, image_paths: List[str]) -> Dict[str, Any]:
        """Analyze images using OpenRouter API"""
        try:
            insights = []
            
            for i, path in enumerate(image_paths):
                try:
                    # Load image to get basic info
                    image = PILImage.open(path).convert("RGB")
                    width, height = image.size
                    
                    # Use OpenRouter for image analysis
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{self.openrouter_base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.openrouter_api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "google/gemini-2.0-flash-exp:free",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": "Analyze this medical image and provide: 1) Image description, 2) Suggested modality (CT, MRI, X-ray, ultrasound, etc.), 3) Key observations, 4) Quality assessment (0-1)"
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/png;base64,{base64_image}"
                                                }
                                            }
                                        ]
                                    }
                                ]
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        analysis_text = result["choices"][0]["message"]["content"]
                        
                        # Parse analysis (simplified)
                        insight = {
                            "image_index": i,
                            "file_path": path,
                            "dimensions": {"width": width, "height": height},
                            "aspect_ratio": round(width / height, 2),
                            "description": analysis_text[:200],  # Truncate for brevity
                            "suggested_modality": self._suggest_modality_from_aspect(width/height),
                            "quality_score": 0.85  # Default quality
                        }
                        insights.append(insight)
                        
                except Exception as e:
                    logger.warning(f"Could not analyze image {path} with OpenRouter: {e}")
                    # Fallback to local analysis
                    image = PILImage.open(path).convert("RGB")
                    width, height = image.size
                    insights.append({
                        "image_index": i,
                        "file_path": path,
                        "dimensions": {"width": width, "height": height},
                        "aspect_ratio": round(width / height, 2),
                        "description": f"Medical image with dimensions {width}x{height}",
                        "suggested_modality": self._suggest_modality_from_aspect(width/height),
                        "quality_score": 0.7
                    })
            
            return {"insights": insights}
        except Exception as e:
            logger.error(f"Error analyzing images with OpenRouter: {e}")
            return {"insights": [], "error": str(e)}

# Global instance
ai_service = AIService()