"""Azure OpenAI Embeddings helper"""

from typing import List
import requests

from config import get_settings


class EmbeddingClient:
    def __init__(self):
        settings = get_settings()
        self.endpoint = settings.azure_openai_endpoint.rstrip("/")
        self.api_key = settings.azure_openai_api_key
        self.deployment = settings.azure_openai_embedding_deployment
        # API version from settings
        self.api_version = settings.azure_openai_api_version

        if not self.endpoint or not self.api_key or not self.deployment:
            raise RuntimeError("Azure OpenAI settings are not configured. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")

    def embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/embeddings?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        payload = {"input": texts}
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        vectors = [item["embedding"] for item in data.get("data", [])]
        return vectors

"""Azure OpenAI embeddings service"""

import logging
from typing import List, Optional
import os
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


class AzureEmbeddingsService:
    """Service for generating embeddings using Azure OpenAI"""
    
    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: str = "2024-02-15-preview"
    ):
        """Initialize Azure OpenAI client
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            azure_deployment: Deployment name for embeddings model
            api_version: API version to use
        """
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-ada-002")
        self.api_version = api_version
        
        if not self.azure_endpoint or not self.azure_api_key:
            logger.warning("Azure OpenAI credentials not configured. Embeddings will fail.")
            self.client = None
        else:
            try:
                self.client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.api_version
                )
                logger.info(f"Azure OpenAI client initialized with deployment: {self.azure_deployment}")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                self.client = None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not self.client:
            logger.error("Azure OpenAI client not initialized")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.azure_deployment
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (or None for failed embeddings)
        """
        if not self.client:
            logger.error("Azure OpenAI client not initialized")
            return [None] * len(texts)
        
        results = []
        # Process in batches to avoid rate limits
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Filter out empty texts
                valid_texts = [(idx, text) for idx, text in enumerate(batch) if text and text.strip()]
                if not valid_texts:
                    results.extend([None] * len(batch))
                    continue
                
                valid_indices = [idx for idx, _ in valid_texts]
                valid_texts_only = [text for _, text in valid_texts]
                
                response = self.client.embeddings.create(
                    input=valid_texts_only,
                    model=self.azure_deployment
                )
                
                # Map results back to original positions
                batch_results = [None] * len(batch)
                for idx, embedding_obj in enumerate(response.data):
                    original_idx = valid_indices[idx]
                    batch_results[original_idx] = embedding_obj.embedding
                
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                results.extend([None] * len(batch))
        
        return results


# Global service instance
_embeddings_service: Optional[AzureEmbeddingsService] = None


def get_embeddings_service() -> AzureEmbeddingsService:
    """Get embeddings service instance (singleton pattern)"""
    global _embeddings_service
    if _embeddings_service is None:
        _embeddings_service = AzureEmbeddingsService()
    return _embeddings_service

