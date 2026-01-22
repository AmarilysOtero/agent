"""Azure OpenAI client for entity extraction and relationship typing

This module provides a client for interacting with Azure OpenAI to:
- Extract entities from text chunks
- Identify typed relationships between entities
- Generate embeddings for entities
"""

import os
import logging
from typing import List, Dict, Any, Optional
import json
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """Client for Azure OpenAI operations"""
    
    def __init__(self):
        """Initialize Azure OpenAI client from environment variables"""
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
        self.embedding_deployment = os.getenv("AZURE_AI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in environment"
            )
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        logger.info(f"Azure OpenAI client initialized with endpoint: {self.endpoint}")
    
    def extract_entities_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from a text chunk using Azure OpenAI
        
        Args:
            chunk_text: The text content to extract entities from
            chunk_id: Unique identifier for the chunk
            entity_types: Optional list of entity types to extract (default: Person, Organization, Location, Concept)
        
        Returns:
            List of entity dictionaries with keys: name, type, confidence, context
        """
        if entity_types is None:
            entity_types = ["Person", "Organization", "Location", "Concept", "Event", "Product"]
        
        from .prompts import ENTITY_EXTRACTION_PROMPT
        
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=", ".join(entity_types),
            chunk_text=chunk_text
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured entities from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            entities = result.get("entities", [])
            
            # Add chunk_id to each entity
            for entity in entities:
                entity["source_chunk_id"] = chunk_id
                entity["extraction_method"] = "llm"
            
            logger.info(f"Extracted {len(entities)} entities from chunk {chunk_id}")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities from chunk {chunk_id}: {e}")
            return []
    
    def extract_relationships_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract typed relationships between entities in a chunk
        
        Args:
            chunk_text: The text content
            chunk_id: Unique identifier for the chunk
            entities: List of entities already extracted from this chunk
        
        Returns:
            List of relationship dictionaries with keys: subject, relationship_type, object, confidence
        """
        if not entities or len(entities) < 2:
            return []
        
        from .prompts import RELATIONSHIP_EXTRACTION_PROMPT
        
        entity_names = [e["name"] for e in entities]
        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
            entity_names=", ".join(entity_names),
            chunk_text=chunk_text
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured relationships from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            relationships = result.get("relationships", [])
            
            # Add metadata to each relationship
            for rel in relationships:
                rel["source_chunk_id"] = chunk_id
                rel["extraction_method"] = "llm"
            
            logger.info(f"Extracted {len(relationships)} relationships from chunk {chunk_id}")
            return relationships
            
        except Exception as e:
            logger.error(f"Error extracting relationships from chunk {chunk_id}: {e}")
            return []
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using Azure OpenAI
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
