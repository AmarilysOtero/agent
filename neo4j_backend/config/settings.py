"""Application settings and configuration"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""
    
    def __init__(self):
        # Neo4j Connection Settings
        self.neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
        # Neo4j encryption (set NEO4J_ENCRYPTED=true when using bolt+s://)
        self.neo4j_encrypted: bool = os.getenv("NEO4J_ENCRYPTED", "false").lower() in ("1", "true", "yes")
        
        # API Settings
        self.api_title: str = "RAG File Scanner Neo4j API"
        self.api_description: str = "API for storing file structures in Neo4j graph database"
        self.api_version: str = "1.0.0"
        
        # CORS Settings
        self.cors_origins: List[str] = [
            "http://localhost:3000",
            "http://localhost:3001"
        ]
        self.cors_allow_credentials: bool = True
        self.cors_allow_methods: List[str] = ["*"]
        self.cors_allow_headers: List[str] = ["*"]
        
        # Azure OpenAI Settings (for embeddings)
        self.azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
        # Embedding deployment name (do NOT fall back to chat deployment vars)
        self.azure_openai_embedding_deployment: str = (
            os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME")
            or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
            or "text-embedding-3-small"
        )
        # API version if caller wants to specify (e.g., 2024-02-15-preview)
        self.azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")


# Global settings instance
_settings: Settings = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

