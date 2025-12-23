# src/news_reporter/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

def _split_list(val: str | None) -> list[str]:
    if not val:
        return []
    return [p.strip() for p in val.split(",") if p.strip()]

@dataclass
class Settings:
    # === Existing agents (required fields first) ===
    agent_id_triage: str
    agent_id_aisearch: str
    reporter_ids: list[str]
    agent_id_reviewer: str
    
    # === Optional agent settings ===
    agent_id_neo4j_search: str | None = None  # Optional Neo4j GraphRAG agent
    agent_id_aisearch_sql: str | None = None  # Optional SQL agent (PostgreSQL → CSV → Vector)
    multi_route_always: bool = False
    use_neo4j_search: bool = False  # Toggle to use Neo4j instead of Azure Search

    # === Foundry (Azure AI Project) — your format ===
    ai_project_endpoint: str | None = None         # may include /api/projects/<name>
    ai_subscription_id: str | None = None
    ai_resource_group: str | None = None
    ai_account_name: str | None = None
    ai_project_name: str | None = None
    ai_chat_deployment: str = "gpt-4o-mini"
    ai_embedding_deployment: str = "text-embedding-3-large"

    # === Hybrid storage ===
    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None
    azure_search_index: str = "pdf_chunks"
    embedding_vector_dim: int = 3072

    cosmos_endpoint: str | None = None
    cosmos_key: str | None = None
    cosmos_db: str = "ragdb"
    cosmos_container: str = "docs"

    azure_blob_conn_str: str | None = None
    blob_container_raw: str = "raw"
    blob_container_chunks: str = "chunks"

    # === Neo4j GraphRAG ===
    neo4j_api_url: str | None = None  # Neo4j backend API URL (e.g., "http://localhost:8000")

    # === Auth API ===
    auth_api_url: str | None = None
    
    @classmethod
    def from_env(cls) -> "Settings":
        triage = os.getenv("AGENT_ID_TRIAGE") or ""
        aisearch = os.getenv("AGENT_ID_AISEARCH") or ""
        neo4j_search = os.getenv("AGENT_ID_NEO4J_SEARCH")  # Optional
        aisearch_sql = os.getenv("AGENT_ID_AISEARCHSQL")  # Optional SQL agent
        reviewer = os.getenv("AGENT_ID_REVIEWER") or ""
        reporters = _split_list(os.getenv("AGENT_ID_REPORTER_LIST"))
        if not reporters:
            single = os.getenv("AGENT_ID_REPORTER")
            if single:
                reporters = [single]
        use_neo4j = os.getenv("USE_NEO4J_SEARCH", "false").lower() in {"1", "true", "yes"}
        
        # Validation: need at least one search agent
        if not aisearch and not neo4j_search:
            raise RuntimeError("Missing required search agent ID in .env (AISEARCH or NEO4J_SEARCH)")
        if not (triage and reviewer and reporters):
            raise RuntimeError("Missing one or more required agent IDs in .env (TRIAGE/REPORTER(S)/REVIEWER)")

        # Foundry
        ai_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")  # can include project path
        ai_sub = os.getenv("AZURE_AI_SUBSCRIPTION_ID")
        ai_rg = os.getenv("AZURE_AI_RESOURCE_GROUP")
        ai_account = os.getenv("AZURE_AI_ACCOUNT_NAME")
        ai_project = os.getenv("AZURE_AI_PROJECT_NAME")
        ai_chat = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")
        ai_embed = os.getenv("AZURE_AI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")

        # Hybrid storage
        search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        search_key = os.getenv("AZURE_SEARCH_API_KEY")
        search_index = os.getenv("AZURE_SEARCH_INDEX", "pdf_chunks")
        vec_dim = int(os.getenv("EMBEDDING_VECTOR_DIM", "3072"))

        cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
        cosmos_key = os.getenv("COSMOS_KEY")
        cosmos_db = os.getenv("COSMOS_DB", "ragdb")
        cosmos_container = os.getenv("COSMOS_CONTAINER", "docs")

        blob_cs = os.getenv("AZURE_BLOB_CONN_STR")
        blob_raw = os.getenv("BLOB_CONTAINER_RAW", "raw")
        blob_chunks = os.getenv("BLOB_CONTAINER_CHUNKS", "chunks")

        # Neo4j GraphRAG
        neo4j_url = os.getenv("NEO4J_API_URL")  # e.g., "http://localhost:8000"

        # Auth
        auth_url = os.getenv("MONGO_AUTH_URL")

        # minimal validation (you likely already have these set)
        missing = []
        if not (ai_endpoint and ai_sub and ai_rg and ai_account and (ai_project or "/api/projects/" in (ai_endpoint or ""))):
            missing.append("Azure AI Project envs (endpoint/subscription/resource_group/account/project)")
        
        # Only require Azure Search if not using Neo4j
        if not use_neo4j and not (search_endpoint and search_key and search_index):
            missing.append("Azure AI Search (endpoint/api_key/index) - required when not using Neo4j")
        
        # Require Neo4j URL if using Neo4j search
        if use_neo4j and not neo4j_url:
            missing.append("Neo4j API URL (NEO4J_API_URL) - required when using Neo4j search")
        
        # if not (cosmos_endpoint and cosmos_key and cosmos_db and cosmos_container):
        #     missing.append("Cosmos (endpoint/key/db/container)")
        if not (blob_cs and blob_raw and blob_chunks):
            missing.append("Blob (conn_str + containers)")
        if missing:
            raise RuntimeError("Missing required settings: " + "; ".join(missing))

        multi_flag = (os.getenv("MULTI_ROUTE_ALWAYS", "false").lower() in {"1","true","yes"})

        return cls(
            agent_id_triage=triage,
            agent_id_aisearch=aisearch,
            agent_id_neo4j_search=neo4j_search,
            agent_id_aisearch_sql=aisearch_sql,
            reporter_ids=reporters,
            agent_id_reviewer=reviewer,
            multi_route_always=multi_flag,
            use_neo4j_search=use_neo4j,
            ai_project_endpoint=ai_endpoint,
            ai_subscription_id=ai_sub,
            ai_resource_group=ai_rg,
            ai_account_name=ai_account,
            ai_project_name=ai_project,
            ai_chat_deployment=ai_chat,
            ai_embedding_deployment=ai_embed,
            azure_search_endpoint=search_endpoint,
            azure_search_api_key=search_key,
            azure_search_index=search_index,
            embedding_vector_dim=vec_dim,
            # cosmos_endpoint=cosmos_endpoint,
            cosmos_key=cosmos_key,
            cosmos_db=cosmos_db,
            cosmos_container=cosmos_container,
            azure_blob_conn_str=blob_cs,
            blob_container_raw=blob_raw,
            blob_container_chunks=blob_chunks,
            neo4j_api_url=neo4j_url,
            auth_api_url=auth_url,
        )

    @classmethod
    def load(cls) -> "Settings":
        return cls.from_env()
