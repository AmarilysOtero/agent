"""Triage Agent for intent classification and database routing."""

import json
import logging
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

from ..foundry_runner import run_foundry_agent

logger = logging.getLogger(__name__)


class IntentResult(BaseModel):
    intents: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    targets: List[str] = Field(default_factory=list)
    database_type: Optional[str] = None  # "postgresql", "csv", "other"
    database_id: Optional[str] = None  # Best matching database ID
    preferred_agent: Optional[str] = None  # "sql", "csv", "vector"


class TriageAgent:
    def __init__(self, foundry_agent_id: str):
        self._id = foundry_agent_id

    async def run(self, goal: str) -> IntentResult:
        logger.info(f"🤖 [AGENT INVOKED] TriageAgent (ID: {self._id})")
        print(f"🤖 [AGENT INVOKED] TriageAgent (ID: {self._id})")
        content = f"Classify and return JSON only. User goal: {goal}"
        print("TriageAgent: using Foundry agent:", self._id)  # keep print
        try:
            raw = run_foundry_agent(self._id, content).strip()
        except RuntimeError as e:
            logger.error("TriageAgent Foundry error: %s", e)
            raise RuntimeError(
                f"Triage agent failed: {str(e)}. "
                "Please check your Foundry access and agent configuration."
            ) from e
        print("Triage raw:", raw)  # keep print
        try:
            data = json.loads(raw)
            
            # If "ai_search" intent OR "unknown" intent, detect best database/schema for routing
            # (unknown queries might be search queries that weren't classified correctly)
            intents = data.get("intents", [])
            if "ai_search" in intents or ("unknown" in intents and len(intents) == 1):
                print(f"🔍 TriageAgent: Starting schema detection for query: '{goal}'")  # Always print
                try:
                    from ..tools_sql.schema_retrieval import SchemaRetriever
                    logger.info(f"🔍 TriageAgent: Starting schema detection for query: '{goal}'")
                    schema_retriever = SchemaRetriever()
                    
                    # List all available databases for debug
                    print("📊 TriageAgent: Listing available databases...")  # Always print
                    all_databases = schema_retriever.list_databases()
                    logger.info(f"📊 TriageAgent: Found {len(all_databases)} available databases")
                    print(f"📊 TriageAgent: Found {len(all_databases)} available databases")
                    for db in all_databases:
                        db_id = db.get("id") or db.get("database_id") or db.get("_id")
                        db_name = db.get("name", "Unknown")
                        db_type = db.get("databaseType") or db.get("database_type", "Unknown")
                        logger.info(f"   - Database: {db_name} (ID: {db_id}, Type: {db_type})")
                        print(f"   - Database: {db_name} (ID: {db_id}, Type: {db_type})")
                    
                    print(f"🔎 TriageAgent: Searching for best database...")
                    best_db_id = schema_retriever.find_best_database(query=goal)
                    print(f"🔎 TriageAgent: Best database ID: {best_db_id}")
                    
                    if best_db_id:
                        # Get database type from database list
                        databases = schema_retriever.list_databases()
                        db_info = next(
                            (db for db in databases 
                             if (db.get("id") or db.get("database_id") or db.get("_id")) == best_db_id),
                            {}
                        )
                        db_type = (db_info.get("databaseType") or db_info.get("database_type") or "").lower()
                        db_name = ((db_info.get("name") or best_db_id) or "").lower()
                        
                        logger.info(f"✅ TriageAgent: Selected database '{db_name}' (ID: {best_db_id}, Type: {db_type})")
                        print(f"✅ TriageAgent: Selected database '{db_name}' (ID: {best_db_id}, Type: {db_type})")
                        
                        if "postgresql" in db_type or "postgres" in db_type:
                            data["database_type"] = "postgresql"
                            data["preferred_agent"] = "sql"
                            print(f"✅ TriageAgent: Set preferred_agent='sql', database_type='postgresql'")
                        elif "csv" in db_type or "csv" in db_name or ".csv" in db_name:
                            data["database_type"] = "csv"
                            data["preferred_agent"] = "csv"
                            print(f"✅ TriageAgent: Set preferred_agent='csv', database_type='csv'")
                        else:
                            data["database_type"] = "other"
                            data["preferred_agent"] = "vector"
                            print(f"✅ TriageAgent: Set preferred_agent='vector', database_type='other'")
                        
                        data["database_id"] = best_db_id
                        logger.info(f"TriageAgent detected database_type={data.get('database_type')}, preferred_agent={data.get('preferred_agent')}, database_id={best_db_id}")
                        print(f"TriageAgent final: database_type={data.get('database_type')}, preferred_agent={data.get('preferred_agent')}, database_id={best_db_id}")
                    else:
                        logger.info("TriageAgent: No best database found, will use default routing")
                        print("TriageAgent: No best database found, will use default routing")
                except Exception as e:
                    logger.error(f"TriageAgent: Schema detection failed, falling back to default: {e}", exc_info=True)
                    print(f"❌ TriageAgent: Schema detection failed: {e}")
                    import traceback
                    print(traceback.format_exc())
            
            result = IntentResult(**data)
            logger.info(f"📋 TriageAgent: Final IntentResult - intents={result.intents}, preferred_agent={result.preferred_agent}, database_id={result.database_id}")
            print(f"📋 TriageAgent: Final IntentResult - intents={result.intents}, preferred_agent={result.preferred_agent}, database_id={result.database_id}")
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("Triage parse error: %s", e)
            return IntentResult(intents=["unknown"], confidence=0.0, rationale="parse_error")
