from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # Foundry
    ai_project_connection_string: str

    # Agent IDs (from Foundry)
    agent_id_triage: str
    agent_id_websearch: str
    reporter_ids: list[str]
    agent_id_reviewer: str

    # Routing
    multi_route_always: bool

    @staticmethod
    def load() -> "Settings":
        def need(name: str) -> str:
            v = os.getenv(name)
            if not v:
                raise RuntimeError(f"Missing required env var: {name}")
            return v

        def get_bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

        reporter_raw = need("AGENT_ID_REPORTER_LIST")
        reporter_ids = [x.strip() for x in reporter_raw.split(";") if x.strip()]

        return Settings(
            ai_project_connection_string=need("AI_PROJECT_CONNECTION_STRING"),
            agent_id_triage=need("AGENT_ID_TRIAGE"),
            agent_id_websearch=need("AGENT_ID_WEBSEARCH"),
            reporter_ids=reporter_ids,
            agent_id_reviewer=need("AGENT_ID_REVIEWER"),
            multi_route_always=get_bool("MULTI_ROUTE_ALWAYS", False),
        )
