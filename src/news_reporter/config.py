from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

def _split_list(val: str | None) -> list[str]:
    if not val:
        return []
    return [p.strip() for p in val.split(",") if p.strip()]

@dataclass
class Settings:
    agent_id_triage: str
    agent_id_websearch: str
    reporter_ids: list[str]
    agent_id_reviewer: str
    multi_route_always: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        triage = os.getenv("AGENT_ID_TRIAGE") or ""
        web = os.getenv("AGENT_ID_WEBSEARCH") or ""
        reviewer = os.getenv("AGENT_ID_REVIEWER") or ""
        reporters = _split_list(os.getenv("AGENT_ID_REPORTER_LIST"))
        if not reporters:
            single = os.getenv("AGENT_ID_REPORTER")
            if single:
                reporters = [single]
        if not (triage and web and reviewer and reporters):
            raise RuntimeError("Missing one or more agent IDs in .env (TRIAGE/WEBSEARCH/REPORTER(S)/REVIEWER)")
        multi_flag = (os.getenv("MULTI_ROUTE_ALWAYS", "false").lower() in {"1", "true", "yes"})
        return cls(
            agent_id_triage=triage,
            agent_id_websearch=web,
            reporter_ids=reporters,
            agent_id_reviewer=reviewer,
            multi_route_always=multi_flag,
        )

    # Alias for your app.py compatibility
    @classmethod
    def load(cls) -> "Settings":
        return cls.from_env()
