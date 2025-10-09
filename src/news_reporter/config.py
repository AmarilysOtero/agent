# src/news_reporter/config.py
from __future__ import annotations   # must be first!

from dataclasses import dataclass, field
from typing import List
import os

# helper functions -----------------------------
def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

def _list_env(name: str) -> List[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return []
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]

# main settings class --------------------------
@dataclass
class Settings:
    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    default_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "o3-mini"))

    routing_deployments: List[str] = field(default_factory=lambda: _list_env("ROUTING_DEPLOYMENTS"))
    multi_route_always: bool = field(default_factory=lambda: _bool_env("MULTI_ROUTE_ALWAYS", False))

    bing_api_key: str = field(default_factory=lambda: os.getenv("BING_API_KEY", ""))
    serpapi_key: str = field(default_factory=lambda: os.getenv("SERPAPI_API_KEY", ""))

    def validate(self) -> None:
        missing = []
        if not self.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.api_version:
            missing.append("AZURE_OPENAI_API_VERSION")
        if not self.default_deployment:
            missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Ensure .env is loaded before creating Settings()."
            )

    @classmethod
    def from_env(cls) -> "Settings":
        cfg = cls()
        cfg.validate()
        return cfg

    def redacted_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "default_deployment": self.default_deployment,
            "routing_deployments": self.routing_deployments,
            "multi_route_always": self.multi_route_always,
            "bing_api_key": "***" if self.bing_api_key else "",
            "serpapi_key": "***" if self.serpapi_key else "",
            "api_key": "***" if self.api_key else "",
        }
