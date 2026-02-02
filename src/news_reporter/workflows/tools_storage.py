"""Tools storage - MongoDB backend for tools registry and agent-tool relations."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import logging
import uuid
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    _MONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    _MONGO_AVAILABLE = False

logger = logging.getLogger(__name__)


def _get_mongo_url() -> Optional[str]:
    url = os.getenv("MONGO_WORKFLOW_URL")
    if url:
        return url
    agent_url = os.getenv("MONGO_AGENT_URL")
    if not agent_url:
        return None
    url = agent_url.replace("/agent_db?", "/workflow_db?").replace("/agent_db", "/workflow_db")
    if "workflow_db" not in url:
        if "?" in agent_url:
            url = agent_url.split("?")[0].replace("/agent_db", "/workflow_db") + "?" + agent_url.split("?")[1]
        else:
            url = agent_url.replace("/agent_db", "/workflow_db") + "?authSource=workflow_db"
    return url


class ToolsStorage:
    """MongoDB storage for tools and agent-tool relations."""

    def __init__(self, mongo_url: Optional[str] = None) -> None:
        self.mongo_url = mongo_url or _get_mongo_url()
        self.client: Any = None
        self.db: Any = None
        self.tools_collection: Any = None
        self.agent_tools_collection: Any = None
        self._connected = False
        self._use_memory = False
        self._last_error: Optional[str] = None
        self._memory_tools: Dict[str, Dict[str, Any]] = {}
        self._memory_agent_tools: List[tuple] = []

    def connect(self) -> bool:
        self._last_error = None
        if not _MONGO_AVAILABLE:
            self._last_error = "pymongo not installed. Using in-memory fallback (tools will not persist)."
            logger.warning("Tools storage: pymongo not installed. Using in-memory fallback.")
            self._use_memory = True
            return False
        if not self.mongo_url:
            self._last_error = (
                "MONGO_WORKFLOW_URL or MONGO_AGENT_URL not set. "
                "Set one in .env. Using in-memory fallback."
            )
            logger.warning("Tools storage: no MongoDB URL. Using in-memory fallback (tools will not persist).")
            self._use_memory = True
            return False
        if self._connected and self.client:
            try:
                self.client.admin.command("ping")
                return True
            except Exception:
                self._connected = False
        try:
            parsed = urlparse(self.mongo_url)
            db_name = (parsed.path or "/workflow_db").strip("/").split("?")[0] or "workflow_db"
            qs = parse_qs(parsed.query)
            auth_source = qs.get("authSource", [db_name])[0]
            password = unquote(parsed.password) if parsed.password else ""
            self.client = MongoClient(
                host=parsed.hostname or "127.0.0.1",
                port=parsed.port or 27017,
                username=parsed.username,
                password=password,
                authSource=auth_source,
                authMechanism="SCRAM-SHA-256",
                serverSelectionTimeoutMS=5000,
            )
            self.client.admin.command("ping")
            self.db = self.client[db_name]
            self.tools_collection = self.db["tools"]
            self.agent_tools_collection = self.db["agent_tools"]
            self.agent_tools_collection.create_index([("agent_id", 1), ("tool_id", 1)], unique=True)
            self._connected = True
            logger.info("ToolsStorage connected to MongoDB %s", db_name)
            return True
        except Exception as e:
            err = str(e)
            self._last_error = f"MongoDB connection failed: {err}. Check host, port, credentials, and that MongoDB is running."
            logger.warning("ToolsStorage MongoDB connection failed: %s. Using in-memory fallback.", e)
            self._connected = False
            self._use_memory = True
            return False

    def connection_failure_detail(self) -> str:
        """Short message explaining why storage is unavailable (for 503 responses)."""
        if self._connected:
            return "Tools storage not available"
        if self._last_error:
            return self._last_error
        if not _MONGO_AVAILABLE:
            return "pymongo not installed"
        if not self.mongo_url:
            return "MONGO_WORKFLOW_URL / MONGO_AGENT_URL not set"
        return "MongoDB connection failed"

    def _ensure(self) -> bool:
        if self._connected:
            return True
        if self._use_memory:
            return True
        if not self.connect() and not self._use_memory:
            return False
        return True

    def _list_tools_mongo(self) -> List[Dict[str, Any]]:
        cursor = self.tools_collection.find({})
        out = []
        for doc in cursor:
            doc["id"] = doc.get("id") or str(doc.get("_id", ""))
            out.append(self._tool_doc_to_response(doc))
        return out

    def _list_tools_memory(self) -> List[Dict[str, Any]]:
        return [self._tool_doc_to_response(d) for d in self._memory_tools.values()]

    def list_tools(self) -> List[Dict[str, Any]]:
        if not self._ensure():
            return []
        try:
            raw = self._list_tools_memory() if self._use_memory else self._list_tools_mongo()
            return [t for t in raw if t.get("id") and (t.get("name") is not None)]
        except Exception as e:
            logger.exception("list_tools failed: %s", e)
            return []

    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        if not self._ensure():
            return None
        try:
            if self._use_memory:
                doc = self._memory_tools.get(tool_id)
                return self._tool_doc_to_response(doc) if doc else None
            doc = self.tools_collection.find_one({"id": tool_id})
            if not doc:
                return None
            return self._tool_doc_to_response(doc)
        except Exception as e:
            logger.exception("get_tool failed: %s", e)
            return None

    def create_tool(
        self,
        name: str,
        description: str,
        tool_type: str,
        spec: Dict[str, Any],
        source: str = "app",
        foundry_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._ensure():
            raise RuntimeError("Tools storage not connected")
        tid = str(uuid.uuid4())
        now = datetime.utcnow()
        doc = {
            "id": tid,
            "name": name,
            "description": description,
            "type": tool_type,
            "spec": spec,
            "source": source,
            "created_at": now,
            "updated_at": now,
        }
        if foundry_ref is not None:
            doc["foundry_ref"] = foundry_ref
        if self._use_memory:
            self._memory_tools[tid] = doc.copy()
            return self._tool_doc_to_response(doc)
        self.tools_collection.insert_one(doc)
        return self._tool_doc_to_response(doc)

    def upsert_tool_by_foundry_ref(
        self,
        foundry_ref: str,
        name: str,
        description: str,
        tool_type: str,
        spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._ensure():
            raise RuntimeError("Tools storage not connected")
        # 1) Already linked by foundry_ref
        existing = self.find_tool_by_foundry_ref(foundry_ref)
        if existing:
            if self._use_memory:
                for d in self._memory_tools.values():
                    if d.get("foundry_ref") == foundry_ref:
                        d.update({"name": name, "description": description, "type": tool_type, "spec": spec, "updated_at": datetime.utcnow()})
                        break
                return existing
            now = datetime.utcnow()
            self.tools_collection.update_one(
                {"foundry_ref": foundry_ref},
                {"$set": {"name": name, "description": description, "type": tool_type, "spec": spec, "updated_at": now}},
            )
            return self._tool_doc_to_response(self.tools_collection.find_one({"foundry_ref": foundry_ref}))
        # 2) App-created tool with same name+type (we pushed it to Foundry) - link it instead of duplicating
        app_tool = self.find_tool_by_name_and_type(name, tool_type)
        if app_tool:
            tid = app_tool["id"]
            if self._use_memory:
                d = self._memory_tools.get(tid)
                if d:
                    d["foundry_ref"] = foundry_ref
                    d.update({"name": name, "description": description, "type": tool_type, "spec": spec, "updated_at": datetime.utcnow()})
                return app_tool
            self.tools_collection.update_one({"id": tid}, {"$set": {"foundry_ref": foundry_ref, "name": name, "description": description, "type": tool_type, "spec": spec, "updated_at": datetime.utcnow()}})
            return self._tool_doc_to_response(self.tools_collection.find_one({"id": tid}))
        # 3) New tool from Foundry - create it
        if self._use_memory:
            tid = str(uuid.uuid4())
            doc = {"id": tid, "name": name, "description": description, "type": tool_type, "spec": spec, "source": "foundry", "foundry_ref": foundry_ref, "created_at": datetime.utcnow(), "updated_at": datetime.utcnow()}
            self._memory_tools[tid] = doc
            return self._tool_doc_to_response(doc)
        now = datetime.utcnow()
        tid = str(uuid.uuid4())
        doc = {"id": tid, "name": name, "description": description, "type": tool_type, "spec": spec, "source": "foundry", "foundry_ref": foundry_ref, "created_at": now, "updated_at": now}
        self.tools_collection.insert_one(doc)
        return self._tool_doc_to_response(doc)

    def find_tool_by_name_and_type(self, name: str, tool_type: str) -> Optional[Dict[str, Any]]:
        """Find app-created tool (no foundry_ref) with matching name and type - avoids duplicates when syncing from Foundry."""
        if not self._ensure():
            return None
        if self._use_memory:
            for d in self._memory_tools.values():
                if d.get("foundry_ref") is None and d.get("name") == name and d.get("type") == tool_type:
                    return self._tool_doc_to_response(d)
            return None
        doc = self.tools_collection.find_one({"name": name, "type": tool_type, "source": "app", "$or": [{"foundry_ref": {"$exists": False}}, {"foundry_ref": None}]})
        return self._tool_doc_to_response(doc) if doc else None

    def find_tool_by_foundry_ref(self, foundry_ref: str) -> Optional[Dict[str, Any]]:
        if not self._ensure():
            return None
        if self._use_memory:
            for d in self._memory_tools.values():
                if d.get("foundry_ref") == foundry_ref:
                    return self._tool_doc_to_response(d)
            return None
        doc = self.tools_collection.find_one({"foundry_ref": foundry_ref})
        return self._tool_doc_to_response(doc) if doc else None

    def update_tool(
        self,
        tool_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tool_type: Optional[str] = None,
        spec: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._ensure():
            raise RuntimeError("Tools storage not connected")
        if self._use_memory:
            doc = self._memory_tools.get(tool_id)
            if not doc:
                return None
            if name is not None:
                doc["name"] = name
            if description is not None:
                doc["description"] = description
            if tool_type is not None:
                doc["type"] = tool_type
            if spec is not None:
                doc["spec"] = spec
            doc["updated_at"] = datetime.utcnow()
            return self._tool_doc_to_response(doc)
        doc = self.tools_collection.find_one({"id": tool_id})
        if not doc:
            return None
        updates: Dict[str, Any] = {"updated_at": datetime.utcnow()}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if tool_type is not None:
            updates["type"] = tool_type
        if spec is not None:
            updates["spec"] = spec
        self.tools_collection.update_one({"id": tool_id}, {"$set": updates})
        updated = self.tools_collection.find_one({"id": tool_id})
        return self._tool_doc_to_response(updated) if updated else None

    def delete_tool(self, tool_id: str) -> bool:
        if not self._ensure():
            raise RuntimeError("Tools storage not connected")
        if self._use_memory:
            self._memory_agent_tools = [(a, t) for a, t in self._memory_agent_tools if t != tool_id]
            if tool_id in self._memory_tools:
                del self._memory_tools[tool_id]
                return True
            return False
        self.agent_tools_collection.delete_many({"tool_id": tool_id})
        result = self.tools_collection.delete_one({"id": tool_id})
        return result.deleted_count > 0

    def _tool_doc_to_response(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": doc.get("id", ""),
            "name": doc.get("name", ""),
            "description": doc.get("description", ""),
            "type": doc.get("type", "function"),
            "spec": doc.get("spec", {}),
            "source": doc.get("source", "app"),
            "foundry_ref": doc.get("foundry_ref"),
        }

    def list_agent_tool_ids(self, agent_id: str) -> List[str]:
        if not self._ensure():
            return []
        try:
            if self._use_memory:
                return [t for a, t in self._memory_agent_tools if a == agent_id]
            cursor = self.agent_tools_collection.find({"agent_id": agent_id})
            return [r["tool_id"] for r in cursor]
        except Exception as e:
            logger.exception("list_agent_tool_ids failed: %s", e)
            return []

    def set_agent_tools(self, agent_id: str, tool_ids: List[str]) -> None:
        if not self._ensure():
            raise RuntimeError("Tools storage not connected")
        unique_ids = list(dict.fromkeys(tid for tid in tool_ids if tid))
        if self._use_memory:
            self._memory_agent_tools = [(a, t) for a, t in self._memory_agent_tools if a != agent_id]
            for tid in unique_ids:
                self._memory_agent_tools.append((agent_id, tid))
            return
        self.agent_tools_collection.delete_many({"agent_id": agent_id})
        for tid in unique_ids:
            self.agent_tools_collection.insert_one({"agent_id": agent_id, "tool_id": tid, "created_at": datetime.utcnow()})

    def add_agent_tool(self, agent_id: str, tool_id: str) -> None:
        if not self._ensure():
            raise RuntimeError("Tools storage not connected")
        if self._use_memory:
            if (agent_id, tool_id) not in self._memory_agent_tools:
                self._memory_agent_tools.append((agent_id, tool_id))
            return
        self.agent_tools_collection.update_one(
            {"agent_id": agent_id, "tool_id": tool_id},
            {"$setOnInsert": {"agent_id": agent_id, "tool_id": tool_id, "created_at": datetime.utcnow()}},
            upsert=True,
        )

    def list_agent_ids_for_tool(self, tool_id: str) -> List[str]:
        if not self._ensure():
            return []
        try:
            if self._use_memory:
                return [a for a, t in self._memory_agent_tools if t == tool_id]
            cursor = self.agent_tools_collection.find({"tool_id": tool_id})
            return [r["agent_id"] for r in cursor]
        except Exception as e:
            logger.exception("list_agent_ids_for_tool failed: %s", e)
            return []

    def merge_duplicate_tools(self) -> int:
        """
        Merge duplicate tools (same name+type): keep one, update agent_tools refs, delete extras.
        Returns number of duplicates removed.
        """
        if not self._ensure():
            return 0
        tools = self._list_tools_memory() if self._use_memory else self._list_tools_mongo()
        by_key: Dict[tuple, List[Dict[str, Any]]] = {}
        for t in tools:
            key = (t.get("name", ""), t.get("type", "function"))
            by_key.setdefault(key, []).append(t)
        removed = 0
        for key, group in by_key.items():
            if len(group) <= 1:
                continue
            # Prefer: app with foundry_ref, then app without, then foundry; then first by id
            def _sort_key(x: Dict[str, Any]) -> tuple:
                has_ref = 1 if x.get("foundry_ref") else 0
                is_app = 0 if x.get("source") == "app" else 1
                return (is_app, -has_ref, x.get("id", ""))
            group.sort(key=_sort_key)
            canonical_id = group[0]["id"]
            for dup in group[1:]:
                dup_id = dup["id"]
                if dup_id == canonical_id:
                    continue
                # Redirect agent_tools from dup_id to canonical_id (avoid duplicate rows)
                if self._use_memory:
                    seen: set[tuple[str, str]] = set()
                    new_list: List[tuple[str, str]] = []
                    for a, t in self._memory_agent_tools:
                        tid = canonical_id if t == dup_id else t
                        if (a, tid) not in seen:
                            seen.add((a, tid))
                            new_list.append((a, tid))
                    self._memory_agent_tools = new_list
                    if dup_id in self._memory_tools:
                        del self._memory_tools[dup_id]
                else:
                    for doc in list(self.agent_tools_collection.find({"tool_id": dup_id})):
                        aid = doc["agent_id"]
                        if not self.agent_tools_collection.find_one({"agent_id": aid, "tool_id": canonical_id}):
                            self.agent_tools_collection.insert_one(
                                {"agent_id": aid, "tool_id": canonical_id, "created_at": datetime.utcnow()}
                            )
                    self.agent_tools_collection.delete_many({"tool_id": dup_id})
                    self.tools_collection.delete_one({"id": dup_id})
                removed += 1
        return removed


_tools_storage: Optional[ToolsStorage] = None


def get_tools_storage() -> ToolsStorage:
    global _tools_storage
    if _tools_storage is None:
        _tools_storage = ToolsStorage()
        _tools_storage.connect()
    return _tools_storage
