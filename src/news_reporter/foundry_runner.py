from __future__ import annotations

import os
import time
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.core.exceptions import (
    ResourceNotFoundError,
    ClientAuthenticationError,
    HttpResponseError,
)

_client: Optional[AIProjectClient] = None
_validated: bool = False


# ============================== Utilities ==============================
def _choose_credential():
    """Prefer the same identity as `az login`; fall back to DefaultAzureCredential."""
    try:
        return AzureCliCredential()
    except Exception:
        return DefaultAzureCredential()


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def _explain_http(e: HttpResponseError, context: str) -> RuntimeError:
    sc = getattr(e, "status_code", None)
    if sc == 404:
        return RuntimeError(f"{context}: 404 Not Found — check endpoint/project/agent ID.")
    if sc in (401, 403):
        return RuntimeError(f"{context}: {sc} Auth — check your `az login` or RBAC permissions.")
    if sc in (429, 500, 502, 503, 504):
        return RuntimeError(f"{context}: {sc} transient — retry later.")
    return RuntimeError(f"{context}: HTTP {sc or 'unknown'}")


def _with_retries(fn: Callable[[], Any], *, attempts: int = 3) -> Any:
    last = None
    for i in range(attempts):
        try:
            return fn()
        except (HttpResponseError, ResourceNotFoundError) as e:
            last = e
            if isinstance(e, HttpResponseError) and getattr(e, "status_code", None) not in {
                404,
                429,
                500,
                502,
                503,
                504,
            }:
                raise
            time.sleep(0.5 * (2 ** i) + random.random() * 0.2)
    if isinstance(last, HttpResponseError):
        raise _explain_http(last, "operation")
    raise last


def _get_id(obj: Any) -> str:
    if obj is None:
        return ""
    return (
        getattr(obj, "id", None)
        or getattr(obj, "value", None)
        or (obj.get("id", "") if isinstance(obj, dict) else "")
        or ""
    )


def _resolve(ops: Any, *candidates: str) -> Callable[..., Any]:
    """Return the first callable attribute found among candidates."""
    for cand in candidates:
        target = ops
        ok = True
        for part in cand.split("."):
            if not hasattr(target, part):
                ok = False
                break
            target = getattr(target, part)
        if ok and callable(target):
            return target
    raise AttributeError(f"None of the candidate methods exist: {candidates!r} on {ops}.")


def _call_with_agent_kw(fn: Callable[..., Any], *, thread_id: str, agent_id: str):
    """Handle SDK differences for agent_id vs assistant_id."""
    try:
        return fn(thread_id=thread_id, agent_id=agent_id)
    except TypeError:
        return fn(thread_id=thread_id, assistant_id=agent_id)


# =============================== Client ===============================
def get_foundry_client() -> AIProjectClient:
    """Build and validate AIProjectClient for Foundry project."""
    global _client, _validated
    if _client is not None:
        return _client

    endpoint = _require_env("AZURE_AI_PROJECT_ENDPOINT").rstrip("/")
    if not endpoint.startswith("https://"):
        raise RuntimeError("AZURE_AI_PROJECT_ENDPOINT must start with https://")

    cred = _choose_credential()
    _client = AIProjectClient(endpoint=endpoint, credential=cred)

    if not _validated:
        try:
            _validated = True
            print("[Foundry] Client ready (project endpoint).")
        except Exception as e:
            raise RuntimeError(f"Validation failed: {e}") from e

    return _client


# =============================== Core Run ===============================
def run_foundry_agent(agent_id: str, user_content: str, *, system_hint: str | None = None) -> str:
    """Send a message to a Foundry agent and return its response text."""
    client = get_foundry_client()
    agents = client.agents

    # 1) Create thread
    create_thread = _resolve(agents, "create_thread", "threads.create")
    thread = _with_retries(lambda: create_thread())
    thread_id = _get_id(thread)
    if not thread_id:
        raise RuntimeError("Failed to obtain thread id from create_thread() result")

    # 2) Add messages
    create_message = _resolve(agents, "create_message", "messages.create")
    if system_hint:
        try:
            create_message(thread_id=thread_id, role="system", content=system_hint)
        except Exception:
            pass
    create_message(thread_id=thread_id, role="user", content=user_content)

    # 3) Run assistant
    try:
        create_and_process_run = _resolve(agents, "create_and_process_run", "runs.create_and_process")
        _with_retries(lambda: _call_with_agent_kw(create_and_process_run, thread_id=thread_id, agent_id=agent_id))
    except AttributeError:
        create_run = _resolve(agents, "create_run", "runs.create")
        run = _with_retries(lambda: _call_with_agent_kw(create_run, thread_id=thread_id, agent_id=agent_id))
        run_id = _get_id(run)
        try:
            get_run = _resolve(agents, "get_run", "runs.get")
            status = _poll_run(get_run, thread_id=thread_id, run_id=run_id)
            if status not in {"succeeded", "completed"}:
                raise RuntimeError(f"Run did not complete successfully (status={status}).")
        except AttributeError:
            pass

    # 4) Get latest message
    list_messages = _resolve(agents, "list_messages", "messages.list")
    msgs = list_messages(thread_id=thread_id)
    data = list(getattr(msgs, "data", None) or getattr(msgs, "value", None) or msgs or [])

    if not data:
        raise RuntimeError("No messages returned from agent thread")

    def _extract_text(m: Any) -> Optional[str]:
        parts = getattr(m, "content", None) or []
        texts: list[str] = []
        for p in parts:
            t = getattr(getattr(p, "text", None), "value", None) or getattr(p, "text", None)
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
        if texts:
            return "\n".join(texts)
        t = getattr(getattr(m, "text", None), "value", None) or getattr(m, "text", None)
        if isinstance(t, str) and t.strip():
            return t.strip()
        return None

    for m in reversed(data):
        role = (getattr(m, "role", None) or getattr(m, "author", None) or "").lower()
        if role == "assistant":
            out = _extract_text(m)
            if out:
                return out

    for m in reversed(data):
        out = _extract_text(m)
        if out:
            return out

    return str(data[-1])


def _poll_run(get_run: Callable[..., Any], *, thread_id: str, run_id: str, timeout_s: float = 20.0, interval_s: float = 0.6) -> str:
    """Poll run status until terminal or timeout."""
    deadline = time.time() + timeout_s
    terminal = {"succeeded", "failed", "cancelled", "completed"}
    while time.time() < deadline:
        r = get_run(thread_id=thread_id, run_id=run_id)
        status = getattr(r, "status", None) or getattr(r, "value", None)
        if isinstance(status, str) and status.lower() in terminal:
            return status.lower()
        time.sleep(interval_s)
    return "timeout"


# ========================== JSON Helper ==========================
def run_foundry_agent_json(agent_id: str, user_content: str, *, system_hint: str | None = "Reply with STRICT JSON only.") -> dict:
    txt = run_foundry_agent(agent_id, user_content, system_hint=system_hint).strip()
    try:
        return json.loads(txt)
    except Exception as e:
        preview = txt if len(txt) < 300 else (txt[:280] + " …")
        raise RuntimeError(f"Agent did not return valid JSON. Got: {preview}") from e


# ================================ Probe =================================
if __name__ == "__main__":
    try:
        client = get_foundry_client()
        print("OK: client built.")
    except Exception as e:
        print("Client build/validation failed:", e)
        raise

    # All known agents (from .env)
    agents = {
        "TRIAGE": os.getenv("AGENT_ID_TRIAGE"),
        "WEBSEARCH": os.getenv("AGENT_ID_WEBSEARCH"),
        "REPORTER_LIST": os.getenv("AGENT_ID_REPORTER_LIST"),
        "REVIEWER": os.getenv("AGENT_ID_REVIEWER"),
    }

    for name, aid in agents.items():
        if not aid:
            continue
        try:
            get_agent = _resolve(client.agents, "get_agent", "agents.get")
            ag = get_agent(agent_id=aid)
            print(f"Agent OK: {name} ({_get_id(ag)})")
        except Exception as e:
            print(f"Agent lookup failed for {name}: {e}")
            continue

        try:
            print(f"Probing {name} agent run...")
            out = run_foundry_agent(aid, "Ping from probe. Reply with 'pong'.", system_hint="Be concise.")
            print(f"{name} agent replied:\n{out}\n")
        except Exception as e:
            print(f"{name} agent probe failed:", e)
