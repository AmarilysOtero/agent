from __future__ import annotations
import os, time, json, random, logging
from pathlib import Path
from typing import Any, Callable, Optional
from dotenv import load_dotenv

# Lazy import for azure-ai-projects (Python 3.8 compatibility)
# This will be imported when needed
try:
    from azure.ai.projects import AIProjectClient
    from azure.identity import AzureCliCredential, DefaultAzureCredential
    from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
    _azure_available = True
except (ImportError, TypeError) as e:
    # Python 3.8 compatibility issue - azure-ai-projects requires Python 3.9+
    if "subscriptable" in str(e) or "ABCMeta" in str(e):
        logging.warning(
            "azure-ai-projects requires Python 3.9+. Please upgrade Python or use Python 3.9+ environment. "
            "Error: %s", e
        )
    _azure_available = False
    AIProjectClient = None
    AzureCliCredential = None
    DefaultAzureCredential = None
    ResourceNotFoundError = Exception
    HttpResponseError = Exception

# Load .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

# Mute Azure SDK logs
if os.getenv("QUIET_AZURE_LOGS", "1").lower() in {"1", "true", "yes"}:
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

_client: Optional[AIProjectClient] = None
_validated: bool = False

def _choose_credential():
    try:
        return AzureCliCredential()
    except Exception:
        return DefaultAzureCredential()

def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v

def _get_id(obj: Any) -> str:
    if obj is None:
        return ""
    return getattr(obj, "id", None) or getattr(obj, "value", None) or (obj.get("id", "") if isinstance(obj, dict) else "")

def _resolve(ops: Any, *candidates: str) -> Callable[..., Any]:
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
    raise AttributeError(f"None of these methods exist: {candidates!r}")

def _call_with_agent_kw(fn: Callable[..., Any], *, thread_id: str, agent_id: str, **kw):
    try:
        return fn(thread_id=thread_id, agent_id=agent_id, **kw)
    except TypeError:
        return fn(thread_id=thread_id, assistant_id=agent_id, **kw)

def _with_retries(fn: Callable[[], Any], *, attempts: int = 3) -> Any:
    for i in range(attempts):
        try:
            return fn()
        except (HttpResponseError, ResourceNotFoundError):
            time.sleep(0.5 * (2 ** i) + random.random() * 0.2)
    raise RuntimeError("Operation failed after retries")

def get_foundry_client() -> AIProjectClient:
    global _client, _validated
    if not _azure_available:
        raise RuntimeError(
            "Azure AI Projects SDK is not available. "
            "This is likely due to Python 3.8 compatibility. "
            "Please upgrade to Python 3.9+ or use a Python 3.9+ environment."
        )
    if _client:
        return _client
    endpoint = _require_env("AZURE_AI_PROJECT_ENDPOINT").rstrip("/")
    cred = _choose_credential()
    _client = AIProjectClient(endpoint=endpoint, credential=cred)
    if not _validated:
        _validated = True
        print("[Foundry] Client ready (project endpoint).")
    return _client

def run_foundry_agent(agent_id: str, user_content: str, *, system_hint: str | None = None) -> str:
    client = get_foundry_client()
    agents = client.agents

    create_thread = _resolve(agents, "create_thread", "threads.create")
    thread = _with_retries(lambda: create_thread(logging_enable=False))
    thread_id = _get_id(thread)

    create_message = _resolve(agents, "create_message", "messages.create")
    if system_hint:
        try:
            create_message(thread_id=thread_id, role="system", content=system_hint, logging_enable=False)
        except Exception:
            pass
    create_message(thread_id=thread_id, role="user", content=user_content, logging_enable=False)

    try:
        create_and_process_run = _resolve(agents, "create_and_process_run", "runs.create_and_process")
        _with_retries(lambda: _call_with_agent_kw(create_and_process_run, thread_id=thread_id, agent_id=agent_id, logging_enable=False))
    except AttributeError:
        create_run = _resolve(agents, "create_run", "runs.create")
        run = _with_retries(lambda: _call_with_agent_kw(create_run, thread_id=thread_id, agent_id=agent_id, logging_enable=False))
        run_id = _get_id(run)
        get_run = _resolve(agents, "get_run", "runs.get")
        _poll_run(get_run, thread_id=thread_id, run_id=run_id)

    list_messages = _resolve(agents, "list_messages", "messages.list")
    msgs = list_messages(thread_id=thread_id, logging_enable=False)
    data = list(getattr(msgs, "data", None) or getattr(msgs, "value", None) or msgs or [])

    def _extract_text(m: Any) -> Optional[str]:
        parts = getattr(m, "content", None) or []
        texts = []
        for p in parts:
            t = getattr(getattr(p, "text", None), "value", None) or getattr(p, "text", None)
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
        if texts:
            return "\n".join(texts)
        t = getattr(getattr(m, "text", None), "value", None) or getattr(m, "text", None)
        return t.strip() if isinstance(t, str) and t.strip() else None

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
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = get_run(thread_id=thread_id, run_id=run_id, logging_enable=False)
        status = getattr(r, "status", None) or getattr(r, "value", None)
        if isinstance(status, str) and status.lower() in {"succeeded", "failed", "cancelled", "completed"}:
            return status.lower()
        time.sleep(interval_s)
    return "timeout"

def run_foundry_agent_json(agent_id: str, user_content: str, *, system_hint: str | None = "Reply with JSON only.") -> dict:
    txt = run_foundry_agent(agent_id, user_content, system_hint=system_hint).strip()
    try:
        return json.loads(txt)
    except Exception as e:
        raise RuntimeError(f"Agent did not return valid JSON: {txt[:200]}...") from e

if __name__ == "__main__":
    client = get_foundry_client()
    print("Client test OK.")
    test_agent = os.getenv("AGENT_ID_TRIAGE")
    if test_agent:
        result = run_foundry_agent(test_agent, "Say hello.")
        print(result)
