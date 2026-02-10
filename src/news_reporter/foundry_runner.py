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

# Load .env (only if file exists - in Docker, env vars come from docker-compose)
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    # In Docker, env vars are set by docker-compose, so just ensure they're loaded from environment
    load_dotenv(override=False)  # This will load from environment without overriding

# Mute Azure SDK logs
if os.getenv("QUIET_AZURE_LOGS", "1").lower() in {"1", "true", "yes"}:
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

_client: Optional[AIProjectClient] = None
_validated: bool = False

def _choose_credential():
    """
    Choose Azure credential - prioritize service principal (tenant-specific) over az login.
    
    Priority:
    1. DefaultAzureCredential (if service principal env vars are set) - ensures correct tenant
    2. AzureCliCredential (az login) - fallback for local development
    """
    # Check if service principal credentials are configured
    has_service_principal = all([
        os.getenv("AZURE_CLIENT_ID"),
        os.getenv("AZURE_CLIENT_SECRET"),
        os.getenv("AZURE_TENANT_ID")
    ])
    
    is_docker = os.getenv("DOCKER_ENV", "").lower() in {"1", "true", "yes"}
    
    # Always prefer DefaultAzureCredential if service principal is configured
    # This ensures we use the correct tenant, even when running locally
    if has_service_principal or is_docker:
        # Use DefaultAzureCredential which supports:
        # - Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
        # - Managed Identity (if running on Azure)
        # - Other credential types that work in containers
        try:
            return DefaultAzureCredential()
        except Exception as e:
            # Fallback to AzureCliCredential if DefaultAzureCredential fails
            if is_docker:
                # In Docker, AzureCliCredential likely won't work, so provide clear error
                raise RuntimeError(
                    "Failed to authenticate with Azure. In Docker, set environment variables:\n"
                    "  AZURE_CLIENT_ID=<your-client-id>\n"
                    "  ***REMOVED***
                    "  AZURE_TENANT_ID=<your-tenant-id>\n"
                    "Or configure managed identity if running on Azure.\n"
                    f"Error: {str(e)}"
                )
            # Outside Docker, try AzureCliCredential as fallback
            try:
                logging.warning(
                    "DefaultAzureCredential failed, falling back to AzureCliCredential. "
                    "Error: %s", str(e)
                )
                return AzureCliCredential()
            except Exception:
                raise RuntimeError(
                    "Failed to authenticate with Azure. Both DefaultAzureCredential and "
                    "AzureCliCredential failed. Please check your Azure credentials.\n"
                    f"DefaultAzureCredential error: {str(e)}"
                )
    else:
        # No service principal configured - use AzureCliCredential for local development
        try:
            return AzureCliCredential()
        except Exception as e:
            # Fallback to DefaultAzureCredential (might work with managed identity or other methods)
            try:
                logging.warning(
                    "AzureCliCredential failed, falling back to DefaultAzureCredential. "
                    "Error: %s", str(e)
                )
                return DefaultAzureCredential()
            except Exception:
                raise RuntimeError(
                    "Failed to authenticate with Azure. Please either:\n"
                    "  1. Run 'az login' to authenticate with Azure CLI, or\n"
                    "  2. Set service principal credentials:\n"
                    "     AZURE_CLIENT_ID=<your-client-id>\n"
                    "     ***REMOVED***
                    "     AZURE_TENANT_ID=<your-tenant-id>\n"
                    f"AzureCliCredential error: {str(e)}"
                )

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
    last_error = None
    for i in range(attempts):
        try:
            return fn()
        except (HttpResponseError, ResourceNotFoundError) as e:
            last_error = e
            if i < attempts - 1:  # Don't log on last attempt
                logging.debug(f"Retry {i+1}/{attempts} after error: {type(e).__name__}: {str(e)}")
            time.sleep(0.5 * (2 ** i) + random.random() * 0.2)
        except Exception as e:
            # For non-HTTP errors, log and re-raise immediately
            last_error = e
            logging.error(f"Non-retryable error in _with_retries: {type(e).__name__}: {str(e)}")
            raise
    # If we get here, all retries failed
    error_msg = f"Operation failed after {attempts} retries"
    if last_error:
        error_details = str(last_error)
        status_code = getattr(last_error, 'status_code', None)
        if status_code:
            error_msg += f" (HTTP {status_code}: {error_details})"
        else:
            error_msg += f" (Error: {error_details})"
    raise RuntimeError(error_msg) from last_error

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
    try:
        endpoint = _require_env("AZURE_AI_PROJECT_ENDPOINT").rstrip("/")
    except RuntimeError as e:
        raise RuntimeError(
            "Foundry access is not configured. "
            "Please set AZURE_AI_PROJECT_ENDPOINT in your .env file. "
            "If you don't have access to Foundry, you can use Neo4j search only mode."
        ) from e
    try:
        cred = _choose_credential()
        _client = AIProjectClient(endpoint=endpoint, credential=cred)
        if not _validated:
            _validated = True
            print("[Foundry] Client ready (project endpoint).")
        return _client
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "credential" in error_msg.lower():
            raise RuntimeError(
                "Failed to authenticate with Foundry. "
                "Please check your Azure credentials. "
                "Run 'az login' to authenticate, or check your DefaultAzureCredential configuration. "
                f"Error: {error_msg}"
            ) from e
        elif "not found" in error_msg.lower() or "404" in error_msg.lower():
            raise RuntimeError(
                "Foundry project not found or you don't have access to it. "
                "Please verify your AZURE_AI_PROJECT_ENDPOINT and ensure you have access to the project. "
                f"Error: {error_msg}"
            ) from e
        else:
            raise RuntimeError(
                f"Failed to connect to Foundry: {error_msg}. "
                "Please check your Foundry configuration and access permissions."
            ) from e

def run_foundry_agent(agent_id: str, user_content: str, *, system_hint: str | None = None) -> str:
    try:
        client = get_foundry_client()
    except RuntimeError as e:
        # Re-raise with clearer message
        raise RuntimeError(
            f"Foundry access error: {str(e)}. "
            "The chat feature requires Foundry access. "
            "Please configure Foundry or contact your administrator for access."
        ) from e
    agents = client.agents


    # Use the available thread creation method from the current SDK
    try:
        create_thread_and_run = _resolve(agents, "create_thread_and_run", "create_thread_and_process_run")
        thread_run = _with_retries(lambda: create_thread_and_run(agent_id=agent_id, logging_enable=False))
        # Extract thread_id and run_id from the response
        thread_id = getattr(thread_run, "thread_id", None) or (thread_run.get("thread_id") if isinstance(thread_run, dict) else None)
        run_id = getattr(thread_run, "id", None) or getattr(thread_run, "run_id", None) or (thread_run.get("id") if isinstance(thread_run, dict) else None)
        if not thread_id:
            # Fallback: try to extract from nested attributes or log for debugging
            thread_id = _get_id(thread_run)
        if not thread_id or not str(thread_id).startswith("thread"):
            raise RuntimeError(f"Could not extract valid thread_id from create_thread_and_run response: {thread_run}")
        # Wait for run to complete before adding messages
        get_run = _resolve(agents, "get_run", "runs.get")
        _poll_run(get_run, thread_id=thread_id, run_id=run_id)
    except Exception as e:
        error_msg = str(e)
        status_code = getattr(e, 'status_code', None)
        if status_code == 401 or status_code == 403 or "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            raise RuntimeError(
                f"Access denied to Foundry (HTTP {status_code if status_code else 'N/A'}). "
                "Please check your Azure credentials and permissions. "
                "In Docker, ensure AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_TENANT_ID are set correctly. "
                "The service principal needs 'Cognitive Services User' or 'AI Developer' role on the Foundry project. "
                f"Error: {error_msg}"
            ) from e
        elif status_code == 404 or "not found" in error_msg.lower():
            raise RuntimeError(
                f"Foundry resource not found (HTTP {status_code if status_code else 'N/A'}). "
                "Please verify your AZURE_AI_PROJECT_ENDPOINT and ensure you have access to the project. "
                f"Error: {error_msg}"
            ) from e
        else:
            raise RuntimeError(
                f"Failed to create Foundry thread (HTTP {status_code if status_code else 'N/A'}): {error_msg}. "
                "Please check your Foundry access and configuration."
            ) from e

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
