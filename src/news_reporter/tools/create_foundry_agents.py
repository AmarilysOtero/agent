# src/news_reporter/tools/create_foundry_agents.py
from __future__ import annotations
import os
import inspect
from pathlib import Path
from typing import Any, Dict, Tuple

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError


# -------------------- Robust .env loading (repo root) --------------------
def _load_env():
    """
    Load .env from the project root regardless of CWD:
    Expected root: ...\news-reporter-af\.env
    File path:     ...\src\news_reporter\tools\create_foundry_agents.py
    """
    try:
        from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv
        here = Path(__file__).resolve()
        candidates = [
            here.parents[3] / ".env",  # repo root
            here.parents[2] / ".env",  # ...\src\.env
            here.parents[1] / ".env",  # ...\src\news_reporter\.env
            Path.cwd() / ".env",
        ]
        for p in candidates:
            if p.exists():
                load_dotenv(p)
                print(f"[env] Loaded .env from: {p}")
                break
        else:
            found = find_dotenv(usecwd=True)
            if found:
                load_dotenv(found)
                print(f"[env] Loaded .env via find_dotenv: {found}")
            else:
                print("[env] ⚠️  No .env found. Tried:\n - " + "\n - ".join(map(str, candidates)))
    except Exception as e:
        print(f"[env] ⚠️  Failed to load .env automatically: {e}")

_load_env()

# -------------------- helpers --------------------
def parse_conn(conn: str) -> Dict[str, str]:
    parts: Dict[str, str] = {}
    for chunk in conn.split(";"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            parts[k.strip().lower()] = v.strip()
    required = ["endpoint", "project", "subscription_id", "resource_group", "account"]
    miss = [k for k in required if not parts.get(k)]
    if miss:
        raise SystemExit(f"AI_PROJECT_CONNECTION_STRING missing: {', '.join(miss)}")
    ep = parts["endpoint"].rstrip("/")
    if not ep.startswith("https://"):
        raise SystemExit(f"Endpoint must start with https://   Got: {ep}")
    parts["endpoint"] = ep
    return parts

def build_client(parts: Dict[str, str]) -> AIProjectClient:
    cred = DefaultAzureCredential()
    try:
        return AIProjectClient(
            endpoint=parts["endpoint"],
            project=parts["project"],
            subscription_id=parts["subscription_id"],
            resource_group_name=parts["resource_group"],
            account_name=parts["account"],
            credential=cred,
        )
    except TypeError:
        return AIProjectClient(
            endpoint=parts["endpoint"],
            project_name=parts["project"],
            subscription_id=parts["subscription_id"],
            resource_group=parts["resource_group"],
            account=parts["account"],
            credential=cred,
        )

def get_id(obj: Any) -> str:
    return getattr(obj, "id", None) or getattr(obj, "value", None) or ""

def explain_http(e: HttpResponseError, ctx: str) -> str:
    sc = getattr(e, "status_code", None)
    if sc == 404:
        return f"{ctx}: 404 Not Found — programmatic creation may be disabled for this hub/project/region or path. Create via Studio UI."
    if sc in (401, 403):
        return f"{ctx}: {sc} Auth — ensure you’re logged in and have RBAC on the hub/project."
    if sc in (429, 500, 502, 503, 504):
        return f"{ctx}: {sc} transient — retry later."
    return f"{ctx}: HTTP {sc or 'unknown'} — {e}"

def try_methods(agents_ops: Any, name: str, model: str, instructions: str):
    body = dict(model=model, name=name, description=name, instructions=instructions)
    trials = [
        ("agents.create_agent(model=..., name=..., instructions=...)",  lambda: agents_ops.create_agent(**body)),
        ("agents.create(model=..., name=..., instructions=...)",        lambda: agents_ops.create(**body)),
        ("agents.create_agent(body={...})",                             lambda: agents_ops.create_agent(body=body)),
        ("agents.create(body={...})",                                   lambda: agents_ops.create(body=body)),
    ]
    errors = []
    for label, fn in trials:
        try:
            return fn()
        except AttributeError as e:
            errors.append(f"{label} -> AttributeError: {e}")
        except TypeError as e:
            errors.append(f"{label} -> TypeError: {e}")
        except HttpResponseError as e:
            # bubble up for caller to decide (we’ll provide manual fallback on 404)
            raise e
    # No working method → show available callables for this SDK
    members = sorted([m for m in dir(agents_ops) if not m.startswith("_")])
    sigs = []
    for m in members:
        try:
            obj = getattr(agents_ops, m)
            if callable(obj):
                sigs.append(f"{m}{inspect.signature(obj)}")
        except Exception:
            pass
    diag = "\n".join(errors)
    sigdump = "\n  ".join(sigs[:50])
    raise SystemExit(
        "Could not find a working create method on client.agents.\n"
        "Tried:\n" + diag + "\n\n"
        "Available callables on client.agents (first 50):\n  " + sigdump
    )

def agent_specs(default_model: str) -> Tuple[tuple, ...]:
    return (
        ("TriageAgent",       default_model,
         "You classify a user goal and return STRICT JSON only with keys: "
         "intents (string list), confidence (0-1), rationale (string), targets (string list)."),
        ("AiSearchAgent",    default_model,
         "You run grounded ai searches and return only relevant and accurate findings."),
        ("NewsReporterAgent", default_model,
         "You write concise 60–90s neutral news scripts with explicit dates and sources."),
        ("ReviewAgent",       default_model,
         "You strictly review a news script for factuality and return STRICT JSON only: "
         '{"decision":"accept|revise","reason":string,"suggested_changes":string,"revised_script":string}'),
        ("TextToQueryAgent", default_model,
         "You analyze natural language and convert it into a valid SQL query for cosulting databases."),
        ("TextToAPIAgent", default_model,
         "You analyze natural language and convert it into a valid API call for consulting databases."),
        ("QueryResultInterpreter", default_model,
         "You interpret the result of a database query using a query and write the result in natural language."),
        ("APIResultInterpreter", default_model,
         "You interpret the result of a database API call using an API call and write the result in natural language."),       
        ("ResultInterpreter", default_model,
         "You interpret two paragraphs and join them in such a way that it is just one paragraph in natural language")       
    )

def print_manual_instructions(endpoint: str, project: str, model: str):
    print("\n[Manual creation required]")
    print("It looks like programmatic creation is disabled for this project/region or your account.")
    print("Create the following agents in **Azure AI Foundry Studio → your Hub → Project → Agents → + New Agent**:")
    print(f"- Model for all (suggested): {model}\n")
    for name, _, instructions in agent_specs(model):
        print(f"Name: {name}\nInstructions:\n{instructions}\n---")
    print("\nAfter creating, copy the agent IDs into your .env as:")
    print("TRIAGE_AGENT_ID=<id of TriageAgent>")
    print("AISEARCH_AGENT_ID=<id of AiSearchAgent>")
    print("REPORTER_AGENT_IDS=<id of NewsReporterAgent>   # add more if you create multiple")
    print("REVIEWER_AGENT_ID=<id of ReviewAgent>")
    print("TEXT_TO_QUERY_AGENT_ID=<id of TextToQueryAgent>")
    print("TEXT_TO_API_AGENT_ID=<id of TextToAPIAgent>")
    print("QUERY_RESULT_INTERPRETER_ID=<id of QueryResultInterpreter>")
    print("API_RESULT_INTERPRETER_ID=<id of APIResultInterpreter>")
    print("RESULT_INTERPRETER_ID=<id of ResultInterpreter>")

# -------------------- main --------------------
def main():
    conn = os.getenv("AI_PROJECT_CONNECTION_STRING")
    print(f"[debug] AI_PROJECT_CONNECTION_STRING loaded? {bool(conn)}")
    if not conn:
        raise SystemExit("Set AI_PROJECT_CONNECTION_STRING in your .env first.")
    parts = parse_conn(conn)
    client = build_client(parts)

    # Confirm scope/auth
    listing = client.agents.list_agents()
    data = getattr(listing, "data", None) or getattr(listing, "value", None) or []
    print(f"[Scope OK] endpoint={parts['endpoint']} sub={parts['subscription_id']} rg={parts['resource_group']} account={parts['account']} project={parts['project']} (agents={len(data)})")

    default_model = (
        os.getenv("FOUNDRY_DEFAULT_MODEL")
        or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or "gpt-4o-mini"
    )

    # Try to create each agent; on 404 fall back to Studio instructions (but keep going for all)
    created_ids = {}
    had_404 = False
    for name, model, instructions in agent_specs(default_model):
        try:
            ag = try_methods(client.agents, name=name, model=model, instructions=instructions)
            created_ids[name] = get_id(ag)
            print(f"[OK] Created {name}: {created_ids[name]}")
        except HttpResponseError as e:
            msg = explain_http(e, f"Creating agent '{name}'")
            print(f"[WARN] {msg}")
            if getattr(e, "status_code", None) == 404:
                had_404 = True
            # continue to next agent

    if created_ids:
        print("\n[Created — add these to your .env]")
        if "TriageAgent" in created_ids:
            print("TRIAGE_AGENT_ID=", created_ids["TriageAgent"])
        if "AiSearchAgent" in created_ids:
            print("AISEARCH_AGENT_ID=", created_ids["AiSearchAgent"])
        if "NewsReporterAgent" in created_ids:
            print("REPORTER_AGENT_IDS=", created_ids["NewsReporterAgent"])
        if "ReviewAgent" in created_ids:
            print("REVIEWER_AGENT_ID=", created_ids["ReviewAgent"])
        if "TextToQueryAgent" in created_ids:
            print("TEXT_TO_QUERY_AGENT_ID=", created_ids["TextToQueryAgent"])
        if "TextToAPIAgent" in created_ids:
            print("TEXT_TO_API_AGENT_ID=", created_ids["TextToAPIAgent"])
        if "QueryResultInterpreter" in created_ids:
            print("QUERY_RESULT_INTERPRETER_ID=", created_ids["QueryResultInterpreter"])
        if "APIResultInterpreter" in created_ids:
            print("API_RESULT_INTERPRETER_ID=", created_ids["APIResultInterpreter"])
        if "ResultInterpreter" in created_ids:
            print("RESULT_INTERPRETER_ID=", created_ids["ResultInterpreter"])
        
    if had_404 and not created_ids:
        # Nothing could be created → print manual steps once
        print_manual_instructions(parts["endpoint"], parts["project"], default_model)

if __name__ == "__main__":
    main()
