"""Register Foundry tools with agents

This script registers text-to-SQL and CSV/Excel schema tools with Foundry agents.
Tools are registered using FunctionTool and ToolSet from azure.ai.agents.models.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env
try:
    from dotenv import load_dotenv, find_dotenv
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
except Exception as e:
    print(f"[env] ⚠️  Failed to load .env automatically: {e}")

# Import Azure SDK
try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.projects import AIProjectClient
    from azure.core.exceptions import HttpResponseError
except ImportError as e:
    print(f"[ERROR] Failed to import Azure SDK: {e}")
    print("Please install: pip install azure-ai-projects azure-identity")
    sys.exit(1)

# Import tool functions
try:
    from ..tools_sql.text_to_sql_tool import query_database
    from .csv_schema_tool import get_file_schema
except ImportError:
    # Fallback for direct execution
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.tools_sql.text_to_sql_tool import query_database
    from src.news_reporter.tools.csv_schema_tool import get_file_schema

# Try to import FunctionTool and ToolSet
try:
    from azure.ai.agents.models import FunctionTool, ToolSet
    _TOOLS_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        from azure.ai.projects.models import FunctionTool, ToolSet
        _TOOLS_AVAILABLE = True
    except ImportError:
        _TOOLS_AVAILABLE = False
        print("[WARN] FunctionTool and ToolSet not available. Tools may need to be registered manually via Azure AI Studio.")


def parse_conn(conn: str) -> Dict[str, str]:
    """Parse AI_PROJECT_CONNECTION_STRING"""
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


def get_credential():
    """Choose Azure credential - prefer DefaultAzureCredential in Docker, AzureCliCredential otherwise."""
    import os
    # In Docker, AzureCliCredential doesn't work reliably, so prefer DefaultAzureCredential
    is_docker = os.getenv("DOCKER_ENV", "").lower() in {"1", "true", "yes"}
    
    if is_docker:
        # In Docker, use DefaultAzureCredential which supports:
        # - Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
        # - Managed Identity (if running on Azure)
        try:
            return DefaultAzureCredential()
        except Exception:
            # Fallback to AzureCliCredential if DefaultAzureCredential fails
            try:
                from azure.identity import AzureCliCredential
                return AzureCliCredential()
            except Exception:
                raise RuntimeError(
                    "Failed to authenticate with Azure. In Docker, set environment variables:\n"
                    "  AZURE_CLIENT_ID=<your-client-id>\n"
                    "  ***REMOVED***
                    "  AZURE_TENANT_ID=<your-tenant-id>\n"
                    "Or configure managed identity if running on Azure."
                )
    else:
        # Outside Docker, try AzureCliCredential first (for local development)
        try:
            from azure.identity import AzureCliCredential
            return AzureCliCredential()
        except Exception:
            return DefaultAzureCredential()


def build_client(parts: Dict[str, str]) -> AIProjectClient:
    """Build AIProjectClient - try AZURE_AI_PROJECT_ENDPOINT first (working approach), fallback to connection string"""
    cred = get_credential()
    
    # First, try using AZURE_AI_PROJECT_ENDPOINT (same as check_agent_reachability.py)
    project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if project_endpoint:
        endpoint = project_endpoint.rstrip("/")
        if endpoint.startswith("https://") and "/api/projects/" in endpoint:
            print(f"[DEBUG] Using AZURE_AI_PROJECT_ENDPOINT: {endpoint}")
            return AIProjectClient(endpoint=endpoint, credential=cred)
    
    # Fallback to connection string approach
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
    """Extract ID from object"""
    return getattr(obj, "id", None) or getattr(obj, "value", None) or ""


def create_toolset() -> Optional[ToolSet]:
    """Create ToolSet with registered tools"""
    if not _TOOLS_AVAILABLE:
        print("[ERROR] FunctionTool and ToolSet not available in SDK")
        return None
    
    try:
        # Create FunctionTool with both tools
        print("[INFO] Creating FunctionTool with query_database and get_file_schema...")
        function_tool = FunctionTool(functions={query_database, get_file_schema})
        
        # Create ToolSet and add FunctionTool
        toolset = ToolSet()
        toolset.add(function_tool)
        
        print("[OK] ToolSet created successfully")
        return toolset
    except Exception as e:
        print(f"[ERROR] Failed to create ToolSet: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_agent(client: AIProjectClient, agent_id: str) -> Optional[Any]:
    """Get agent by ID to verify it exists"""
    try:
        agents_ops = client.agents
        
        # First, let's see what methods are available
        available_methods = [m for m in dir(agents_ops) if not m.startswith("_") and callable(getattr(agents_ops, m, None))]
        print(f"[DEBUG] Available methods on agents_ops: {', '.join(available_methods[:15])}")
        
        # Try different get methods - check if methods exist first
        get_methods = []
        if hasattr(agents_ops, "get_agent"):
            get_methods.append(("agents.get_agent(agent_id)", lambda: agents_ops.get_agent(agent_id=agent_id)))
        if hasattr(agents_ops, "get"):
            get_methods.append(("agents.get(agent_id)", lambda: agents_ops.get(agent_id=agent_id)))
        if hasattr(agents_ops, "get_assistant"):
            get_methods.append(("agents.get_assistant(agent_id)", lambda: agents_ops.get_assistant(agent_id=agent_id)))
        
        if not get_methods:
            print(f"[DEBUG] No get methods found. Available: {available_methods}")
            # If we can't get individual agents, try to find it in the list
            print(f"[DEBUG] Attempting to find agent in list...")
            all_agents = list_agents(client)
            for agent_info in all_agents:
                if agent_info["id"] == agent_id:
                    print(f"[DEBUG] Found agent {agent_info['name']} in list")
                    return {"id": agent_id, "name": agent_info["name"]}
            return None
        
        last_error = None
        for method_name, method_func in get_methods:
            try:
                agent = method_func()
                print(f"[DEBUG] Successfully retrieved agent using {method_name}")
                return agent
            except (AttributeError, TypeError) as e:
                last_error = f"{method_name}: {type(e).__name__}: {str(e)}"
                continue
            except HttpResponseError as e:
                last_error = f"{method_name}: HTTP {e.status_code}: {str(e)}"
                if e.status_code == 404:
                    # Agent not found with this method, try next
                    continue
                # Other HTTP errors, try next method
                continue
        
        print(f"[DEBUG] Could not retrieve agent. Last error: {last_error}")
        # Fallback: try to find in list
        all_agents = list_agents(client)
        for agent_info in all_agents:
            if agent_info["id"] == agent_id:
                print(f"[DEBUG] Found agent {agent_info['name']} in list (fallback)")
                return {"id": agent_id, "name": agent_info["name"]}
        return None
    except Exception as e:
        print(f"[DEBUG] Exception in get_agent: {e}")
        return None


def verify_agent_accessible(client: AIProjectClient, agent_id: str, agent_name: str) -> bool:
    """Verify that an agent is accessible via the SDK"""
    print(f"\n{'='*60}")
    print(f"[VERIFY] Checking if agent {agent_name} ({agent_id}) is accessible...")
    print(f"{'='*60}")
    
    try:
        agents_ops = client.agents
        
        # Step 1: Check if we can list agents at all - use EXACT pattern from check_agent_reachability.py
        print(f"[STEP 1] Attempting to list all agents to verify SDK connectivity...")
        print(f"[DEBUG] Using exact pattern from check_agent_reachability.py (line 80)...")
        try:
            # Match check_agent_reachability.py exactly - iterate the ItemPaged object
            agents_iter = client.agents.list_agents()
            print(f"[DEBUG] list_agents() returned: {type(agents_iter)}")
            
            agents_list = []
            for agent in agents_iter:
                agent_id_found = get_id(agent)
                agent_name_found = getattr(agent, "name", None) or getattr(agent, "display_name", None) or "Unknown"
                agents_list.append({"id": agent_id_found, "name": agent_name_found})
            
            if agents_list:
                print(f"[OK] Successfully listed {len(agents_list)} agent(s) from SDK:")
                for agent_info in agents_list:
                    print(f"  - {agent_info['name']}: {agent_info['id']}")
                
                # Check if our target agent is in the list
                found_in_list = any(a["id"] == agent_id for a in agents_list)
                if found_in_list:
                    print(f"[OK] Target agent {agent_name} found in agent list!")
                else:
                    print(f"[WARN] Target agent {agent_name} ({agent_id}) NOT found in agent list")
                    print(f"[WARN] Available agent IDs: {[a['id'] for a in agents_list]}")
            else:
                print(f"[WARN] list_agents() returned empty list (but no error)")
        except HttpResponseError as e:
            print(f"[ERROR] list_agents() returned HTTP {e.status_code}: {str(e)}")
            print(f"[DEBUG] This might be a project/endpoint configuration issue.")
        except Exception as e:
            print(f"[ERROR] Could not list agents: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Step 2: Try to get the specific agent
        print(f"\n[STEP 2] Attempting to get specific agent by ID: {agent_id}...")
        
        if not hasattr(agents_ops, "get_agent"):
            print(f"[ERROR] get_agent method not available on agents_ops")
            available_methods = [m for m in dir(agents_ops) if not m.startswith("_") and callable(getattr(agents_ops, m, None))]
            print(f"[DEBUG] Available methods: {', '.join(available_methods[:10])}")
            return False
        
        # Try multiple get methods
        get_methods = [
            ("get_agent(agent_id=...)", lambda: agents_ops.get_agent(agent_id=agent_id)),
            ("get_agent(id=...)", lambda: agents_ops.get_agent(id=agent_id)),
        ]
        
        if hasattr(agents_ops, "get"):
            get_methods.append(("get(agent_id=...)", lambda: agents_ops.get(agent_id=agent_id)))
        
        agent = None
        method_used = None
        for method_name, method_func in get_methods:
            try:
                print(f"[DEBUG] Trying {method_name}...")
                agent = method_func()
                method_used = method_name
                print(f"[OK] Successfully retrieved agent using {method_name}")
                break
            except HttpResponseError as e:
                print(f"[DEBUG] {method_name} returned HTTP {e.status_code}: {str(e)}")
                if e.status_code == 404:
                    continue  # Try next method
                else:
                    print(f"[ERROR] Unexpected HTTP error {e.status_code}")
                    return False
            except Exception as e:
                print(f"[DEBUG] {method_name} failed: {type(e).__name__}: {str(e)}")
                continue
        
        if agent is None:
            print(f"[ERROR] Could not retrieve agent using any method")
            print(f"[INFO] Agent may exist in Foundry UI but SDK cannot access it")
            return False
        
        # Step 3: Show agent details
        print(f"\n[STEP 3] Agent details retrieved successfully:")
        agent_id_found = get_id(agent)
        agent_name_found = getattr(agent, "name", None) or getattr(agent, "display_name", None) or "Unknown"
        print(f"  - Method used: {method_used}")
        print(f"  - ID: {agent_id_found}")
        print(f"  - Name: {agent_name_found}")
        print(f"  - Type: {type(agent).__name__}")
        
        # Show agent attributes
        agent_attrs = [a for a in dir(agent) if not a.startswith("_")][:20]
        print(f"  - Available attributes: {', '.join(agent_attrs)}")
        
        # Check if agent has toolset/tools attributes
        if hasattr(agent, "toolset"):
            print(f"  - Has 'toolset' attribute: Yes")
            try:
                current_toolset = getattr(agent, "toolset")
                print(f"  - Current toolset: {current_toolset}")
            except:
                print(f"  - Current toolset: (could not read)")
        if hasattr(agent, "tools"):
            print(f"  - Has 'tools' attribute: Yes")
            try:
                current_tools = getattr(agent, "tools")
                print(f"  - Current tools: {current_tools}")
            except:
                print(f"  - Current tools: (could not read)")
        
        print(f"\n[OK] Agent {agent_name} is accessible and ready for update!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to verify agent: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def update_agent_with_tools(
    client: AIProjectClient,
    agent_id: str,
    agent_name: str,
    toolset: ToolSet
) -> bool:
    """Update an existing agent with tools"""
    try:
        agents_ops = client.agents
        
        print(f"\n[INFO] Attempting to update agent {agent_name} ({agent_id})...")
        
        # First, verify agent is accessible
        if not verify_agent_accessible(client, agent_id, agent_name):
            print(f"[WARN] Could not verify agent {agent_name} is accessible. Skipping update.")
            return False
        
        # Check available methods first
        available_methods = [m for m in dir(agents_ops) if not m.startswith("_") and callable(getattr(agents_ops, m, None))]
        print(f"[DEBUG] Available methods on agents_ops: {', '.join(available_methods)}")
        
        # Inspect update_agent method signature
        if hasattr(agents_ops, "update_agent"):
            import inspect
            sig = inspect.signature(agents_ops.update_agent)
            print(f"[DEBUG] update_agent signature: {sig}")
        
        # Try different update methods - check if they exist first
        update_methods = []
        if hasattr(agents_ops, "update_agent"):
            # Try with toolset parameter
            update_methods.append(("agents.update_agent(agent_id, toolset=...)", 
                 lambda: agents_ops.update_agent(agent_id=agent_id, toolset=toolset)))
            # Try with tools parameter
            update_methods.append(("agents.update_agent(agent_id, tools=...)", 
                 lambda: agents_ops.update_agent(agent_id=agent_id, tools=toolset)))
            # Try with body parameter containing toolset
            update_methods.append(("agents.update_agent(agent_id, body={'toolset': ...})", 
                 lambda: agents_ops.update_agent(agent_id=agent_id, body={"toolset": toolset})))
            # Try getting agent first, then updating with toolset
            try:
                agent = agents_ops.get_agent(agent_id=agent_id)
                # Try updating the agent object directly
                if hasattr(agent, "toolset") or hasattr(agent, "tools"):
                    update_methods.append(("agents.update_agent(agent with toolset)", 
                         lambda: agents_ops.update_agent(agent_id=agent_id, toolset=toolset)))
            except:
                pass
        if hasattr(agents_ops, "update"):
            update_methods.append(("agents.update(agent_id, toolset=...)", 
                 lambda: agents_ops.update(agent_id=agent_id, toolset=toolset)))
        if hasattr(agents_ops, "modify"):
            update_methods.append(("agents.modify(agent_id, toolset=...)", 
                 lambda: agents_ops.modify(agent_id=agent_id, toolset=toolset)))
        
        if not update_methods:
            print(f"[ERROR] No update methods found on agents_ops")
            print(f"[DEBUG] All available methods: {available_methods[:20]}")
            return False
        
        for method_name, method_func in update_methods:
            try:
                result = method_func()
                print(f"[OK] Updated {agent_name} ({agent_id}) using {method_name}")
                return True
            except (AttributeError, TypeError) as e:
                continue
            except HttpResponseError as e:
                error_msg = str(e)
                if e.status_code == 404:
                    print(f"[WARN] Update returned 404 for {agent_name}. Error: {error_msg}")
                    # Continue to try other methods
                    continue
                elif e.status_code in (400, 403):
                    print(f"[WARN] Update failed for {agent_name} with {e.status_code}. Error: {error_msg}")
                    # Try next method
                    continue
                else:
                    print(f"[WARN] HTTP error {e.status_code} for {method_name}: {error_msg}")
                    continue
        
        print(f"[WARN] Could not find working update method for {agent_name}")
        print(f"[INFO] Agent exists but update methods failed. You may need to update manually via Azure AI Studio.")
        
        # Debug: Show available methods
        print(f"[DEBUG] Available methods on agents_ops:")
        available_methods = [m for m in dir(agents_ops) if not m.startswith("_") and callable(getattr(agents_ops, m, None))]
        for method in available_methods[:10]:  # Show first 10
            print(f"  - {method}")
        
        return False
        
    except Exception as e:
        print(f"[ERROR] Failed to update {agent_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_agents(client: AIProjectClient) -> List[Dict[str, str]]:
    """List all agents to help verify IDs - matches EXACT pattern from check_agent_reachability.py"""
    try:
        # Use the EXACT same pattern as check_agent_reachability.py line 80 which works
        # list_agents() returns an ItemPaged iterator that must be iterated, not accessed via .data/.value
        agents_iter = client.agents.list_agents()
        
        agents_list = []
        for agent in agents_iter:
            agent_id = get_id(agent)
            agent_name = getattr(agent, "name", None) or getattr(agent, "display_name", None) or "Unknown"
            agents_list.append({"id": agent_id, "name": agent_name})
        
        return agents_list
    except HttpResponseError as e:
        # 404 might mean agents are in a different location or require different access
        # But agents exist, so this might be a project/endpoint configuration issue
        print(f"[DEBUG] list_agents() returned HTTP {e.status_code}: {str(e)}")
        print(f"[DEBUG] This might be a project/endpoint configuration issue.")
        print(f"[DEBUG] Agents exist in Foundry, but SDK can't access them via this endpoint.")
        return []
    except Exception as e:
        print(f"[WARN] Could not list agents: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_agent_ids_from_env() -> Dict[str, str]:
    """Get agent IDs from environment variables"""
    return {
        "TriageAgent": os.getenv("AGENT_ID_TRIAGE", ""),
        "AiSearchAgent": os.getenv("AGENT_ID_AISEARCH", ""),
        "Neo4jGraphRAGAgent": os.getenv("AGENT_ID_NEO4J_SEARCH", ""),
    }


def main():
    """Main function to register tools with agents"""
    print("=" * 60)
    print("Foundry Tool Registration Script")
    print("=" * 60)
    
    # Check if tools are available
    if not _TOOLS_AVAILABLE:
        print("\n[ERROR] FunctionTool and ToolSet are not available in the SDK.")
        print("This may be due to:")
        print("  1. SDK version doesn't support tools yet")
        print("  2. Tools need to be registered manually via Azure AI Studio")
        print("\nTo register tools manually:")
        print("  1. Go to Azure AI Foundry Studio")
        print("  2. Navigate to your Project → Agents")
        print("  3. Edit each agent (AiSearchAgent, TriageAgent, etc.)")
        print("  4. Add the following functions as tools:")
        print("     - query_database(natural_language_query: str, database_id: str) -> str")
        print("     - get_file_schema(file_path: str) -> str")
        sys.exit(1)
    
    # Get connection string
    conn = os.getenv("AI_PROJECT_CONNECTION_STRING")
    if not conn:
        print("[ERROR] AI_PROJECT_CONNECTION_STRING not set in .env")
        sys.exit(1)
    
    # Parse connection and build client
    try:
        parts = parse_conn(conn)
        client = build_client(parts)
        print(f"[OK] Connected to Foundry: {parts['project']}")
        
        # Verify connection by listing agents (same as check_agent_reachability.py line 80)
        print(f"[DEBUG] Verifying connection by listing agents (matching check_agent_reachability.py)...")
        try:
            agents_iter = client.agents.list_agents()
            agent_count = sum(1 for _ in agents_iter)  # Count by iterating
            print(f"[OK] Connection verified: endpoint={parts['endpoint']} sub={parts['subscription_id']} "
                  f"rg={parts['resource_group']} account={parts['account']} project={parts['project']} "
                  f"(agents={agent_count})")
        except Exception as e:
            print(f"[WARN] Could not verify connection by listing agents: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Foundry: {e}")
        sys.exit(1)
    
    # Create toolset
    toolset = create_toolset()
    if not toolset:
        print("[ERROR] Failed to create toolset")
        sys.exit(1)
    
    # Get agent IDs from .env
    agent_ids = get_agent_ids_from_env()
    agents_to_update = {name: agent_id for name, agent_id in agent_ids.items() if agent_id}
    
    if not agents_to_update:
        print("\n[WARN] No agent IDs found in environment variables")
        print("Set the following in your .env:")
        print("  AGENT_ID_TRIAGE=<triage_agent_id>")
        print("  AGENT_ID_AISEARCH=<aisearch_agent_id>")
        print("  AGENT_ID_NEO4J_SEARCH=<neo4j_agent_id> (optional)")
        
        # Try to list agents as a fallback (but don't fail if it doesn't work)
        print("\n[INFO] Attempting to list agents to help find IDs...")
        available_agents = list_agents(client)
        if available_agents:
            print(f"[OK] Found {len(available_agents)} agent(s) in Foundry:")
            for agent in available_agents:
                print(f"  - {agent['name']}: {agent['id']}")
        sys.exit(1)
    
    print(f"\n[INFO] Will attempt to register tools with {len(agents_to_update)} agent(s):")
    for name, agent_id in agents_to_update.items():
        print(f"  - {name}: {agent_id}")
    
    # Update agents
    print(f"\n[INFO] Registering tools with {len(agents_to_update)} agent(s)...")
    success_count = 0
    for agent_name, agent_id in agents_to_update.items():
        if update_agent_with_tools(client, agent_id, agent_name, toolset):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Registration complete: {success_count}/{len(agents_to_update)} agents updated")
    print("=" * 60)
    
    if success_count < len(agents_to_update):
        print("\n" + "=" * 60)
        print("MANUAL REGISTRATION REQUIRED")
        print("=" * 60)
        print("\nThe Azure AI Projects SDK doesn't support programmatic agent updates")
        print("in this version. Please register tools manually via Azure AI Studio.\n")
        print("Steps:")
        print("1. Go to: https://ai.azure.com → Your Hub → AgentFrameworkProject → Agents")
        print("2. For each agent (AiSearchAgent, TriageAgent, etc.):")
        print("   a. Click on the agent name")
        print("   b. Click 'Edit' or go to 'Tools' section")
        print("   c. Click 'Add Tool' or 'Add Function'")
        print("   d. Add the following two functions:\n")
        print("   Function 1: query_database")
        print("   - Name: query_database")
        print("   - Description: Converts natural language to SQL, executes it, and returns results")
        print("   - Parameters:")
        print("     * natural_language_query (string, required)")
        print("     * database_id (string, required)")
        print("   - Returns: JSON string with SQL query, execution results, and metadata\n")
        print("   Function 2: get_file_schema")
        print("   - Name: get_file_schema")
        print("   - Description: Gets schema information from CSV or Excel files")
        print("   - Parameters:")
        print("     * file_path (string, required)")
        print("   - Returns: JSON string with column information\n")
        print("3. Enable 'Automatic function calling' or 'Auto tool selection'")
        print("4. Save the agent\n")
        print("To get detailed tool definitions, run:")
        print("  python -m src.news_reporter.tools.generate_tool_definitions")
    else:
        print("\n[SUCCESS] All agents updated with tools!")
        print("Tools are now available in Chat conversations.")


if __name__ == "__main__":
    main()






