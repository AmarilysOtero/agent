"""
Test file to debug why default_workflow.json doesn't find information
when the sequential fallback code does.

This test compares:
1. Sequential workflow (fallback) - run_sequential_goal
2. Graph workflow using default_workflow.json - run_graph_workflow

Input: "Tell me about Alexis Torres"
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.news_reporter.config import Settings
from src.news_reporter.workflows.workflow_factory import run_sequential_goal, run_graph_workflow
from src.news_reporter.workflows.workflow_state import WorkflowState
from src.news_reporter.workflows.condition_evaluator import ConditionEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_sequential_workflow(cfg: Settings, goal: str):
    """Test the sequential (fallback) workflow"""
    print("\n" + "="*80)
    print("TESTING SEQUENTIAL WORKFLOW (FALLBACK)")
    print("="*80)
    
    try:
        print("⏳ Starting sequential workflow (this may take 1-2 minutes)...")
        result = await asyncio.wait_for(
            run_sequential_goal(cfg, goal),
            timeout=300.0  # 5 minute timeout
        )
        print(f"\n✅ Sequential workflow SUCCESS")
        print(f"Result length: {len(result)}")
        print(f"Result preview (first 500 chars):\n{result[:500]}")
        return result
    except asyncio.TimeoutError:
        print(f"\n❌ Sequential workflow TIMED OUT after 5 minutes")
        return None
    except Exception as e:
        print(f"\n❌ Sequential workflow FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_graph_workflow(cfg: Settings, goal: str):
    """Test the graph workflow using default_workflow.json"""
    print("\n" + "="*80)
    print("TESTING GRAPH WORKFLOW (default_workflow.json)")
    print("="*80)
    
    # Load default_workflow.json
    workflow_path = Path(__file__).parent / "src" / "news_reporter" / "workflows" / "default_workflow.json"
    
    try:
        with open(workflow_path, 'r') as f:
            workflow_definition = json.load(f)
        
        print(f"Loaded workflow: {workflow_definition.get('name')}")
        print(f"Entry node: {workflow_definition.get('entry_node_id')}")
        print(f"Nodes: {len(workflow_definition.get('nodes', []))}")
        print(f"Edges: {len(workflow_definition.get('edges', []))}")
        
        print("⏳ Starting graph workflow (this may take 1-2 minutes)...")
        result = await asyncio.wait_for(
            run_graph_workflow(
                cfg=cfg,
                goal=goal,
                workflow_definition=workflow_definition
            ),
            timeout=300.0  # 5 minute timeout
        )
        
        print(f"\n✅ Graph workflow SUCCESS")
        print(f"Result length: {len(result)}")
        print(f"Result preview (first 500 chars):\n{result[:500]}")
        return result
    except asyncio.TimeoutError:
        print(f"\n❌ Graph workflow TIMED OUT after 5 minutes")
        return None
    except Exception as e:
        print(f"\n❌ Graph workflow FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_condition_evaluation():
    """Test condition evaluation with sample state"""
    print("\n" + "="*80)
    print("TESTING CONDITION EVALUATION")
    print("="*80)
    
    # Create a sample state similar to what triage would produce
    state = WorkflowState(goal="Tell me about Alexis Torres")
    
    # Simulate triage results (what the sequential workflow would have)
    state.triage = {
        "intents": ["ai_search", "unknown"],
        "preferred_agent": None,
        "database_id": None
    }
    
    # Test the should_search condition from the JSON
    condition = '"ai_search" in triage.intents or ("unknown" in triage.intents and triage.preferred_agent is not None and triage.database_id is not None)'
    
    print(f"State triage: {state.triage}")
    print(f"State.get('triage.intents'): {state.get('triage.intents')}")
    print(f"Condition: {condition}")
    
    try:
        result = ConditionEvaluator.evaluate(condition, state)
        print(f"✅ Condition evaluation result: {result}")
    except Exception as e:
        print(f"❌ Condition evaluation FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with state set via adapter (like in graph workflow)
    state2 = WorkflowState(goal="Tell me about Alexis Torres")
    # Simulate what TriageAdapter.parse_output does
    state2.set("triage", {
        "intents": ["ai_search"],
        "preferred_agent": None,
        "database_id": None
    })
    
    print(f"\nState2 triage (via set): {state2.get('triage')}")
    print(f"State2.get('triage.intents'): {state2.get('triage.intents')}")
    try:
        result2 = ConditionEvaluator.evaluate(condition, state2)
        print(f"✅ Condition evaluation result (state2): {result2}")
    except Exception as e:
        print(f"❌ Condition evaluation FAILED (state2): {e}")
        import traceback
        traceback.print_exc()
    
    # Test with nested set (what adapter might do)
    state3 = WorkflowState(goal="Tell me about Alexis Torres")
    state3.set("triage", {})  # Initialize as dict
    state3.set("triage.intents", ["ai_search"])
    state3.set("triage.preferred_agent", None)
    state3.set("triage.database_id", None)
    
    print(f"\nState3 triage (nested set): {state3.get('triage')}")
    print(f"State3.get('triage.intents'): {state3.get('triage.intents')}")
    try:
        result3 = ConditionEvaluator.evaluate(condition, state3)
        print(f"✅ Condition evaluation result (state3): {result3}")
    except Exception as e:
        print(f"❌ Condition evaluation FAILED (state3): {e}")
        import traceback
        traceback.print_exc()
    
    # Test the should_search condition result reference
    state4 = WorkflowState(goal="Tell me about Alexis Torres")
    state4.set("conditional.should_search.result", True)
    condition_ref = "should_search condition result"
    
    print(f"\nState4 conditional: {state4.get('conditional')}")
    print(f"Condition reference: {condition_ref}")
    try:
        result4 = ConditionEvaluator.evaluate(condition_ref, state4)
        print(f"✅ Condition reference evaluation result: {result4}")
    except Exception as e:
        print(f"❌ Condition reference evaluation FAILED: {e}")
        import traceback
        traceback.print_exc()


async def debug_workflow_execution(cfg: Settings, goal: str):
    """Debug the workflow execution step by step"""
    print("\n" + "="*80)
    print("DEBUGGING WORKFLOW EXECUTION")
    print("="*80)
    
    from src.news_reporter.workflows.graph_executor import GraphExecutor
    from src.news_reporter.workflows.graph_loader import load_graph_definition
    from src.news_reporter.workflows.workflow_state import WorkflowState
    
    # Load workflow
    workflow_path = Path(__file__).parent / "src" / "news_reporter" / "workflows" / "default_workflow.json"
    graph_def = load_graph_definition(graph_path=str(workflow_path), config=cfg)
    
    # Create executor
    executor = GraphExecutor(graph_def, cfg)
    
    # Execute and capture state at each step
    print(f"Executing workflow with goal: {goal}")
    print(f"Entry nodes: {graph_def.get_entry_nodes()}")
    
    # Monkey-patch to capture state
    original_execute = executor._execute_queue_based
    captured_states = []
    
    async def capture_state_execute(state, entry_nodes, root_context):
        # Capture state before execution
        captured_states.append(("before", state.model_dump()))
        result = await original_execute(state, entry_nodes, root_context)
        # Capture state after execution
        captured_states.append(("after", state.model_dump()))
        return result
    
    executor._execute_queue_based = capture_state_execute
    
    try:
        result = await executor.execute(goal)
        
        print(f"\n✅ Execution completed")
        print(f"Result: {result[:500] if result else 'None'}")
        
        # Print state at key points
        if captured_states:
            print(f"\n📊 State snapshots:")
            for label, state_dict in captured_states:
                print(f"\n  {label.upper()}:")
                print(f"    triage: {state_dict.get('triage')}")
                print(f"    latest: {state_dict.get('latest', '')[:100] if state_dict.get('latest') else 'None'}")
                print(f"    conditional: {state_dict.get('conditional')}")
        
        return result
    except asyncio.TimeoutError:
        print(f"\n❌ Execution TIMED OUT after 5 minutes")
        return None
    except Exception as e:
        print(f"\n❌ Execution FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Print state even on failure
        if captured_states:
            print(f"\n📊 State at failure:")
            for label, state_dict in captured_states[-1:]:
                print(f"  {label}:")
                print(f"    triage: {state_dict.get('triage')}")
                print(f"    latest: {state_dict.get('latest', '')[:100] if state_dict.get('latest') else 'None'}")
        
        return None


async def main():
    """Main test function"""
    print("\n" + "="*80)
    print("WORKFLOW DEBUG TEST")
    print("="*80)
    print("Goal: 'Tell me about Alexis Torres'")
    print("="*80)
    
    # Load config
    try:
        cfg = Settings.from_env()
        print(f"\n✅ Config loaded")
        print(f"  - Triage agent: {cfg.agent_id_triage}")
        print(f"  - AI Search agent: {cfg.agent_id_aisearch}")
        print(f"  - Reporter agents: {cfg.reporter_ids}")
        print(f"  - Reviewer agent: {cfg.agent_id_reviewer}")
    except Exception as e:
        print(f"\n❌ Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        return
    
    goal = "Tell me about Alexis Torres"
    
    # Test condition evaluation first
    test_condition_evaluation()
    
    # Test state update application
    print("\n" + "="*80)
    print("TESTING STATE UPDATE APPLICATION")
    print("="*80)
    
    # Simulate what TriageAdapter.parse_output returns
    state = WorkflowState(goal="Tell me about Alexis Torres")
    triage_data = {
        "intents": ["ai_search"],
        "preferred_agent": None,
        "database_id": None
    }
    
    # Simulate the adapter's state_updates
    updates = {
        "triage": triage_data,
        "triage.preferred_agent": triage_data.get("preferred_agent"),
        "triage.database_id": triage_data.get("database_id"),
        "triage.intents": triage_data.get("intents")
    }
    
    print(f"Updates to apply: {updates}")
    
    # Apply updates (like graph_executor does)
    for path, value in updates.items():
        state.set(path, value)
        print(f"  Set {path} = {value}")
    
    print(f"\nFinal state:")
    print(f"  state.triage: {state.triage}")
    print(f"  state.get('triage.intents'): {state.get('triage.intents')}")
    print(f"  state.get('triage.preferred_agent'): {state.get('triage.preferred_agent')}")
    
    # Test condition evaluation on this state
    condition = '"ai_search" in triage.intents or ("unknown" in triage.intents and triage.preferred_agent is not None and triage.database_id is not None)'
    try:
        result = ConditionEvaluator.evaluate(condition, state)
        print(f"\n✅ Condition evaluation on updated state: {result}")
    except Exception as e:
        print(f"\n❌ Condition evaluation FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Ask user if they want to skip sequential workflow (faster testing)
    import sys
    skip_sequential = "--skip-sequential" in sys.argv or "-s" in sys.argv
    
    # Test sequential workflow (optional)
    sequential_result = None
    if not skip_sequential:
        sequential_result = await test_sequential_workflow(cfg, goal)
    else:
        print("\n⏭️  Skipping sequential workflow (use --skip-sequential to skip)")
    
    # Test graph workflow
    graph_result = await test_graph_workflow(cfg, goal)
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    if sequential_result and graph_result:
        print(f"Sequential result length: {len(sequential_result)}")
        print(f"Graph result length: {len(graph_result)}")
        
        if len(sequential_result) > 100 and len(graph_result) < 100:
            print("\n⚠️  WARNING: Graph workflow produced much shorter result!")
            print("This suggests the workflow may have stopped early or not executed search.")
        elif len(sequential_result) < 100 and len(graph_result) > 100:
            print("\n⚠️  WARNING: Sequential workflow produced shorter result!")
        else:
            print("\n✅ Results are similar in length")
    elif sequential_result and not graph_result:
        print("\n❌ Graph workflow failed but sequential succeeded")
    elif not sequential_result and graph_result:
        print("\n❌ Sequential workflow failed but graph succeeded")
    else:
        print("\n❌ Both workflows failed")
    
    # Debug execution
    print("\n" + "="*80)
    print("DETAILED DEBUG")
    print("="*80)
    await debug_workflow_execution(cfg, goal)


if __name__ == "__main__":
    asyncio.run(main())
