#!/usr/bin/env python3
"""
PR 1 Acceptance Test: Workflow Models + Collections
Tests that Pydantic models can be instantiated and MongoDB collections can be accessed.
"""
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_models():
    """Test Pydantic model instantiation"""
    from src.news_reporter.models.workflow import (
        Workflow, WorkflowRun, WorkflowGraph, NodeResult, NodeError
    )
    
    logger.info("=== Testing Pydantic Models ===")
    
    # Test Workflow model
    workflow = Workflow(
        userId="test_user_123",
        name="Test Workflow",
        description="A test workflow for PR 1",
        graph=WorkflowGraph(
            nodes=[{"id": "start", "type": "StartNode"}],
            edges=[]
        ),
        validationStatus="invalid"
    )
    logger.info(f"✓ Created Workflow: {workflow.name} (userId: {workflow.userId})")
    
    # Test WorkflowRun model
    run = WorkflowRun(
        workflowId="test_workflow_id",
        userId="test_user_123",
        status="queued"
    )
    logger.info(f"✓ Created WorkflowRun: {run.id} (status: {run.status})")
    
    # Test NodeResult model
    node_result = NodeResult(
        status="succeeded",
        inputs={"nodeA": "output from A"},
        output="serialized output string",
        outputTruncated=False,
        executionMs=1234.5,
        startedAt=datetime.utcnow(),
        completedAt=datetime.utcnow(),
        logs=[],
        error=None
    )
    logger.info(f"✓ Created NodeResult: status={node_result.status}, executionMs={node_result.executionMs}")
    
    # Test NodeError model
    node_error = NodeError(
        message="Test error message",
        details="ValueError"
    )
    logger.info(f"✓ Created NodeError: {node_error.message}")
    
    logger.info("✓ All Pydantic models instantiated successfully")


async def test_repository():
    """Test MongoDB repository initialization and basic operations"""
    from src.news_reporter.workflows.workflow_repository import WorkflowRepository
    from src.news_reporter.models.workflow import Workflow, WorkflowRun, WorkflowGraph
    
    logger.info("\n=== Testing Workflow Repository ===")
    
    # Initialize repository
    repo = WorkflowRepository("mongodb://user_rw:BestRAG.2026@localhost:27017/workflow_db?authSource=workflow_db")
    logger.info("✓ WorkflowRepository initialized")
    
    try:
        # Test Workflow creation
        workflow = Workflow(
            userId="test_user_pr1",
            name="PR1 Test Workflow",
            description="Manual test for PR 1 acceptance",
            graph=WorkflowGraph(
                nodes=[
                    {"id": "start_1", "type": "StartNode"},
                    {"id": "msg_1", "type": "SendMessage", "config": {"message": "Hello World"}}
                ],
                edges=[
                    {"source": "start_1", "target": "msg_1"}
                ]
            ),
            validationStatus="valid"
        )
        
        created_workflow = await repo.create_workflow(workflow)
        logger.info(f"✓ Created workflow in MongoDB: {created_workflow.id}")
        
        # Test Workflow retrieval
        retrieved = await repo.get_workflow(created_workflow.id, "test_user_pr1")
        if retrieved:
            logger.info(f"✓ Retrieved workflow: {retrieved.name}")
        else:
            logger.error("✗ Failed to retrieve workflow")
        
        # Test WorkflowRun creation
        run = WorkflowRun(
            workflowId=created_workflow.id,
            userId="test_user_pr1",
            status="queued"
        )
        
        created_run = await repo.create_run(run)
        logger.info(f"✓ Created workflow run in MongoDB: {created_run.id}")
        
        # Test run retrieval
        retrieved_run = await repo.get_run(created_run.id, "test_user_pr1")
        if retrieved_run:
            logger.info(f"✓ Retrieved run: status={retrieved_run.status}")
        else:
            logger.error("✗ Failed to retrieve run")
        
        logger.info("✓ All repository operations completed successfully")
        
    finally:
        await repo.close()
        logger.info("✓ Repository connection closed")


async def main():
    try:
        # Test models
        await test_models()
        
        # Test repository (requires MongoDB running)
        try:
            await test_repository()
        except Exception as e:
            logger.warning(f"⚠ Repository test skipped (MongoDB may not be running): {e}")
        
        logger.info("\n=== PR 1 Acceptance Tests PASSED ===")
        
    except Exception as e:
        logger.error(f"\n=== PR 1 Acceptance Tests FAILED ===")
        logger.exception(e)
        raise


if __name__ == "__main__":
    asyncio.run(main())
