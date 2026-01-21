"""State Checkpointing - Save and restore workflow state for long-running workflows"""

from __future__ import annotations
from typing import Dict, Any, Optional
import json
import time
import logging
from pathlib import Path

from .workflow_state import WorkflowState

logger = logging.getLogger(__name__)


class StateCheckpoint:
    """Manages checkpointing of workflow state"""
    
    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints (default: ./checkpoints)
        """
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path("./checkpoints")
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        run_id: str,
        state: WorkflowState,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint of the workflow state.
        
        Args:
            run_id: Unique run identifier
            state: WorkflowState to checkpoint
            metadata: Additional metadata to store
        
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_data = {
            "run_id": run_id,
            "timestamp": time.time(),
            "state": state.model_dump(),
            "metadata": metadata or {}
        }
        
        checkpoint_file = self.checkpoint_dir / f"{run_id}_checkpoint.json"
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Saved checkpoint to {checkpoint_file}")
        return str(checkpoint_file)
    
    def load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint by run_id.
        
        Args:
            run_id: Unique run identifier
        
        Returns:
            Checkpoint data dict or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{run_id}_checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"Loaded checkpoint from {checkpoint_file}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def restore_state(self, run_id: str) -> Optional[WorkflowState]:
        """
        Restore WorkflowState from checkpoint.
        
        Args:
            run_id: Unique run identifier
        
        Returns:
            Restored WorkflowState or None if checkpoint not found
        """
        checkpoint_data = self.load_checkpoint(run_id)
        if not checkpoint_data:
            return None
        
        try:
            state_data = checkpoint_data["state"]
            state = WorkflowState(**state_data)
            logger.info(f"Restored state for run {run_id}")
            return state
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            return None
    
    def list_checkpoints(self) -> list[str]:
        """List all available checkpoint run_ids"""
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
            run_id = checkpoint_file.stem.replace("_checkpoint", "")
            checkpoints.append(run_id)
        return sorted(checkpoints)
    
    def delete_checkpoint(self, run_id: str) -> bool:
        """Delete a checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"{run_id}_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info(f"Deleted checkpoint {run_id}")
            return True
        return False
