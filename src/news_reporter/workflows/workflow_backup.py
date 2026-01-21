"""Workflow Backup - Backup and disaster recovery"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import os
from pathlib import Path

from .workflow_persistence import WorkflowPersistence, WorkflowRecord

logger = logging.getLogger(__name__)


class BackupType(str, Enum):
    """Types of backups"""
    FULL = "full"  # Complete backup
    INCREMENTAL = "incremental"  # Only changes since last backup
    DIFFERENTIAL = "differential"  # Changes since last full backup


@dataclass
class Backup:
    """A backup record"""
    backup_id: str
    backup_type: BackupType
    workflow_ids: List[str] = field(default_factory=list)  # Empty = all workflows
    backup_path: str = ""
    size_bytes: int = 0
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowBackupManager:
    """Manages workflow backups and disaster recovery"""
    
    def __init__(self, backup_dir: Optional[str] = None):
        self.backup_dir = Path(backup_dir) if backup_dir else Path("./backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backups: Dict[str, Backup] = {}
        self.persistence: Optional[WorkflowPersistence] = None
    
    def set_persistence(self, persistence: WorkflowPersistence) -> None:
        """Set the persistence layer to use"""
        self.persistence = persistence
    
    def create_backup(
        self,
        backup_id: Optional[str] = None,
        backup_type: BackupType = BackupType.FULL,
        workflow_ids: Optional[List[str]] = None
    ) -> Backup:
        """Create a backup"""
        if not self.persistence:
            raise ValueError("Persistence layer not set")
        
        if not backup_id:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / f"{backup_id}.json"
        
        # Collect workflows to backup
        if workflow_ids:
            workflows = [self.persistence.get_workflow(wid) for wid in workflow_ids if self.persistence.get_workflow(wid)]
        else:
            workflows = self.persistence.list_workflows()
        
        # Serialize workflows
        backup_data = {
            "backup_id": backup_id,
            "backup_type": backup_type.value,
            "created_at": datetime.now().isoformat(),
            "workflows": [w.to_dict() for w in workflows]
        }
        
        # Write to file
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        size_bytes = backup_path.stat().st_size
        
        backup = Backup(
            backup_id=backup_id,
            backup_type=backup_type,
            workflow_ids=[w.workflow_id for w in workflows],
            backup_path=str(backup_path),
            size_bytes=size_bytes,
            created_at=datetime.now()
        )
        
        self.backups[backup_id] = backup
        logger.info(f"Created backup: {backup_id} ({size_bytes} bytes)")
        
        return backup
    
    def restore_backup(
        self,
        backup_id: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Restore a backup"""
        backup = self.backups.get(backup_id)
        if not backup:
            raise ValueError(f"Backup {backup_id} not found")
        
        if not os.path.exists(backup.backup_path):
            raise FileNotFoundError(f"Backup file not found: {backup.backup_path}")
        
        if not self.persistence:
            raise ValueError("Persistence layer not set")
        
        # Load backup data
        with open(backup.backup_path, 'r') as f:
            backup_data = json.load(f)
        
        restored_count = 0
        skipped_count = 0
        errors = []
        
        # Restore workflows
        for workflow_data in backup_data.get("workflows", []):
            workflow_id = workflow_data.get("workflow_id")
            
            # Check if workflow exists
            existing = self.persistence.get_workflow(workflow_id)
            if existing and not overwrite:
                skipped_count += 1
                continue
            
            try:
                workflow = WorkflowRecord(**workflow_data)
                self.persistence.save_workflow(workflow)
                restored_count += 1
            except Exception as e:
                errors.append(f"Error restoring {workflow_id}: {e}")
        
        logger.info(f"Restored backup {backup_id}: {restored_count} workflows restored, {skipped_count} skipped")
        
        return {
            "backup_id": backup_id,
            "restored_count": restored_count,
            "skipped_count": skipped_count,
            "errors": errors
        }
    
    def list_backups(self) -> List[Backup]:
        """List all backups"""
        return list(self.backups.values())
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        backup = self.backups.get(backup_id)
        if not backup:
            return False
        
        # Delete file
        if os.path.exists(backup.backup_path):
            os.remove(backup.backup_path)
        
        del self.backups[backup_id]
        logger.info(f"Deleted backup: {backup_id}")
        return True
    
    def get_backup_info(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a backup"""
        backup = self.backups.get(backup_id)
        if not backup:
            return None
        
        return {
            "backup_id": backup.backup_id,
            "backup_type": backup.backup_type.value,
            "workflow_count": len(backup.workflow_ids),
            "size_bytes": backup.size_bytes,
            "size_mb": backup.size_bytes / (1024 * 1024),
            "created_at": backup.created_at.isoformat() if backup.created_at else None,
            "backup_path": backup.backup_path
        }


# Global backup manager instance
_global_backup_manager = WorkflowBackupManager()


def get_workflow_backup_manager() -> WorkflowBackupManager:
    """Get the global workflow backup manager"""
    return _global_backup_manager
