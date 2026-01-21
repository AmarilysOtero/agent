"""Workflow Collaboration - Sharing, permissions, and team features"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .workflow_security import Permission, Role

logger = logging.getLogger(__name__)


class ShareLevel(str, Enum):
    """Sharing levels"""
    PRIVATE = "private"  # Only owner
    TEAM = "team"  # Team members
    ORGANIZATION = "organization"  # All org members
    PUBLIC = "public"  # Anyone


@dataclass
class Team:
    """Team entity"""
    team_id: str
    name: str
    description: Optional[str] = None
    members: List[str] = field(default_factory=list)  # user_ids
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None


@dataclass
class WorkflowShare:
    """Workflow sharing configuration"""
    workflow_id: str
    owner_id: str
    share_level: ShareLevel
    shared_with_users: List[str] = field(default_factory=list)  # user_ids
    shared_with_teams: List[str] = field(default_factory=list)  # team_ids
    permissions: Dict[str, List[Permission]] = field(default_factory=dict)  # user_id/team_id -> [permissions]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WorkflowCollaboration:
    """Manages workflow collaboration, sharing, and team features"""
    
    def __init__(self):
        self.teams: Dict[str, Team] = {}
        self.workflow_shares: Dict[str, WorkflowShare] = {}  # workflow_id -> WorkflowShare
        self.user_teams: Dict[str, List[str]] = {}  # user_id -> [team_ids]
    
    def create_team(
        self,
        team_id: str,
        name: str,
        description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Team:
        """Create a new team"""
        team = Team(
            team_id=team_id,
            name=name,
            description=description,
            created_by=created_by,
            created_at=datetime.now()
        )
        self.teams[team_id] = team
        
        if created_by:
            self.add_team_member(team_id, created_by)
        
        logger.info(f"Created team: {team_id}")
        return team
    
    def add_team_member(self, team_id: str, user_id: str) -> bool:
        """Add a member to a team"""
        team = self.teams.get(team_id)
        if not team:
            return False
        
        if user_id not in team.members:
            team.members.append(user_id)
            
            # Track teams per user
            if user_id not in self.user_teams:
                self.user_teams[user_id] = []
            if team_id not in self.user_teams[user_id]:
                self.user_teams[user_id].append(team_id)
            
            logger.info(f"Added user {user_id} to team {team_id}")
        
        return True
    
    def remove_team_member(self, team_id: str, user_id: str) -> bool:
        """Remove a member from a team"""
        team = self.teams.get(team_id)
        if not team:
            return False
        
        if user_id in team.members:
            team.members.remove(user_id)
            
            if user_id in self.user_teams:
                if team_id in self.user_teams[user_id]:
                    self.user_teams[user_id].remove(team_id)
            
            logger.info(f"Removed user {user_id} from team {team_id}")
            return True
        
        return False
    
    def get_user_teams(self, user_id: str) -> List[Team]:
        """Get all teams a user belongs to"""
        team_ids = self.user_teams.get(user_id, [])
        return [self.teams[tid] for tid in team_ids if tid in self.teams]
    
    def share_workflow(
        self,
        workflow_id: str,
        owner_id: str,
        share_level: ShareLevel = ShareLevel.PRIVATE,
        shared_with_users: Optional[List[str]] = None,
        shared_with_teams: Optional[List[str]] = None
    ) -> WorkflowShare:
        """Share a workflow with users or teams"""
        share = WorkflowShare(
            workflow_id=workflow_id,
            owner_id=owner_id,
            share_level=share_level,
            shared_with_users=shared_with_users or [],
            shared_with_teams=shared_with_teams or [],
            created_at=datetime.now()
        )
        
        self.workflow_shares[workflow_id] = share
        logger.info(f"Shared workflow {workflow_id} with level {share_level.value}")
        
        return share
    
    def update_share_permissions(
        self,
        workflow_id: str,
        user_or_team_id: str,
        permissions: List[Permission]
    ) -> bool:
        """Update permissions for a user or team on a workflow"""
        share = self.workflow_shares.get(workflow_id)
        if not share:
            return False
        
        share.permissions[user_or_team_id] = permissions
        share.updated_at = datetime.now()
        logger.info(f"Updated permissions for {user_or_team_id} on workflow {workflow_id}")
        return True
    
    def can_access_workflow(
        self,
        workflow_id: str,
        user_id: str
    ) -> bool:
        """Check if a user can access a workflow"""
        share = self.workflow_shares.get(workflow_id)
        if not share:
            return False
        
        # Owner always has access
        if share.owner_id == user_id:
            return True
        
        # Check share level
        if share.share_level == ShareLevel.PRIVATE:
            return False
        elif share.share_level == ShareLevel.PUBLIC:
            return True
        elif share.share_level == ShareLevel.TEAM:
            # Check if user is in any shared team
            user_teams = self.user_teams.get(user_id, [])
            return any(tid in share.shared_with_teams for tid in user_teams)
        elif share.share_level == ShareLevel.ORGANIZATION:
            # Would check organization membership (simplified - always true for now)
            return True
        
        # Check explicit user sharing
        if user_id in share.shared_with_users:
            return True
        
        # Check team sharing
        user_teams = self.user_teams.get(user_id, [])
        if any(tid in share.shared_with_teams for tid in user_teams):
            return True
        
        return False
    
    def get_shared_workflows(self, user_id: str) -> List[str]:
        """Get list of workflow IDs shared with a user"""
        shared_workflows = []
        
        for workflow_id, share in self.workflow_shares.items():
            if self.can_access_workflow(workflow_id, user_id):
                shared_workflows.append(workflow_id)
        
        return shared_workflows
    
    def unshare_workflow(
        self,
        workflow_id: str,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None
    ) -> bool:
        """Unshare a workflow from a user or team"""
        share = self.workflow_shares.get(workflow_id)
        if not share:
            return False
        
        if user_id:
            if user_id in share.shared_with_users:
                share.shared_with_users.remove(user_id)
            if user_id in share.permissions:
                del share.permissions[user_id]
        
        if team_id:
            if team_id in share.shared_with_teams:
                share.shared_with_teams.remove(team_id)
            if team_id in share.permissions:
                del share.permissions[team_id]
        
        share.updated_at = datetime.now()
        logger.info(f"Unshared workflow {workflow_id} from {user_id or team_id}")
        return True


# Global collaboration instance
_global_collaboration = WorkflowCollaboration()


def get_workflow_collaboration() -> WorkflowCollaboration:
    """Get the global workflow collaboration instance"""
    return _global_collaboration
