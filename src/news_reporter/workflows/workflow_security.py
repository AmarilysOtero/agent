"""Workflow Security - Authentication, authorization, and access control"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Workflow permissions"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class Role(str, Enum):
    """User roles"""
    VIEWER = "viewer"  # Read-only access
    USER = "user"  # Can execute workflows
    EDITOR = "editor"  # Can create/edit workflows
    ADMIN = "admin"  # Full access


@dataclass
class User:
    """User entity"""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: List[Role] = field(default_factory=list)
    permissions: Dict[str, List[Permission]] = field(default_factory=dict)  # workflow_id -> [permissions]
    created_at: Optional[datetime] = None
    is_active: bool = True


@dataclass
class AccessToken:
    """Access token for API authentication"""
    token: str
    user_id: str
    expires_at: datetime
    scopes: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now() > self.expires_at


class WorkflowSecurity:
    """Manages security, authentication, and authorization for workflows"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.tokens: Dict[str, AccessToken] = {}
        self.workflow_permissions: Dict[str, Dict[str, List[Permission]]] = {}  # workflow_id -> {user_id: [permissions]}
    
    def create_user(
        self,
        user_id: str,
        username: str,
        email: Optional[str] = None,
        roles: Optional[List[Role]] = None
    ) -> User:
        """Create a new user"""
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles or [Role.USER],
            created_at=datetime.now()
        )
        self.users[user_id] = user
        logger.info(f"Created user: {user_id}")
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def assign_role(self, user_id: str, role: Role) -> bool:
        """Assign a role to a user"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        if role not in user.roles:
            user.roles.append(role)
            logger.info(f"Assigned role {role.value} to user {user_id}")
        return True
    
    def grant_permission(
        self,
        user_id: str,
        workflow_id: str,
        permission: Permission
    ) -> bool:
        """Grant a permission to a user for a workflow"""
        if workflow_id not in self.workflow_permissions:
            self.workflow_permissions[workflow_id] = {}
        
        if user_id not in self.workflow_permissions[workflow_id]:
            self.workflow_permissions[workflow_id][user_id] = []
        
        if permission not in self.workflow_permissions[workflow_id][user_id]:
            self.workflow_permissions[workflow_id][user_id].append(permission)
            logger.info(f"Granted {permission.value} permission on {workflow_id} to {user_id}")
        
        return True
    
    def revoke_permission(
        self,
        user_id: str,
        workflow_id: str,
        permission: Permission
    ) -> bool:
        """Revoke a permission from a user"""
        if workflow_id in self.workflow_permissions:
            if user_id in self.workflow_permissions[workflow_id]:
                if permission in self.workflow_permissions[workflow_id][user_id]:
                    self.workflow_permissions[workflow_id][user_id].remove(permission)
                    logger.info(f"Revoked {permission.value} permission on {workflow_id} from {user_id}")
                    return True
        return False
    
    def has_permission(
        self,
        user_id: str,
        workflow_id: str,
        permission: Permission
    ) -> bool:
        """Check if user has a specific permission"""
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        
        # Admins have all permissions
        if Role.ADMIN in user.roles:
            return True
        
        # Check role-based permissions
        if permission == Permission.READ:
            if Role.VIEWER in user.roles or Role.USER in user.roles or Role.EDITOR in user.roles:
                return True
        elif permission == Permission.EXECUTE:
            if Role.USER in user.roles or Role.EDITOR in user.roles:
                return True
        elif permission in [Permission.WRITE, Permission.DELETE]:
            if Role.EDITOR in user.roles:
                return True
        
        # Check explicit workflow permissions
        if workflow_id in self.workflow_permissions:
            if user_id in self.workflow_permissions[workflow_id]:
                if permission in self.workflow_permissions[workflow_id][user_id]:
                    return True
        
        # Check user-specific permissions
        if workflow_id in user.permissions:
            if permission in user.permissions[workflow_id]:
                return True
        
        return False
    
    def create_token(
        self,
        user_id: str,
        expires_in_hours: int = 24,
        scopes: Optional[List[str]] = None
    ) -> AccessToken:
        """Create an access token for a user"""
        import secrets
        token = secrets.token_urlsafe(32)
        
        access_token = AccessToken(
            token=token,
            user_id=user_id,
            expires_at=datetime.now() + timedelta(hours=expires_in_hours),
            scopes=scopes or [],
            created_at=datetime.now()
        )
        
        self.tokens[token] = access_token
        logger.info(f"Created access token for user {user_id}")
        return access_token
    
    def validate_token(self, token: str) -> Optional[User]:
        """Validate an access token and return the user"""
        access_token = self.tokens.get(token)
        if not access_token:
            return None
        
        if access_token.is_expired():
            del self.tokens[token]
            return None
        
        return self.users.get(access_token.user_id)
    
    def revoke_token(self, token: str) -> bool:
        """Revoke an access token"""
        if token in self.tokens:
            del self.tokens[token]
            logger.info("Revoked access token")
            return True
        return False
    
    def get_user_permissions(self, user_id: str, workflow_id: str) -> List[Permission]:
        """Get all permissions for a user on a workflow"""
        permissions = []
        
        user = self.users.get(user_id)
        if not user:
            return permissions
        
        # Role-based permissions
        if Role.ADMIN in user.roles:
            return [p for p in Permission]
        
        if Role.EDITOR in user.roles:
            permissions.extend([Permission.READ, Permission.WRITE, Permission.EXECUTE])
        elif Role.USER in user.roles:
            permissions.extend([Permission.READ, Permission.EXECUTE])
        elif Role.VIEWER in user.roles:
            permissions.append(Permission.READ)
        
        # Explicit workflow permissions
        if workflow_id in self.workflow_permissions:
            if user_id in self.workflow_permissions[workflow_id]:
                permissions.extend(self.workflow_permissions[workflow_id][user_id])
        
        # User-specific permissions
        if workflow_id in user.permissions:
            permissions.extend(user.permissions[workflow_id])
        
        return list(set(permissions))  # Remove duplicates


# Global security instance
_global_security = WorkflowSecurity()


def get_workflow_security() -> WorkflowSecurity:
    """Get the global workflow security instance"""
    return _global_security
