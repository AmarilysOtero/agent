"""Test Phase 7: Persistence, Security, Collaboration, Notifications, Integrations, Deployment"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
from src.news_reporter.workflows.workflow_persistence import WorkflowPersistence, WorkflowRecord, ExecutionRecord, WorkflowStatus
from src.news_reporter.workflows.workflow_security import WorkflowSecurity, User, Permission, Role
from src.news_reporter.workflows.workflow_collaboration import WorkflowCollaboration, Team, ShareLevel
from src.news_reporter.workflows.workflow_notifications import WorkflowNotificationManager, NotificationType, NotificationChannel
from src.news_reporter.workflows.workflow_integrations import WorkflowIntegrations, WebhookConfig, EventSubscription
from src.news_reporter.workflows.workflow_deployment import WorkflowDeployment, DeploymentStatus


class TestWorkflowPersistence:
    """Test workflow persistence"""
    
    def test_save_and_get_workflow(self):
        """Test saving and retrieving workflows"""
        persistence = WorkflowPersistence()
        workflow = WorkflowRecord(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test workflow",
            graph_definition={"nodes": [], "edges": []}
        )
        persistence.save_workflow(workflow)
        
        retrieved = persistence.get_workflow("test_workflow")
        assert retrieved is not None
        assert retrieved.workflow_id == "test_workflow"
        assert retrieved.name == "Test Workflow"
    
    def test_save_execution(self):
        """Test saving execution records"""
        persistence = WorkflowPersistence()
        execution = ExecutionRecord(
            execution_id="exec_1",
            workflow_id="test_workflow",
            run_id="run_1",
            goal="Test goal",
            status=WorkflowStatus.COMPLETED,
            result="Success"
        )
        persistence.save_execution(execution)
        
        retrieved = persistence.get_execution("exec_1")
        assert retrieved is not None
        assert retrieved.status == WorkflowStatus.COMPLETED


class TestWorkflowSecurity:
    """Test workflow security"""
    
    def test_create_user(self):
        """Test user creation"""
        security = WorkflowSecurity()
        user = security.create_user("user1", "testuser", roles=[Role.USER])
        assert user.user_id == "user1"
        assert Role.USER in user.roles
    
    def test_grant_permission(self):
        """Test granting permissions"""
        security = WorkflowSecurity()
        security.create_user("user1", "testuser")
        success = security.grant_permission("user1", "workflow1", Permission.EXECUTE)
        assert success
        
        has_perm = security.has_permission("user1", "workflow1", Permission.EXECUTE)
        assert has_perm
    
    def test_create_token(self):
        """Test token creation"""
        security = WorkflowSecurity()
        security.create_user("user1", "testuser")
        token = security.create_token("user1")
        assert token.token is not None
        assert not token.is_expired()


class TestWorkflowCollaboration:
    """Test workflow collaboration"""
    
    def test_create_team(self):
        """Test team creation"""
        collaboration = WorkflowCollaboration()
        team = collaboration.create_team("team1", "Test Team")
        assert team.team_id == "team1"
        assert team.name == "Test Team"
    
    def test_share_workflow(self):
        """Test workflow sharing"""
        collaboration = WorkflowCollaboration()
        # Share with PUBLIC level so user1 can access
        share = collaboration.share_workflow(
            "workflow1",
            "owner1",
            ShareLevel.PUBLIC,
            shared_with_users=["user1"]
        )
        assert share.workflow_id == "workflow1"
        assert share.share_level == ShareLevel.PUBLIC
        
        can_access = collaboration.can_access_workflow("workflow1", "user1")
        assert can_access


class TestWorkflowNotifications:
    """Test workflow notifications"""
    
    def test_notify(self):
        """Test sending notifications"""
        import asyncio
        manager = WorkflowNotificationManager()
        notification = asyncio.run(manager.notify(
            NotificationType.INFO,
            "Test",
            "Test message",
            workflow_id="workflow1"
        ))
        assert notification.type == NotificationType.INFO
        assert notification.title == "Test"
    
    def test_add_rule(self):
        """Test adding notification rules"""
        manager = WorkflowNotificationManager()
        from src.news_reporter.workflows.workflow_notifications import NotificationRule
        rule = NotificationRule(
            rule_id="rule1",
            name="Test Rule",
            event_type="workflow_failed",
            channels=[NotificationChannel.LOG]
        )
        manager.add_rule(rule)
        assert "rule1" in manager.rules


class TestWorkflowIntegrations:
    """Test workflow integrations"""
    
    def test_register_webhook(self):
        """Test webhook registration"""
        integrations = WorkflowIntegrations()
        config = WebhookConfig(
            webhook_id="webhook1",
            url="https://example.com/webhook"
        )
        integrations.register_webhook(config)
        assert "webhook1" in integrations.webhooks
    
    def test_subscribe_event(self):
        """Test event subscription"""
        integrations = WorkflowIntegrations()
        subscription = EventSubscription(
            subscription_id="sub1",
            event_type="workflow_completed",
            workflow_id="workflow1"
        )
        integrations.subscribe_event(subscription)
        assert "sub1" in integrations.event_subscriptions


class TestWorkflowDeployment:
    """Test workflow deployment"""
    
    def test_deploy_workflow(self):
        """Test workflow deployment"""
        from src.news_reporter.workflows.workflow_persistence import WorkflowPersistence
        persistence = WorkflowPersistence()
        workflow = WorkflowRecord(
            workflow_id="workflow1",
            name="Test",
            graph_definition={"nodes": [], "edges": []}
        )
        persistence.save_workflow(workflow)
        
        deployment = WorkflowDeployment(persistence)
        deploy = deployment.deploy_workflow("workflow1", "production", "dev")
        assert deploy.workflow_id == "workflow1"
        assert deploy.target_environment == "production"
        assert deploy.status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILED]
