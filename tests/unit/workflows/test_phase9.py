"""Test Phase 9: Marketplace, Patterns, Migration, Alerting, Multi-Tenant, Gateway"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pytest
from src.news_reporter.workflows.workflow_marketplace import WorkflowMarketplace, MarketplaceCategory, ListingStatus
from src.news_reporter.workflows.workflow_patterns import WorkflowPatterns, PatternType
from src.news_reporter.workflows.workflow_migration import WorkflowMigration, MigrationType
from src.news_reporter.workflows.workflow_alerting import WorkflowAlerting, AlertSeverity, AlertType
from src.news_reporter.workflows.workflow_multitenant import WorkflowMultiTenant, TenantTier
from src.news_reporter.workflows.workflow_gateway import WorkflowGateway, RateLimitStrategy
from src.news_reporter.workflows.performance_metrics import WorkflowMetrics


class TestWorkflowMarketplace:
    """Test workflow marketplace"""
    
    def test_create_listing(self):
        """Test creating marketplace listings"""
        marketplace = WorkflowMarketplace()
        listing = marketplace.create_listing(
            "listing1",
            "workflow1",
            "Test Workflow",
            "A test workflow",
            MarketplaceCategory.AUTOMATION,
            "user1"
        )
        assert listing.listing_id == "listing1"
        assert listing.category == MarketplaceCategory.AUTOMATION
    
    def test_search_listings(self):
        """Test searching listings"""
        marketplace = WorkflowMarketplace()
        listing = marketplace.create_listing(
            "listing1",
            "workflow1",
            "Test Workflow",
            "A test workflow",
            MarketplaceCategory.AUTOMATION,
            "user1"
        )
        marketplace.publish_listing("listing1")
        results = marketplace.search_listings(query="test")
        assert len(results) > 0


class TestWorkflowPatterns:
    """Test workflow patterns"""
    
    def test_create_state_machine(self):
        """Test creating state machines"""
        patterns = WorkflowPatterns()
        from src.news_reporter.workflows.workflow_patterns import StateMachineState
        states = {
            "start": StateMachineState(state_id="start", name="Start"),
            "end": StateMachineState(state_id="end", name="End")
        }
        machine = patterns.create_state_machine("machine1", "start", states)
        assert machine.machine_id == "machine1"
        assert machine.current_state == "start"
    
    def test_emit_event(self):
        """Test emitting events"""
        patterns = WorkflowPatterns()
        event = patterns.emit_event("test_event", "source1", {"data": "test"})
        assert event.event_type == "test_event"
        assert event.source == "source1"


class TestWorkflowMigration:
    """Test workflow migration"""
    
    def test_add_migration_rule(self):
        """Test adding migration rules"""
        migration = WorkflowMigration()
        from src.news_reporter.workflows.graph_schema import GraphDefinition
        rule = migration.add_migration_rule(
            "rule1",
            "Test Rule",
            MigrationType.VERSION_UPGRADE,
            "1.0",
            "2.0",
            lambda g: g  # Identity transform
        )
        assert rule.rule_id == "rule1"
        assert rule.migration_type == MigrationType.VERSION_UPGRADE


class TestWorkflowAlerting:
    """Test workflow alerting"""
    
    def test_add_alert_rule(self):
        """Test adding alert rules"""
        alerting = WorkflowAlerting()
        rule = alerting.add_alert_rule(
            "rule1",
            "Test Alert",
            AlertType.PERFORMANCE,
            AlertSeverity.HIGH,
            lambda m: m.total_duration_ms > 1000,
            threshold=1000
        )
        assert rule.rule_id == "rule1"
        assert rule.severity == AlertSeverity.HIGH
    
    def test_evaluate_metrics(self):
        """Test evaluating metrics for alerts"""
        import time
        alerting = WorkflowAlerting()
        metrics = WorkflowMetrics(
            run_id="run1",
            goal="test",
            total_duration_ms=50000,
            total_nodes_executed=5,
            successful_nodes=5,
            failed_nodes=0,
            skipped_nodes=0,
            total_retries=0,
            cache_hits=0,
            cache_misses=0,
            start_time=time.time(),
            end_time=time.time()
        )
        alerts = alerting.evaluate_metrics(metrics, "workflow1")
        # Should not trigger for 50s duration
        assert isinstance(alerts, list)


class TestWorkflowMultiTenant:
    """Test workflow multi-tenant"""
    
    def test_create_tenant(self):
        """Test creating tenants"""
        multitenant = WorkflowMultiTenant()
        tenant = multitenant.create_tenant("tenant1", "Test Tenant", TenantTier.BASIC)
        assert tenant.tenant_id == "tenant1"
        assert tenant.tier == TenantTier.BASIC
    
    def test_check_quota(self):
        """Test quota checking"""
        multitenant = WorkflowMultiTenant()
        multitenant.create_tenant("tenant1", "Test", TenantTier.BASIC)
        has_quota = multitenant.check_quota("tenant1", "workflows", 1)
        assert has_quota


class TestWorkflowGateway:
    """Test workflow gateway"""
    
    def test_create_api_key(self):
        """Test creating API keys"""
        gateway = WorkflowGateway()
        api_key = gateway.create_api_key("key1", user_id="user1")
        assert api_key.key_id == "key1"
        assert api_key.user_id == "user1"
        assert api_key.key_value is not None
    
    def test_validate_api_key(self):
        """Test validating API keys"""
        gateway = WorkflowGateway()
        api_key = gateway.create_api_key("key1")
        validated = gateway.validate_api_key(api_key.key_value)
        assert validated is not None
        assert validated.key_id == "key1"
