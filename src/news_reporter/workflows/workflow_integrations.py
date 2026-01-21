"""Workflow Integrations - Webhooks, events, and external system integration"""

from __future__ import annotations
from typing import Dict, List, Optional, Callable, Any
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class IntegrationType(str, Enum):
    """Types of integrations"""
    WEBHOOK = "webhook"
    EVENT = "event"
    API = "api"
    MESSAGE_QUEUE = "message_queue"


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    webhook_id: str
    url: str
    method: str = "POST"  # GET, POST, PUT, DELETE
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    retry_count: int = 3
    enabled: bool = True


@dataclass
class EventSubscription:
    """Event subscription configuration"""
    subscription_id: str
    event_type: str  # "workflow_started", "workflow_completed", "node_failed", etc.
    workflow_id: Optional[str] = None  # None = all workflows
    handler: Optional[Callable] = None
    webhook_url: Optional[str] = None
    enabled: bool = True


class WebhookHandler:
    """Handles webhook calls"""
    
    async def call(
        self,
        config: WebhookConfig,
        payload: Dict[str, Any]
    ) -> bool:
        """Call a webhook"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    config.method,
                    config.url,
                    json=payload,
                    headers=config.headers,
                    timeout=aiohttp.ClientTimeout(total=config.timeout_seconds)
                ) as response:
                    if response.status >= 200 and response.status < 300:
                        logger.info(f"Webhook {config.webhook_id} called successfully")
                        return True
                    else:
                        logger.warning(f"Webhook {config.webhook_id} returned status {response.status}")
                        return False
        except asyncio.TimeoutError:
            logger.error(f"Webhook {config.webhook_id} timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to call webhook {config.webhook_id}: {e}")
            return False


class WorkflowIntegrations:
    """Manages external integrations for workflows"""
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.event_subscriptions: Dict[str, EventSubscription] = {}
        self.webhook_handler = WebhookHandler()
        self._event_handlers: Dict[str, List[Callable]] = {}  # event_type -> [handlers]
    
    def register_webhook(self, config: WebhookConfig) -> None:
        """Register a webhook"""
        self.webhooks[config.webhook_id] = config
        logger.info(f"Registered webhook: {config.webhook_id}")
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Unregistered webhook: {webhook_id}")
            return True
        return False
    
    async def trigger_webhook(
        self,
        webhook_id: str,
        payload: Dict[str, Any]
    ) -> bool:
        """Trigger a webhook"""
        config = self.webhooks.get(webhook_id)
        if not config:
            logger.warning(f"Webhook {webhook_id} not found")
            return False
        
        if not config.enabled:
            logger.debug(f"Webhook {webhook_id} is disabled")
            return False
        
        # Retry logic
        for attempt in range(config.retry_count):
            success = await self.webhook_handler.call(config, payload)
            if success:
                return True
            
            if attempt < config.retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def subscribe_event(
        self,
        subscription: EventSubscription
    ) -> None:
        """Subscribe to workflow events"""
        self.event_subscriptions[subscription.subscription_id] = subscription
        
        # Register handler
        if subscription.event_type not in self._event_handlers:
            self._event_handlers[subscription.event_type] = []
        
        if subscription.handler:
            self._event_handlers[subscription.event_type].append(subscription.handler)
        
        logger.info(f"Subscribed to event: {subscription.event_type}")
    
    def unsubscribe_event(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        subscription = self.event_subscriptions.get(subscription_id)
        if not subscription:
            return False
        
        # Remove handler
        if subscription.event_type in self._event_handlers:
            if subscription.handler in self._event_handlers[subscription.event_type]:
                self._event_handlers[subscription.event_type].remove(subscription.handler)
        
        del self.event_subscriptions[subscription_id]
        logger.info(f"Unsubscribed from event: {subscription_id}")
        return True
    
    async def emit_event(
        self,
        event_type: str,
        workflow_id: Optional[str],
        data: Dict[str, Any]
    ) -> None:
        """Emit an event to all subscribers"""
        # Find matching subscriptions
        matching_subscriptions = []
        
        for subscription in self.event_subscriptions.values():
            if not subscription.enabled:
                continue
            
            if subscription.event_type != event_type:
                continue
            
            if subscription.workflow_id and subscription.workflow_id != workflow_id:
                continue
            
            matching_subscriptions.append(subscription)
        
        # Call handlers
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, workflow_id, data)
                else:
                    handler(event_type, workflow_id, data)
            except Exception as e:
                logger.error(f"Error in event handler: {e}", exc_info=True)
        
        # Trigger webhooks for matching subscriptions
        for subscription in matching_subscriptions:
            if subscription.webhook_url:
                webhook_config = WebhookConfig(
                    webhook_id=f"event_{subscription.subscription_id}",
                    url=subscription.webhook_url
                )
                await self.webhook_handler.call(webhook_config, {
                    "event_type": event_type,
                    "workflow_id": workflow_id,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
    
    def list_webhooks(self) -> List[WebhookConfig]:
        """List all registered webhooks"""
        return list(self.webhooks.values())
    
    def list_event_subscriptions(
        self,
        event_type: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> List[EventSubscription]:
        """List event subscriptions with optional filtering"""
        subscriptions = list(self.event_subscriptions.values())
        
        if event_type:
            subscriptions = [s for s in subscriptions if s.event_type == event_type]
        
        if workflow_id:
            subscriptions = [s for s in subscriptions if s.workflow_id == workflow_id or s.workflow_id is None]
        
        return subscriptions


# Global integrations instance
_global_integrations = WorkflowIntegrations()


def get_workflow_integrations() -> WorkflowIntegrations:
    """Get the global workflow integrations instance"""
    return _global_integrations
