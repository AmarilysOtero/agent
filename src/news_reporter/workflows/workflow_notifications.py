"""Workflow Notifications - Error handling, alerts, and notifications"""

from __future__ import annotations
from typing import Dict, List, Optional, Callable, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"
    ALERT = "alert"


class NotificationChannel(str, Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    LOG = "log"
    IN_APP = "in_app"


@dataclass
class NotificationRule:
    """Rule for when to send notifications"""
    rule_id: str
    name: str
    event_type: str  # "workflow_failed", "node_failed", "workflow_completed", etc.
    workflow_id: Optional[str] = None  # None = all workflows
    conditions: Dict[str, Any] = field(default_factory=dict)  # Additional conditions
    channels: List[NotificationChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)  # user_ids or email addresses
    enabled: bool = True


@dataclass
class Notification:
    """A notification message"""
    notification_id: str
    type: NotificationType
    title: str
    message: str
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    channels: List[NotificationChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)


class NotificationHandler:
    """Base class for notification handlers"""
    
    async def send(self, notification: Notification) -> bool:
        """Send a notification"""
        raise NotImplementedError


class LogNotificationHandler(NotificationHandler):
    """Log-based notification handler"""
    
    async def send(self, notification: Notification) -> bool:
        """Send notification to log"""
        level = {
            NotificationType.ERROR: logging.ERROR,
            NotificationType.WARNING: logging.WARNING,
            NotificationType.ALERT: logging.WARNING,
            NotificationType.INFO: logging.INFO,
            NotificationType.SUCCESS: logging.INFO
        }.get(notification.type, logging.INFO)
        
        logger.log(
            level,
            f"[{notification.type.value}] {notification.title}: {notification.message}"
        )
        return True


class WebhookNotificationHandler(NotificationHandler):
    """Webhook-based notification handler"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send(self, notification: Notification) -> bool:
        """Send notification via webhook"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "type": notification.type.value,
                    "title": notification.title,
                    "message": notification.message,
                    "workflow_id": notification.workflow_id,
                    "execution_id": notification.execution_id,
                    "metadata": notification.metadata
                }
                async with session.post(self.webhook_url, json=payload) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class WorkflowNotificationManager:
    """Manages workflow notifications and alerts"""
    
    def __init__(self):
        self.rules: Dict[str, NotificationRule] = {}
        self.notifications: List[Notification] = []
        self.handlers: Dict[NotificationChannel, NotificationHandler] = {
            NotificationChannel.LOG: LogNotificationHandler()
        }
        self._notification_counter = 0
    
    def register_handler(
        self,
        channel: NotificationChannel,
        handler: NotificationHandler
    ) -> None:
        """Register a notification handler"""
        self.handlers[channel] = handler
        logger.info(f"Registered notification handler for {channel.value}")
    
    def add_rule(self, rule: NotificationRule) -> None:
        """Add a notification rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added notification rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a notification rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed notification rule: {rule_id}")
            return True
        return False
    
    async def notify(
        self,
        type: NotificationType,
        title: str,
        message: str,
        workflow_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Create and send a notification"""
        notification = Notification(
            notification_id=f"notif_{self._notification_counter}",
            type=type,
            title=title,
            message=message,
            workflow_id=workflow_id,
            execution_id=execution_id,
            node_id=node_id,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        self._notification_counter += 1
        
        # Find matching rules
        matching_rules = self._find_matching_rules(notification)
        
        # Collect channels and recipients
        channels = set()
        recipients = set()
        
        for rule in matching_rules:
            if rule.enabled:
                channels.update(rule.channels)
                recipients.update(rule.recipients)
        
        notification.channels = list(channels)
        notification.recipients = list(recipients)
        
        # Send notifications
        for channel in notification.channels:
            handler = self.handlers.get(channel)
            if handler:
                try:
                    await handler.send(notification)
                    notification.sent_at = datetime.now()
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel.value}: {e}")
        
        self.notifications.append(notification)
        
        # Keep only last 1000 notifications
        if len(self.notifications) > 1000:
            self.notifications = self.notifications[-1000:]
        
        return notification
    
    def _find_matching_rules(self, notification: Notification) -> List[NotificationRule]:
        """Find rules that match the notification"""
        matching = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check workflow match
            if rule.workflow_id and rule.workflow_id != notification.workflow_id:
                continue
            
            # Check event type (simplified - would match based on notification metadata)
            # For now, match based on notification type
            if notification.type == NotificationType.ERROR:
                if "error" in rule.event_type.lower() or "failed" in rule.event_type.lower():
                    matching.append(rule)
            elif notification.type == NotificationType.SUCCESS:
                if "completed" in rule.event_type.lower() or "success" in rule.event_type.lower():
                    matching.append(rule)
            else:
                matching.append(rule)
        
        return matching
    
    def get_notifications(
        self,
        workflow_id: Optional[str] = None,
        type: Optional[NotificationType] = None,
        limit: int = 100
    ) -> List[Notification]:
        """Get notifications with optional filtering"""
        notifications = list(self.notifications)
        
        if workflow_id:
            notifications = [n for n in notifications if n.workflow_id == workflow_id]
        
        if type:
            notifications = [n for n in notifications if n.type == type]
        
        # Sort by created_at descending
        notifications.sort(key=lambda n: n.created_at or datetime.min, reverse=True)
        
        return notifications[:limit]


# Global notification manager
_global_notifications = WorkflowNotificationManager()


def get_notification_manager() -> WorkflowNotificationManager:
    """Get the global notification manager"""
    return _global_notifications
