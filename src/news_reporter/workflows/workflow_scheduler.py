"""Workflow Scheduler - Schedule and automate workflow execution"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Callable
import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Types of schedules"""
    ONCE = "once"
    INTERVAL = "interval"  # Every N seconds/minutes/hours
    CRON = "cron"  # Cron expression
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled workflow"""
    schedule_id: str
    workflow_id: str
    schedule_type: ScheduleType
    enabled: bool = True
    next_run: Optional[float] = None  # Timestamp
    last_run: Optional[float] = None
    run_count: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    
    # For interval schedules
    interval_seconds: Optional[float] = None
    
    # For cron/daily/weekly
    time_of_day: Optional[str] = None  # "HH:MM" format
    days_of_week: Optional[List[int]] = None  # 0=Monday, 6=Sunday
    
    # For cron
    cron_expression: Optional[str] = None


class WorkflowScheduler:
    """Schedules and manages automated workflow execution"""
    
    def __init__(self):
        self.schedules: Dict[str, ScheduleConfig] = {}
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    def add_schedule(
        self,
        schedule_id: str,
        workflow_id: str,
        schedule_type: ScheduleType,
        **kwargs
    ) -> ScheduleConfig:
        """
        Add a new schedule.
        
        Args:
            schedule_id: Unique schedule identifier
            workflow_id: Workflow to execute
            schedule_type: Type of schedule
            **kwargs: Additional schedule parameters
        
        Returns:
            ScheduleConfig
        """
        config = ScheduleConfig(
            schedule_id=schedule_id,
            workflow_id=workflow_id,
            schedule_type=schedule_type,
            **kwargs
        )
        
        # Calculate next run time
        config.next_run = self._calculate_next_run(config)
        
        self.schedules[schedule_id] = config
        logger.info(f"Added schedule {schedule_id} for workflow {workflow_id}")
        
        return config
    
    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule"""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            logger.info(f"Removed schedule {schedule_id}")
            return True
        return False
    
    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule"""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = True
            self.schedules[schedule_id].next_run = self._calculate_next_run(self.schedules[schedule_id])
            return True
        return False
    
    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule"""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = False
            return True
        return False
    
    def _calculate_next_run(self, config: ScheduleConfig) -> Optional[float]:
        """Calculate next run time for a schedule"""
        now = time.time()
        
        if config.schedule_type == ScheduleType.ONCE:
            return now if config.next_run is None else config.next_run
        
        elif config.schedule_type == ScheduleType.INTERVAL:
            if config.interval_seconds:
                if config.last_run:
                    return config.last_run + config.interval_seconds
                return now + config.interval_seconds
            return None
        
        elif config.schedule_type == ScheduleType.DAILY:
            if config.time_of_day:
                # Parse time and calculate next occurrence
                hour, minute = map(int, config.time_of_day.split(":"))
                next_run = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= datetime.now():
                    next_run += timedelta(days=1)
                return next_run.timestamp()
            return None
        
        elif config.schedule_type == ScheduleType.WEEKLY:
            if config.time_of_day and config.days_of_week:
                hour, minute = map(int, config.time_of_day.split(":"))
                today = datetime.now()
                current_weekday = today.weekday()  # 0=Monday
                
                # Find next matching day
                for day_offset in range(7):
                    target_day = (current_weekday + day_offset) % 7
                    if target_day in config.days_of_week:
                        next_run = today + timedelta(days=day_offset)
                        next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        if next_run > today:
                            return next_run.timestamp()
                return None
        
        return None
    
    async def start(self, execute_callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        """
        Start the scheduler.
        
        Args:
            execute_callback: Function to call when a schedule triggers
        """
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self._task = asyncio.create_task(self._scheduler_loop(execute_callback))
        logger.info("Workflow scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Workflow scheduler stopped")
    
    async def _scheduler_loop(self, execute_callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        """Main scheduler loop"""
        while self.running:
            try:
                now = time.time()
                ready_schedules = []
                
                # Find schedules ready to run
                for schedule_id, config in self.schedules.items():
                    if not config.enabled:
                        continue
                    
                    if config.next_run and config.next_run <= now:
                        ready_schedules.append((schedule_id, config))
                
                # Execute ready schedules
                for schedule_id, config in ready_schedules:
                    try:
                        logger.info(f"Executing scheduled workflow {config.workflow_id} (schedule: {schedule_id})")
                        await execute_callback(config.workflow_id, config.params)
                        
                        # Update schedule
                        config.last_run = now
                        config.run_count += 1
                        config.next_run = self._calculate_next_run(config)
                        
                        # Remove one-time schedules
                        if config.schedule_type == ScheduleType.ONCE:
                            self.remove_schedule(schedule_id)
                    
                    except Exception as e:
                        logger.error(f"Error executing scheduled workflow {schedule_id}: {e}", exc_info=True)
                
                # Sleep for a short interval before checking again
                await asyncio.sleep(1.0)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # Wait longer on error
    
    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all schedules"""
        return [
            {
                "schedule_id": s.schedule_id,
                "workflow_id": s.workflow_id,
                "schedule_type": s.schedule_type.value,
                "enabled": s.enabled,
                "next_run": datetime.fromtimestamp(s.next_run).isoformat() if s.next_run else None,
                "last_run": datetime.fromtimestamp(s.last_run).isoformat() if s.last_run else None,
                "run_count": s.run_count
            }
            for s in self.schedules.values()
        ]
    
    def get_schedule(self, schedule_id: str) -> Optional[ScheduleConfig]:
        """Get a schedule by ID"""
        return self.schedules.get(schedule_id)


# Global scheduler instance
_global_scheduler = WorkflowScheduler()


def get_workflow_scheduler() -> WorkflowScheduler:
    """Get the global workflow scheduler"""
    return _global_scheduler
