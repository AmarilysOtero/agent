"""Workflow Patterns - Advanced orchestration patterns (state machines, event-driven)"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .graph_schema import GraphDefinition
from .workflow_state import WorkflowState

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """Types of workflow patterns"""
    STATE_MACHINE = "state_machine"
    EVENT_DRIVEN = "event_driven"
    SAGA = "saga"
    CHOREOGRAPHY = "choreography"
    ORCHESTRATION = "orchestration"
    PIPELINE = "pipeline"


@dataclass
class StateMachineState:
    """A state in a state machine"""
    state_id: str
    name: str
    entry_action: Optional[Callable] = None
    exit_action: Optional[Callable] = None
    transitions: Dict[str, str] = field(default_factory=dict)  # event -> target_state


@dataclass
class StateMachine:
    """A state machine pattern"""
    machine_id: str
    initial_state: str
    states: Dict[str, StateMachineState] = field(default_factory=dict)
    current_state: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Event:
    """An event in event-driven pattern"""
    event_id: str
    event_type: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class EventHandler:
    """An event handler"""
    handler_id: str
    event_type: str
    handler: Callable[[Event], Any]
    workflow_id: Optional[str] = None
    enabled: bool = True


class WorkflowPatterns:
    """Manages advanced orchestration patterns"""
    
    def __init__(self):
        self.state_machines: Dict[str, StateMachine] = {}
        self.event_handlers: Dict[str, EventHandler] = {}
        self.event_queue: List[Event] = []
        self._event_counter = 0
    
    def create_state_machine(
        self,
        machine_id: str,
        initial_state: str,
        states: Dict[str, StateMachineState]
    ) -> StateMachine:
        """Create a state machine"""
        machine = StateMachine(
            machine_id=machine_id,
            initial_state=initial_state,
            states=states,
            current_state=initial_state
        )
        self.state_machines[machine_id] = machine
        logger.info(f"Created state machine: {machine_id}")
        return machine
    
    def transition_state(
        self,
        machine_id: str,
        event: str
    ) -> Optional[str]:
        """Transition state machine based on event"""
        machine = self.state_machines.get(machine_id)
        if not machine:
            return None
        
        current_state_obj = machine.states.get(machine.current_state)
        if not current_state_obj:
            return None
        
        target_state = current_state_obj.transitions.get(event)
        if not target_state:
            return None
        
        # Execute exit action
        if current_state_obj.exit_action:
            try:
                current_state_obj.exit_action()
            except Exception as e:
                logger.error(f"Error in exit action: {e}")
        
        # Transition
        old_state = machine.current_state
        machine.current_state = target_state
        
        # Record history
        machine.history.append({
            "from_state": old_state,
            "to_state": target_state,
            "event": event,
            "timestamp": datetime.now().isoformat()
        })
        
        # Execute entry action
        new_state_obj = machine.states.get(target_state)
        if new_state_obj and new_state_obj.entry_action:
            try:
                new_state_obj.entry_action()
            except Exception as e:
                logger.error(f"Error in entry action: {e}")
        
        logger.info(f"State machine {machine_id} transitioned: {old_state} -> {target_state}")
        return target_state
    
    def register_event_handler(
        self,
        handler_id: str,
        event_type: str,
        handler: Callable[[Event], Any],
        workflow_id: Optional[str] = None
    ) -> EventHandler:
        """Register an event handler"""
        event_handler = EventHandler(
            handler_id=handler_id,
            event_type=event_type,
            handler=handler,
            workflow_id=workflow_id
        )
        self.event_handlers[handler_id] = event_handler
        logger.info(f"Registered event handler: {handler_id}")
        return event_handler
    
    def emit_event(
        self,
        event_type: str,
        source: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Emit an event"""
        event = Event(
            event_id=f"event_{self._event_counter}",
            event_type=event_type,
            source=source,
            payload=payload or {},
            timestamp=datetime.now()
        )
        
        self._event_counter += 1
        self.event_queue.append(event)
        
        # Process event handlers
        for handler in self.event_handlers.values():
            if not handler.enabled:
                continue
            
            if handler.event_type == event_type:
                try:
                    handler.handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler {handler.handler_id}: {e}")
        
        logger.debug(f"Emitted event: {event_type} from {source}")
        return event
    
    def get_state_machine(self, machine_id: str) -> Optional[StateMachine]:
        """Get a state machine by ID"""
        return self.state_machines.get(machine_id)
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """Get event history"""
        events = list(self.event_queue)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if source:
            events = [e for e in events if e.source == source]
        
        events.sort(key=lambda e: e.timestamp or datetime.min, reverse=True)
        return events[:limit]


# Global patterns instance
_global_patterns = WorkflowPatterns()


def get_workflow_patterns() -> WorkflowPatterns:
    """Get the global workflow patterns instance"""
    return _global_patterns
