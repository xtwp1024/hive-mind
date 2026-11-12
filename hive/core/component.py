# -*- coding: utf-8 -*-
"""
Hive V2.0 Core Component System
-------------------------------
Replaces the monolithic `BaseBee` with a composable `HiveActor` + `HiveComponent` architecture.
Inspired by modern entity-component systems (ECS) and LangChain's Tool pattern.
"""

import abc
from typing import Any, Dict, List, Optional, Type

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class ComponentConfig(BaseModel):
    """Configuration base for all components"""

    enabled: bool = True


class HiveComponent(abc.ABC):
    """
    Atomic unit of functionality.
    E.g., FileSystemComponent, LLMComponent, ShellComponent.
    """

    def __init__(self, config: ComponentConfig = None):
        self.config = config or ComponentConfig()
        self.logger = logger.bind(component=self.__class__.__name__)

    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this component"""
        pass

    @abc.abstractmethod
    def capabilities(self) -> List[str]:
        """List of capabilities provided by this component"""
        pass

    async def initialize(self) -> bool:
        """Async initialization (connection, setup)"""
        return True

    async def shutdown(self) -> bool:
        """Async cleanup"""
        return True


class HiveActor(abc.ABC):
    """
    The evolvable agent entity (formerly 'Bee').
    It is defined NOT by its class name, but by the components it holds.
    """

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.components: Dict[str, HiveComponent] = {}
        self.memory: Dict[str, Any] = {}  # Short-term working memory
        self.logger = logger.bind(actor=name, role=role)

    def add_component(self, component: HiveComponent):
        """Dynamic skill acquisition"""
        self.components[component.name()] = component
        self.logger.info("Equipped component", component=component.name())

    def get_component(self, name: str) -> Optional[HiveComponent]:
        return self.components.get(name)

    def has_capability(self, capability: str) -> bool:
        for comp in self.components.values():
            if capability in comp.capabilities():
                return True
        return False

    async def perform(self, action: str, **kwargs) -> Any:
        """
        Universal Action Handler.
        Instead of compile_code(), it's perform("compile", code=...)
        The Actor routes this to the correct component.
        """
        self.logger.info("Performing action", action=action)

        # Simple routing logic (can be upgraded to LLM routing later)
        # 1. Check if any component explicitly handles this action?
        #    (We need a mapping logic, for now simpler: component methods)

        # For prototype: Direct method call on components via reflection
        # In V2.1: Semantic Routing

        handled = False
        result = None

        for comp in self.components.values():
            # Look for method named 'action_{action}'
            method_name = f"action_{action}"
            if hasattr(comp, method_name):
                method = getattr(comp, method_name)
                result = await method(**kwargs)
                handled = True
                break

        if not handled:
            raise NotImplementedError(
                f"Actor '{self.name}' cannot perform '{action}'. Missing component?"
            )

        return result

    async def run(self):
        """Lifecycle start"""
        for comp in self.components.values():
            await comp.initialize()

    async def stop(self):
        """Lifecycle stop"""
        for comp in self.components.values():
            await comp.shutdown()
