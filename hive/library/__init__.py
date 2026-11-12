# -*- coding: utf-8 -*-
"""
Hive Library - 扩展库模块

包含:
- agentic_framework.py: Agentic框架核心
- mesh_network.py: 自组织网状网络
"""

from hive.library.agentic_framework import (
    AgenticFrameworkOrchestrator,
    BaseAgent,
    DependencyConfig,
    DependencyValidatorBee,
    IntegrationFixerBee,
    IntegrationStatus,
    ReleaseNote,
    SwarmManifest,
    Tool,
    VersionAnalyzerBee,
    VersionInfo,
    WorkerBee,
)
from hive.library.mesh_network import (
    MeshNetwork,
    MeshNode,
    MessageType,
    NetworkAddress,
    NetworkMessage,
    NetworkNode,
    NodeDNA,
    NodeLoad,
    NodeState,
)

__all__ = [
    # agentic_framework
    "IntegrationStatus",
    "VersionInfo",
    "DependencyConfig",
    "ReleaseNote",
    "Tool",
    "BaseAgent",
    "SwarmManifest",
    "WorkerBee",
    "VersionAnalyzerBee",
    "DependencyValidatorBee",
    "IntegrationFixerBee",
    "AgenticFrameworkOrchestrator",
    # mesh_network
    "NodeState",
    "MessageType",
    "NetworkAddress",
    "NetworkMessage",
    "NodeDNA",
    "NodeLoad",
    "NetworkNode",
    "MeshNode",
    "MeshNetwork",
]
