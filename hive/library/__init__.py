# -*- coding: utf-8 -*-
"""
Hive Library - 扩展库模块

包含:
- agentic_framework.py: Agentic框架核心
- mesh_network.py: 自组织网状网络
"""

from hive.library.agentic_framework import (
    IntegrationStatus,
    VersionInfo,
    DependencyConfig,
    ReleaseNote,
    Tool,
    BaseAgent,
    SwarmManifest,
    WorkerBee,
    VersionAnalyzerBee,
    DependencyValidatorBee,
    IntegrationFixerBee,
    AgenticFrameworkOrchestrator,
)

from hive.library.mesh_network import (
    NodeState,
    MessageType,
    NetworkAddress,
    NetworkMessage,
    NodeDNA,
    NodeLoad,
    NetworkNode,
    MeshNode,
    MeshNetwork,
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
