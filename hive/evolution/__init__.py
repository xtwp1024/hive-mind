#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive Evolution Module - 自我进化引擎

基于 GEP (Genome Evolution Protocol) 的能力进化系统。
参考 autogame-17/evolver 设计理念，为 Hive 蜂群系统提供自我改进能力。

功能模块:
- Evolver: 核心进化引擎
- EvolutionStrategy: 进化策略配置
- DNABridge: DNA 系统集成
- EvoMapBridge: EvoMap 网络连接
- MonitorBridge: 监控系统集成
- TestRunner: 自动测试验证
"""

__version__ = "1.1.0"

from hive.evolution.dna_bridge import DNABridge, get_bridge
from hive.evolution.evolver import EvolutionResult, Evolver
from hive.evolution.strategies import EvolutionStrategy, StrategyType


# 延迟导入可选模块
def get_evomap_bridge():
    """获取 EvoMap 桥接器"""
    from hive.evolution.evomap_bridge import EvoMapBridge
    from hive.evolution.evomap_bridge import get_evomap_bridge as _get

    return _get()


def get_monitor_bridge():
    """获取监控桥接器"""
    from hive.evolution.monitor_bridge import MonitorBridge

    return MonitorBridge()


def get_test_runner():
    """获取测试运行器"""
    from hive.evolution.test_runner import TestRunner

    return TestRunner()


__all__ = [
    # 核心
    "Evolver",
    "EvolutionResult",
    "EvolutionStrategy",
    "StrategyType",
    # 桥接器
    "DNABridge",
    "get_bridge",
    "get_evomap_bridge",
    "get_monitor_bridge",
    "get_test_runner",
]
