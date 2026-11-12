#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化策略配置

定义不同的进化策略，控制进化引擎的行为模式。
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StrategyType(Enum):
    """进化策略类型"""

    BALANCED = "balanced"  # 平衡模式：修复与创新并重
    INNOVATE = "innovate"  # 创新模式：优先尝试新方案
    HARDEN = "harden"  # 加固模式：优先稳定性和安全性
    REPAIR_ONLY = "repair-only"  # 仅修复：只处理已知错误
    EARLY_STABILIZE = "early-stabilize"  # 早期稳定：快速收敛到稳定状态
    STEADY_STATE = "steady-state"  # 稳态模式：维持现状，最小改动
    AUTO = "auto"  # 自动模式：根据系统状态动态选择


@dataclass
class EvolutionStrategy:
    """进化策略配置"""

    strategy_type: StrategyType = StrategyType.BALANCED

    # 修复相关
    repair_threshold: float = 0.3  # 错误率超过此值触发修复
    max_repair_attempts: int = 3  # 单个问题最大修复尝试次数

    # 创新相关
    innovation_probability: float = 0.2  # 稳定时触发创新的概率
    max_innovations_per_cycle: int = 2  # 每周期最大创新数

    # 安全相关
    require_human_review: bool = False  # 是否需要人工审核
    rollback_on_failure: bool = True  # 失败时自动回滚

    # 负载感知
    load_max: float = 2.0  # 最大负载阈值
    backoff_seconds: int = 60  # 高负载时退避秒数

    # 元数据
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "EvolutionStrategy":
        """从环境变量加载策略配置"""
        strategy_name = os.getenv("EVOLVE_STRATEGY", "balanced").lower()

        # 解析策略类型
        try:
            strategy_type = StrategyType(strategy_name)
        except ValueError:
            strategy_type = StrategyType.BALANCED

        # 根据策略类型设置默认参数
        config = cls._get_strategy_defaults(strategy_type)

        # 环境变量覆盖
        config.load_max = float(os.getenv("EVOLVE_LOAD_MAX", config.load_max))
        config.require_human_review = os.getenv("EVOLVE_REVIEW", "false").lower() == "true"

        return config

    @classmethod
    def _get_strategy_defaults(cls, strategy_type: StrategyType) -> "EvolutionStrategy":
        """获取策略类型的默认配置"""
        defaults = {
            StrategyType.BALANCED: cls(
                strategy_type=strategy_type,
                repair_threshold=0.3,
                innovation_probability=0.2,
            ),
            StrategyType.INNOVATE: cls(
                strategy_type=strategy_type,
                repair_threshold=0.5,
                innovation_probability=0.5,
                max_innovations_per_cycle=5,
            ),
            StrategyType.HARDEN: cls(
                strategy_type=strategy_type,
                repair_threshold=0.1,
                innovation_probability=0.05,
                require_human_review=True,
                rollback_on_failure=True,
            ),
            StrategyType.REPAIR_ONLY: cls(
                strategy_type=strategy_type,
                repair_threshold=0.1,
                innovation_probability=0.0,
                max_innovations_per_cycle=0,
            ),
            StrategyType.EARLY_STABILIZE: cls(
                strategy_type=strategy_type,
                repair_threshold=0.2,
                innovation_probability=0.1,
                max_repair_attempts=5,
            ),
            StrategyType.STEADY_STATE: cls(
                strategy_type=strategy_type,
                repair_threshold=0.5,
                innovation_probability=0.02,
                require_human_review=True,
            ),
            StrategyType.AUTO: cls(
                strategy_type=strategy_type,
                repair_threshold=0.3,
                innovation_probability=0.2,
            ),
        }
        return defaults.get(strategy_type, cls())

    def should_innovate(self, error_rate: float, stability_score: float) -> bool:
        """判断是否应该进行创新"""
        if self.strategy_type == StrategyType.REPAIR_ONLY:
            return False

        if error_rate > self.repair_threshold:
            return False  # 有错误时优先修复

        if self.strategy_type == StrategyType.INNOVATE:
            return stability_score > 0.7

        import random

        return random.random() < self.innovation_probability

    def should_repair(self, error_rate: float) -> bool:
        """判断是否应该进行修复"""
        return error_rate > self.repair_threshold
