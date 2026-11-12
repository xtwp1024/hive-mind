"""
AttentionEconomy - 注意力经济系统
================================

基于认知科学的注意力预算管理，模拟人类注意力的有限性。

核心概念:
- 注意力预算: 有限的认知资源
- 操作消耗: 不同认知操作消耗不同量级的预算
- 预算恢复: 随时间缓慢恢复
- 快速路径: 预算不足时切换到 System1

Usage:
    economy = AttentionEconomy()
    if economy.can_afford("planning"):
        economy.consume("planning")
        # 执行规划...
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger("Hive.Brain.AttentionEconomy")


@dataclass
class AttentionBudget:
    """注意力预算"""

    total: float = 100.0  # 总注意力预算
    available: float = 100.0  # 可用预算
    consumed_by: Dict[str, float] = field(default_factory=dict)  # 按操作类型的消耗

    def get_usage_ratio(self) -> float:
        """获取已使用比例 (0-1)"""
        if self.total <= 0:
            return 1.0
        return (self.total - self.available) / self.total

    def is_depleted(self, threshold: float = 10.0) -> bool:
        """检查预算是否耗尽"""
        return self.available < threshold

    def reset(self):
        """重置预算到满额"""
        self.available = self.total
        self.consumed_by.clear()


class AttentionEconomy:
    """
    注意力经济系统

    管理认知操作的注意力消耗，支持预算检查、消耗和恢复。
    """

    # 各操作消耗预算
    ATTENTION_COSTS = {
        "perception": 2.0,
        "memory_recall": 5.0,
        "goal_decomposition": 10.0,
        "planning": 15.0,
        "reflection": 25.0,
        "meta_cognition": 30.0,
    }

    # 默认预算阈值，低于此值切换到 System1 快速响应
    LOW_BUDGET_THRESHOLD = 15.0

    def __init__(self, budget: Optional[AttentionBudget] = None):
        """
        初始化注意力经济系统

        Args:
            budget: 可选的预定义预算，默认创建新的满额预算
        """
        self.budget = budget or AttentionBudget()
        self._last_recharge_time = time.time()
        self._recharge_rate = 0.5  # 每秒恢复的预算量

    def can_afford(self, operation: str) -> bool:
        """
        检查是否足够预算执行操作

        Args:
            operation: 操作名称（如 "planning", "reflection"）

        Returns:
            bool: 预算是否足够
        """
        cost = self.ATTENTION_COSTS.get(operation, 10.0)
        return self.budget.available >= cost

    def consume(self, operation: str) -> bool:
        """
        消耗预算执行操作

        Args:
            operation: 操作名称

        Returns:
            bool: 成功返回 True，预算不足返回 False
        """
        cost = self.ATTENTION_COSTS.get(operation, 10.0)

        if self.budget.available < cost:
            logger.warning(
                f"⚠️ Attention budget insufficient for '{operation}': "
                f"need {cost}, have {self.budget.available:.1f}"
            )
            return False

        self.budget.available -= cost

        # 记录消耗
        if operation not in self.budget.consumed_by:
            self.budget.consumed_by[operation] = 0.0
        self.budget.consumed_by[operation] += cost

        logger.debug(
            f"💰 Consumed {cost:.1f} budget for '{operation}', "
            f"remaining: {self.budget.available:.1f}"
        )
        return True

    def recharge(self, amount: float):
        """
        手动恢复预算

        Args:
            amount: 恢复的预算量
        """
        self.budget.available = min(self.budget.total, self.budget.available + amount)
        self._last_recharge_time = time.time()
        logger.debug(f"⚡ Recharged {amount:.1f} budget, now: {self.budget.available:.1f}")

    def auto_recharge(self, elapsed_seconds: Optional[float] = None):
        """
        自动随时间恢复预算

        Args:
            elapsed_seconds: 经过的时间，不提供则自动计算
        """
        if elapsed_seconds is None:
            elapsed_seconds = time.time() - self._last_recharge_time

        # 衰减恢复率：预算越低恢复越慢
        current_ratio = self.budget.available / self.budget.total
        adaptive_rate = self._recharge_rate * (0.5 + 0.5 * current_ratio)

        recovered = elapsed_seconds * adaptive_rate
        self.budget.available = min(self.budget.total, self.budget.available + recovered)
        self._last_recharge_time = time.time()

        logger.debug(
            f"⚡ Auto-recharged {recovered:.2f} budget in {elapsed_seconds:.1f}s, "
            f"now: {self.budget.available:.1f}"
        )

    def get_budget_status(self) -> Dict:
        """
        获取预算状态

        Returns:
            dict: 包含 total, available, usage_ratio, consumed_by, is_low
        """
        return {
            "total": self.budget.total,
            "available": self.budget.available,
            "usage_ratio": self.budget.get_usage_ratio(),
            "consumed_by": dict(self.budget.consumed_by),
            "is_low": self.budget.is_depleted(self.LOW_BUDGET_THRESHOLD),
            "can_afford": dict({op: self.can_afford(op) for op in self.ATTENTION_COSTS.keys()}),
        }

    def is_low_budget(self) -> bool:
        """检查预算是否过低"""
        return self.budget.is_depleted(self.LOW_BUDGET_THRESHOLD)

    def get_operation_cost(self, operation: str) -> float:
        """获取操作的预算消耗"""
        return self.ATTENTION_COSTS.get(operation, 10.0)

    def reset(self):
        """重置预算"""
        self.budget.reset()
        self._last_recharge_time = time.time()
