"""
StrategySelector - 策略选择器
=============================

功能:
- System1 vs System2 选择
- 风险等级决策
- 执行策略路由
- 认知模式切换

Usage:
    selector = StrategySelector()
    strategy = await selector.select(context)
    # strategy.mode -> "system1" or "system2"
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Hive.Brain.StrategySelector")


class CognitiveMode(str, Enum):
    """认知模式"""

    SYSTEM1 = "system1"  # 快思考 - 反射/直觉/经验
    SYSTEM2 = "system2"  # 慢思考 - 分析/推理/规划


class ExecutionMode(str, Enum):
    """执行模式"""

    REACTIVE = "reactive"  # 响应式 - 直接执行
    DELIBERATIVE = "deliberative"  # 深思熟虑 - 充分规划
    HYBRID = "hybrid"  # 混合 - 先快后慢


@dataclass
class Strategy:
    """
    选择出的策略

    Attributes:
        mode: 认知模式
        execution_mode: 执行模式
        reasoning: 选择理由
        confidence: 置信度
        suggested_bees: 建议使用的Bee类型
        attention_level: 注意力投入程度 (0-1)
        llm_model: 动态路由模型 (Phase 1 新增)
            - SYSTEM1/simple: glm-flash (快、省钱)
            - SYSTEM2/deliberative: deepseek-chat 或 claude (慢、聪明)
    """

    mode: CognitiveMode
    execution_mode: ExecutionMode
    reasoning: str = ""
    confidence: float = 0.5
    suggested_bees: List[str] = field(default_factory=list)
    attention_level: float = 0.5
    max_duration: float = 60.0  # 最大执行时间(秒)
    parallel_allowed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_model: str = "default"  # Phase 1: 动态模型路由

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "execution_mode": self.execution_mode.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "suggested_bees": self.suggested_bees,
            "attention_level": self.attention_level,
            "max_duration": self.max_duration,
            "parallel_allowed": self.parallel_allowed,
            "llm_model": self.llm_model,
        }


@dataclass
class StrategyContext:
    """
    策略选择上下文

    包含影响策略选择的各种因素
    """

    # 输入特征
    input_text: str = ""
    input_length: int = 0
    has_clear_goal: bool = False
    is_urgent: bool = False
    is_risky: bool = False

    # 认知状态
    cognitive_load: float = 0.5  # 0-1, 认知负载
    memory_pressure: float = 0.5  # 0-1, 记忆压力
    confidence: float = 0.5  # 0-1, 全局置信度
    time_available: float = 60.0  # 可用时间(秒)

    # 历史特征
    is_repeated_task: bool = False
    previous_success_rate: float = 1.0  # 历史成功率
    similar_task_count: int = 0  # 相似任务数

    # 任务特征
    task_complexity: str = "medium"  # low, medium, high, very_high
    task_type: str = "unknown"  # analysis, creation, review, etc.
    estimated_duration: float = 60.0  # 预估时长

    # 风险评估
    risk_level: float = 0.3  # 0-1
    requires_confirmation: bool = False


class StrategySelector:
    """
    策略选择器

    根据上下文选择合适的认知模式和执行策略。

    决策逻辑:
    1. 紧急程度 → 是否需要 System1
    2. 风险等级 → 是否需要更多 deliberation
    3. 历史经验 → 是否可以使用反射
    4. 认知负载 → 是否需要降级
    5. 任务复杂度 → 选择执行模式
    """

    # System1 触发条件
    SYSTEM1_TRIGGERS = {
        "greeting", "goodbye", "help", "status",
        "简单查询", "查看状态", "打招呼",
    }

    # 高风险操作
    RISKY_OPERATIONS = {
        "delete", "remove", "drop", "rm",
        "删除", "销毁", "强制", "kill",
    }

    # 复杂任务关键词
    COMPLEX_TASK_KEYWORDS = {
        "分析", "设计", "实现", "重构", "优化",
        "create", "design", "implement", "refactor", "optimize",
    }

    # ===== Phase 1: 动态模型路由 =====
    # System1 / 简单任务 → glm-flash (快、省钱)
    MODEL_SYSTEM1 = "glm-flash"
    # System2 / 复杂任务 → deepseek-chat (聪明)
    MODEL_SYSTEM2 = "deepseek-chat"
    # Hybrid / 中等 → glm-4 (平衡)
    MODEL_HYBRID = "glm-4"

    def __init__(self):
        self._stats = {
            "selections": 0,
            "system1_count": 0,
            "system2_count": 0,
            "hybrid_count": 0,
        }
        self._last_strategy: Optional[Strategy] = None

    async def select(
        self,
        context: StrategyContext,
    ) -> Strategy:
        """
        选择策略

        Args:
            context: 策略选择上下文

        Returns:
            Strategy: 选中的策略
        """
        self._stats["selections"] += 1

        # 1. 检查是否触发 System1
        if self._should_use_system1(context):
            strategy = await self._select_system1(context)
            self._stats["system1_count"] += 1
        # 2. 检查是否是简单任务
        elif self._is_simple_task(context):
            strategy = await self._select_simple_reactive(context)
            self._stats["system1_count"] += 1
        # 3. 检查是否高风险
        elif self._is_high_risk(context):
            strategy = await self._select_deliberative(context)
            self._stats["system2_count"] += 1
        # 4. 检查是否是复杂任务
        elif self._is_complex_task(context):
            strategy = await self._select_deliberative(context)
            self._stats["system2_count"] += 1
        # 5. 默认混合策略
        else:
            strategy = await self._select_hybrid(context)
            self._stats["hybrid_count"] += 1

        self._last_strategy = strategy

        logger.debug(
            f"🧠 Strategy selected: {strategy.mode.value} / "
            f"{strategy.execution_mode.value}, "
            f"confidence={strategy.confidence:.2f}"
        )

        return strategy

    def _should_use_system1(self, context: StrategyContext) -> bool:
        """判断是否应使用 System1"""
        # 紧急任务
        if context.is_urgent and context.time_available < 10:
            return True

        # 明确的 System1 触发词
        input_lower = context.input_text.lower()[:50]
        for trigger in self.SYSTEM1_TRIGGERS:
            if trigger in input_lower:
                return True

        # 重复任务且历史成功率高
        if context.is_repeated_task and context.previous_success_rate > 0.9:
            return True

        # 认知负载高
        if context.cognitive_load > 0.8:
            return True

        return False

    def _is_simple_task(self, context: StrategyContext) -> bool:
        """判断是否是简单任务"""
        # 输入很短
        if context.input_length < 20:
            return True

        # 预估时间很短
        if context.estimated_duration < 10:
            return True

        # 任务类型是查询/状态
        if context.task_type in {"status", "query", "greeting", "help"}:
            return True

        # 复杂度低
        if context.task_complexity == "low":
            return True

        return False

    def _is_high_risk(self, context: StrategyContext) -> bool:
        """判断是否是高风险任务"""
        # 风险等级高
        if context.risk_level > 0.6:
            return True

        # 包含危险操作词
        input_lower = context.input_text.lower()
        for risky in self.RISKY_OPERATIONS:
            if risky in input_lower:
                return True

        # 需要确认
        if context.requires_confirmation:
            return True

        # 历史成功率低
        if context.previous_success_rate < 0.5:
            return True

        return False

    def _is_complex_task(self, context: StrategyContext) -> bool:
        """判断是否是复杂任务"""
        # 复杂度高
        if context.task_complexity in {"high", "very_high"}:
            return True

        # 预估时间长
        if context.estimated_duration > 300:  # > 5分钟
            return True

        # 包含复杂任务关键词
        input_lower = context.input_text.lower()
        complex_count = sum(1 for kw in self.COMPLEX_TASK_KEYWORDS if kw in input_lower)
        if complex_count >= 2:
            return True

        # 相似任务少，说明可能需要更多推理
        if context.similar_task_count < 2 and context.input_length > 100:
            return True

        return False

    async def _select_system1(
        self,
        context: StrategyContext,
    ) -> Strategy:
        """选择 System1 策略"""
        return Strategy(
            mode=CognitiveMode.SYSTEM1,
            execution_mode=ExecutionMode.REACTIVE,
            reasoning="触发 System1: 反射性任务，使用快速响应",
            confidence=0.9,
            suggested_bees=["Cerebellum"],  # 使用小脑反射
            attention_level=0.3,
            max_duration=min(10.0, context.time_available),
            parallel_allowed=False,
            metadata={"fast_path": True},
            llm_model=self.MODEL_SYSTEM1,  # Phase 1: 动态模型路由
        )

    async def _select_simple_reactive(
        self,
        context: StrategyContext,
    ) -> Strategy:
        """选择简单响应式策略"""
        return Strategy(
            mode=CognitiveMode.SYSTEM1,
            execution_mode=ExecutionMode.REACTIVE,
            reasoning="简单任务，直接执行",
            confidence=0.85,
            suggested_bees=["WorkerBee"],
            attention_level=0.4,
            max_duration=min(30.0, context.estimated_duration),
            parallel_allowed=True,
            metadata={"efficiency_priority": True},
            llm_model=self.MODEL_SYSTEM1,  # Phase 1: 动态模型路由
        )

    async def _select_deliberative(
        self,
        context: StrategyContext,
    ) -> Strategy:
        """选择深思熟虑策略"""
        reasoning_parts = []

        if self._is_high_risk(context):
            reasoning_parts.append("高风险操作")
        if self._is_complex_task(context):
            reasoning_parts.append("复杂任务")
        if context.confidence < 0.5:
            reasoning_parts.append("低置信度")

        reasoning = "深思熟虑: " + ", ".join(reasoning_parts) if reasoning_parts else "全面分析"

        return Strategy(
            mode=CognitiveMode.SYSTEM2,
            execution_mode=ExecutionMode.DELIBERATIVE,
            reasoning=reasoning,
            confidence=0.75,
            suggested_bees=["EngineerBee", "AnalystBee"],
            attention_level=0.9,
            max_duration=context.time_available,
            parallel_allowed=True,
            metadata={
                "planning_required": True,
                "review_required": True,
            },
            llm_model=self.MODEL_SYSTEM2,  # Phase 1: 动态模型路由
        )

    async def _select_hybrid(
        self,
        context: StrategyContext,
    ) -> Strategy:
        """选择混合策略"""
        return Strategy(
            mode=CognitiveMode.SYSTEM2,
            execution_mode=ExecutionMode.HYBRID,
            reasoning="混合策略: 先快速响应，再根据结果调整",
            confidence=0.7,
            suggested_bees=["EngineerBee"],
            attention_level=0.6,
            max_duration=context.estimated_duration * 1.5,
            parallel_allowed=True,
            metadata={
                "fast_start": True,
                "adaptive": True,
            },
            llm_model=self.MODEL_HYBRID,  # Phase 1: 动态模型路由
        )

    async def select_for_goal(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Strategy:
        """
        根据目标选择策略（简化接口）

        Args:
            goal: 目标描述
            context: 可选的额外上下文

        Returns:
            Strategy: 选中的策略
        """
        # 构建上下文
        strategy_context = StrategyContext(
            input_text=goal,
            input_length=len(goal),
            has_clear_goal=True,
            task_type=self._infer_task_type(goal),
            estimated_duration=self._estimate_duration(goal),
            task_complexity=self._assess_complexity(goal),
        )

        return await self.select(strategy_context)

    def _infer_task_type(self, goal: str) -> str:
        """推断任务类型"""
        goal_lower = goal.lower()

        if any(k in goal_lower for k in ["分析", "研究", "看看"]):
            return "analysis"
        elif any(k in goal_lower for k in ["创建", "实现", "写"]):
            return "creation"
        elif any(k in goal_lower for k in ["审查", "检查", "review"]):
            return "review"
        elif any(k in goal_lower for k in ["优化", "改进"]):
            return "optimization"
        elif any(k in goal_lower for k in ["调试", "修复", "debug"]):
            return "debug"
        else:
            return "general"

    def _estimate_duration(self, goal: str) -> float:
        """估算任务时长"""
        base = 30.0  # 默认30秒

        if len(goal) > 100:
            base *= 2
        if len(goal) > 200:
            base *= 2

        return base

    def _assess_complexity(self, goal: str) -> str:
        """评估复杂度"""
        if len(goal) < 50:
            return "low"
        elif len(goal) < 100:
            return "medium"
        elif len(goal) < 200:
            return "high"
        else:
            return "very_high"

    async def adapt_strategy(
        self,
        current: Strategy,
        feedback: Dict[str, Any],
    ) -> Strategy:
        """
        根据反馈调整策略

        Args:
            current: 当前策略
            feedback: 反馈信息 (success, duration, errors, etc.)

        Returns:
            Strategy: 调整后的策略
        """
        success = feedback.get("success", False)
        duration = feedback.get("duration", 0)
        errors = feedback.get("errors", [])

        # 如果失败，切换到更谨慎的策略
        if not success and current.mode == CognitiveMode.SYSTEM1:
            logger.info("🧠 Adapting: System1 failed, switching to System2")
            return Strategy(
                mode=CognitiveMode.SYSTEM2,
                execution_mode=ExecutionMode.DELIBERATIVE,
                reasoning="System1 失败，升级到 System2",
                confidence=0.6,
                suggested_bees=["EngineerBee"],
                attention_level=0.8,
                max_duration=current.max_duration * 2,
                parallel_allowed=True,
                metadata={"adapted": True, "previous_mode": "system1"},
            )

        # 如果超时，增加时间预算
        if duration > current.max_duration * 0.9:
            logger.info("🧠 Adapting: Duration approaching limit, extending")
            return Strategy(
                mode=current.mode,
                execution_mode=current.execution_mode,
                reasoning=f"延长执行时间 (was {current.max_duration:.0f}s)",
                confidence=current.confidence * 0.9,
                suggested_bees=current.suggested_bees,
                attention_level=current.attention_level,
                max_duration=current.max_duration * 2,
                parallel_allowed=current.parallel_allowed,
                metadata={**current.metadata, "duration_extended": True},
            )

        # 否则保持当前策略
        return current

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total = self._stats["selections"] or 1
        return {
            **self._stats,
            "system1_ratio": self._stats["system1_count"] / total,
            "system2_ratio": self._stats["system2_count"] / total,
            "last_strategy": self._last_strategy.to_dict() if self._last_strategy else None,
        }
