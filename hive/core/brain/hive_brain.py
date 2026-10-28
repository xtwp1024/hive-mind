"""
HiveBrain - 统一认知大脑
========================

整合所有认知模块，提供统一的思考接口。

架构:
    Perceiver → 感知输入
         ↓
    MemoryHierarchy → 检索记忆上下文
         ↓
    GoalDecomposer → 目标分解
         ↓
    Planner → 生成执行计划
         ↓
    Executor (QueenBee) → 执行
         ↓
    Reflector → 反思学习

Usage:
    brain = HiveBrain()
    result = await brain.think("帮我分析这个项目")
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cognitive_bus import CognitiveBus, Message, MessageType, get_or_create_cognitive_bus
from .working_memory import WorkingMemory
from .memory_hierarchy import MemoryHierarchy, Context, Experience
from .perception import Perceiver, Perception, Intent
from .goal_decomposer import GoalDecomposer, SubGoal
from .planner import Planner, Plan, ExecutionStrategy
from .reflector import Reflector, ReflectionResult, Insight
from .critic import Critic, Evaluation, Issue
from .strategy_selector import StrategySelector, Strategy, CognitiveMode
from .self_monitor import SelfMonitor, CognitiveMetrics, HealthStatus
from .attention_economy import AttentionEconomy, AttentionBudget

logger = logging.getLogger("Hive.Brain")


@dataclass
class BrainConfig:
    """大脑配置"""

    name: str = "HiveBrain"
    max_think_depth: int = 10  # 最大思考深度
    enable_reflection: bool = True  # 是否启用反思
    enable_self_monitor: bool = True  # 是否启用自我监控
    llm_for_decomposition: bool = True  # 是否使用LLM分解
    max_parallel_execution: int = 3  # 最大并行执行数
    reflection_threshold: float = 0.3  # 反思阈值（低置信度时反思）

    # 容量配置
    working_memory_capacity: int = 50
    episodic_limit: int = 1000
    semantic_limit: int = 5000


@dataclass
class CognitiveState:
    """
    认知状态

    反映当前大脑的运行状态
    """

    attention_load: float = 0.0  # 注意力负载 (0-1)
    memory_pressure: float = 0.0  # 记忆压力 (0-1)
    confidence: float = 0.5  # 全局置信度 (0-1)
    active_goals: int = 0  # 活跃目标数
    thinking_depth: int = 0  # 当前思考深度
    is_blocked: bool = False  # 是否阻塞
    blocked_reason: Optional[str] = None
    last_think_duration_ms: float = 0.0


@dataclass
class ThinkResult:
    """思考结果"""

    success: bool
    message: str
    perception: Optional[Perception] = None
    context: Optional[Context] = None
    plan: Optional[Plan] = None
    sub_goals: List[SubGoal] = field(default_factory=list)
    cognitive_state: Optional[CognitiveState] = None
    execution_results: List[Any] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HiveBrain:
    """
    统一认知大脑

    整合感知、记忆、推理、决策模块，提供完整的认知闭环。
    """

    def __init__(self, config: Optional[BrainConfig] = None):
        self.config = config or BrainConfig()
        self._initialized = False

        # 核心组件
        self.bus: Optional[CognitiveBus] = None
        self.working_memory: Optional[WorkingMemory] = None
        self.memory_hierarchy: Optional[MemoryHierarchy] = None
        self.perceiver: Optional[Perceiver] = None
        self.goal_decomposer: Optional[GoalDecomposer] = None
        self.planner: Optional[Planner] = None

        # Phase 2 组件
        self.reflector: Optional[Reflector] = None
        self.critic: Optional[Critic] = None
        self.strategy_selector: Optional[StrategySelector] = None
        self.self_monitor: Optional[SelfMonitor] = None

        # Phase 3: 注意力经济系统
        self.attention_economy: AttentionEconomy = AttentionEconomy()

        # 状态
        self._state = CognitiveState()
        self._think_count = 0
        self._error_count = 0

        # 统计
        self._stats = {
            "thinks": 0,
            "successful_thinks": 0,
            "failed_thinks": 0,
            "total_think_time_ms": 0.0,
            "reflections": 0,
            "evaluations": 0,
            "strategies_selected": 0,
        }

        logger.info(f"🧠 HiveBrain created (config={self.config.name})")

    async def initialize(self):
        """初始化大脑"""
        if self._initialized:
            return

        logger.info("🧠 Initializing HiveBrain...")

        # 初始化消息总线
        self.bus = await get_or_create_cognitive_bus()
        await self.bus.start()

        # 初始化工作记忆
        self.working_memory = WorkingMemory(
            max_items=self.config.working_memory_capacity
        )

        # 初始化记忆层次
        self.memory_hierarchy = MemoryHierarchy(
            episodic_limit=self.config.episodic_limit,
            semantic_limit=self.config.semantic_limit,
        )
        await self.memory_hierarchy.initialize()

        # 初始化感知器
        self.perceiver = Perceiver()

        # 初始化目标分解器
        self.goal_decomposer = GoalDecomposer(
            use_llm=self.config.llm_for_decomposition
        )

        # 初始化规划器
        self.planner = Planner(
            max_parallel=self.config.max_parallel_execution
        )

        # Phase 2: 初始化反思器
        if self.config.enable_reflection:
            self.reflector = Reflector(memory_hierarchy=self.memory_hierarchy)

        # Phase 2: 初始化批评器
        self.critic = Critic()

        # Phase 2: 初始化策略选择器
        self.strategy_selector = StrategySelector()

        # Phase 2: 初始化自我监控器
        if self.config.enable_self_monitor:
            self.self_monitor = SelfMonitor(working_memory=self.working_memory)
            await self.self_monitor.start()

        # 订阅消息
        self._subscribe_to_messages()

        self._initialized = True
        logger.info("🧠 HiveBrain initialized successfully")

    def _subscribe_to_messages(self):
        """订阅认知消息"""
        if not self.bus:
            return

        # 感知消息
        self.bus.subscribe(MessageType.PERCEPTION, self._handle_perception)

        # 记忆消息
        self.bus.subscribe(MessageType.MEMORY_RECALL, self._handle_memory_recall)
        self.bus.subscribe(MessageType.MEMORY_STORE, self._handle_memory_store)

        # 目标消息
        self.bus.subscribe(MessageType.GOAL_DECOMPOSE, self._handle_goal_decompose)

        # 执行结果
        self.bus.subscribe(MessageType.EXECUTION_RESULT, self._handle_execution_result)

    async def think(self, input_text: str) -> ThinkResult:
        """
        主思考循环

        Args:
            input_text: 输入文本

        Returns:
            ThinkResult: 思考结果
        """
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        self._think_count += 1
        self._state.thinking_depth = 0

        # ===== 注意力预算检查 =====
        # 自动恢复预算
        self.attention_economy.auto_recharge()

        # 检查预算是否充足，不足则切换 System1 快速响应
        forced_system1 = False
        if self.attention_economy.is_low_budget():
            logger.warning(
                f"⚠️ Low attention budget ({self.attention_economy.budget.available:.1f}), "
                f"forcing System1 mode"
            )
            forced_system1 = True

        try:
            # ===== 阶段1: 感知 =====
            perception = await self._perceive(input_text)

            # ===== 阶段2: 策略选择 =====
            if forced_system1:
                # 预算不足时强制使用 System1
                strategy = await self._select_strategy(perception)
                if strategy and strategy.mode != CognitiveMode.SYSTEM1:
                    strategy.mode = CognitiveMode.SYSTEM1
                    logger.info("🔄 Switched to System1 due to low attention budget")
            else:
                strategy = await self._select_strategy(perception)

            # ===== 阶段3: 记忆检索 =====
            context = await self._recall_context(perception)

            # ===== 阶段4: 目标分解 =====
            sub_goals = await self._decompose_goal(perception)

            # ===== 阶段5: 规划 =====
            plan = await self._plan(sub_goals, context, strategy)

            # ===== 阶段6: 更新工作记忆 =====
            await self._update_working_memory(perception, sub_goals)

            # ===== 阶段7: 反思（如需要）=====
            insights = []
            if self.config.enable_reflection:
                if perception.confidence < self.config.reflection_threshold:
                    insights = await self._reflect(perception, context, plan)
                elif strategy.mode == CognitiveMode.SYSTEM1:
                    # System1 快速反思
                    insights = [f"使用 System1 快速响应，置信度={perception.confidence:.2f}"]

            # 计算认知状态
            self._update_cognitive_state()

            duration_ms = (time.time() - start_time) * 1000
            self._stats["thinks"] += 1
            self._stats["successful_thinks"] += 1
            self._stats["total_think_time_ms"] += duration_ms
            self._state.last_think_duration_ms = duration_ms

            result = ThinkResult(
                success=True,
                message="思考完成",
                perception=perception,
                context=context,
                plan=plan,
                sub_goals=sub_goals,
                cognitive_state=self._get_cognitive_state_snapshot(),
                insights=insights,
                metadata={
                    "strategy": strategy.to_dict() if strategy else None,
                },
            )

            logger.info(
                f"🧠 Think completed: intent={perception.intent.value}, "
                f"mode={strategy.mode.value if strategy else 'unknown'}, "
                f"goals={len(sub_goals)}, "
                f"confidence={perception.confidence:.2f}, "
                f"duration={duration_ms:.0f}ms"
            )

            return result

        except Exception as e:
            self._error_count += 1
            self._stats["failed_thinks"] += 1
            duration_ms = (time.time() - start_time) * 1000

            logger.error(f"❌ Think failed: {e}", exc_info=True)

            return ThinkResult(
                success=False,
                message=f"思考失败: {str(e)}",
                cognitive_state=self._get_cognitive_state_snapshot(),
            )

    async def _perceive(self, text: str) -> Perception:
        """感知阶段"""
        # 消耗感知预算
        self.attention_economy.consume("perception")

        if self.perceiver:
            perception = await self.perceiver.process(text)
        else:
            perception = Perception(original_text=text)

        # 发布感知消息
        if self.bus:
            await self.bus.publish(
                Message(
                    type=MessageType.PERCEPTION,
                    data=perception.to_dict(),
                ),
                source="HiveBrain",
            )

        return perception

    async def _recall_context(self, perception: Perception) -> Context:
        """记忆检索阶段"""
        # 消耗记忆检索预算
        self.attention_economy.consume("memory_recall")

        if self.memory_hierarchy:
            context = await self.memory_hierarchy.recall(
                query=perception.original_text,
                layers=perception.suggested_layers,
                limit=10,
            )
        else:
            context = Context()

        # 发布记忆检索消息
        if self.bus:
            await self.bus.publish(
                Message(
                    type=MessageType.MEMORY_RECALL,
                    data=context.to_dict(),
                ),
                source="HiveBrain",
            )

        return context

    async def _decompose_goal(self, perception: Perception) -> List[SubGoal]:
        """目标分解阶段"""
        # 消耗目标分解预算
        self.attention_economy.consume("goal_decomposition")

        if not perception.has_goal:
            return []

        if self.goal_decomposer:
            sub_goals = await self.goal_decomposer.decompose(
                goal=perception.original_text,
                context={"intent": perception.intent.value},
            )
        else:
            sub_goals = []

        # 发布目标分解消息
        if self.bus:
            await self.bus.publish(
                Message(
                    type=MessageType.GOAL,
                    data={"sub_goals": [sg.to_dict() for sg in sub_goals]},
                ),
                source="HiveBrain",
            )

        return sub_goals

    async def _plan(
        self,
        sub_goals: List[SubGoal],
        context: Context,
        strategy=None,
    ) -> Optional[Plan]:
        """规划阶段"""
        # 消耗规划预算
        self.attention_economy.consume("planning")

        if not sub_goals:
            return None

        plan_context = {
            "goal": context.working.content if context.working else None,
            "strategy": strategy.to_dict() if strategy else None,
        }

        if self.planner:
            plan = await self.planner.plan(
                sub_goals=sub_goals,
                context=plan_context,
            )
        else:
            plan = None

        # 发布计划消息
        if self.bus and plan:
            await self.bus.publish(
                Message(
                    type=MessageType.PLAN,
                    data=plan.to_dict(),
                ),
                source="HiveBrain",
            )

        return plan

    async def _select_strategy(self, perception: Perception):
        """策略选择阶段"""
        self._stats["strategies_selected"] += 1

        if self.strategy_selector:
            return await self.strategy_selector.select_for_goal(
                goal=perception.original_text,
                context={"perception": perception.to_dict()},
            )
        return None

    async def _update_working_memory(
        self,
        perception: Perception,
        sub_goals: List[SubGoal],
    ):
        """更新工作记忆"""
        if not self.working_memory:
            return

        # 添加感知到工作记忆
        await self.working_memory.add(
            content=perception.original_text,
            tags=[perception.intent.value],
            attention_level=1.0,
        )

        # 添加目标到工作记忆
        for goal in sub_goals:
            await self.working_memory.push_goal(
                description=goal.description,
                priority=goal.priority,
            )

    async def _reflect(
        self,
        perception: Perception,
        context: Context,
        plan: Optional[Plan] = None,
    ) -> List[str]:
        """
        反思阶段

        当置信度低时，进行深度反思
        """
        # 消耗反思预算
        if not self.attention_economy.consume("reflection"):
            logger.warning("⚠️ Insufficient budget for reflection, skipping")
            return ["Skipped reflection due to low attention budget"]

        insights = []
        self._stats["reflections"] += 1

        try:
            # 使用 Reflector 进行深度反思
            if self.reflector:
                result = await self.reflector.reflect(
                    result={
                        "success": True,
                        "description": perception.original_text,
                        "confidence": perception.confidence,
                    },
                    perception=perception,
                )

                # 提取洞察文本
                insights = [i.content for i in result.insights]

                if result.patterns:
                    insights.extend([f"模式: {p}" for p in result.patterns[:3]])

                if result.reflex_updated:
                    insights.append("已更新反射记忆")

            # 简单反思：基于记忆上下文调整置信度
            if context.confidence > perception.confidence:
                insights.append(
                    f"置信度提升: 感知{perception.confidence:.2f} → "
                    f"上下文{context.confidence:.2f}"
                )

            # 发布反思消息
            if self.bus:
                await self.bus.publish(
                    Message(
                        type=MessageType.INSIGHT,
                        data={"insights": insights},
                    ),
                    source="HiveBrain",
                )

        except Exception as e:
            logger.warning(f"⚠️ Reflection error: {e}")

        return insights

    async def store_experience(
        self,
        content: str,
        success: bool = True,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ):
        """存储经验到记忆"""
        if not self.memory_hierarchy:
            return

        experience = Experience(
            content=content,
            success=success,
            importance=importance,
            tags=tags or [],
        )

        await self.memory_hierarchy.store(experience)

    def _update_cognitive_state(self):
        """更新认知状态"""
        if self.working_memory:
            self._state.attention_load = self.working_memory.get_attention_load()
            self._state.memory_pressure = self.working_memory.get_memory_pressure()
            self._state.active_goals = self.working_memory.get_active_goal_count()

        # 从 SelfMonitor 获取并更新置信度
        if self.self_monitor:
            metrics = self.self_monitor.get_current_metrics()
            if metrics:
                self._state.confidence = metrics.confidence
        else:
            # 如果没有 SelfMonitor，基于工作记忆状态计算基础置信度
            base_confidence = 0.5
            if self.working_memory:
                attention = self._state.attention_load
                memory = self._state.memory_pressure
                # 置信度随注意力负载和记忆压力下降而降低
                self._state.confidence = max(0.1, min(0.9,
                    base_confidence * (1.0 - attention * 0.3) * (1.0 - memory * 0.2)
                ))

    def _get_cognitive_state_snapshot(self) -> CognitiveState:
        """获取认知状态快照"""
        return CognitiveState(
            attention_load=self._state.attention_load,
            memory_pressure=self._state.memory_pressure,
            confidence=self._state.confidence,
            active_goals=self._state.active_goals,
            thinking_depth=self._state.thinking_depth,
            is_blocked=self._state.is_blocked,
            blocked_reason=self._state.blocked_reason,
            last_think_duration_ms=self._state.last_think_duration_ms,
        )

    # ===== 消息处理器 =====

    async def _handle_perception(self, message: Message):
        """处理感知消息"""
        logger.debug(f"📨 Received PERCEPTION message: {message.id}")

    async def _handle_memory_recall(self, message: Message):
        """处理记忆检索消息"""
        logger.debug(f"📨 Received MEMORY_RECALL message: {message.id}")

    async def _handle_memory_store(self, message: Message):
        """处理记忆存储消息"""
        logger.debug(f"📨 Received MEMORY_STORE message: {message.id}")

    async def _handle_goal_decompose(self, message: Message):
        """处理目标分解消息"""
        logger.debug(f"📨 Received GOAL_DECOMPOSE message: {message.id}")

    async def _handle_execution_result(self, message: Message):
        """处理执行结果消息"""
        logger.debug(f"📨 Received EXECUTION_RESULT message: {message.id}")

    # ===== 公共API =====

    async def get_status(self) -> Dict[str, Any]:
        """获取大脑状态"""
        status = {
            "initialized": self._initialized,
            "config": {
                "name": self.config.name,
                "enable_reflection": self.config.enable_reflection,
                "enable_self_monitor": self.config.enable_self_monitor,
                "llm_for_decomposition": self.config.llm_for_decomposition,
            },
            "state": self._get_cognitive_state_snapshot().__dict__,
            "stats": self._stats,
            "bus_stats": self.bus.get_stats() if self.bus else {},
            "memory_stats": self.memory_hierarchy.get_stats() if self.memory_hierarchy else {},
        }

        # Phase 2 模块状态
        if self.reflector:
            status["reflector_stats"] = self.reflector.get_stats()
        if self.critic:
            status["critic_stats"] = self.critic.get_stats()
        if self.strategy_selector:
            status["strategy_selector_stats"] = self.strategy_selector.get_stats()
        if self.self_monitor:
            status["self_monitor_stats"] = self.self_monitor.get_stats()
            metrics = self.self_monitor.get_current_metrics()
            status["health_status"] = metrics.to_dict() if metrics else None

        return status

    async def imprint_reflex(self, pattern: str, response: str) -> bool:
        """
        印记学习 - 存储快速反射

        Args:
            pattern: 触发模式
            response: 响应

        Returns:
            bool: 是否成功
        """
        if not self.memory_hierarchy:
            return False

        return await self.memory_hierarchy.imprint(pattern, response)

    async def shutdown(self):
        """关闭大脑"""
        if self.self_monitor:
            await self.self_monitor.stop()

        if self.bus:
            await self.bus.stop()

        if self.memory_hierarchy:
            await self.memory_hierarchy.cleanup()

        self._initialized = False
        logger.info("🧠 HiveBrain shutdown")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "think_count": self._think_count,
            "error_count": self._error_count,
            "avg_think_time_ms": (
                self._stats["total_think_time_ms"] / self._stats["thinks"]
                if self._stats["thinks"] > 0
                else 0
            ),
        }

    def get_budget_status(self) -> Dict[str, Any]:
        """获取注意力预算状态"""
        return self.attention_economy.get_budget_status()


# 全局单例
_global_brain: Optional[HiveBrain] = None


async def get_hive_brain() -> HiveBrain:
    """获取全局HiveBrain单例"""
    global _global_brain
    if _global_brain is None:
        _global_brain = HiveBrain()
        await _global_brain.initialize()
    return _global_brain
