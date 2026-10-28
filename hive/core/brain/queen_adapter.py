"""
QueenAdapter - QueenBee 适配器
==============================

将 HiveBrain 集成到 QueenBee 中，作为蜂群的"大脑皮层"。

工作流程:
    User Input → HiveBrain.think() → 分解目标 → 生成计划
                                              ↓
    QueenBee.execute_plan() ← ← ← ← ← ← ← ← ← ←
         ↓
    执行结果 → HiveBrain.store_experience() → 反思学习

Usage:
    adapter = QueenAdapter(queen_bee, hive_brain)
    await adapter.think_and_execute("帮我分析项目架构")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .hive_brain import HiveBrain, ThinkResult
from .planner import Plan, ExecutionStrategy
from .cognitive_diversity import CognitiveSwarm, CollectiveResult

logger = logging.getLogger("Hive.Brain.QueenAdapter")


@dataclass
class ExecutionBatch:
    """执行批次"""

    batch_id: int
    steps: List[Any]  # PlanStep 或类似对象
    strategy: ExecutionStrategy
    results: List[Any] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed

    @property
    def step_ids(self) -> List[str]:
        return [s.sub_goal.id for s in self.steps]


@dataclass
class QueenAdapterConfig:
    """QueenAdapter 配置"""

    enable_collective_thinking: bool = False  # 是否启用集体思考
    swarm_size: int = 3  # 集体思考时的大脑数量
    complexity_threshold: int = 5  # 复杂度超过此值时启用集体思考
    consensus_threshold: float = 0.6  # 共识阈值


class QueenAdapter:
    """
    QueenBee 适配器

    桥接 HiveBrain 和 QueenBee，实现:
    1. 思考→执行闭环
    2. 经验存储
    3. 反思学习
    4. 错误重试机制
    """

    # 重试配置
    DEFAULT_RETRY_CONFIG = {
        "max_retries": 3,
        "base_delay": 1.0,  # 基础延迟（秒）
        "max_delay": 10.0,  # 最大延迟（秒）
        "exponential_base": 2,  # 指数退避基数
    }

    def __init__(
        self,
        queen_bee,
        hive_brain: HiveBrain,
        retry_config: Optional[Dict[str, Any]] = None,
        collective_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            queen_bee: QueenBee 实例
            hive_brain: HiveBrain 实例
            retry_config: 可选的重试配置
            collective_config: 可选的集体思考配置
                - enable_collective_thinking: bool 是否启用集体思考
                - swarm_size: int 大脑数量
                - complexity_threshold: int 复杂度阈值
                - consensus_threshold: float 共识阈值
        """
        self.queen = queen_bee
        self.brain = hive_brain
        self._execution_batches: List[ExecutionBatch] = []
        self._retry_config = {**self.DEFAULT_RETRY_CONFIG, **(retry_config or {})}

        # 集体思考配置
        self._collective_config = QueenAdapterConfig(
            **{k: v for k, v in (collective_config or {}).items()}
        )
        self._swarm: Optional[CognitiveSwarm] = None
        if self._collective_config.enable_collective_thinking:
            self._swarm = CognitiveSwarm(
                size=self._collective_config.swarm_size
            )

        self._stats = {
            "thinks": 0,
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "collective_thinks": 0,
        }

    async def think_and_execute(self, input_text: str) -> ThinkResult:
        """
        思考并执行

        Args:
            input_text: 输入指令

        Returns:
            ThinkResult: 思考+执行结果
        """
        self._stats["thinks"] += 1

        logger.info(f"🧠 QueenAdapter: thinking '{input_text[:50]}...'")

        # 决定是否使用集体思考
        use_collective = self._should_use_collective(input_text)

        if use_collective and self._swarm:
            think_result = await self._collective_think_and_execute(input_text)
        else:
            think_result = await self._single_think_and_execute(input_text)

        return think_result

    async def _single_think_and_execute(self, input_text: str) -> ThinkResult:
        """单一大脑思考并执行"""
        # 1. 思考
        think_result = await self.brain.think(input_text)

        if not think_result.success:
            self._stats["failures"] += 1
            return think_result

        # 2. 如果有计划，执行
        if think_result.plan and think_result.sub_goals:
            execution_results = await self._execute_plan(think_result.plan)
            think_result.execution_results = execution_results

        # 3. 存储经验
        success = all(
            r.get("success", False) for r in think_result.execution_results if isinstance(r, dict)
        ) if think_result.execution_results else think_result.success

        await self.brain.store_experience(
            content=f"{input_text[:100]}... -> {'成功' if success else '失败'}",
            success=success,
            importance=0.7 if success else 0.9,
            tags=["execution"] + ([think_result.perception.intent.value] if think_result.perception else []),
        )

        if success:
            self._stats["successes"] += 1
        else:
            self._stats["failures"] += 1

        return think_result

    async def _collective_think_and_execute(self, input_text: str) -> ThinkResult:
        """集体思考并执行"""
        self._stats["collective_thinks"] += 1

        logger.info(f"🧠 QueenAdapter: using collective thinking for '{input_text[:50]}...'")

        # 1. 初始化 swarm（如需要）
        if not self._swarm._initialized:
            try:
                await self._swarm.initialize()
            except Exception as e:
                logger.error(f"❌ CognitiveSwarm initialization failed: {e}")
                # 初始化失败，回退到单大脑模式
                return await self._single_think_and_execute(input_text)

        # 2. 集体思考
        collective_result = await self._swarm.collective_think(input_text)

        # 3. 基于最佳大脑结果执行
        best_idx = collective_result.best_brain_index
        if best_idx >= 0 and best_idx < len(collective_result.individual_results):
            think_result = collective_result.individual_results[best_idx]
        else:
            # 回退到第一个成功的结果
            think_result = next(
                (r for r in collective_result.individual_results if r.success),
                collective_result.individual_results[0]
            )

        # 4. 如果有计划且有共识，执行
        if think_result.plan and think_result.sub_goals and collective_result.has_consensus(
            self._collective_config.consensus_threshold
        ):
            execution_results = await self._execute_plan(think_result.plan)
            think_result.execution_results = execution_results

            # 5. 存储经验
            success = all(
                r.get("success", False) for r in execution_results if isinstance(r, dict)
            ) if execution_results else think_result.success

            await self.brain.store_experience(
                content=f"[COLLECTIVE] {input_text[:100]}... -> {'成功' if success else '失败'}",
                success=success,
                importance=0.8 if success else 0.9,
                tags=["collective", "execution"] + (
                    [think_result.perception.intent.value] if think_result.perception else []
                ),
            )

            if success:
                self._stats["successes"] += 1
            else:
                self._stats["failures"] += 1
        else:
            # 无共识或无计划，只存储思考结果
            logger.warning(
                f"⚠️ Collective think had low consensus: {collective_result.confidence:.2f}"
            )
            self._stats["failures"] += 1

        # 添加集体思考元数据
        think_result.metadata["collective_result"] = {
            "confidence": collective_result.confidence,
            "disagreement_points": collective_result.disagreement_points,
            "voting_summary": collective_result.voting_summary,
            "best_brain_index": collective_result.best_brain_index,
        }

        return think_result

    def _should_use_collective(self, task: str) -> bool:
        """
        判断是否应该使用集体思考

        Args:
            task: 任务描述

        Returns:
            bool: 是否使用集体思考
        """
        if not self._collective_config.enable_collective_thinking:
            return False

        # 检查任务复杂度
        task_lower = task.lower()

        # 复杂任务关键词
        complex_keywords = [
            "分析", "评估", "比较", "设计", "规划",
            "复杂", "全面", "深入",
            "analyze", "evaluate", "compare", "design", "plan", "complex",
        ]

        complexity_score = sum(1 for kw in complex_keywords if kw in task_lower)

        return complexity_score >= self._collective_config.complexity_threshold

    async def _execute_plan(self, plan: Plan) -> List[Dict[str, Any]]:
        """
        执行计划

        Args:
            plan: 执行计划

        Returns:
            List[Dict]: 执行结果
        """
        results = []
        self._stats["executions"] += len(plan.steps)

        # 获取执行顺序
        execution_groups = self.brain.planner.suggest_execution_order(
            plan, max_concurrent=self.brain.config.max_parallel_execution
        )

        logger.info(f"🧠 Executing plan: {len(execution_groups)} batches")

        for batch_idx, batch in enumerate(execution_groups):
            batch_results = await self._execute_batch(batch_idx, batch)
            results.extend(batch_results)

            # 如果批次失败，可能需要停止
            if any(not r.get("success", False) for r in batch_results):
                logger.warning(f"⚠️ Batch {batch_idx} had failures")
                # 可以选择停止或继续

        return results

    async def _execute_batch(
        self,
        batch_id: int,
        batch: List[tuple],
    ) -> List[Dict[str, Any]]:
        """
        执行单个批次

        Args:
            batch_id: 批次ID
            batch: [(PlanStep, ExecutionStrategy), ...]

        Returns:
            List[Dict]: 执行结果
        """
        if not batch:
            return []

        first_step, strategy = batch[0]

        if strategy == ExecutionStrategy.PARALLEL:
            # 并行执行
            return await self._execute_parallel(batch_id, batch)
        else:
            # 顺序执行
            return await self._execute_sequential(batch_id, batch)

    async def _execute_sequential(
        self,
        batch_id: int,
        batch: List[tuple],
    ) -> List[Dict[str, Any]]:
        """顺序执行"""
        results = []

        for step, _ in batch:
            result = await self._execute_step(step)
            results.append(result)

            # 如果失败，可以选择停止
            if not result.get("success", False):
                break

        return results

    async def _execute_parallel(
        self,
        batch_id: int,
        batch: List[tuple],
    ) -> List[Dict[str, Any]]:
        """并行执行"""
        tasks = [self._execute_step(step) for step, _ in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "step_id": batch[i][0].sub_goal.id,
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_step(self, step, retry_count: int = 0) -> Dict[str, Any]:
        """
        执行单个步骤（带重试机制）

        Args:
            step: PlanStep
            retry_count: 当前重试次数

        Returns:
            Dict: 执行结果
        """
        max_retries = self._retry_config.get("max_retries", 3)
        base_delay = self._retry_config.get("base_delay", 1.0)
        max_delay = self._retry_config.get("max_delay", 10.0)
        exp_base = self._retry_config.get("exponential_base", 2)

        try:
            goal = step.sub_goal
            logger.debug(f"🧠 Executing step: {goal.description[:50]} (attempt {retry_count + 1})")

            # 根据目标类型路由到不同的执行器
            if hasattr(self.queen, "execute_goal"):
                result = await self.queen.execute_goal(goal)
            elif hasattr(self.queen, "execute_task"):
                # 回退到旧的API
                result = await self.queen.execute_task(goal.description)
            else:
                result = {
                    "success": False,
                    "error": "QueenBee 没有可用的执行方法",
                }

            # 检查是否成功
            if isinstance(result, dict) and result.get("success") is False:
                error = result.get("error", "Unknown error")
                if retry_count < max_retries:
                    # 计算延迟时间（指数退避）
                    delay = min(base_delay * (exp_base ** retry_count), max_delay)
                    logger.warning(
                        f"⚠️ Step failed, retrying in {delay:.1f}s "
                        f"(attempt {retry_count + 1}/{max_retries}): {error}"
                    )
                    self._stats["retries"] += 1
                    await asyncio.sleep(delay)
                    return await self._execute_step(step, retry_count + 1)
                else:
                    logger.error(f"❌ Step failed after {max_retries} retries: {error}")
                    return {
                        "success": False,
                        "step_id": goal.id,
                        "description": goal.description,
                        "error": f"Failed after {max_retries} retries: {error}",
                        "retries_used": retry_count,
                    }

            return {
                "success": True,
                "step_id": goal.id,
                "description": goal.description,
                "result": result,
                "retries_used": retry_count,
            }

        except Exception as e:
            error_msg = str(e)
            if retry_count < max_retries:
                # 计算延迟时间（指数退避）
                delay = min(base_delay * (exp_base ** retry_count), max_delay)
                logger.warning(
                    f"⚠️ Step exception, retrying in {delay:.1f}s "
                    f"(attempt {retry_count + 1}/{max_retries}): {error_msg}"
                )
                self._stats["retries"] += 1
                await asyncio.sleep(delay)
                return await self._execute_step(step, retry_count + 1)
            else:
                logger.error(f"❌ Step execution failed after {max_retries} retries: {error_msg}")
                return {
                    "success": False,
                    "step_id": step.sub_goal.id,
                    "error": f"Failed after {max_retries} retries: {error_msg}",
                    "retries_used": retry_count,
                }

    async def execute_simple(self, command: str) -> Dict[str, Any]:
        """
        简单执行（不经过完整思考流程）

        Args:
            command: 命令

        Returns:
            Dict: 执行结果

        Raises:
            NotImplementedError: 此方法为存根实现，尚未完成
        """
        raise NotImplementedError(
            "execute_simple() is a stub implementation. "
            "Commands are not actually executed - use the full thought flow instead."
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "brain_stats": self.brain.get_stats(),
        }
