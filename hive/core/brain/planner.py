"""
Planner - 执行规划器
====================

功能:
- 子任务排序 (Topological Sort)
- 资源规划 (Resource Planning)
- 风险评估 (Risk Assessment)
- 执行策略选择 (Execution Strategy)

Usage:
    planner = Planner()
    plan = await planner.plan(sub_goals, context)
    # plan.executable_steps -> [(SubGoal, strategy), ...]
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .goal_decomposer import Complexity, SubGoal

logger = logging.getLogger("Hive.Brain.Planner")


class ExecutionStrategy(str, Enum):
    """执行策略"""

    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"  # 并行执行
    CONDITIONAL = "conditional"  # 条件执行
    PIPELINE = "pipeline"  # 流水线
    DIVIDE_AND_CONQUER = "divide_and_conquer"  # 分治


@dataclass
class PlanStep:
    """
    计划步骤

    Attributes:
        sub_goal: 对应的子目标
        strategy: 执行策略
        estimated_duration: 预估执行时间 (秒)
        risk_level: 风险等级 (0-1)
        can_parallelize: 是否可以并行
        resources_needed: 需要的资源
    """

    sub_goal: SubGoal
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    estimated_duration: float = 60.0  # 默认60秒
    risk_level: float = 0.3  # 默认低风险
    can_parallelize: bool = False
    resources_needed: Dict[str, Any] = field(default_factory=dict)
    dependencies_satisfied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.sub_goal.id,
            "description": self.sub_goal.description,
            "strategy": self.strategy.value,
            "estimated_duration": self.estimated_duration,
            "risk_level": self.risk_level,
            "can_parallelize": self.can_parallelize,
        }


@dataclass
class Plan:
    """
    执行计划

    Attributes:
        id: 计划ID
        goal_description: 原始目标描述
        steps: 计划步骤
        total_estimated_duration: 总预估时间
        overall_risk: 整体风险
        parallel_groups: 可以并行的步骤分组
        metadata: 元数据
    """

    id: str
    goal_description: str
    steps: List[PlanStep] = field(default_factory=list)
    total_estimated_duration: float = 0.0
    overall_risk: float = 0.0
    parallel_groups: List[List[str]] = field(default_factory=list)  # step_ids分组
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_executable_order(self) -> List[List[str]]:
        """获取可执行的顺序（分组）"""
        groups = []
        completed = set()

        while len(completed) < len(self.steps):
            ready = []
            for step in self.steps:
                if (
                    step.sub_goal.id not in completed
                    and step.dependencies_satisfied
                    and all(dep in completed for dep in step.sub_goal.dependencies)
                ):
                    ready.append(step.sub_goal.id)

            if not ready:
                # 有循环依赖或其他问题
                break

            groups.append(ready)
            completed.update(ready)

        return groups

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal_description,
            "steps": [s.to_dict() for s in self.steps],
            "estimated_duration": self.total_estimated_duration,
            "overall_risk": self.overall_risk,
            "parallel_groups": self.parallel_groups,
        }


class Planner:
    """
    执行规划器

    功能:
    1. 拓扑排序子目标
    2. 识别可并行的步骤
    3. 评估风险
    4. 生成执行计划
    """

    # 复杂度到预估时间的映射 (秒)
    COMPLEXITY_DURATION = {
        Complexity.LOW: 30.0,
        Complexity.MEDIUM: 120.0,
        Complexity.HIGH: 300.0,
        Complexity.VERY_HIGH: 600.0,
    }

    # 风险关键词
    RISKY_KEYWORDS = {
        "delete",
        "remove",
        "drop",
        "rm",
        "destroy",
        "force",
        "kill",
        "terminate",
        "abort",
        "database",
        "db",
        "disk",
        "partition",
        "root",
        "sudo",
        "admin",
    }

    def __init__(self, max_parallel: int = 5):
        """
        Args:
            max_parallel: 最大并行数
        """
        self.max_parallel = max_parallel
        self._stats = {"plans_created": 0, "parallel_steps": 0}

    async def plan(
        self,
        sub_goals: List[SubGoal],
        context: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        """
        生成执行计划

        Args:
            sub_goals: 子目标列表
            context: 上下文信息

        Returns:
            Plan: 执行计划
        """
        context = context or {}

        if not sub_goals:
            return Plan(
                id=f"plan_{int(time.time() * 1000) % 10000}",
                goal_description="Empty",
            )

        # 1. 拓扑排序
        sorted_goals = self._topological_sort(sub_goals)

        # 2. 识别可并行的组
        parallel_groups = self._identify_parallel_groups(sorted_goals)

        # 3. 评估风险和时间
        plan_steps = []
        for goal in sorted_goals:
            step = self._create_plan_step(goal, parallel_groups)
            plan_steps.append(step)

        # 4. 计算整体风险
        overall_risk = self._calculate_overall_risk(plan_steps)

        # 5. 创建计划
        plan = Plan(
            id=f"plan_{int(time.time() * 1000) % 10000}",
            goal_description=context.get("goal", "Unknown"),
            steps=plan_steps,
            total_estimated_duration=sum(s.estimated_duration for s in plan_steps),
            overall_risk=overall_risk,
            parallel_groups=[[g.id for g in group] for group in parallel_groups],
        )

        self._stats["plans_created"] += 1
        self._stats["parallel_steps"] += sum(len(g) for g in parallel_groups if len(g) > 1)

        logger.info(
            f"🧠 Created plan: {len(plan_steps)} steps, "
            f"~{plan.total_estimated_duration:.0f}s, "
            f"risk={plan.overall_risk:.2f}, "
            f"{len(parallel_groups)} parallel groups"
        )

        return plan

    def _topological_sort(self, goals: List[SubGoal]) -> List[SubGoal]:
        """拓扑排序"""
        # 构建 id→goal 字典，O(1) 查询
        goal_by_id = {g.id: g for g in goals}

        # 构建依赖图
        in_degree = {g.id: 0 for g in goals}
        adj_list = defaultdict(list)

        for goal in goals:
            for dep in goal.dependencies:
                if dep in in_degree:
                    adj_list[dep].append(goal.id)
                    in_degree[goal.id] += 1

        # Kahn's algorithm
        queue = deque([g for g in goals if in_degree[g.id] == 0])
        sorted_goals = []

        while queue:
            goal = queue.popleft()
            sorted_goals.append(goal)

            for neighbor in adj_list[goal.id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(goal_by_id[neighbor])

        # 检查循环依赖
        if len(sorted_goals) != len(goals):
            logger.warning("⚠️ Circular dependency detected in goals")
            # 返回部分排序结果
            remaining = [g for g in goals if g not in sorted_goals]
            sorted_goals.extend(remaining)

        return sorted_goals

    def _identify_parallel_groups(
        self,
        sorted_goals: List[SubGoal],
    ) -> List[List[SubGoal]]:
        """识别可并行的组"""
        groups = []
        current_group = []
        last_level = -1

        for goal in sorted_goals:
            depth = len(goal.dependencies)

            if depth == 0:
                # 根节点
                if current_group and last_level > 0:
                    groups.append(current_group)
                current_group = [goal]
                last_level = 0
            elif depth <= self.max_parallel:
                # 可以并行
                if depth != last_level:
                    if current_group:
                        groups.append(current_group)
                    current_group = [goal]
                    last_level = depth
                else:
                    current_group.append(goal)
            else:
                # 深度太大，不并行
                if current_group:
                    groups.append(current_group)
                groups.append([goal])
                current_group = []
                last_level = -1

        if current_group:
            groups.append(current_group)

        return groups

    def _create_plan_step(
        self,
        goal: SubGoal,
        parallel_groups: List[List[SubGoal]],
    ) -> PlanStep:
        """创建计划步骤"""
        # 估算时间
        duration = self.COMPLEXITY_DURATION.get(goal.complexity, 60.0)

        # 评估风险
        risk = self._assess_risk(goal.description)

        # 判断是否可以并行
        can_parallel = (
            any(goal in group for group in parallel_groups)
            and len([g for g in goal.dependencies if g]) <= 1
        )

        # 选择策略
        if can_parallel and len([g for g in parallel_groups if goal in g]) > 1:
            strategy = ExecutionStrategy.PARALLEL
        else:
            strategy = ExecutionStrategy.SEQUENTIAL

        return PlanStep(
            sub_goal=goal,
            strategy=strategy,
            estimated_duration=duration,
            risk_level=risk,
            can_parallelize=can_parallel,
            dependencies_satisfied=len(goal.dependencies) == 0,
        )

    def _assess_risk(self, description: str) -> float:
        """评估风险 (0-1)"""
        risk = 0.3  # 基础风险
        desc_lower = description.lower()

        # 检查危险关键词
        risky_count = sum(1 for kw in self.RISKY_KEYWORDS if kw in desc_lower)
        risk += risky_count * 0.15

        # 复杂度高
        risk += 0.1

        return min(1.0, risk)

    def _calculate_overall_risk(self, steps: List[PlanStep]) -> float:
        """计算整体风险"""
        if not steps:
            return 0.0

        # 最大风险
        max_risk = max(s.risk_level for s in steps)

        # 平均风险
        avg_risk = sum(s.risk_level for s in steps) / len(steps)

        # 组合
        return max_risk * 0.6 + avg_risk * 0.4

    def suggest_execution_order(
        self,
        plan: Plan,
        max_concurrent: int = 3,
    ) -> List[List[Tuple[PlanStep, ExecutionStrategy]]]:
        """
        建议执行顺序

        Returns:
            List of execution batches, each batch can run in parallel
        """
        batches = []
        completed_ids = set()

        while len(completed_ids) < len(plan.steps):
            batch = []
            batch_ids = set()

            for step in plan.steps:
                if step.sub_goal.id in completed_ids:
                    continue

                # 检查依赖
                deps_satisfied = all(dep in completed_ids for dep in step.sub_goal.dependencies)

                if deps_satisfied:
                    batch.append((step, step.strategy))
                    batch_ids.add(step.sub_goal.id)

                    if len(batch) >= max_concurrent:
                        break

            if not batch:
                # 无法继续，可能是死锁
                logger.warning("⚠️ Execution deadlock detected")
                break

            batches.append(batch)
            completed_ids.update(batch_ids)

        return batches

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()
