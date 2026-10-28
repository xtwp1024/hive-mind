"""
GoalDecomposer - 目标分解器
===========================

将复杂目标分解为可执行的子目标。

功能:
- 目标理解
- 子目标分解
- 依赖分析
- 优先级排序
- 执行计划生成

Usage:
    decomposer = GoalDecomposer()
    sub_goals = await decomposer.decompose("帮我构建一个REST API")
    # [
    #   SubGoal("设计数据模型", dependencies=[]),
    #   SubGoal("定义API端点", dependencies=[0]),
    #   SubGoal("实现CRUD", dependencies=[1]),
    #   SubGoal("编写测试", dependencies=[2]),
    # ]
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Hive.Brain.GoalDecomposer")

try:
    from hive.utils.brain_interface import BrainInterface
except ImportError:
    BrainInterface = None


class GoalType(str, Enum):
    """目标类型"""

    ATOMIC = "atomic"  # 不可分解的原子目标
    COMPOSITE = "composite"  # 可分解的复合目标
    RESEARCH = "research"  # 研究类目标
    IMPLEMENTATION = "implementation"  # 实现类目标
    REVIEW = "review"  # 审查类目标
    REFACTOR = "refactor"  # 重构类目标
    TEST = "test"  # 测试类目标


class Complexity(str, Enum):
    """复杂度"""

    LOW = "low"  # 简单
    MEDIUM = "medium"  # 中等
    HIGH = "high"  # 复杂
    VERY_HIGH = "very_high"  # 非常复杂


@dataclass
class SubGoal:
    """
    子目标

    Attributes:
        id: 唯一标识
        description: 描述
        goal_type: 目标类型
        complexity: 复杂度
        dependencies: 依赖的子目标ID列表
        priority: 优先级 1-5
        estimated_effort: 预估工作量 (相对单位)
        status: 状态 (pending, ready, running, completed, failed, skipped)
        parent_id: 父目标ID
        metadata: 其他元数据
    """

    id: str
    description: str
    goal_type: GoalType = GoalType.ATOMIC
    complexity: Complexity = Complexity.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    priority: int = 3  # 1-5, 5最高
    estimated_effort: float = 1.0
    status: str = "pending"
    parent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_ready(self) -> bool:
        """检查是否准备好执行"""
        return self.status == "pending"

    def can_execute(self, completed_ids: set) -> bool:
        """检查依赖是否满足"""
        if self.status != "pending":
            return False
        return all(dep in completed_ids for dep in self.dependencies)

    def mark_running(self):
        self.status = "running"
        self.started_at = time.time()

    def mark_completed(self, result: Any = None):
        self.status = "completed"
        self.completed_at = time.time()
        self.result = result

    def mark_failed(self, error: str):
        self.status = "failed"
        self.completed_at = time.time()
        self.error = error

    def mark_skipped(self):
        self.status = "skipped"
        self.completed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "goal_type": self.goal_type.value,
            "complexity": self.complexity.value,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "status": self.status,
            "effort": self.estimated_effort,
            "duration_seconds": (
                self.completed_at - self.started_at
                if self.completed_at and self.started_at
                else None
            ),
        }


class GoalDecomposer:
    """
    目标分解器

    将复杂目标分解为可执行的子目标图。
    支持:
    - 自动分解 (LLM-based)
    - 模板分解 (基于规则)
    - 依赖分析
    - 优先级排序
    """

    # 常见项目相关的子目标模板
    SUBGOAL_TEMPLATES = {
        "analyze": [
            ("理解项目结构", GoalType.RESEARCH, Complexity.LOW, []),
            ("识别核心模块", GoalType.RESEARCH, Complexity.MEDIUM, [0]),
            ("分析依赖关系", GoalType.RESEARCH, Complexity.MEDIUM, [0, 1]),
            ("生成分析报告", GoalType.REVIEW, Complexity.LOW, [1, 2]),
        ],
        "create": [
            ("设计实现方案", GoalType.IMPLEMENTATION, Complexity.MEDIUM, []),
            ("编写代码", GoalType.IMPLEMENTATION, Complexity.HIGH, [0]),
            ("编写测试", GoalType.TEST, Complexity.MEDIUM, [1]),
            ("代码审查", GoalType.REVIEW, Complexity.LOW, [2]),
        ],
        "review": [
            ("收集代码", GoalType.ATOMIC, Complexity.LOW, []),
            ("静态分析", GoalType.REVIEW, Complexity.MEDIUM, [0]),
            ("逻辑审查", GoalType.REVIEW, Complexity.HIGH, [0, 1]),
            ("生成审查报告", GoalType.REVIEW, Complexity.MEDIUM, [1, 2]),
        ],
        "refactor": [
            ("识别重构点", GoalType.RESEARCH, Complexity.MEDIUM, []),
            ("设计新结构", GoalType.IMPLEMENTATION, Complexity.HIGH, [0]),
            ("实施重构", GoalType.REFACTOR, Complexity.HIGH, [1]),
            ("验证功能", GoalType.TEST, Complexity.MEDIUM, [2]),
        ],
        "debug": [
            ("复现问题", GoalType.ATOMIC, Complexity.MEDIUM, []),
            ("定位根因", GoalType.RESEARCH, Complexity.HIGH, [0]),
            ("修复问题", GoalType.IMPLEMENTATION, Complexity.MEDIUM, [1]),
            ("验证修复", GoalType.TEST, Complexity.LOW, [2]),
        ],
        "optimize": [
            ("识别性能瓶颈", GoalType.RESEARCH, Complexity.HIGH, []),
            ("分析根因", GoalType.RESEARCH, Complexity.MEDIUM, [0]),
            ("实施优化", GoalType.IMPLEMENTATION, Complexity.HIGH, [1]),
            ("性能测试", GoalType.TEST, Complexity.MEDIUM, [2]),
        ],
    }

    # 复杂度到工作量的映射
    COMPLEXITY_EFFORT = {
        Complexity.LOW: 1.0,
        Complexity.MEDIUM: 2.0,
        Complexity.HIGH: 4.0,
        Complexity.VERY_HIGH: 8.0,
    }

    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: 是否使用LLM进行智能分解
        """
        self.use_llm = use_llm and BrainInterface is not None
        self._stats = {"decomposed": 0, "llm_used": 0, "template_used": 0}

    async def decompose(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> List[SubGoal]:
        """
        分解目标

        Args:
            goal: 目标描述
            context: 上下文信息
            parent_id: 父目标ID

        Returns:
            List[SubGoal]: 子目标列表
        """
        context = context or {}
        self._stats["decomposed"] += 1

        # 1. 检测目标类型
        goal_type, complexity = self._classify_goal(goal)

        # 2. 选择分解策略
        if goal_type == GoalType.ATOMIC:
            # 原子目标，直接返回
            sub_goals = [self._create_atomic_goal(goal, parent_id)]
        elif self.use_llm and complexity in {Complexity.HIGH, Complexity.VERY_HIGH}:
            # 复杂目标使用LLM分解
            sub_goals = await self._decompose_with_llm(goal, context, parent_id)
            self._stats["llm_used"] += 1
        else:
            # 使用模板分解
            sub_goals = await self._decompose_with_template(goal, goal_type, parent_id)
            self._stats["template_used"] += 1

        # 3. 添加根目标作为第一个
        root_goal = SubGoal(
            id=f"goal_{int(time.time() * 1000) % 10000}_root",
            description=goal,
            goal_type=GoalType.COMPOSITE,
            complexity=complexity,
            priority=5,
            parent_id=parent_id,
        )
        for sg in sub_goals:
            sg.parent_id = root_goal.id

        logger.info(
            f"🧠 Decomposed '{goal[:50]}...' into {len(sub_goals)} sub-goals "
            f"(llm={self.use_llm}, type={goal_type.value})"
        )

        return sub_goals

    def _classify_goal(self, goal: str) -> tuple[GoalType, Complexity]:
        """分类目标"""
        goal_lower = goal.lower()

        # 检测类型
        if any(kw in goal_lower for kw in ["分析", "研究", "了解", "学习"]):
            goal_type = GoalType.RESEARCH
        elif any(kw in goal_lower for kw in ["创建", "实现", "写", "开发"]):
            goal_type = GoalType.IMPLEMENTATION
        elif any(kw in goal_lower for kw in ["审查", "检查", "审计"]):
            goal_type = GoalType.REVIEW
        elif any(kw in goal_lower for kw in ["重构", "优化", "改进"]):
            goal_type = GoalType.REFACTOR
        else:
            goal_type = GoalType.COMPOSITE

        # 评估复杂度
        complexity_score = 0

        # 长度指标
        if len(goal) > 100:
            complexity_score += 1
        if len(goal) > 200:
            complexity_score += 1

        # 多步骤指示词
        multi_step_keywords = ["然后", "接着", "再", "并且", "and then", "after that"]
        if any(kw in goal_lower for kw in multi_step_keywords):
            complexity_score += 2

        # 多目标指示词
        multi_goal_keywords = ["以及", "还有", "并且", "和", "以及", "also", "and"]
        if any(kw in goal_lower for kw in multi_goal_keywords):
            complexity_score += 1

        # 专业术语密度
        tech_terms = ["api", "database", "cache", "async", "concurrent", "refactor"]
        tech_count = sum(1 for t in tech_terms if t in goal_lower)
        complexity_score += tech_count

        if complexity_score <= 2:
            complexity = Complexity.LOW
        elif complexity_score <= 4:
            complexity = Complexity.MEDIUM
        elif complexity_score <= 6:
            complexity = Complexity.HIGH
        else:
            complexity = Complexity.VERY_HIGH

        return goal_type, complexity

    def _create_atomic_goal(
        self,
        description: str,
        parent_id: Optional[str] = None,
    ) -> SubGoal:
        """创建原子目标"""
        return SubGoal(
            id=f"goal_{int(time.time() * 1000) % 10000}_atomic",
            description=description,
            goal_type=GoalType.ATOMIC,
            complexity=Complexity.LOW,
            parent_id=parent_id,
        )

    async def _decompose_with_template(
        self,
        goal: str,
        goal_type: GoalType,
        parent_id: Optional[str] = None,
    ) -> List[SubGoal]:
        """使用模板分解"""
        # 匹配模板
        matched_template = None
        for intent, template in self.SUBGOAL_TEMPLATES.items():
            if intent in goal.lower():
                matched_template = template
                break

        if not matched_template:
            # 默认模板
            matched_template = [
                ("理解需求", GoalType.RESEARCH, Complexity.LOW, []),
                ("制定计划", GoalType.IMPLEMENTATION, Complexity.MEDIUM, [0]),
                ("执行任务", GoalType.IMPLEMENTATION, Complexity.MEDIUM, [1]),
                ("验证结果", GoalType.REVIEW, Complexity.LOW, [2]),
            ]

        # 生成子目标
        sub_goals = []
        base_id = int(time.time() * 1000) % 10000

        for i, (desc, g_type, complexity, deps) in enumerate(matched_template):
            # 创建子目标，先不设置依赖
            goal_id = f"goal_{base_id + i}"
            sub_goal = SubGoal(
                id=goal_id,
                description=desc,
                goal_type=g_type,
                complexity=complexity,
                dependencies=[],  # 暂不设置依赖，下一步再映射
                estimated_effort=self.COMPLEXITY_EFFORT[complexity],
            )
            sub_goals.append(sub_goal)

        # 第二步：建立依赖关系，确保索引正确映射到实际 goal id
        goal_id_map = {i: sub_goals[i].id for i in range(len(sub_goals)) if i < len(matched_template)}
        for i, sub_goal in enumerate(sub_goals):
            if i < len(matched_template):
                _, _, _, template_deps = matched_template[i]
                sub_goal.dependencies = [goal_id_map[d] for d in template_deps if d in goal_id_map]

        return sub_goals

    async def _decompose_with_llm(
        self,
        goal: str,
        context: Dict[str, Any],
        parent_id: Optional[str] = None,
    ) -> List[SubGoal]:
        """使用LLM智能分解"""
        try:
            # 构建提示
            prompt = f"""分解以下目标为子目标，返回JSON数组格式:
目标: {goal}

要求:
1. 每个子目标应该是原子性的、可以独立执行
2. 明确标注子目标之间的依赖关系
3. 评估每个子目标的复杂度(LOW/MEDIUM/HIGH)
4. 给出优先级(1-5)

输出格式(JSON数组):
[
  {{"description": "子目标描述", "complexity": "MEDIUM", "dependencies": [], "priority": 3}},
  ...
]"""

            # 调用LLM
            response = await BrainInterface.consult(
                prompt=prompt,
                system_prompt="你是一个任务分解专家，擅长将复杂目标分解为可执行的子目标。",
            )

            # 解析响应
            sub_goals = self._parse_llm_response(response, parent_id)
            return sub_goals

        except Exception as e:
            logger.error(f"❌ LLM decomposition failed: {e}")
            # 回退到模板分解
            return await self._decompose_with_template(
                goal, GoalType.COMPOSITE, parent_id
            )

    def _parse_llm_response(
        self,
        response: str,
        parent_id: Optional[str] = None,
    ) -> List[SubGoal]:
        """解析LLM响应"""
        sub_goals = []
        base_id = int(time.time() * 1000) % 10000

        try:
            # 尝试提取JSON
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())

                # 第一步：创建所有子目标，暂不设置依赖
                goal_id_map = {}
                for i, item in enumerate(items):
                    goal_id = f"goal_{base_id + i}"
                    goal_id_map[i] = goal_id
                    sub_goal = SubGoal(
                        id=goal_id,
                        description=item.get("description", ""),
                        goal_type=GoalType.IMPLEMENTATION,
                        complexity=Complexity(item.get("complexity", "MEDIUM")),
                        dependencies=[],  # 暂不设置依赖
                        priority=item.get("priority", 3),
                        estimated_effort=self.COMPLEXITY_EFFORT.get(
                            Complexity(item.get("complexity", "MEDIUM")),
                            1.0,
                        ),
                        parent_id=parent_id,
                    )
                    sub_goals.append(sub_goal)

                # 第二步：建立依赖关系，确保索引正确映射到实际 goal id
                for i, item in enumerate(items):
                    if i < len(sub_goals):
                        raw_deps = item.get("dependencies", [])
                        sub_goals[i].dependencies = [
                            goal_id_map[d] for d in raw_deps if d in goal_id_map
                        ]
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse LLM response: {e}")

        # 如果解析失败，使用默认模板
        if not sub_goals:
            logger.warning("⚠️ LLM parse failed, using template fallback")
            return [
                SubGoal(
                    id=f"goal_{base_id}",
                    description=response[:100] if response else "parsed_failed",
                    parent_id=parent_id,
                )
            ]

        return sub_goals

    def get_ready_goals(
        self,
        sub_goals: List[SubGoal],
        completed_ids: set,
    ) -> List[SubGoal]:
        """获取可以执行的子目标"""
        ready = []
        for sg in sub_goals:
            if sg.can_execute(completed_ids):
                ready.append(sg)
        # 按优先级和依赖排序
        ready.sort(key=lambda g: (-g.priority, len(g.dependencies)))
        return ready

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()
