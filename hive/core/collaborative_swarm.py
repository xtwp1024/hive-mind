# -*- coding: utf-8 -*-
"""
Collaborative Swarm (协作蜂群) - Phase 2: 群体智能进化

实现功能:
1. 任务分解机制 - 将复杂任务自动分解为子任务
2. 协作执行调度 - 基于 DAG 的并行/串行调度
3. 专业互补机制 - 不同专长的蜜蜂协作
4. 知识聚合 - 多个蜜蜂的知识融合

灵感来自 ADR-002 中的涌现行为设计。
"""

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("Hive.CollaborativeSwarm")


# ===== 枚举类型 =====


class TaskStatus(Enum):
    """任务状态"""

    PENDING = "pending"  # 等待执行
    READY = "ready"  # 就绪（依赖已满足）
    RUNNING = "running"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    SKIPPED = "skipped"  # 已跳过（依赖失败）


class CapabilityCategory(Enum):
    """能力类别"""

    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    BUG_DETECTION = "bug_detection"
    SECURITY_AUDIT = "security_audit"
    PATTERN_RECOGNITION = "pattern_recognition"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    CODE_OPTIMIZATION = "code_optimization"
    ARCHITECTURE_DESIGN = "architecture_design"
    SYSTEM_ANALYSIS = "system_analysis"
    VISION = "vision"
    AUDIO = "audio"
    COMMUNICATION = "communication"


# ===== 数据类 =====


@dataclass
class Subtask:
    """子任务"""

    subtask_id: str
    task_name: str
    task_description: str

    # 能力需求
    required_capabilities: List[CapabilityCategory] = field(default_factory=list)
    complexity: int = 1  # 1-5, 5最复杂
    estimated_duration: float = 5.0  # 预计时间（秒）

    # 依赖关系
    dependencies: List[str] = field(default_factory=list)  # 依赖的子任务ID列表

    # 执行状态
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None  # 分配的蜜蜂ID
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    actual_duration: Optional[float] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_duration(self) -> float:
        """获取实际或预计时间"""
        if self.actual_duration is not None:
            return self.actual_duration
        return self.estimated_duration

    def is_completed(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    def is_ready(
        self,
        completed_subtasks: Optional[Set[str]] = None,
        completed_ids: Optional[Set[str]] = None,
    ) -> bool:
        """检查是否所有依赖都已完成

        Args:
            completed_subtasks: 已完成的子任务ID集合
            completed_ids: 别名（向后兼容）
        """
        if completed_ids is not None:
            completed = completed_ids
        elif completed_subtasks is not None:
            completed = completed_subtasks
        else:
            completed = set()
        return all(dep in completed for dep in self.dependencies)


@dataclass
class TaskDecomposition:
    """任务分解结果 (Phase 2 新增)"""

    original_task_id: str
    original_task_name: str
    original_task_description: str

    # 分解结果
    subtasks: List[Subtask] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # subtask_id -> [depends_on]

    # 估计信息
    estimated_total_time: float = 0.0
    estimated_parallel_time: float = 0.0  # 考虑并行的时间

    # 元数据
    decomposition_method: str = "rule_based"  # rule_based, ai_assisted
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_subtask(self, subtask_id: str) -> Optional[Subtask]:
        """根据ID获取子任务"""
        for subtask in self.subtasks:
            if subtask.subtask_id == subtask_id:
                return subtask
        return None

    def get_ready_tasks(self, completed_ids: Set[str]) -> List[Subtask]:
        """获取就绪任务（依赖已满足且尚未开始）

        Args:
            completed_ids: 已完成的子任务ID集合
        """
        ready = []
        for subtask in self.subtasks:
            if (
                subtask.subtask_id not in completed_ids
                and subtask.status == TaskStatus.PENDING
                and subtask.is_ready(completed_ids)
            ):
                ready.append(subtask)
        return ready

    def get_critical_path(self) -> List[str]:
        """获取关键路径（决定总时长的路径）"""
        if not self.subtasks:
            return []

        # 构建邻接表
        adjacency: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {}

        for subtask in self.subtasks:
            adjacency[subtask.subtask_id] = []
            in_degree[subtask.subtask_id] = 0

        for subtask in self.subtasks:
            for dep in subtask.dependencies:
                if dep in adjacency:
                    adjacency[dep].append(subtask.subtask_id)
                    in_degree[subtask.subtask_id] += 1

        # 拓扑排序 + 计算最长路径
        duration: Dict[str, float] = {s.subtask_id: s.get_duration() for s in self.subtasks}
        longest_path: Dict[str, List[str]] = {s.subtask_id: [s.subtask_id] for s in self.subtasks}

        # 拓扑排序队列
        queue = deque([s for s in in_degree if in_degree[s] == 0])

        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, []):
                if duration[neighbor] < duration[node] + self.get_subtask(neighbor).get_duration():
                    duration[neighbor] = duration[node] + self.get_subtask(neighbor).get_duration()
                    longest_path[neighbor] = longest_path[node] + [neighbor]

                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 找到最长路径
        max_duration = max(duration.values())
        for node, dur in duration.items():
            if dur == max_duration:
                return longest_path[node]

        return []


@dataclass
class BeeCapability:
    """蜜蜂能力描述"""

    bee_id: str
    bee_type: str  # e.g., "EngineerBee", "SecurityBee", "DarwinBee"

    # 能力评分 (0.0 - 1.0)
    capabilities: Dict[CapabilityCategory, float] = field(default_factory=dict)

    # 历史表现
    total_tasks_completed: int = 0
    average_score: float = 0.0
    recent_tasks: List[Dict] = field(default_factory=list)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_capability_score(self, capability: CapabilityCategory) -> float:
        """获取指定能力的分数"""
        return self.capabilities.get(capability, 0.5)

    def get_weighted_score(self, required_capabilities: List[CapabilityCategory]) -> float:
        """计算加权分数（针对特定任务需求）"""
        if not required_capabilities:
            return 0.5

        total = 0.0
        for cap in required_capabilities:
            total += self.get_capability_score(cap)

        return total / len(required_capabilities)

    def update_with_task_result(self, task_result: Dict[str, Any], score: float):
        """用任务结果更新能力评估"""
        self.total_tasks_completed += 1

        # 移动平均
        old_total = self.average_score * (self.total_tasks_completed - 1)
        self.average_score = (old_total + score) / self.total_tasks_completed

        # 保留最近的任务记录
        self.recent_tasks.append(
            {
                "timestamp": time.time(),
                "task_id": task_result.get("task_id"),
                "score": score,
            }
        )

        # 只保留最近 20 个任务
        if len(self.recent_tasks) > 20:
            self.recent_tasks = self.recent_tasks[-20:]


@dataclass
class KnowledgeFragment:
    """知识片段"""

    fragment_id: str
    source_bee_id: str
    content: str
    confidence: float  # 0.0 - 1.0

    # 元数据
    source_task: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    references: List[str] = field(default_factory=list)

    # 向量表示（可选）
    vector: Optional[List[float]] = None

    def get_summary(self, max_length: int = 100) -> str:
        """获取摘要"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."


@dataclass
class AggregatedKnowledge:
    """聚合后的知识"""

    aggregation_id: str

    # 来源
    source_fragments: List[KnowledgeFragment] = field(default_factory=list)

    # 聚合结果
    summary: str = ""
    confidence: float = 0.0
    consensus_level: float = 0.0  # 来源之间的一致性程度

    # 元数据
    created_at: float = field(default_factory=time.time)
    topics: List[str] = field(default_factory=list)


# ===== 核心类 =====


class TaskDecomposer:
    """任务分解器

    负责将复杂任务分解为可执行的子任务。
    """

    def __init__(self):
        # 分解规则
        self.decomposition_rules: List[Callable] = [
            self._rule_code_project,
            self._rule_refactoring_task,
            self._rule_security_audit,
            self._rule_bug_fix,
        ]

        # 复杂度阈值
        self.COMPLEXITY_THRESHOLD = 3

    def decompose(self, task: Dict[str, Any]) -> TaskDecomposition:
        """
        分解任务

        Args:
            task: 原始任务描述，包含:
                - task_id: 任务ID
                - task_name: 任务名称
                - task_description: 任务描述
                - complexity: 复杂度估计（可选）
                - task_type: 任务类型（可选）

        Returns:
            TaskDecomposition 实例
        """
        task_id = task.get("task_id", f"task_{uuid.uuid4().hex[:8]}")
        task_name = task.get("task_name", "Unnamed Task")
        task_description = task.get("task_description", "")
        complexity = task.get("complexity", 1)
        task_type = task.get("task_type", "unknown")

        logger.info(f"🔧 [TaskDecomposer] 分解任务: {task_name} (复杂度: {complexity})")

        # 简单任务不需要分解
        if complexity < self.COMPLEXITY_THRESHOLD:
            logger.info(f"🔧 [TaskDecomposer] 任务复杂度低于阈值，不分解")
            subtask = Subtask(
                subtask_id=f"{task_id}_main",
                task_name=task_name,
                task_description=task_description,
                required_capabilities=self._infer_capabilities(task_type, task_description),
                complexity=complexity,
            )
            return TaskDecomposition(
                original_task_id=task_id,
                original_task_name=task_name,
                original_task_description=task_description,
                subtasks=[subtask],
                estimated_total_time=subtask.estimated_duration,
                estimated_parallel_time=subtask.estimated_duration,
            )

        # 尝试基于规则分解
        for rule in self.decomposition_rules:
            result = rule(task_type, task_description, task_id)
            if result:
                return TaskDecomposition(
                    original_task_id=task_id,
                    original_task_name=task_name,
                    original_task_description=task_description,
                    subtasks=result["subtasks"],
                    dependencies=result.get("dependencies", {}),
                    estimated_total_time=result["estimated_total_time"],
                    estimated_parallel_time=result.get(
                        "estimated_parallel_time", result["estimated_total_time"]
                    ),
                    decomposition_method="rule_based",
                )

        # 默认分解策略
        return self._default_decomposition(task_id, task_name, task_description, complexity)

    def _rule_code_project(self, task_type: str, description: str, task_id: str) -> Optional[Dict]:
        """规则: 新项目创建任务"""
        keywords = ["创建项目", "新项目", "create project", "new project"]
        if not any(kw.lower() in description.lower() for kw in keywords):
            return None

        subtasks = [
            Subtask(
                subtask_id=f"{task_id}_setup",
                task_name="项目结构搭建",
                task_description="创建项目基础目录结构和配置文件",
                required_capabilities=[CapabilityCategory.ARCHITECTURE_DESIGN],
                complexity=2,
                estimated_duration=5.0,
            ),
            Subtask(
                subtask_id=f"{task_id}_core",
                task_name="核心功能实现",
                task_description="实现主要业务逻辑和功能",
                required_capabilities=[CapabilityCategory.CODE_GENERATION],
                complexity=4,
                estimated_duration=15.0,
                dependencies=[f"{task_id}_setup"],
            ),
            Subtask(
                subtask_id=f"{task_id}_test",
                task_name="测试编写",
                task_description="编写单元测试和集成测试",
                required_capabilities=[CapabilityCategory.BUG_DETECTION],
                complexity=2,
                estimated_duration=8.0,
                dependencies=[f"{task_id}_core"],
            ),
            Subtask(
                subtask_id=f"{task_id}_review",
                task_name="代码审查",
                task_description="审查代码质量和安全性",
                required_capabilities=[
                    CapabilityCategory.CODE_ANALYSIS,
                    CapabilityCategory.SECURITY_AUDIT,
                ],
                complexity=3,
                estimated_duration=5.0,
                dependencies=[f"{task_id}_test"],
            ),
        ]

        return {
            "subtasks": subtasks,
            "dependencies": {
                f"{task_id}_core": [f"{task_id}_setup"],
                f"{task_id}_test": [f"{task_id}_core"],
                f"{task_id}_review": [f"{task_id}_test"],
            },
            "estimated_total_time": 5 + 15 + 8 + 5,
            "estimated_parallel_time": 5 + 15 + 8 + 5,  # 串行依赖
        }

    def _rule_refactoring_task(
        self, task_type: str, description: str, task_id: str
    ) -> Optional[Dict]:
        """规则: 重构任务"""
        keywords = ["重构", "refactor", "优化", "optimize", "improve"]
        if not any(kw.lower() in description.lower() for kw in keywords):
            return None

        subtasks = [
            Subtask(
                subtask_id=f"{task_id}_analysis",
                task_name="代码分析",
                task_description="分析现有代码结构和问题",
                required_capabilities=[CapabilityCategory.CODE_ANALYSIS],
                complexity=3,
                estimated_duration=8.0,
            ),
            Subtask(
                subtask_id=f"{task_id}_design",
                task_name="重构方案设计",
                task_description="设计重构方案和架构改进",
                required_capabilities=[CapabilityCategory.ARCHITECTURE_DESIGN],
                complexity=3,
                estimated_duration=6.0,
                dependencies=[f"{task_id}_analysis"],
            ),
            Subtask(
                subtask_id=f"{task_id}_implement",
                task_name="重构实现",
                task_description="执行重构，保持功能不变",
                required_capabilities=[
                    CapabilityCategory.CODE_GENERATION,
                    CapabilityCategory.CODE_OPTIMIZATION,
                ],
                complexity=4,
                estimated_duration=12.0,
                dependencies=[f"{task_id}_design"],
            ),
            Subtask(
                subtask_id=f"{task_id}_verify",
                task_name="验证测试",
                task_description="运行测试验证重构正确性",
                required_capabilities=[CapabilityCategory.BUG_DETECTION],
                complexity=2,
                estimated_duration=5.0,
                dependencies=[f"{task_id}_implement"],
            ),
        ]

        return {
            "subtasks": subtasks,
            "dependencies": {
                f"{task_id}_design": [f"{task_id}_analysis"],
                f"{task_id}_implement": [f"{task_id}_design"],
                f"{task_id}_verify": [f"{task_id}_implement"],
            },
            "estimated_total_time": 8 + 6 + 12 + 5,
            "estimated_parallel_time": 8 + 6 + 12 + 5,
        }

    def _rule_security_audit(
        self, task_type: str, description: str, task_id: str
    ) -> Optional[Dict]:
        """规则: 安全审计任务"""
        keywords = ["安全审计", "security audit", "漏洞扫描", "安全检查"]
        if not any(kw.lower() in description.lower() for kw in keywords):
            return None

        subtasks = [
            Subtask(
                subtask_id=f"{task_id}_static",
                task_name="静态代码分析",
                task_description="使用静态分析工具扫描漏洞",
                required_capabilities=[
                    CapabilityCategory.CODE_ANALYSIS,
                    CapabilityCategory.SECURITY_AUDIT,
                ],
                complexity=2,
                estimated_duration=4.0,
            ),
            Subtask(
                subtask_id=f"{task_id}_dynamic",
                task_name="动态安全测试",
                task_description="运行时安全检测和渗透测试",
                required_capabilities=[CapabilityCategory.SECURITY_AUDIT],
                complexity=3,
                estimated_duration=6.0,
            ),
            Subtask(
                subtask_id=f"{task_id}_dependency",
                task_name="依赖安全检查",
                task_description="检查第三方依赖的安全漏洞",
                required_capabilities=[
                    CapabilityCategory.SECURITY_AUDIT,
                    CapabilityCategory.PATTERN_RECOGNITION,
                ],
                complexity=2,
                estimated_duration=3.0,
            ),
            Subtask(
                subtask_id=f"{task_id}_report",
                task_name="安全报告生成",
                task_description="汇总发现，生成修复建议",
                required_capabilities=[
                    CapabilityCategory.KNOWLEDGE_EXTRACTION,
                    CapabilityCategory.SECURITY_AUDIT,
                ],
                complexity=3,
                estimated_duration=5.0,
                dependencies=[
                    f"{task_id}_static",
                    f"{task_id}_dynamic",
                    f"{task_id}_dependency",
                ],
            ),
        ]

        return {
            "subtasks": subtasks,
            "dependencies": {
                f"{task_id}_report": [
                    f"{task_id}_static",
                    f"{task_id}_dynamic",
                    f"{task_id}_dependency",
                ],
            },
            "estimated_total_time": 4 + 6 + 3 + 5,
            "estimated_parallel_time": max(4, 6, 3) + 5,  # 前三个可以并行
        }

    def _rule_bug_fix(self, task_type: str, description: str, task_id: str) -> Optional[Dict]:
        """规则: Bug 修复任务"""
        keywords = ["修复", "bug", "问题", "error", "issue"]
        if not any(kw.lower() in description.lower() for kw in keywords):
            return None

        subtasks = [
            Subtask(
                subtask_id=f"{task_id}_reproduce",
                task_name="问题复现",
                task_description="复现问题并理解根本原因",
                required_capabilities=[
                    CapabilityCategory.BUG_DETECTION,
                    CapabilityCategory.CODE_ANALYSIS,
                ],
                complexity=3,
                estimated_duration=5.0,
            ),
            Subtask(
                subtask_id=f"{task_id}_fix",
                task_name="修复实现",
                task_description="实现修复方案",
                required_capabilities=[
                    CapabilityCategory.CODE_GENERATION,
                    CapabilityCategory.BUG_DETECTION,
                ],
                complexity=3,
                estimated_duration=6.0,
                dependencies=[f"{task_id}_reproduce"],
            ),
            Subtask(
                subtask_id=f"{task_id}_validate",
                task_name="验证测试",
                task_description="验证修复有效且无回归",
                required_capabilities=[
                    CapabilityCategory.BUG_DETECTION,
                    CapabilityCategory.CODE_ANALYSIS,
                ],
                complexity=2,
                estimated_duration=4.0,
                dependencies=[f"{task_id}_fix"],
            ),
        ]

        return {
            "subtasks": subtasks,
            "dependencies": {
                f"{task_id}_fix": [f"{task_id}_reproduce"],
                f"{task_id}_validate": [f"{task_id}_fix"],
            },
            "estimated_total_time": 5 + 6 + 4,
            "estimated_parallel_time": 5 + 6 + 4,
        }

    def _default_decomposition(
        self, task_id: str, task_name: str, description: str, complexity: int
    ) -> TaskDecomposition:
        """默认分解策略"""
        subtasks = [
            Subtask(
                subtask_id=f"{task_id}_plan",
                task_name="任务计划",
                task_description="分析任务并制定执行计划",
                required_capabilities=self._infer_capabilities("unknown", description),
                complexity=max(1, complexity - 2),
                estimated_duration=3.0,
            ),
            Subtask(
                subtask_id=f"{task_id}_execute",
                task_name="任务执行",
                task_description="执行主要任务逻辑",
                required_capabilities=self._infer_capabilities("unknown", description),
                complexity=complexity,
                estimated_duration=complexity * 5.0,
                dependencies=[f"{task_id}_plan"],
            ),
            Subtask(
                subtask_id=f"{task_id}_review",
                task_name="结果审查",
                task_description="审查执行结果，确保质量",
                required_capabilities=[CapabilityCategory.CODE_ANALYSIS],
                complexity=max(1, complexity - 1),
                estimated_duration=3.0,
                dependencies=[f"{task_id}_execute"],
            ),
        ]

        estimated_total = sum(s.estimated_duration for s in subtasks)

        return TaskDecomposition(
            original_task_id=task_id,
            original_task_name=task_name,
            original_task_description=description,
            subtasks=subtasks,
            dependencies={
                f"{task_id}_execute": [f"{task_id}_plan"],
                f"{task_id}_review": [f"{task_id}_execute"],
            },
            estimated_total_time=estimated_total,
            estimated_parallel_time=estimated_total,
        )

    def _infer_capabilities(self, task_type: str, description: str) -> List[CapabilityCategory]:
        """从任务描述推断所需能力"""
        capabilities: List[CapabilityCategory] = []
        desc_lower = description.lower()

        # 基于关键词推断
        if any(kw in desc_lower for kw in ["代码", "code", "编写", "生成", "create"]):
            capabilities.append(CapabilityCategory.CODE_GENERATION)

        if any(kw in desc_lower for kw in ["分析", "analyze", "review", "审查"]):
            capabilities.append(CapabilityCategory.CODE_ANALYSIS)

        if any(kw in desc_lower for kw in ["bug", "问题", "修复", "fix", "调试", "debug"]):
            capabilities.append(CapabilityCategory.BUG_DETECTION)

        if any(kw in desc_lower for kw in ["安全", "security", "漏洞", "vulnerability"]):
            capabilities.append(CapabilityCategory.SECURITY_AUDIT)

        if any(kw in desc_lower for kw in ["架构", "设计", "架构图", "architecture", "design"]):
            capabilities.append(CapabilityCategory.ARCHITECTURE_DESIGN)

        if any(kw in desc_lower for kw in ["优化", "optimize", "性能", "performance"]):
            capabilities.append(CapabilityCategory.CODE_OPTIMIZATION)

        if any(kw in desc_lower for kw in ["知识", "知识图谱", "knowledge"]):
            capabilities.append(CapabilityCategory.KNOWLEDGE_EXTRACTION)

        # 默认能力
        if not capabilities:
            capabilities = [CapabilityCategory.CODE_GENERATION]

        return list(set(capabilities))


class TaskAssignmentEngine:
    """任务分配引擎

    负责将子任务分配给最合适的蜜蜂。
    """

    def __init__(self):
        self.bee_capabilities: Dict[str, BeeCapability] = {}

    def register_bee(self, capability: BeeCapability):
        """注册蜜蜂能力"""
        self.bee_capabilities[capability.bee_id] = capability
        logger.info(f"🐝 [TaskAssignment] 注册蜜蜂: {capability.bee_id} ({capability.bee_type})")

    def unregister_bee(self, bee_id: str):
        """注销蜜蜂"""
        if bee_id in self.bee_capabilities:
            del self.bee_capabilities[bee_id]
            logger.info(f"🐝 [TaskAssignment] 注销蜜蜂: {bee_id}")

    def find_best_bee(
        self,
        subtask: Subtask,
        available_bees: List[str],
    ) -> Optional[str]:
        """
        为子任务找到最合适的蜜蜂

        Args:
            subtask: 子任务
            available_bees: 可用蜜蜂ID列表

        Returns:
            最佳蜜蜂ID，或 None
        """
        if not available_bees:
            return None

        # 计算每个可用蜜蜂的匹配分数
        scores: Dict[str, float] = {}

        for bee_id in available_bees:
            if bee_id not in self.bee_capabilities:
                continue

            cap = self.bee_capabilities[bee_id]

            # 基础匹配分数
            base_score = cap.get_weighted_score(subtask.required_capabilities)

            # 历史表现加分
            history_bonus = min(0.2, cap.average_score * 0.2)

            # 复杂度惩罚（避免新手做复杂任务）
            complexity_penalty = 0.0
            if subtask.complexity >= 4 and cap.total_tasks_completed < 10:
                complexity_penalty = 0.15

            final_score = base_score + history_bonus - complexity_penalty
            scores[bee_id] = final_score

        if not scores:
            # 没有注册的蜜蜂，随机选一个
            return available_bees[0] if available_bees else None

        # 选择分数最高的
        best_bee_id = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_bee_id]

        logger.info(
            f"🐝 [TaskAssignment] 分配任务 '{subtask.task_name}' 给 {best_bee_id} (分数: {best_score:.2f})"
        )

        return best_bee_id

    def batch_assign(
        self,
        subtasks: List[Subtask],
        available_bees: List[str],
    ) -> Dict[str, str]:
        """
        批量分配任务

        当有多个任务和多个蜜蜂时，尝试进行最优配对

        Args:
            subtasks: 待分配的子任务列表
            available_bees: 可用蜜蜂ID列表

        Returns:
            subtask_id -> bee_id 映射
        """
        assignments: Dict[str, str] = {}
        assigned_bees: Set[str] = set()

        # 按复杂度排序，先分配复杂任务
        sorted_subtasks = sorted(subtasks, key=lambda s: s.complexity, reverse=True)

        for subtask in sorted_subtasks:
            remaining_bees = [b for b in available_bees if b not in assigned_bees]

            if not remaining_bees:
                # 蜜蜂不够，重复使用（允许并行？）
                remaining_bees = available_bees

            best_bee = self.find_best_bee(subtask, remaining_bees)
            if best_bee:
                assignments[subtask.subtask_id] = best_bee
                assigned_bees.add(best_bee)

        return assignments

    def update_bee_performance(
        self,
        bee_id: str,
        task_result: Dict[str, Any],
        score: float,
    ):
        """更新蜜蜂的表现记录"""
        if bee_id not in self.bee_capabilities:
            # 自动注册
            self.bee_capabilities[bee_id] = BeeCapability(
                bee_id=bee_id,
                bee_type=task_result.get("bee_type", "Unknown"),
            )

        cap = self.bee_capabilities[bee_id]
        cap.update_with_task_result(task_result, score)


class KnowledgeAggregator:
    """知识聚合器

    负责将多个蜜蜂的知识片段聚合为统一的知识。
    """

    def __init__(self):
        # 已聚合的知识缓存
        self.aggregated_cache: Dict[str, AggregatedKnowledge] = {}

        # 相似度阈值
        self.SIMILARITY_THRESHOLD = 0.7

    def create_fragment(
        self,
        source_bee_id: str,
        content: str,
        confidence: float,
        source_task: Optional[str] = None,
        vector: Optional[List[float]] = None,
    ) -> KnowledgeFragment:
        """创建知识片段"""
        fragment = KnowledgeFragment(
            fragment_id=f"frag_{uuid.uuid4().hex[:12]}",
            source_bee_id=source_bee_id,
            content=content,
            confidence=confidence,
            source_task=source_task,
            vector=vector,
        )

        logger.info(f"🧠 [Knowledge] 创建知识片段: {fragment.fragment_id} (来源: {source_bee_id})")
        return fragment

    def aggregate(
        self,
        fragments: List[KnowledgeFragment],
        topic: Optional[str] = None,
    ) -> AggregatedKnowledge:
        """
        聚合多个知识片段

        策略:
        1. 基于置信度加权
        2. 检测来源之间的一致性
        3. 生成统一的摘要

        Args:
            fragments: 知识片段列表
            topic: 主题（可选）

        Returns:
            聚合后的知识
        """
        if not fragments:
            return AggregatedKnowledge(
                aggregation_id=f"agg_{uuid.uuid4().hex[:12]}",
                summary="",
                confidence=0.0,
                consensus_level=0.0,
            )

        logger.info(f"🧠 [Knowledge] 聚合 {len(fragments)} 个知识片段")

        # 计算一致性（简单版本：比较内容相似度）
        # 实际实现可以使用向量相似度
        consensus_level = self._calculate_consensus(fragments)

        # 加权置信度
        total_confidence = sum(f.confidence for f in fragments)
        avg_confidence = total_confidence / len(fragments)

        # 生成摘要（实际可以使用 LLM）
        summary = self._generate_summary(fragments, topic)

        # 提取主题
        topics = [topic] if topic else []
        for frag in fragments:
            if frag.source_task and frag.source_task not in topics:
                topics.append(frag.source_task)

        aggregated = AggregatedKnowledge(
            aggregation_id=f"agg_{uuid.uuid4().hex[:12]}",
            source_fragments=fragments,
            summary=summary,
            confidence=avg_confidence,
            consensus_level=consensus_level,
            topics=list(set(topics)),
        )

        # 缓存
        self.aggregated_cache[aggregated.aggregation_id] = aggregated

        logger.info(
            f"🧠 [Knowledge] 聚合完成: 置信度 {avg_confidence:.2f}, 一致性 {consensus_level:.2f}"
        )

        return aggregated

    def _calculate_consensus(self, fragments: List[KnowledgeFragment]) -> float:
        """计算片段之间的一致性（简化版）"""
        if len(fragments) <= 1:
            return 1.0

        # 简单版本：检查置信度的方差
        # 实际实现应该比较内容相似度
        confidences = [f.confidence for f in fragments]
        avg_conf = sum(confidences) / len(confidences)

        # 计算变异系数作为不一致性的度量
        variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
        std_dev = variance**0.5

        # 变异系数越小，一致性越高
        if avg_conf == 0:
            return 0.0

        cv = std_dev / avg_conf
        consensus = max(0.0, 1.0 - min(1.0, cv))

        return consensus

    def _generate_summary(self, fragments: List[KnowledgeFragment], topic: Optional[str]) -> str:
        """生成摘要（简化版）"""
        # 实际实现应该使用 LLM 来生成自然语言摘要
        # 这里使用简单的拼接策略

        # 按置信度排序
        sorted_frags = sorted(fragments, key=lambda f: f.confidence, reverse=True)

        if len(sorted_frags) == 1:
            return sorted_frags[0].content

        # 取置信度最高的前几个片段
        top_frags = sorted_frags[: min(3, len(sorted_frags))]

        summary_parts = [
            f"聚合自 {len(fragments)} 个知识片段。",
        ]

        for i, frag in enumerate(top_frags):
            summary_parts.append(f"[{i+1}] {frag.get_summary(150)} (置信度: {frag.confidence:.2f})")

        return "\n".join(summary_parts)

    def get_cached(self, aggregation_id: str) -> Optional[AggregatedKnowledge]:
        """获取缓存的聚合结果"""
        return self.aggregated_cache.get(aggregation_id)


class CollaborativeSwarm:
    """协作蜂群 (Phase 2 核心类)

    整合任务分解、分配、执行和知识聚合。
    """

    def __init__(self, communication_bus: Any = None):
        self.communication_bus = communication_bus

        # 子系统
        self.decomposer = TaskDecomposer()
        self.assignment_engine = TaskAssignmentEngine()
        self.knowledge_aggregator = KnowledgeAggregator()

        # 当前执行中的任务
        self.active_decompositions: Dict[str, TaskDecomposition] = {}
        self.completed_decompositions: List[TaskDecomposition] = []

        # 执行状态
        self.is_running = False

        logger.info(f"🐝 [CollaborativeSwarm] 协作蜂群初始化完成")

    async def execute_collaboratively(
        self,
        task: Dict[str, Any],
        available_bees: List[str],
    ) -> Dict[str, Any]:
        """
        协作执行任务

        完整流程:
        1. 任务分解
        2. 任务分配
        3. DAG 调度执行
        4. 知识聚合

        Args:
            task: 原始任务
            available_bees: 可用蜜蜂ID列表

        Returns:
            执行结果
        """
        task_id = task.get("task_id", f"task_{uuid.uuid4().hex[:8]}")

        logger.info(f"🐝 [CollaborativeSwarm] 开始协作执行任务: {task.get('task_name', 'unnamed')}")
        start_time = time.time()

        # ===== 1. 任务分解 =====
        decomposition = self.decomposer.decompose(task)
        self.active_decompositions[task_id] = decomposition

        logger.info(f"🐝 [CollaborativeSwarm] 任务分解完成: {len(decomposition.subtasks)} 个子任务")

        # ===== 2. 执行调度 =====
        completed_ids: Set[str] = set()
        failed_ids: Set[str] = set()

        # 准备就绪任务队列
        ready_tasks = decomposition.get_ready_tasks(completed_ids)
        in_progress: Dict[str, Subtask] = {}

        # 模拟执行循环（实际应该通过通信总线调度）
        while len(completed_ids) + len(failed_ids) < len(decomposition.subtasks):
            # 分配就绪任务
            for subtask in ready_tasks:
                if subtask.subtask_id not in in_progress:
                    best_bee = self.assignment_engine.find_best_bee(subtask, available_bees)
                    if best_bee:
                        subtask.assigned_to = best_bee
                        subtask.status = TaskStatus.RUNNING
                        subtask.start_time = time.time()
                        in_progress[subtask.subtask_id] = subtask

                        logger.info(
                            f"🐝 [CollaborativeSwarm] 开始执行: {subtask.task_name} (蜜蜂: {best_bee})"
                        )

            # 模拟任务完成（实际应该异步等待）
            await self._simulate_task_execution(in_progress, completed_ids, failed_ids)

            # 获取新的就绪任务
            ready_tasks = decomposition.get_ready_tasks(completed_ids)

            # 短暂延迟
            import asyncio

            await asyncio.sleep(0.1)

        # ===== 3. 汇总结果 =====
        total_duration = time.time() - start_time

        # 检查是否有失败
        success = len(failed_ids) == 0

        # 收集子任务结果
        subtask_results: Dict[str, Any] = {}
        for subtask in decomposition.subtasks:
            subtask_results[subtask.subtask_id] = {
                "task_name": subtask.task_name,
                "status": subtask.status.value,
                "assigned_to": subtask.assigned_to,
                "result": subtask.result,
                "error": subtask.error,
                "duration": subtask.actual_duration,
            }

        # ===== 4. 知识聚合（如果有知识片段） =====
        aggregated_knowledge = None

        # 从实际任务结果中提取知识（简化版本）
        fragments: List[KnowledgeFragment] = []
        for subtask in decomposition.subtasks:
            if subtask.result and "knowledge" in subtask.result:
                frag = self.knowledge_aggregator.create_fragment(
                    source_bee_id=subtask.assigned_to or "unknown",
                    content=str(subtask.result["knowledge"]),
                    confidence=subtask.result.get("confidence", 0.8),
                    source_task=subtask.subtask_id,
                )
                fragments.append(frag)

        if fragments:
            aggregated_knowledge = self.knowledge_aggregator.aggregate(fragments)

        # 记录完成
        self.completed_decompositions.append(decomposition)
        del self.active_decompositions[task_id]

        logger.info(
            f"🐝 [CollaborativeSwarm] 任务完成: {'成功' if success else '失败'} "
            f"(耗时: {total_duration:.2f}秒, 子任务: {len(decomposition.subtasks)})"
        )

        return {
            "task_id": task_id,
            "success": success,
            "total_duration": total_duration,
            "estimated_total_time": decomposition.estimated_total_time,
            "estimated_parallel_time": decomposition.estimated_parallel_time,
            "subtask_count": len(decomposition.subtasks),
            "completed_count": len(completed_ids),
            "failed_count": len(failed_ids),
            "subtask_results": subtask_results,
            "critical_path": decomposition.get_critical_path(),
            "aggregated_knowledge": (
                {
                    "aggregation_id": aggregated_knowledge.aggregation_id,
                    "summary": aggregated_knowledge.summary,
                    "confidence": aggregated_knowledge.confidence,
                    "consensus_level": aggregated_knowledge.consensus_level,
                    "fragment_count": len(aggregated_knowledge.source_fragments),
                }
                if aggregated_knowledge
                else None
            ),
        }

    async def _simulate_task_execution(
        self,
        in_progress: Dict[str, Subtask],
        completed_ids: Set[str],
        failed_ids: Set[str],
    ):
        """
        模拟任务执行（简化版）

        实际实现应该:
        1. 通过通信总线发送任务给蜜蜂
        2. 监听任务完成事件
        3. 更新子任务状态
        """
        import asyncio
        import random

        completed_this_cycle: List[str] = []

        for subtask_id, subtask in list(in_progress.items()):
            # 模拟执行时间
            elapsed = time.time() - subtask.start_time

            # 简单完成条件：超过预计时间的 90%
            if elapsed >= subtask.estimated_duration * 0.9:
                # 模拟成功率 90%
                if random.random() < 0.9:
                    subtask.status = TaskStatus.COMPLETED
                    subtask.end_time = time.time()
                    subtask.actual_duration = subtask.end_time - subtask.start_time
                    subtask.result = {
                        "success": True,
                        "output": f"完成任务: {subtask.task_name}",
                        "knowledge": f"关于 {subtask.task_name} 的知识",
                        "confidence": 0.7 + random.random() * 0.3,
                    }
                    completed_ids.add(subtask_id)

                    logger.info(
                        f"✅ [CollaborativeSwarm] 任务完成: {subtask.task_name} (耗时: {subtask.actual_duration:.2f}s)"
                    )
                else:
                    subtask.status = TaskStatus.FAILED
                    subtask.end_time = time.time()
                    subtask.actual_duration = subtask.end_time - subtask.start_time
                    subtask.error = "模拟执行错误"
                    failed_ids.add(subtask_id)

                    logger.warning(f"❌ [CollaborativeSwarm] 任务失败: {subtask.task_name}")

                completed_this_cycle.append(subtask_id)

        # 移除已完成的
        for subtask_id in completed_this_cycle:
            del in_progress[subtask_id]

    def register_bee_capability(self, capability: BeeCapability):
        """注册蜜蜂能力"""
        self.assignment_engine.register_bee(capability)

    def get_execution_status(self, task_id: str) -> Optional[Dict]:
        """获取任务执行状态"""
        if task_id not in self.active_decompositions:
            return None

        decomposition = self.active_decompositions[task_id]

        completed = [s for s in decomposition.subtasks if s.is_completed()]
        running = [s for s in decomposition.subtasks if s.status == TaskStatus.RUNNING]
        pending = [s for s in decomposition.subtasks if s.status == TaskStatus.PENDING]

        return {
            "task_id": task_id,
            "original_task_name": decomposition.original_task_name,
            "total_subtasks": len(decomposition.subtasks),
            "completed": len(completed),
            "running": len(running),
            "pending": len(pending),
            "progress": (len(completed) / len(decomposition.subtasks)) * 100,
            "subtasks": [
                {
                    "id": s.subtask_id,
                    "name": s.task_name,
                    "status": s.status.value,
                    "assigned_to": s.assigned_to,
                }
                for s in decomposition.subtasks
            ],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_completed = len(self.completed_decompositions)
        total_subtasks = sum(len(d.subtasks) for d in self.completed_decompositions)

        if self.completed_decompositions:
            avg_time = sum(d.estimated_total_time for d in self.completed_decompositions) / len(
                self.completed_decompositions
            )
        else:
            avg_time = 0.0

        return {
            "active_tasks": len(self.active_decompositions),
            "completed_tasks": total_completed,
            "total_subtasks_executed": total_subtasks,
            "average_task_duration": avg_time,
            "registered_bees": len(self.assignment_engine.bee_capabilities),
            "knowledge_fragments_aggregated": len(self.knowledge_aggregator.aggregated_cache),
        }


# ===== 便捷函数 =====

_collaborative_swarm_instance: Optional[CollaborativeSwarm] = None


def get_collaborative_swarm(communication_bus: Any = None) -> CollaborativeSwarm:
    """获取协作蜂群单例"""
    global _collaborative_swarm_instance

    if _collaborative_swarm_instance is None:
        _collaborative_swarm_instance = CollaborativeSwarm(communication_bus=communication_bus)
    elif communication_bus is not None:
        _collaborative_swarm_instance.communication_bus = communication_bus

    return _collaborative_swarm_instance
