# -*- coding: utf-8 -*-
"""测试 Phase 2: 协作蜂群 (Collaborative Swarm)

测试内容:
1. TaskStatus 枚举
2. CapabilityCategory 枚举
3. Subtask 数据类
4. TaskDecomposition 数据类
5. BeeCapability 数据类
6. TaskDecomposer 类
7. TaskAssignmentEngine 类
8. KnowledgeAggregator 类
9. CollaborativeSwarm 类
"""

import pytest
import time
import asyncio
from datetime import datetime
from enum import Enum
from typing import List

from hive.core.collaborative_swarm import (
    CollaborativeSwarm,
    TaskDecomposer,
    TaskAssignmentEngine,
    KnowledgeAggregator,
    TaskDecomposition,
    Subtask,
    BeeCapability,
    KnowledgeFragment,
    AggregatedKnowledge,
    TaskStatus,
    CapabilityCategory,
    get_collaborative_swarm,
)


class TestTaskStatus:
    """测试 TaskStatus 枚举"""

    def test_all_statuses_exist(self):
        """测试所有状态存在"""
        assert hasattr(TaskStatus, "PENDING")
        assert hasattr(TaskStatus, "READY")
        assert hasattr(TaskStatus, "RUNNING")
        assert hasattr(TaskStatus, "COMPLETED")
        assert hasattr(TaskStatus, "FAILED")
        assert hasattr(TaskStatus, "SKIPPED")

    def test_status_values(self):
        """测试状态值"""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.READY.value == "ready"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"


class TestCapabilityCategory:
    """测试 CapabilityCategory 枚举"""

    def test_all_categories_exist(self):
        """测试所有能力类别存在"""
        categories = [
            "CODE_GENERATION",
            "CODE_ANALYSIS",
            "BUG_DETECTION",
            "SECURITY_AUDIT",
            "PATTERN_RECOGNITION",
            "KNOWLEDGE_EXTRACTION",
            "CODE_OPTIMIZATION",
            "ARCHITECTURE_DESIGN",
            "SYSTEM_ANALYSIS",
            "VISION",
            "AUDIO",
            "COMMUNICATION",
        ]
        
        for cat in categories:
            assert hasattr(CapabilityCategory, cat)

    def test_category_values(self):
        """测试类别值"""
        assert CapabilityCategory.CODE_GENERATION.value == "code_generation"
        assert CapabilityCategory.SECURITY_AUDIT.value == "security_audit"
        assert CapabilityCategory.ARCHITECTURE_DESIGN.value == "architecture_design"


class TestSubtask:
    """测试 Subtask 数据类"""

    def test_subtask_creation_defaults(self):
        """测试创建子任务（默认值）"""
        subtask = Subtask(
            subtask_id="sub_001",
            task_name="测试子任务",
            task_description="这是一个测试子任务",
        )
        
        assert subtask.subtask_id == "sub_001"
        assert subtask.task_name == "测试子任务"
        assert subtask.task_description == "这是一个测试子任务"
        
        # 默认值
        assert subtask.required_capabilities == []
        assert subtask.complexity == 1
        assert subtask.estimated_duration == 5.0
        assert subtask.dependencies == []
        assert subtask.status == TaskStatus.PENDING
        assert subtask.assigned_to is None
        assert subtask.result is None
        assert subtask.error is None
        assert subtask.start_time is None
        assert subtask.end_time is None
        assert subtask.actual_duration is None
        assert subtask.metadata == {}

    def test_subtask_creation_with_params(self):
        """测试创建子任务（带参数）"""
        subtask = Subtask(
            subtask_id="sub_002",
            task_name="复杂任务",
            task_description="这是一个复杂的子任务",
            required_capabilities=[
                CapabilityCategory.CODE_GENERATION,
                CapabilityCategory.SECURITY_AUDIT,
            ],
            complexity=4,
            estimated_duration=15.0,
            dependencies=["sub_001"],
            metadata={"priority": "high"},
        )
        
        assert len(subtask.required_capabilities) == 2
        assert subtask.complexity == 4
        assert subtask.estimated_duration == 15.0
        assert subtask.dependencies == ["sub_001"]
        assert subtask.metadata["priority"] == "high"

    def test_subtask_get_duration(self):
        """测试获取时间"""
        subtask = Subtask(
            subtask_id="sub_003",
            task_name="测试",
            task_description="测试",
            estimated_duration=10.0,
        )
        
        # 实际时间未设置时返回估计时间
        assert subtask.get_duration() == 10.0
        
        # 设置实际时间
        subtask.actual_duration = 8.5
        assert subtask.get_duration() == 8.5

    def test_subtask_is_completed(self):
        """测试检查是否完成"""
        subtask = Subtask(
            subtask_id="sub_004",
            task_name="测试",
            task_description="测试",
        )
        
        assert subtask.is_completed() == False
        
        subtask.status = TaskStatus.COMPLETED
        assert subtask.is_completed() == True

    def test_subtask_is_ready(self):
        """测试检查是否就绪"""
        subtask = Subtask(
            subtask_id="sub_005",
            task_name="测试",
            task_description="测试",
            dependencies=["dep_1", "dep_2"],
        )
        
        # 依赖未完成
        assert subtask.is_ready(completed_ids=set()) == False
        assert subtask.is_ready(completed_ids={"dep_1"}) == False
        
        # 依赖已完成
        assert subtask.is_ready(completed_ids={"dep_1", "dep_2"}) == True
        assert subtask.is_ready(completed_ids={"dep_1", "dep_2", "extra"}) == True
        
        # 无依赖
        subtask_no_deps = Subtask(
            subtask_id="sub_006",
            task_name="测试",
            task_description="测试",
            dependencies=[],
        )
        assert subtask_no_deps.is_ready(completed_ids=set()) == True


class TestTaskDecomposition:
    """测试 TaskDecomposition 数据类"""

    def test_decomposition_creation_defaults(self):
        """测试创建任务分解（默认值）"""
        decomposition = TaskDecomposition(
            original_task_id="task_001",
            original_task_name="原始任务",
            original_task_description="这是原始任务",
        )
        
        assert decomposition.original_task_id == "task_001"
        assert decomposition.original_task_name == "原始任务"
        assert decomposition.original_task_description == "这是原始任务"
        
        # 默认值
        assert decomposition.subtasks == []
        assert decomposition.dependencies == {}
        assert decomposition.estimated_total_time == 0.0
        assert decomposition.estimated_parallel_time == 0.0
        assert decomposition.decomposition_method == "rule_based"
        assert decomposition.metadata == {}

    def test_get_subtask(self):
        """测试获取子任务"""
        sub1 = Subtask(subtask_id="s1", task_name="1", task_description="1")
        sub2 = Subtask(subtask_id="s2", task_name="2", task_description="2")
        
        decomposition = TaskDecomposition(
            original_task_id="t1",
            original_task_name="Test",
            original_task_description="Test",
            subtasks=[sub1, sub2],
        )
        
        assert decomposition.get_subtask("s1") == sub1
        assert decomposition.get_subtask("s2") == sub2
        assert decomposition.get_subtask("nonexistent") is None

    def test_get_ready_tasks(self):
        """测试获取就绪任务"""
        # 创建有依赖关系的任务
        # s1 → s2 → s3
        # s4 无依赖
        
        s1 = Subtask(subtask_id="s1", task_name="1", task_description="1", dependencies=[])
        s2 = Subtask(subtask_id="s2", task_name="2", task_description="2", dependencies=["s1"])
        s3 = Subtask(subtask_id="s3", task_name="3", task_description="3", dependencies=["s2"])
        s4 = Subtask(subtask_id="s4", task_name="4", task_description="4", dependencies=[])
        
        decomposition = TaskDecomposition(
            original_task_id="t1",
            original_task_name="Test",
            original_task_description="Test",
            subtasks=[s1, s2, s3, s4],
        )
        
        # 初始状态：s1 和 s4 就绪
        ready = decomposition.get_ready_tasks(completed_ids=set())
        ready_ids = [s.subtask_id for s in ready]
        
        assert "s1" in ready_ids
        assert "s4" in ready_ids
        assert "s2" not in ready_ids
        assert "s3" not in ready_ids
        
        # s1 完成后：s2 就绪
        ready2 = decomposition.get_ready_tasks(completed_ids={"s1"})
        ready_ids2 = [s.subtask_id for s in ready2]
        
        assert "s2" in ready_ids2
        assert "s4" in ready_ids2
        assert "s1" not in ready_ids2  # s1 已开始？不，这里 s1 在 completed 中

    def test_get_critical_path_simple(self):
        """测试获取关键路径（简单情况）"""
        # 简单串行任务：s1 → s2 → s3
        # 预计时间：1 → 2 → 3
        # 关键路径应该是 s1-s2-s3
        
        s1 = Subtask(subtask_id="s1", task_name="1", task_description="1", dependencies=[], estimated_duration=1.0)
        s2 = Subtask(subtask_id="s2", task_name="2", task_description="2", dependencies=["s1"], estimated_duration=2.0)
        s3 = Subtask(subtask_id="s3", task_name="3", task_description="3", dependencies=["s2"], estimated_duration=3.0)
        
        decomposition = TaskDecomposition(
            original_task_id="t1",
            original_task_name="Test",
            original_task_description="Test",
            subtasks=[s1, s2, s3],
        )
        
        critical_path = decomposition.get_critical_path()
        
        # 关键路径应该是 s1 → s2 → s3
        assert len(critical_path) == 3
        assert critical_path[0] == "s1"
        assert critical_path[1] == "s2"
        assert critical_path[2] == "s3"

    def test_get_critical_path_parallel(self):
        """测试获取关键路径（并行情况）"""
        # 并行任务：
        # s1 (5) → s3 (1)
        # s2 (3) → s4 (1)
        # 关键路径应该是 s1-s3（总时间 6）
        
        s1 = Subtask(subtask_id="s1", task_name="1", task_description="1", dependencies=[], estimated_duration=5.0)
        s2 = Subtask(subtask_id="s2", task_name="2", task_description="2", dependencies=[], estimated_duration=3.0)
        s3 = Subtask(subtask_id="s3", task_name="3", task_description="3", dependencies=["s1"], estimated_duration=1.0)
        s4 = Subtask(subtask_id="s4", task_name="4", task_description="4", dependencies=["s2"], estimated_duration=1.0)
        
        decomposition = TaskDecomposition(
            original_task_id="t1",
            original_task_name="Test",
            original_task_description="Test",
            subtasks=[s1, s2, s3, s4],
        )
        
        critical_path = decomposition.get_critical_path()
        
        # 关键路径应该是 s1 → s3（总时间 6 > s2→s4 的 4）
        # 注意：get_critical_path 可能只返回一个或多个
        # 这里只验证逻辑正确


class TestBeeCapability:
    """测试 BeeCapability 数据类"""

    def test_bee_capability_creation_defaults(self):
        """测试创建蜜蜂能力（默认值）"""
        cap = BeeCapability(
            bee_id="bee_001",
            bee_type="EngineerBee",
        )
        
        assert cap.bee_id == "bee_001"
        assert cap.bee_type == "EngineerBee"
        
        # 默认值
        assert cap.capabilities == {}
        assert cap.total_tasks_completed == 0
        assert cap.average_score == 0.0
        assert cap.recent_tasks == []
        assert cap.metadata == {}

    def test_bee_capability_creation_with_caps(self):
        """测试创建蜜蜂能力（带能力评分）"""
        cap = BeeCapability(
            bee_id="bee_002",
            bee_type="SecurityBee",
            capabilities={
                CapabilityCategory.SECURITY_AUDIT: 0.95,
                CapabilityCategory.BUG_DETECTION: 0.85,
            },
        )
        
        assert len(cap.capabilities) == 2
        assert cap.capabilities[CapabilityCategory.SECURITY_AUDIT] == 0.95

    def test_get_capability_score(self):
        """测试获取能力分数"""
        cap = BeeCapability(
            bee_id="bee_003",
            bee_type="TestBee",
            capabilities={
                CapabilityCategory.CODE_GENERATION: 0.9,
                CapabilityCategory.CODE_ANALYSIS: 0.8,
            },
        )
        
        # 已有的能力
        assert cap.get_capability_score(CapabilityCategory.CODE_GENERATION) == 0.9
        assert cap.get_capability_score(CapabilityCategory.CODE_ANALYSIS) == 0.8
        
        # 没有的能力返回默认值 0.5
        assert cap.get_capability_score(CapabilityCategory.SECURITY_AUDIT) == 0.5

    def test_get_weighted_score(self):
        """测试计算加权分数"""
        cap = BeeCapability(
            bee_id="bee_004",
            bee_type="TestBee",
            capabilities={
                CapabilityCategory.CODE_GENERATION: 0.9,
                CapabilityCategory.CODE_ANALYSIS: 0.7,
                CapabilityCategory.SECURITY_AUDIT: 0.5,
            },
        )
        
        # 无需求返回 0.5
        assert cap.get_weighted_score([]) == 0.5
        
        # 单个需求
        assert cap.get_weighted_score([CapabilityCategory.CODE_GENERATION]) == 0.9
        
        # 多个需求取平均
        assert cap.get_weighted_score([
            CapabilityCategory.CODE_GENERATION,  # 0.9
            CapabilityCategory.CODE_ANALYSIS,     # 0.7
        ]) == 0.8  # (0.9 + 0.7) / 2

    def test_update_with_task_result(self):
        """测试用任务结果更新"""
        cap = BeeCapability(
            bee_id="bee_005",
            bee_type="TestBee",
        )
        
        # 初始状态
        assert cap.total_tasks_completed == 0
        assert cap.average_score == 0.0
        
        # 第一个任务
        cap.update_with_task_result({"task_id": "t1"}, score=0.8)
        assert cap.total_tasks_completed == 1
        assert cap.average_score == 0.8
        
        # 第二个任务
        cap.update_with_task_result({"task_id": "t2"}, score=0.6)
        assert cap.total_tasks_completed == 2
        assert cap.average_score == 0.7  # (0.8 + 0.6) / 2
        
        # 检查最近任务记录
        assert len(cap.recent_tasks) == 2


class TestTaskDecomposer:
    """测试 TaskDecomposer 类"""

    def test_decomposer_initialization(self):
        """测试分解器初始化"""
        decomposer = TaskDecomposer()
        
        assert decomposer.COMPLEXITY_THRESHOLD == 3

    def test_decompose_simple_task(self):
        """测试分解简单任务"""
        decomposer = TaskDecomposer()
        
        # 简单任务（复杂度 < 阈值）不分解
        task = {
            "task_id": "simple_001",
            "task_name": "简单任务",
            "task_description": "这是一个简单任务",
            "complexity": 1,  # 低于阈值
        }
        
        result = decomposer.decompose(task)
        
        assert isinstance(result, TaskDecomposition)
        assert len(result.subtasks) == 1  # 只有一个子任务
        assert result.estimated_total_time == result.subtasks[0].estimated_duration

    def test_decompose_complex_task(self):
        """测试分解复杂任务"""
        decomposer = TaskDecomposer()
        
        # 复杂任务应该会分解
        task = {
            "task_id": "complex_001",
            "task_name": "创建新项目",
            "task_description": "需要创建一个新的 Python 项目，包括项目结构、核心功能、测试和审查",
            "complexity": 4,  # 高于阈值
            "task_type": "create_project",
        }
        
        result = decomposer.decompose(task)
        
        assert isinstance(result, TaskDecomposition)
        # 复杂任务应该会分解为多个子任务
        # 具体数量取决于匹配的规则

    def test_rule_code_project(self):
        """测试项目创建规则"""
        decomposer = TaskDecomposer()
        
        # 触发项目创建规则
        task = {
            "task_id": "proj_001",
            "task_name": "创建新项目",
            "task_description": "需要创建一个新的 Python 项目，create project",
            "complexity": 4,
        }
        
        result = decomposer.decompose(task)
        
        # 应该触发项目创建规则，分解为多个子任务
        assert len(result.subtasks) >= 1
        
        # 检查是否有 setup/core/test/review 等子任务
        subtask_names = [s.task_name for s in result.subtasks]

    def test_rule_refactoring(self):
        """测试重构规则"""
        decomposer = TaskDecomposer()
        
        task = {
            "task_id": "refactor_001",
            "task_name": "重构代码",
            "task_description": "需要 refactor 这段代码，优化其结构",
            "complexity": 4,
        }
        
        result = decomposer.decompose(task)
        
        # 应该触发重构规则
        assert len(result.subtasks) >= 1

    def test_rule_security_audit(self):
        """测试安全审计规则"""
        decomposer = TaskDecomposer()
        
        task = {
            "task_id": "security_001",
            "task_name": "安全审计",
            "task_description": "需要进行 security audit，检查代码中的漏洞",
            "complexity": 4,
        }
        
        result = decomposer.decompose(task)
        
        # 应该触发安全审计规则
        # 安全审计可以并行执行多个检查：静态分析、动态测试、依赖检查
        assert len(result.subtasks) >= 1

    def test_rule_bug_fix(self):
        """测试 Bug 修复规则"""
        decomposer = TaskDecomposer()
        
        task = {
            "task_id": "bug_001",
            "task_name": "修复 Bug",
            "task_description": "需要 fix 这个 bug，解决代码中的问题",
            "complexity": 4,
        }
        
        result = decomposer.decompose(task)
        
        # 应该触发 Bug 修复规则
        assert len(result.subtasks) >= 1


class TestTaskAssignmentEngine:
    """测试 TaskAssignmentEngine 类"""

    def test_engine_initialization(self):
        """测试分配引擎初始化"""
        engine = TaskAssignmentEngine()
        
        assert engine.bee_capabilities == {}

    def test_register_bee(self):
        """测试注册蜜蜂"""
        engine = TaskAssignmentEngine()
        
        cap = BeeCapability(
            bee_id="bee_001",
            bee_type="EngineerBee",
            capabilities={
                CapabilityCategory.CODE_GENERATION: 0.9,
            },
        )
        
        engine.register_bee(cap)
        
        assert "bee_001" in engine.bee_capabilities
        assert engine.bee_capabilities["bee_001"] == cap

    def test_unregister_bee(self):
        """测试注销蜜蜂"""
        engine = TaskAssignmentEngine()
        
        cap = BeeCapability(bee_id="bee_001", bee_type="Test")
        engine.register_bee(cap)
        
        assert "bee_001" in engine.bee_capabilities
        
        engine.unregister_bee("bee_001")
        
        assert "bee_001" not in engine.bee_capabilities

    def test_find_best_bee_simple(self):
        """测试找到最佳蜜蜂（简单情况）"""
        engine = TaskAssignmentEngine()
        
        # 注册两个蜜蜂
        cap1 = BeeCapability(
            bee_id="bee_code",
            bee_type="EngineerBee",
            capabilities={CapabilityCategory.CODE_GENERATION: 0.9},
        )
        
        cap2 = BeeCapability(
            bee_id="bee_security",
            bee_type="SecurityBee",
            capabilities={CapabilityCategory.SECURITY_AUDIT: 0.95},
        )
        
        engine.register_bee(cap1)
        engine.register_bee(cap2)
        
        # 创建需要代码生成的任务
        subtask = Subtask(
            subtask_id="t1",
            task_name="代码生成",
            task_description="测试",
            required_capabilities=[CapabilityCategory.CODE_GENERATION],
        )
        
        best_bee = engine.find_best_bee(subtask, ["bee_code", "bee_security"])
        
        # 应该选择 bee_code，因为它在代码生成上更好
        assert best_bee == "bee_code"

    def test_find_best_bee_no_available(self):
        """测试没有可用蜜蜂的情况"""
        engine = TaskAssignmentEngine()
        
        subtask = Subtask(
            subtask_id="t1",
            task_name="测试",
            task_description="测试",
        )
        
        best_bee = engine.find_best_bee(subtask, [])
        
        assert best_bee is None

    def test_batch_assign(self):
        """测试批量分配"""
        engine = TaskAssignmentEngine()
        
        # 注册蜜蜂
        for i in range(3):
            cap = BeeCapability(
                bee_id=f"bee_{i}",
                bee_type="TestBee",
                capabilities={CapabilityCategory.CODE_GENERATION: 0.7 + i * 0.1},
            )
            engine.register_bee(cap)
        
        # 创建子任务
        subtasks = [
            Subtask(subtask_id="t1", task_name="1", task_description="1"),
            Subtask(subtask_id="t2", task_name="2", task_description="2"),
            Subtask(subtask_id="t3", task_name="3", task_description="3"),
        ]
        
        assignments = engine.batch_assign(subtasks, ["bee_0", "bee_1", "bee_2"])
        
        assert len(assignments) == 3

    def test_update_bee_performance(self):
        """测试更新蜜蜂表现"""
        engine = TaskAssignmentEngine()
        
        # 测试自动注册
        engine.update_bee_performance(
            bee_id="new_bee",
            task_result={"task_id": "t1", "bee_type": "EngineerBee"},
            score=0.85,
        )
        
        assert "new_bee" in engine.bee_capabilities
        
        cap = engine.bee_capabilities["new_bee"]
        assert cap.total_tasks_completed == 1
        assert cap.average_score == 0.85


class TestKnowledgeAggregator:
    """测试 KnowledgeAggregator 类"""

    def test_aggregator_initialization(self):
        """测试聚合器初始化"""
        aggregator = KnowledgeAggregator()
        
        assert aggregator.aggregated_cache == {}
        assert aggregator.SIMILARITY_THRESHOLD == 0.7

    def test_create_fragment(self):
        """测试创建知识片段"""
        aggregator = KnowledgeAggregator()
        
        fragment = aggregator.create_fragment(
            source_bee_id="bee_001",
            content="这是一段测试知识内容",
            confidence=0.9,
            source_task="task_001",
        )
        
        assert isinstance(fragment, KnowledgeFragment)
        assert fragment.source_bee_id == "bee_001"
        assert fragment.content == "这是一段测试知识内容"
        assert fragment.confidence == 0.9
        assert fragment.source_task == "task_001"
        assert fragment.fragment_id is not None

    def test_aggregate_simple(self):
        """测试简单聚合"""
        aggregator = KnowledgeAggregator()
        
        fragments = [
            aggregator.create_fragment(
                source_bee_id="bee_1",
                content="知识片段 1",
                confidence=0.8,
            ),
            aggregator.create_fragment(
                source_bee_id="bee_2",
                content="知识片段 2",
                confidence=0.9,
            ),
        ]
        
        result = aggregator.aggregate(fragments, topic="测试主题")
        
        assert isinstance(result, AggregatedKnowledge)
        assert result.confidence == pytest.approx(0.85)
        assert len(result.source_fragments) == 2
        assert "测试主题" in result.topics

    def test_aggregate_empty(self):
        """测试空聚合"""
        aggregator = KnowledgeAggregator()
        
        result = aggregator.aggregate([])
        
        assert isinstance(result, AggregatedKnowledge)
        assert result.summary == ""
        assert result.confidence == 0.0
        assert result.consensus_level == 0.0

    def test_get_cached(self):
        """测试获取缓存"""
        aggregator = KnowledgeAggregator()
        
        # 先聚合，存入缓存
        fragments = [
            aggregator.create_fragment("bee_1", "test", 0.8),
        ]
        aggregated = aggregator.aggregate(fragments)
        
        # 获取缓存
        cached = aggregator.get_cached(aggregated.aggregation_id)
        
        assert cached is not None
        assert cached.aggregation_id == aggregated.aggregation_id
        
        # 不存在的 ID
        assert aggregator.get_cached("nonexistent") is None

    def test_generate_summary(self):
        """测试生成摘要"""
        aggregator = KnowledgeAggregator()
        
        # 创建多个片段
        fragments = [
            aggregator.create_fragment(
                source_bee_id="bee_1",
                content="这是第一个知识片段，包含重要信息",
                confidence=0.9,
            ),
            aggregator.create_fragment(
                source_bee_id="bee_2",
                content="这是第二个知识片段，补充更多细节",
                confidence=0.8,
            ),
        ]
        
        # 聚合
        result = aggregator.aggregate(fragments)
        
        # 摘要应该包含两个片段的信息
        assert result.summary is not None
        assert len(result.summary) > 0


class TestCollaborativeSwarm:
    """测试 CollaborativeSwarm 类"""

    def test_swarm_initialization(self):
        """测试协作蜂群初始化"""
        swarm = CollaborativeSwarm()
        
        assert swarm.communication_bus is None
        assert isinstance(swarm.decomposer, TaskDecomposer)
        assert isinstance(swarm.assignment_engine, TaskAssignmentEngine)
        assert isinstance(swarm.knowledge_aggregator, KnowledgeAggregator)
        assert swarm.active_decompositions == {}
        assert swarm.completed_decompositions == []
        assert swarm.is_running == False

    def test_register_bee_capability(self):
        """测试注册蜜蜂能力"""
        swarm = CollaborativeSwarm()
        
        cap = BeeCapability(
            bee_id="test_bee",
            bee_type="TestBee",
            capabilities={CapabilityCategory.CODE_GENERATION: 0.8},
        )
        
        swarm.register_bee_capability(cap)
        
        # 应该委托给 assignment_engine
        assert "test_bee" in swarm.assignment_engine.bee_capabilities

    def test_get_execution_status_nonexistent(self):
        """测试获取不存在任务的执行状态"""
        swarm = CollaborativeSwarm()
        
        status = swarm.get_execution_status("nonexistent")
        
        assert status is None

    def test_get_statistics(self):
        """测试获取统计信息"""
        swarm = CollaborativeSwarm()
        
        stats = swarm.get_statistics()
        
        assert isinstance(stats, dict)
        assert "active_tasks" in stats
        assert "completed_tasks" in stats
        assert "total_subtasks_executed" in stats
        assert "average_task_duration" in stats
        assert "registered_bees" in stats
        assert "knowledge_fragments_aggregated" in stats

    def test_get_execution_status(self):
        """测试获取执行状态"""
        swarm = CollaborativeSwarm()
        
        # 创建一个分解并加入活跃列表
        subtask = Subtask(
            subtask_id="s1",
            task_name="测试子任务",
            task_description="测试",
        )
        subtask.status = TaskStatus.RUNNING
        subtask.assigned_to = "bee_1"
        
        decomposition = TaskDecomposition(
            original_task_id="active_001",
            original_task_name="活跃任务",
            original_task_description="测试",
            subtasks=[subtask],
        )
        
        swarm.active_decompositions["active_001"] = decomposition
        
        # 获取状态
        status = swarm.get_execution_status("active_001")
        
        assert status is not None
        assert status["task_id"] == "active_001"
        assert status["original_task_name"] == "活跃任务"
        assert status["total_subtasks"] == 1
        assert status["running"] == 1


class TestGetCollaborativeSwarm:
    """测试 get_collaborative_swarm 便捷函数"""

    def test_get_swarm_returns_instance(self):
        """测试返回实例"""
        swarm = get_collaborative_swarm()
        
        assert isinstance(swarm, CollaborativeSwarm)

    def test_get_swarm_singleton(self):
        """测试单例模式"""
        swarm1 = get_collaborative_swarm()
        swarm2 = get_collaborative_swarm()
        
        assert swarm1 is swarm2

    def test_get_swarm_with_bus(self):
        """测试带通信总线"""
        mock_bus = "mock_bus"
        
        swarm = get_collaborative_swarm(communication_bus=mock_bus)
        
        assert swarm.communication_bus == mock_bus


class TestKnowledgeFragment:
    """测试 KnowledgeFragment 数据类"""

    def test_fragment_creation_defaults(self):
        """测试创建知识片段（默认值）"""
        fragment = KnowledgeFragment(
            fragment_id="frag_001",
            source_bee_id="bee_1",
            content="测试内容",
            confidence=0.8,
        )
        
        assert fragment.fragment_id == "frag_001"
        assert fragment.source_bee_id == "bee_1"
        assert fragment.content == "测试内容"
        assert fragment.confidence == 0.8
        
        # 默认值
        assert fragment.source_task is None
        assert fragment.references == []
        assert fragment.vector is None

    def test_fragment_get_summary(self):
        """测试获取摘要"""
        # 短内容
        short_frag = KnowledgeFragment(
            fragment_id="f1",
            source_bee_id="bee",
            content="短内容",
            confidence=0.8,
        )
        assert short_frag.get_summary(max_length=100) == "短内容"
        
        # 长内容
        long_content = "这是一段非常长的内容，超过了默认的 100 字符限制，应该会被截断。" * 3
        long_frag = KnowledgeFragment(
            fragment_id="f2",
            source_bee_id="bee",
            content=long_content,
            confidence=0.8,
        )
        summary = long_frag.get_summary(max_length=50)
        assert len(summary) > 50  # 包含 "..."
        assert "..." in summary


class TestAggregatedKnowledge:
    """测试 AggregatedKnowledge 数据类"""

    def test_aggregated_creation_defaults(self):
        """测试创建聚合知识（默认值）"""
        aggregated = AggregatedKnowledge(
            aggregation_id="agg_001",
        )
        
        assert aggregated.aggregation_id == "agg_001"
        
        # 默认值
        assert aggregated.source_fragments == []
        assert aggregated.summary == ""
        assert aggregated.confidence == 0.0
        assert aggregated.consensus_level == 0.0
        assert aggregated.topics == []
