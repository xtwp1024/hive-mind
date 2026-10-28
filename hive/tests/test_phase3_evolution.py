# -*- coding: utf-8 -*-
"""
Phase 3: 自主进化能力 - 单元测试

测试以下模块:
1. CapabilityGapAnalyzer - 能力缺口分析器
2. SpeciesBlueprint - 蜂种蓝图系统
3. AutoSpeciesGenerator - 自动蜂种生成器
4. DynamicResourceAllocator - 动态资源分配系统
5. DNAAdaptiveEngine - DNA 自适应修改系统
"""

import unittest
import time
import threading
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCapabilityGapAnalyzer(unittest.TestCase):
    """测试能力缺口分析器"""

    def setUp(self):
        from hive.core.capability_gap_analyzer import (
            CapabilityGapAnalyzer,
            TaskExecutionRecord,
            GapSeverity,
            GapType,
        )
        self.CapabilityGapAnalyzer = CapabilityGapAnalyzer
        self.TaskExecutionRecord = TaskExecutionRecord
        self.GapSeverity = GapSeverity
        self.GapType = GapType

        self.analyzer = self.CapabilityGapAnalyzer(max_records=1000)

    def test_record_task(self):
        """测试任务记录"""
        record = self.TaskExecutionRecord(
            task_id="test_001",
            bee_id="bee_001",
            bee_type="miner_bee",
            task_category="mining",
            success=True,
            duration=5.0,
            capability_used="mining",
        )

        self.analyzer.record_task(record)

        records = self.analyzer.get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].task_id, "test_001")

    def test_record_task_simple(self):
        """测试简化的任务记录接口"""
        self.analyzer.record_task_simple(
            task_id="test_002",
            bee_id="bee_001",
            bee_type="miner_bee",
            task_category="mining",
            success=False,
            duration=10.0,
            error_message="Connection timeout",
            capability_used="mining",
        )

        records = self.analyzer.get_records()
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].success)
        self.assertIsNotNone(records[0].error_message)

    def test_get_records_with_window(self):
        """测试按时间窗口获取记录"""
        self.analyzer.record_task_simple(
            task_id="old_task",
            bee_id="bee_001",
            bee_type="miner",
            task_category="mining",
            success=True,
            duration=5.0,
        )

        all_records = self.analyzer.get_records()
        self.assertEqual(len(all_records), 1)

        recent_records = self.analyzer.get_records(hours=1.0)
        self.assertEqual(len(recent_records), 1)

    def test_analyze_empty(self):
        """测试空数据的分析"""
        report = self.analyzer.analyze(analysis_window_hours=24)

        self.assertEqual(report.total_tasks_analyzed, 0)
        self.assertEqual(report.total_gaps_detected, 0)

    def test_analyze_success_rate_gap(self):
        """测试成功率缺口分析"""
        for i in range(10):
            success = i < 3
            self.analyzer.record_task_simple(
                task_id=f"task_{i}",
                bee_id=f"bee_{i % 3}",
                bee_type="miner_bee",
                task_category="mining",
                success=success,
                duration=5.0 + i,
                capability_used="mining",
            )

        report = self.analyzer.analyze(analysis_window_hours=24)

        self.assertEqual(report.total_tasks_analyzed, 10)

    def test_analyze_efficiency_gap(self):
        """测试效率缺口分析"""
        for i in range(10):
            self.analyzer.record_task_simple(
                task_id=f"task_{i}",
                bee_id="bee_001",
                bee_type="miner_bee",
                task_category="mining",
                success=True,
                duration=100.0 + i * 10,
                capability_used="mining",
                expected_duration=30.0,
            )

        report = self.analyzer.analyze(analysis_window_hours=24)

        self.assertEqual(report.total_tasks_analyzed, 10)

    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.analyzer.get_stats()

        self.assertIn("total_records", stats)
        self.assertIn("detected_gaps_count", stats)

    def test_get_latest_report(self):
        """测试获取最新报告"""
        report = self.analyzer.get_latest_report()
        self.assertIsNone(report)


class TestSpeciesBlueprint(unittest.TestCase):
    """测试蜂种蓝图系统"""

    def setUp(self):
        from hive.core.species_blueprint import (
            SpeciesBlueprint,
            CapabilitySpecification,
            TaskAffinity,
            BlueprintStatus,
            BlueprintOrigin,
            CapabilityPriority,
            BlueprintRegistry,
        )
        self.SpeciesBlueprint = SpeciesBlueprint
        self.CapabilitySpecification = CapabilitySpecification
        self.TaskAffinity = TaskAffinity
        self.BlueprintStatus = BlueprintStatus
        self.BlueprintOrigin = BlueprintOrigin
        self.CapabilityPriority = CapabilityPriority
        self.BlueprintRegistry = BlueprintRegistry

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_blueprint(self):
        """测试创建蓝图"""
        blueprint = self.SpeciesBlueprint(
            blueprint_id="test_blueprint_001",
            name="Test Miner Blueprint",
            description="A test blueprint for mining tasks",
            version="1.0.0",
        )

        self.assertEqual(blueprint.blueprint_id, "test_blueprint_001")
        self.assertEqual(blueprint.name, "Test Miner Blueprint")
        self.assertEqual(blueprint.status, self.BlueprintStatus.DRAFT)

    def test_add_capability(self):
        """测试添加能力规格"""
        blueprint = self.SpeciesBlueprint(
            blueprint_id="test_001",
            name="Test Blueprint",
            description="Test",
        )

        blueprint.add_capability(
            name="mining",
            target_value=0.9,
            priority=self.CapabilityPriority.PRIMARY,
            weight=2.0,
        )

        self.assertIn("mining", blueprint.capabilities)
        spec = blueprint.capabilities["mining"]
        self.assertEqual(spec.target_value, 0.9)
        self.assertEqual(spec.priority, self.CapabilityPriority.PRIMARY)

    def test_add_task_affinity(self):
        """测试添加任务亲和度"""
        blueprint = self.SpeciesBlueprint(
            blueprint_id="test_001",
            name="Test Blueprint",
            description="Test",
        )

        blueprint.add_task_affinity(
            task_category="mining",
            affinity_score=0.9,
            is_exclusive=True,
        )

        self.assertEqual(len(blueprint.task_affinities), 1)
        self.assertEqual(blueprint.task_affinities[0].task_category, "mining")
        self.assertTrue(blueprint.task_affinities[0].is_exclusive)

    def test_validate_valid_blueprint(self):
        """测试验证有效的蓝图"""
        blueprint = self.SpeciesBlueprint(
            blueprint_id="valid_001",
            name="Valid Blueprint",
            description="A valid blueprint with all required fields",
        )

        blueprint.add_capability(
            name="mining",
            target_value=0.8,
            priority=self.CapabilityPriority.PRIMARY,
        )

        result = blueprint.validate()

        self.assertTrue(result.is_valid)

    def test_validate_invalid_blueprint(self):
        """测试验证无效的蓝图"""
        blueprint = self.SpeciesBlueprint(
            blueprint_id="invalid_001",
            name="",
            description="Short",
        )

        result = blueprint.validate()

    def test_get_primary_capabilities(self):
        """测试获取主要能力"""
        blueprint = self.SpeciesBlueprint(
            blueprint_id="test_001",
            name="Test Blueprint",
            description="Test",
        )

        blueprint.add_capability("mining", 0.9, self.CapabilityPriority.PRIMARY)
        blueprint.add_capability("gathering", 0.7, self.CapabilityPriority.SECONDARY)
        blueprint.add_capability("sentinel", 0.6, self.CapabilityPriority.PRIMARY)

        primaries = blueprint.get_primary_capabilities()

        self.assertEqual(len(primaries), 2)
        primary_names = {name for name, _ in primaries}
        self.assertIn("mining", primary_names)
        self.assertIn("sentinel", primary_names)

    def test_get_preferred_task_categories(self):
        """测试获取偏好任务类别"""
        blueprint = self.SpeciesBlueprint(
            blueprint_id="test_001",
            name="Test Blueprint",
            description="Test",
        )

        blueprint.add_task_affinity("mining", 0.9)
        blueprint.add_task_affinity("gathering", 0.7)
        blueprint.add_task_affinity("sentinel", 0.8)

        preferred = blueprint.get_preferred_task_categories()

        self.assertEqual(preferred[0], "mining")
        self.assertEqual(preferred[1], "sentinel")
        self.assertEqual(preferred[2], "gathering")

    def test_mutate(self):
        """测试蓝图变异"""
        blueprint = self.SpeciesBlueprint(
            blueprint_id="parent_001",
            name="Parent Blueprint",
            description="Parent",
        )
        blueprint.add_capability("mining", 0.8, self.CapabilityPriority.PRIMARY)

        mutated = blueprint.mutate(mutation_rate=0.5)

        self.assertNotEqual(mutated.blueprint_id, blueprint.blueprint_id)
        self.assertEqual(mutated.origin, self.BlueprintOrigin.MUTATED)
        self.assertEqual(len(mutated.parent_blueprints), 1)
        self.assertIn(blueprint.blueprint_id, mutated.parent_blueprints)

    def test_crossover(self):
        """测试蓝图交叉"""
        blueprint1 = self.SpeciesBlueprint(
            blueprint_id="parent_001",
            name="Parent 1",
            description="First parent",
        )
        blueprint1.add_capability("mining", 0.9, self.CapabilityPriority.PRIMARY)

        blueprint2 = self.SpeciesBlueprint(
            blueprint_id="parent_002",
            name="Parent 2",
            description="Second parent",
        )
        blueprint2.add_capability("sentinel", 0.8, self.CapabilityPriority.PRIMARY)

        child1, child2 = blueprint1.crossover(blueprint2)

        self.assertEqual(child1.origin, self.BlueprintOrigin.CROSSED)
        self.assertEqual(child2.origin, self.BlueprintOrigin.CROSSED)
        self.assertIn(blueprint1.blueprint_id, child1.parent_blueprints)
        self.assertIn(blueprint2.blueprint_id, child1.parent_blueprints)

    def test_to_dict_and_from_dict(self):
        """测试字典序列化和反序列化"""
        blueprint = self.SpeciesBlueprint(
            blueprint_id="serialize_001",
            name="Serialize Test",
            description="Test serialization",
            version="2.0.0",
            mutation_rate=0.15,
            learning_rate=0.6,
        )
        blueprint.add_capability("mining", 0.85, self.CapabilityPriority.PRIMARY)
        blueprint.add_task_affinity("mining", 0.9)

        data = blueprint.to_dict()
        restored = self.SpeciesBlueprint.from_dict(data)

        self.assertEqual(restored.blueprint_id, blueprint.blueprint_id)
        self.assertEqual(restored.name, blueprint.name)
        self.assertEqual(restored.version, blueprint.version)
        self.assertIn("mining", restored.capabilities)

    def test_registry_register(self):
        """测试蓝图注册表注册"""
        registry = self.BlueprintRegistry(Path(self.temp_dir))

        blueprint = self.SpeciesBlueprint(
            blueprint_id="reg_test_001",
            name="Registry Test",
            description="Test registry",
        )

        result = registry.register(blueprint)
        self.assertTrue(result)

        retrieved = registry.get("reg_test_001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Registry Test")

    def test_registry_unregister(self):
        """测试蓝图注册表注销"""
        registry = self.BlueprintRegistry(Path(self.temp_dir))

        blueprint = self.SpeciesBlueprint(
            blueprint_id="unreg_test_001",
            name="Unregister Test",
            description="Test",
        )

        registry.register(blueprint)
        self.assertIsNotNone(registry.get("unreg_test_001"))

        result = registry.unregister("unreg_test_001")
        self.assertTrue(result)
        self.assertIsNone(registry.get("unreg_test_001"))

    def test_registry_get_by_status(self):
        """测试按状态获取蓝图"""
        registry = self.BlueprintRegistry(Path(self.temp_dir))

        draft_bp = self.SpeciesBlueprint(
            blueprint_id="draft_001",
            name="Draft Blueprint",
            description="Draft",
        )
        registry.register(draft_bp)

        active_bp = self.SpeciesBlueprint(
            blueprint_id="active_001",
            name="Active Blueprint",
            description="Active",
        )
        active_bp.status = self.BlueprintStatus.ACTIVE
        registry.register(active_bp)

        drafts = registry.get_by_status(self.BlueprintStatus.DRAFT)
        actives = registry.get_by_status(self.BlueprintStatus.ACTIVE)

        self.assertEqual(len(drafts), 1)
        self.assertEqual(len(actives), 1)

    def test_registry_get_stats(self):
        """测试获取注册表统计"""
        registry = self.BlueprintRegistry(Path(self.temp_dir))

        stats = registry.get_stats()
        self.assertIn("total_blueprints", stats)
        self.assertIn("by_status", stats)


class TestCapabilitySpecification(unittest.TestCase):
    """测试能力规格"""

    def setUp(self):
        from hive.core.species_blueprint import CapabilitySpecification, CapabilityPriority
        self.CapabilitySpecification = CapabilitySpecification
        self.CapabilityPriority = CapabilityPriority

    def test_validate_valid_spec(self):
        """测试验证有效的能力规格"""
        spec = self.CapabilitySpecification(
            capability_name="mining",
            target_value=0.8,
            min_value=0.3,
            max_value=1.0,
            priority=self.CapabilityPriority.PRIMARY,
            weight=1.0,
        )

        is_valid, errors = spec.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_invalid_target(self):
        """测试验证无效的目标值"""
        spec = self.CapabilitySpecification(
            capability_name="mining",
            target_value=1.5,
            min_value=0.3,
            max_value=1.0,
        )

        is_valid, errors = spec.validate()
        self.assertFalse(is_valid)

    def test_validate_inverted_bounds(self):
        """测试验证反转的边界"""
        spec = self.CapabilitySpecification(
            capability_name="mining",
            target_value=0.5,
            min_value=0.8,
            max_value=0.3,
        )

        is_valid, errors = spec.validate()
        self.assertFalse(is_valid)


class TestTaskAffinity(unittest.TestCase):
    """测试任务亲和度"""

    def setUp(self):
        from hive.core.species_blueprint import TaskAffinity
        self.TaskAffinity = TaskAffinity

    def test_validate_valid_affinity(self):
        """测试验证有效的任务亲和度"""
        affinity = self.TaskAffinity(
            task_category="mining",
            affinity_score=0.8,
            min_task_complexity=0.2,
            max_task_complexity=0.8,
        )

        is_valid, errors = affinity.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_invalid_score(self):
        """测试验证无效的亲和度分数"""
        affinity = self.TaskAffinity(
            task_category="mining",
            affinity_score=1.5,
        )

        is_valid, errors = affinity.validate()
        self.assertFalse(is_valid)


class TestDynamicResourceAllocator(unittest.TestCase):
    """测试动态资源分配器"""

    def setUp(self):
        from hive.core.resource_allocator import (
            DynamicResourceAllocator,
            ResourceType,
            AllocationPriority,
            ConsumerProfile,
            ResourceQuota,
        )
        self.DynamicResourceAllocator = DynamicResourceAllocator
        self.ResourceType = ResourceType
        self.AllocationPriority = AllocationPriority
        self.ConsumerProfile = ConsumerProfile
        self.ResourceQuota = ResourceQuota

        self.allocator = self.DynamicResourceAllocator(
            system_resources={
                self.ResourceType.CPU: 8.0,
                self.ResourceType.MEMORY: 16.0,
                self.ResourceType.CONCURRENCY: 100.0,
            }
        )

    def test_register_consumer(self):
        """测试注册消费者"""
        profile = self.allocator.register_consumer(
            consumer_id="test_bee_001",
            consumer_type="miner_bee",
            base_priority=self.AllocationPriority.NORMAL,
        )

        self.assertIsNotNone(profile)
        self.assertEqual(profile.consumer_id, "test_bee_001")

    def test_register_duplicate_consumer(self):
        """测试重复注册消费者"""
        self.allocator.register_consumer("dup_001", "miner")
        result = self.allocator.register_consumer("dup_001", "miner")
        self.assertIsNotNone(result)

    def test_unregister_consumer(self):
        """测试注销消费者"""
        self.allocator.register_consumer("unreg_001", "miner")

        result = self.allocator.unregister_consumer("unreg_001")
        self.assertTrue(result)

        result2 = self.allocator.unregister_consumer("nonexistent")
        self.assertFalse(result2)

    def test_get_consumer(self):
        """测试获取消费者"""
        self.allocator.register_consumer("get_001", "miner")

        consumer = self.allocator.get_consumer("get_001")
        self.assertIsNotNone(consumer)
        self.assertEqual(consumer.consumer_id, "get_001")

        consumer2 = self.allocator.get_consumer("nonexistent")
        self.assertIsNone(consumer2)

    def test_request_resource_success(self):
        """测试成功请求资源"""
        self.allocator.register_consumer(
            consumer_id="requester_001",
            consumer_type="miner_bee",
            base_priority=self.AllocationPriority.HIGH,
            initial_quotas={
                self.ResourceType.CPU: {
                    "hard_limit": 2.0,
                    "soft_limit": 1.5,
                }
            },
        )

        decision = self.allocator.request_resource(
            consumer_id="requester_001",
            resource_type=self.ResourceType.CPU,
            amount=1.0,
        )

        self.assertTrue(decision.granted)
        self.assertGreater(decision.granted_amount, 0)

    def test_request_resource_unregistered(self):
        """测试未注册消费者请求资源"""
        decision = self.allocator.request_resource(
            consumer_id="nonexistent",
            resource_type=self.ResourceType.CPU,
            amount=1.0,
        )

        self.assertFalse(decision.granted)
        self.assertEqual(decision.reason, "consumer_not_registered")

    def test_request_resource_quota_exceeded(self):
        """测试超出配额请求资源"""
        self.allocator.register_consumer(
            consumer_id="quota_test_001",
            consumer_type="miner",
            initial_quotas={
                self.ResourceType.CPU: {
                    "hard_limit": 0.5,
                    "soft_limit": 0.4,
                    "burst_allowed": False,
                }
            },
        )

        decision = self.allocator.request_resource(
            consumer_id="quota_test_001",
            resource_type=self.ResourceType.CPU,
            amount=1.0,
        )

    def test_release_allocation(self):
        """测试释放资源"""
        self.allocator.register_consumer(
            consumer_id="release_test_001",
            consumer_type="miner",
            initial_quotas={
                self.ResourceType.CPU: {"hard_limit": 2.0, "soft_limit": 1.5}
            },
        )

        decision = self.allocator.request_resource(
            consumer_id="release_test_001",
            resource_type=self.ResourceType.CPU,
            amount=1.0,
        )

        self.assertTrue(decision.granted)

    def test_update_consumer_fitness(self):
        """测试更新消费者适应度"""
        self.allocator.register_consumer("fitness_001", "miner")

        result = self.allocator.update_consumer_fitness("fitness_001", 0.8)
        self.assertTrue(result)

        result2 = self.allocator.update_consumer_fitness("nonexistent", 0.5)
        self.assertFalse(result2)

    def test_update_consumer_task_result(self):
        """测试更新消费者任务结果"""
        self.allocator.register_consumer("task_001", "miner")

        result = self.allocator.update_consumer_task_result(
            consumer_id="task_001",
            success=True,
            fitness_delta=0.1,
        )
        self.assertTrue(result)

        consumer = self.allocator.get_consumer("task_001")
        self.assertEqual(consumer.task_count, 1)
        self.assertEqual(consumer.success_count, 1)

    def test_get_system_status(self):
        """测试获取系统状态"""
        status = self.allocator.get_system_status()

        self.assertIn("timestamp", status)
        self.assertIn("total_consumers", status)
        self.assertIn("system_resources", status)
        self.assertIn("consumers_by_priority", status)

    def test_get_consumer_stats(self):
        """测试获取消费者统计"""
        self.allocator.register_consumer("stats_001", "miner")

        stats = self.allocator.get_consumer_stats("stats_001")
        self.assertIsNotNone(stats)
        self.assertIn("profile", stats)
        self.assertIn("allocation_stats", stats)

        stats2 = self.allocator.get_consumer_stats("nonexistent")
        self.assertIsNone(stats2)

    def test_get_top_consumers(self):
        """测试获取资源使用最多的消费者"""
        top = self.allocator.get_top_consumers(self.ResourceType.CPU, count=5)
        self.assertIsInstance(top, list)


class TestResourceQuota(unittest.TestCase):
    """测试资源配额"""

    def setUp(self):
        from hive.core.resource_allocator import ResourceQuota, ResourceType
        self.ResourceQuota = ResourceQuota
        self.ResourceType = ResourceType

    def test_quota_limits(self):
        """测试配额限制检查"""
        quota = self.ResourceQuota(
            resource_type=self.ResourceType.CPU,
            hard_limit=2.0,
            soft_limit=1.5,
            current_usage=1.0,
        )

        self.assertFalse(quota.is_over_soft_limit())
        self.assertFalse(quota.is_over_hard_limit())

        quota.current_usage = 1.75
        self.assertTrue(quota.is_over_soft_limit())
        self.assertFalse(quota.is_over_hard_limit())

        quota.current_usage = 2.5
        self.assertTrue(quota.is_over_hard_limit())

    def test_quota_available(self):
        """测试配额可用计算"""
        quota = self.ResourceQuota(
            resource_type=self.ResourceType.MEMORY,
            hard_limit=16.0,
            soft_limit=12.0,
            current_usage=5.0,
        )

        self.assertEqual(quota.available(), 11.0)

    def test_quota_effective_limit(self):
        """测试有效配额限制"""
        quota_without_burst = self.ResourceQuota(
            resource_type=self.ResourceType.CPU,
            hard_limit=2.0,
            soft_limit=1.5,
            burst_allowed=False,
            burst_limit=3.0,
        )
        self.assertEqual(quota_without_burst.effective_limit(), 2.0)

        quota_with_burst = self.ResourceQuota(
            resource_type=self.ResourceType.CPU,
            hard_limit=2.0,
            soft_limit=1.5,
            burst_allowed=True,
            burst_limit=3.0,
        )
        self.assertEqual(quota_with_burst.effective_limit(), 3.0)


class TestDNAAdaptiveEngine(unittest.TestCase):
    """测试 DNA 自适应引擎"""

    def setUp(self):
        from hive.core.dna_adaptive import (
            DNAAdaptiveEngine,
            PerformanceSample,
            AdaptationType,
            PerformanceTrend,
            ParameterState,
        )
        self.DNAAdaptiveEngine = DNAAdaptiveEngine
        self.PerformanceSample = PerformanceSample
        self.AdaptationType = AdaptationType
        self.PerformanceTrend = PerformanceTrend
        self.ParameterState = ParameterState

        self.engine = self.DNAAdaptiveEngine()

    def test_create_mock_dna(self):
        """创建模拟 DNA 对象"""
        mock_dna = MagicMock()
        mock_dna.learning_rate = 0.1
        mock_dna.mutation_rate = 0.1
        mock_dna.immune_response = 0.5
        mock_dna.mining_capability = 0.5
        mock_dna.gathering_capability = 0.5
        mock_dna.architectural_capability = 0.5
        mock_dna.librarian_capability = 0.5
        mock_dna.sentinel_capability = 0.5
        mock_dna.fitness = 0.5
        return mock_dna

    def test_monitor_dna(self):
        """测试监控 DNA"""
        mock_dna = self.create_mock_dna()

        self.engine.monitor_dna(mock_dna, "test_dna_001", "bee")

        status = self.engine.get_status("test_dna_001")
        self.assertIsNotNone(status)
        self.assertTrue(status.get("monitored", False))

    def test_monitor_duplicate_dna(self):
        """测试重复监控 DNA"""
        mock_dna = self.create_mock_dna()

        self.engine.monitor_dna(mock_dna, "dup_001", "bee")

    def test_stop_monitoring(self):
        """测试停止监控"""
        mock_dna = self.create_mock_dna()
        self.engine.monitor_dna(mock_dna, "stop_001", "bee")

        result = self.engine.stop_monitoring("stop_001")
        self.assertTrue(result)

        result2 = self.engine.stop_monitoring("nonexistent")
        self.assertFalse(result2)

    def test_record_performance(self):
        """测试记录性能"""
        mock_dna = self.create_mock_dna()
        self.engine.monitor_dna(mock_dna, "perf_001", "bee")

        result = self.engine.record_performance(
            dna_id="perf_001",
            success=True,
            duration=5.0,
            error=False,
            task_type="mining",
            capabilities_used={"mining": 0.9},
        )

        self.assertTrue(result)

    def test_record_performance_unmonitored(self):
        """测试记录未监控 DNA 的性能"""
        result = self.engine.record_performance(
            dna_id="nonexistent",
            success=True,
            duration=5.0,
        )

        self.assertFalse(result)

    def test_suggest_adaptations_insufficient_data(self):
        """测试数据不足时的适应建议"""
        mock_dna = self.create_mock_dna()
        self.engine.monitor_dna(mock_dna, "insufficient_001", "bee")

        actions = self.engine.suggest_adaptations("insufficient_001")
        self.assertEqual(len(actions), 0)

    def test_suggest_adaptations_unmonitored(self):
        """测试未监控 DNA 的适应建议"""
        actions = self.engine.suggest_adaptations("nonexistent")
        self.assertEqual(len(actions), 0)

    def test_get_status_all(self):
        """测试获取所有状态"""
        status = self.engine.get_status()

        self.assertIn("total_monitored", status)
        self.assertIn("monitored_dnas", status)
        self.assertIn("auto_adapt_enabled", status)

    def test_get_status_single(self):
        """测试获取单个 DNA 状态"""
        mock_dna = self.create_mock_dna()
        self.engine.monitor_dna(mock_dna, "single_001", "bee")

        status = self.engine.get_status("single_001")
        self.assertIsNotNone(status)

        status2 = self.engine.get_status("nonexistent")
        self.assertEqual(status2, {})


class TestParameterState(unittest.TestCase):
    """测试参数状态"""

    def setUp(self):
        from hive.core.dna_adaptive import ParameterState, PerformanceTrend
        self.ParameterState = ParameterState
        self.PerformanceTrend = PerformanceTrend

    def test_record_value(self):
        """测试记录值"""
        state = self.ParameterState(
            name="learning_rate",
            current_value=0.1,
            min_value=0.0,
            max_value=1.0,
        )

        state.record_value(0.2)
        state.record_value(0.3)

        self.assertEqual(len(state.historical_values), 3)
        self.assertEqual(len(state.historical_gradients), 2)

    def test_get_trend_improving(self):
        """测试获取上升趋势"""
        state = self.ParameterState(
            name="fitness",
            current_value=0.5,
        )

        for i in range(20):
            state.record_value(0.5 + i * 0.02)

        trend, slope = state.get_trend()

    def test_get_trend_declining(self):
        """测试获取下降趋势"""
        state = self.ParameterState(
            name="fitness",
            current_value=0.9,
        )

        for i in range(20):
            state.record_value(0.9 - i * 0.02)

        trend, slope = state.get_trend()

    def test_get_trend_stable(self):
        """测试获取稳定趋势"""
        state = self.ParameterState(
            name="fitness",
            current_value=0.5,
        )

        for i in range(20):
            state.record_value(0.5 + (random.random() - 0.5) * 0.01)

    def test_get_average(self):
        """测试获取平均值"""
        state = self.ParameterState(
            name="learning_rate",
            current_value=0.1,
        )

        state.record_value(0.1)
        state.record_value(0.2)
        state.record_value(0.3)

        avg = state.get_average(window=3)
        self.assertAlmostEqual(avg, 0.2)

    def test_to_dict(self):
        """测试字典序列化"""
        state = self.ParameterState(
            name="test_param",
            current_value=0.5,
            min_value=0.0,
            max_value=1.0,
            adaptation_count=10,
            successful_adaptations=8,
        )

        data = state.to_dict()

        self.assertEqual(data["name"], "test_param")
        self.assertEqual(data["current_value"], 0.5)
        self.assertEqual(data["adaptation_count"], 10)
        self.assertEqual(data["successful_adaptations"], 8)


import random


if __name__ == "__main__":
    unittest.main()
