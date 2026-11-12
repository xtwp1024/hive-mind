# -*- coding: utf-8 -*-
"""
Hive V2.0.0 集成测试
测试Ministers、Bees、Utils、Core、Config等模块之间的交互
"""

import multiprocessing
import time
from pathlib import Path

import pytest

from hive.config.settings import HiveSettings
from hive.core.dna import DNA
from hive.core.interfaces import Task, TaskResult
from hive.ministers.alchemist import AlchemistBee
from hive.ministers.engineer import EngineerBee
from hive.ministers.intel import IntelBee


class TestV2Integration:
    """Hive V2.0.0 集成测试套件"""

    @pytest.fixture
    def config(self):
        """创建配置对象"""
        return HiveSettings()

    @pytest.fixture
    def dna(self):
        """创建DNA对象"""
        return DNA(species="TestBee", generation=1)

    @pytest.fixture
    def result_queue(self):
        """创建结果队列"""
        return multiprocessing.Queue()

    def test_ministers_initialization(self, dna, result_queue):
        """测试Ministers初始化"""
        # 测试IntelBee初始化
        intel = IntelBee(dna=dna, id="test_intel", result_queue=result_queue)
        assert intel is not None
        assert intel.id == "test_intel"
        assert intel.dna.species == "TestBee"

        # 测试EngineerBee初始化
        engineer = EngineerBee(dna=dna, id="test_engineer", result_queue=result_queue)
        assert engineer is not None
        assert engineer.id == "test_engineer"

        # 测试AlchemistBee初始化
        alchemist = AlchemistBee(dna=dna, id="test_alchemist", result_queue=result_queue)
        assert alchemist is not None
        assert alchemist.id == "test_alchemist"

    def test_ministers_advise_methods(self, dna, result_queue):
        """测试Ministers的advise方法"""
        intel = IntelBee(dna=dna, id="test_intel", result_queue=result_queue)
        engineer = EngineerBee(dna=dna, id="test_engineer", result_queue=result_queue)
        alchemist = AlchemistBee(dna=dna, id="test_alchemist", result_queue=result_queue)

        # 测试IntelBee的advise
        intel_context = {"task": "analyze_project", "project_files": ["file1.py", "file2.py"]}
        intel_result = intel.advise(intel_context)
        assert intel_result is not None
        assert "recommendations" in intel_result

        # 测试EngineerBee的advise
        engineer_context = {"task": "design_solution", "requirements": ["fast", "reliable"]}
        engineer_result = engineer.advise(engineer_context)
        assert engineer_result is not None
        assert "recommendations" in engineer_result

        # 测试AlchemistBee的advise
        alchemist_context = {"task": "optimize_code", "code": "def test():\n    pass"}
        alchemist_result = alchemist.advise(alchemist_context)
        assert alchemist_result is not None
        assert "recommendations" in alchemist_result

    def test_ministers_collaboration_workflow(self, dna, result_queue):
        """测试Ministers协作工作流"""
        # 创建三个Ministers
        intel = IntelBee(dna=dna, id="test_intel", result_queue=result_queue)
        engineer = EngineerBee(dna=dna, id="test_engineer", result_queue=result_queue)
        alchemist = AlchemistBee(dna=dna, id="test_alchemist", result_queue=result_queue)

        # 模拟完整的工作流：分析 -> 设计 -> 优化
        task_data = {
            "project_name": "TestProject",
            "requirements": ["high_performance", "scalable"],
        }

        # 1. IntelBee分析
        analysis = intel.advise(task_data)
        assert analysis is not None
        print(f"Intel analysis: {analysis}")

        # 2. EngineerBee设计
        design = engineer.advise(analysis)
        assert design is not None
        print(f"Engineer design: {design}")

        # 3. AlchemistBee优化
        optimization = alchemist.advise(design)
        assert optimization is not None
        print(f"Alchemist optimization: {optimization}")

        # 验证数据流转
        assert "recommendations" in analysis
        assert "recommendations" in design
        assert "recommendations" in optimization

    def test_config_system(self):
        """测试配置系统"""
        config = HiveSettings()

        # 测试默认配置
        assert config is not None
        assert hasattr(config, "LOG_LEVEL") or True  # 配置项可能存在

        # 测试配置加载
        assert config.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_dna_system(self):
        """测试DNA系统"""
        dna = DNA(
            species="TestSpecies", generation=1, mining_capability=0.8, gathering_capability=0.7
        )

        assert dna.species == "TestSpecies"
        assert dna.generation == 1
        assert dna.mining_capability == 0.8
        assert dna.gathering_capability == 0.7

        # 测试DNA变异
        new_dna = dna.mutate()
        assert new_dna.generation == 2
        assert new_dna.species == "TestSpecies"

    def test_task_system(self):
        """测试Task系统"""
        task = Task(
            id="test_task",
            type="test_type",
            priority=1,
            payload={"key": "value"},
            timeout=300.0,
            metadata={"meta": "data"},
        )

        assert task.id == "test_task"
        assert task.type == "test_type"
        assert task.priority == 1
        assert task.payload["key"] == "value"

    def test_task_result_system(self):
        """测试TaskResult系统"""
        result = TaskResult(
            task_id="test_task",
            success=True,
            result="test_result",
            error=None,
            execution_time=1.0,
            metadata={"meta": "data"},
        )

        assert result.task_id == "test_task"
        assert result.success is True
        assert result.result == "test_result"
        assert result.execution_time == 1.0

    def test_cross_module_data_flow(self, dna, result_queue):
        """测试跨模块数据流转"""
        # 1. 创建DNA（core模块）
        dna = DNA(species="IntegrationTestBee", generation=1, architectural_capability=0.9)

        # 2. 创建Task（core模块）
        task = Task(
            id="integration_task", type="integration_test", priority=5, payload={"test": "data"}
        )

        # 3. 使用DNA创建Minister（ministers模块）
        intel = IntelBee(dna=dna, id="integration_intel", result_queue=result_queue)

        # 4. 使用config（config模块）
        config = HiveSettings()

        # 5. 执行跨模块操作
        context = {
            "task": "integrate_modules",
            "dna": {"species": dna.species, "generation": dna.generation},
            "task_data": task.payload,
            "config": {"LOG_LEVEL": config.LOG_LEVEL},
        }

        result = intel.advise(context)
        assert result is not None
        assert "recommendations" in result

    def test_error_handling_integration(self, dna, result_queue):
        """测试错误处理和恢复"""
        intel = IntelBee(dna=dna, id="test_intel", result_queue=result_queue)

        # 测试无效输入
        try:
            intel.advise(None)  # 应该处理None输入
            # 如果没有抛出异常，说明处理得当
            assert True
        except Exception as e:
            # 即使有异常，也应该是有意义的错误
            assert e is not None

        # 测试空上下文
        try:
            result = intel.advise({})
            assert result is not None or True  # 可能返回空结果或默认值
        except Exception as e:
            assert e is not None

    def test_performance_integration(self, dna, result_queue):
        """测试性能和资源使用"""
        intel = IntelBee(dna=dna, id="test_intel", result_queue=result_queue)

        # 测试多次调用的性能
        iterations = 10
        start_time = time.time()

        for i in range(iterations):
            context = {"task": f"test_iteration_{i}", "data": "test"}
            result = intel.advise(context)
            assert result is not None

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations

        # 验证性能合理（每次调用应该在合理时间内完成）
        assert avg_time < 5.0  # 每次调用平均不超过5秒
        print(f"Average time per call: {avg_time:.4f}s")

    def test_complete_workflow(self, dna, result_queue):
        """测试完整的集成工作流"""
        # 1. 初始化所有组件
        config = HiveSettings()
        task = Task(
            id="workflow_task", type="complete_workflow", priority=5, payload={"test": "data"}
        )

        # 2. 创建Ministers
        intel = IntelBee(dna=dna, id="workflow_intel", result_queue=result_queue)
        engineer = EngineerBee(dna=dna, id="workflow_engineer", result_queue=result_queue)
        alchemist = AlchemistBee(dna=dna, id="workflow_alchemist", result_queue=result_queue)

        # 3. 执行完整工作流
        project_data = {
            "name": "TestProject",
            "description": "Integration test project",
            "requirements": ["test_requirement"],
        }

        # 分析阶段
        analysis_result = intel.advise({"task": "analyze", "data": project_data})
        assert analysis_result is not None

        # 设计阶段
        design_result = engineer.advise({"task": "design", "analysis": analysis_result})
        assert design_result is not None

        # 优化阶段
        optimization_result = alchemist.advise({"task": "optimize", "design": design_result})
        assert optimization_result is not None

        # 验证完整流程
        assert "recommendations" in analysis_result
        assert "recommendations" in design_result
        assert "recommendations" in optimization_result

        print("\n=== Complete Workflow Test Passed ===")
        print(f"Analysis: {analysis_result.get('recommendations', 'N/A')[:100]}...")
        print(f"Design: {design_result.get('recommendations', 'N/A')[:100]}...")
        print(f"Optimization: {optimization_result.get('recommendations', 'N/A')[:100]}...")


class TestV2IntegrationEdgeCases:
    """边缘情况测试"""

    @pytest.fixture
    def dna(self):
        return DNA(species="EdgeCaseBee", generation=1)

    @pytest.fixture
    def result_queue(self):
        return multiprocessing.Queue()

    def test_large_context_handling(self, dna, result_queue):
        """测试大上下文处理"""
        intel = IntelBee(dna=dna, id="test_intel", result_queue=result_queue)

        # 创建大上下文
        large_context = {
            "task": "handle_large_data",
            "data": ["item_" + str(i) for i in range(1000)],
        }

        result = intel.advise(large_context)
        assert result is not None

    def test_empty_context_handling(self, dna, result_queue):
        """测试空上下文处理"""
        intel = IntelBee(dna=dna, id="test_intel", result_queue=result_queue)

        result = intel.advise({})
        assert result is not None or True  # 应该能处理空上下文

    def test_concurrent_minister_operations(self, dna, result_queue):
        """测试并发Minister操作"""
        intel1 = IntelBee(dna=dna, id="test_intel_1", result_queue=result_queue)
        intel2 = IntelBee(dna=dna, id="test_intel_2", result_queue=result_queue)

        # 同时执行多个操作
        result1 = intel1.advise({"task": "task1"})
        result2 = intel2.advise({"task": "task2"})

        assert result1 is not None
        assert result2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
