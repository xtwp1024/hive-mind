# -*- coding: utf-8 -*-
"""测试 BenchmarkSystem 性能基准系统

测试内容:
1. BenchmarkCategory 枚举
2. BenchmarkTask 数据类
3. BenchmarkResult 数据类
4. StandardBenchmarks 标准任务集
5. BenchmarkSystem 核心功能
6. 历史对比功能
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime

from hive.utils.benchmark_system import (
    BenchmarkCategory,
    BenchmarkTask,
    BenchmarkResult,
    BenchmarkComparison,
    StandardBenchmarks,
    BenchmarkSystem,
)


class TestBenchmarkCategory:
    """测试基准类别枚举"""

    def test_all_categories_exist(self):
        """测试所有基准类别都存在"""
        expected_categories = [
            "code_generation", "code_analysis", "bug_detection",
            "security_audit", "pattern_recognition", "knowledge_extraction",
            "code_optimization", "architecture_design", "system_analysis",
            "communication", "memory",
        ]
        
        for cat in expected_categories:
            assert hasattr(BenchmarkCategory, cat.upper())

    def test_category_values(self):
        """测试类别值"""
        assert BenchmarkCategory.CODE_GENERATION.value == "code_generation"
        assert BenchmarkCategory.BUG_DETECTION.value == "bug_detection"
        assert BenchmarkCategory.SECURITY_AUDIT.value == "security_audit"
        assert BenchmarkCategory.COMMUNICATION.value == "communication"
        assert BenchmarkCategory.MEMORY.value == "memory"


class TestBenchmarkTask:
    """测试基准任务"""

    def test_task_creation(self):
        """测试创建基准任务"""
        task = BenchmarkTask(
            task_id="test_001",
            name="测试任务",
            category=BenchmarkCategory.CODE_GENERATION,
            description="一个测试任务",
            difficulty=2,
            expected_duration=5.0,
            timeout=10.0,
            input_data={"param": "value"},
            scoring_criteria={"correctness": 0.7, "efficiency": 0.3},
        )
        
        assert task.task_id == "test_001"
        assert task.name == "测试任务"
        assert task.category == BenchmarkCategory.CODE_GENERATION
        assert task.difficulty == 2
        assert task.expected_duration == 5.0
        assert task.timeout == 10.0
        assert task.input_data == {"param": "value"}
        assert task.scoring_criteria == {"correctness": 0.7, "efficiency": 0.3}

    def test_task_defaults(self):
        """测试任务默认值"""
        task = BenchmarkTask(
            task_id="test_002",
            name="默认值测试",
            category=BenchmarkCategory.CODE_ANALYSIS,
            description="测试默认值",
        )
        
        assert task.difficulty == 1
        assert task.expected_duration == 5.0
        assert task.timeout == 30.0
        assert task.input_data == {}
        assert task.scoring_criteria == {"correctness": 1.0}


class TestBenchmarkResult:
    """测试基准结果"""

    def test_result_creation(self):
        """测试创建基准结果"""
        task = BenchmarkTask(
            task_id="res_test_001",
            name="结果测试",
            category=BenchmarkCategory.PATTERN_RECOGNITION,
            description="测试",
        )
        
        result = BenchmarkResult(
            task_id="res_test_001",
            task_name="结果测试",
            category=BenchmarkCategory.PATTERN_RECOGNITION,
            score=0.85,
            confidence=0.90,
            duration=3.5,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=True,
            metrics={"accuracy": 0.88, "precision": 0.92},
            error=None,
        )
        
        assert result.task_id == "res_test_001"
        assert result.task_name == "结果测试"
        assert result.category == BenchmarkCategory.PATTERN_RECOGNITION
        assert result.score == 0.85
        assert result.confidence == 0.90
        assert result.duration == 3.5
        assert result.success == True
        assert result.metrics == {"accuracy": 0.88, "precision": 0.92}
        assert result.error is None

    def test_result_failure(self):
        """测试失败的基准结果"""
        result = BenchmarkResult(
            task_id="fail_001",
            task_name="失败测试",
            category=BenchmarkCategory.SECURITY_AUDIT,
            score=0.0,
            confidence=0.0,
            duration=1.5,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=False,
            metrics={},
            error="Timeout occurred",
        )
        
        assert result.success == False
        assert result.score == 0.0
        assert result.error == "Timeout occurred"


class TestBenchmarkComparison:
    """测试基准对比"""

    def test_comparison_creation(self):
        """测试创建基准对比"""
        comparison = BenchmarkComparison(
            task_id="comp_001",
            task_name="对比测试",
            category=BenchmarkCategory.CODE_GENERATION,
            current_score=0.85,
            previous_score=0.75,
            score_delta=0.10,
            current_duration=5.0,
            previous_duration=6.0,
            duration_delta=-1.0,
            improvement_trend="improving",
            analysis="分数提升 10%，时间减少 1 秒",
        )
        
        assert comparison.task_id == "comp_001"
        assert comparison.score_delta == 0.10
        assert comparison.duration_delta == -1.0
        assert comparison.improvement_trend == "improving"


class TestStandardBenchmarks:
    """测试标准基准任务集"""

    def test_get_all_tasks_returns_list(self):
        """测试 get_all_tasks 返回任务列表"""
        tasks = StandardBenchmarks.get_all_tasks()
        
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        
        for task in tasks:
            assert isinstance(task, BenchmarkTask)

    def test_all_task_categories(self):
        """测试所有任务的类别"""
        tasks = StandardBenchmarks.get_all_tasks()
        
        categories = set()
        for task in tasks:
            categories.add(task.category)
        
        assert BenchmarkCategory.CODE_GENERATION in categories
        assert BenchmarkCategory.CODE_ANALYSIS in categories

    def test_code_generation_simple(self):
        """测试简单代码生成任务"""
        task = StandardBenchmarks.code_generation_simple()
        
        assert isinstance(task, BenchmarkTask)
        assert task.category == BenchmarkCategory.CODE_GENERATION
        assert task.difficulty == 1

    def test_code_generation_complex(self):
        """测试复杂代码生成任务"""
        task = StandardBenchmarks.code_generation_complex()
        
        assert isinstance(task, BenchmarkTask)
        assert task.category == BenchmarkCategory.CODE_GENERATION
        assert task.difficulty == 3

    def test_communication_latency(self):
        """测试通信延迟任务"""
        task = StandardBenchmarks.communication_latency()
        
        assert isinstance(task, BenchmarkTask)
        assert task.category == BenchmarkCategory.COMMUNICATION

    def test_memory_read_write(self):
        """测试内存读写任务"""
        task = StandardBenchmarks.memory_read_write()
        
        assert isinstance(task, BenchmarkTask)
        assert task.category == BenchmarkCategory.MEMORY

    def test_bug_detection_simple(self):
        """测试简单 Bug 检测任务"""
        task = StandardBenchmarks.bug_detection_simple()
        
        assert isinstance(task, BenchmarkTask)
        assert task.category == BenchmarkCategory.BUG_DETECTION

    def test_security_audit_simple(self):
        """测试简单安全审计任务"""
        task = StandardBenchmarks.security_audit_simple()
        
        assert isinstance(task, BenchmarkTask)
        assert task.category == BenchmarkCategory.SECURITY_AUDIT

    def test_pattern_recognition_simple(self):
        """测试简单模式识别任务"""
        task = StandardBenchmarks.pattern_recognition_simple()
        
        assert isinstance(task, BenchmarkTask)
        assert task.category == BenchmarkCategory.PATTERN_RECOGNITION

    def test_knowledge_extraction_simple(self):
        """测试简单知识提取任务"""
        task = StandardBenchmarks.knowledge_extraction_simple()
        
        assert isinstance(task, BenchmarkTask)
        assert task.category == BenchmarkCategory.KNOWLEDGE_EXTRACTION

    def test_code_analysis_simple(self):
        """测试简单代码分析任务"""
        task = StandardBenchmarks.code_analysis_simple()
        
        assert isinstance(task, BenchmarkTask)
        assert task.category == BenchmarkCategory.CODE_ANALYSIS

    def test_code_analysis_complex(self):
        """测试复杂代码分析任务"""
        task = StandardBenchmarks.code_analysis_complex()
        
        assert isinstance(task, BenchmarkTask)
        assert task.category == BenchmarkCategory.CODE_ANALYSIS


class TestBenchmarkSystem:
    """测试基准系统"""

    def test_system_creation(self):
        """测试创建基准系统"""
        system = BenchmarkSystem()
        
        assert isinstance(system, BenchmarkSystem)
        assert system._executor is not None

    def test_run_task(self):
        """测试运行单个基准任务"""
        system = BenchmarkSystem()
        
        task = BenchmarkTask(
            task_id="run_test_001",
            name="运行测试",
            category=BenchmarkCategory.CODE_GENERATION,
            description="测试运行",
        )
        
        result = system.run_task(task)
        
        assert isinstance(result, BenchmarkResult)
        assert result.task_id == "run_test_001"
        assert result.task_name == "运行测试"
        assert result.category == BenchmarkCategory.CODE_GENERATION
        assert 0.0 <= result.score <= 1.0
        assert result.duration >= 0.0

    def test_run_all_benchmarks(self):
        """测试运行所有基准"""
        system = BenchmarkSystem()
        
        results = system.run_all_benchmarks()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, BenchmarkResult)

    def test_calculate_average_score(self):
        """测试计算平均分数"""
        system = BenchmarkSystem()
        
        results = [
            BenchmarkResult(
                task_id="a",
                task_name="A",
                category=BenchmarkCategory.CODE_GENERATION,
                score=0.8,
                confidence=0.9,
                duration=1.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                metrics={},
            ),
            BenchmarkResult(
                task_id="b",
                task_name="B",
                category=BenchmarkCategory.CODE_GENERATION,
                score=0.6,
                confidence=0.7,
                duration=2.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                metrics={},
            ),
        ]
        
        avg = system.calculate_average_score(results)
        
        assert avg == pytest.approx(0.7)

    def test_calculate_average_score_empty(self):
        """测试空结果的平均分数"""
        system = BenchmarkSystem()
        
        avg = system.calculate_average_score([])
        
        assert avg == 0.0

    def test_get_run_stats(self):
        """测试获取运行统计"""
        system = BenchmarkSystem()
        
        stats = system.get_run_stats()
        
        assert isinstance(stats, dict)
        assert "total_runs" in stats
        assert "total_time" in stats
        assert "average_score" in stats
        assert "categories" in stats

    def test_get_category_breakdown(self):
        """测试获取类别分解"""
        system = BenchmarkSystem()
        
        results = [
            BenchmarkResult(
                task_id="cg1",
                task_name="CG1",
                category=BenchmarkCategory.CODE_GENERATION,
                score=0.9,
                confidence=0.95,
                duration=1.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                metrics={},
            ),
            BenchmarkResult(
                task_id="cg2",
                task_name="CG2",
                category=BenchmarkCategory.CODE_GENERATION,
                score=0.8,
                confidence=0.85,
                duration=1.5,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                metrics={},
            ),
            BenchmarkResult(
                task_id="ca1",
                task_name="CA1",
                category=BenchmarkCategory.CODE_ANALYSIS,
                score=0.7,
                confidence=0.75,
                duration=2.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                metrics={},
            ),
        ]
        
        breakdown = system.get_category_breakdown(results)
        
        assert BenchmarkCategory.CODE_GENERATION in breakdown
        assert BenchmarkCategory.CODE_ANALYSIS in breakdown
        
        cg_data = breakdown[BenchmarkCategory.CODE_GENERATION]
        assert cg_data["count"] == 2
        assert cg_data["average_score"] == pytest.approx(0.85)
        assert cg_data["average_duration"] == pytest.approx(1.25)

    def test_get_category_breakdown_empty(self):
        """测试空结果的类别分解"""
        system = BenchmarkSystem()
        
        breakdown = system.get_category_breakdown([])
        
        assert breakdown == {}

    def test_compare_with_history_no_history(self):
        """测试无历史数据时的对比"""
        system = BenchmarkSystem()
        
        results = [
            BenchmarkResult(
                task_id="hist_test_001",
                task_name="历史测试",
                category=BenchmarkCategory.CODE_GENERATION,
                score=0.8,
                confidence=0.9,
                duration=1.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                metrics={},
            ),
        ]
        
        comparisons = system.compare_with_history(results)
        
        assert isinstance(comparisons, list)
        assert len(comparisons) == 0

    def test_benchmark_executor(self):
        """测试基准执行器"""
        system = BenchmarkSystem()
        
        task = BenchmarkTask(
            task_id="exec_test_001",
            name="执行器测试",
            category=BenchmarkCategory.KNOWLEDGE_EXTRACTION,
            description="测试执行器",
        )
        
        result = system._benchmark_executor(task)
        
        assert isinstance(result, BenchmarkResult)
        assert result.task_id == "exec_test_001"
        assert result.success == True
        assert 0.0 <= result.score <= 1.0

    def test_executor_generates_metrics(self):
        """测试执行器生成指标"""
        system = BenchmarkSystem()
        
        task = BenchmarkTask(
            task_id="metrics_test_001",
            name="指标测试",
            category=BenchmarkCategory.COMMUNICATION,
            description="测试指标生成",
            input_data={"test": "value"},
        )
        
        result = system._benchmark_executor(task)
        
        assert "tasks_completed" in result.metrics
        assert "errors_encountered" in result.metrics
        assert "average_latency_ms" in result.metrics

    def test_full_benchmark_run(self):
        """测试完整基准运行"""
        system = BenchmarkSystem()
        
        results = system.run_all_benchmarks()
        
        assert len(results) > 0
        
        for result in results:
            assert result.success == True
            assert result.duration >= 0
            assert 0 <= result.score <= 1

    def test_run_stats_updated_after_run(self):
        """测试运行后统计信息更新"""
        system = BenchmarkSystem()
        
        stats_before = system.get_run_stats()
        runs_before = stats_before["total_runs"]
        
        results = system.run_all_benchmarks()
        
        stats_after = system.get_run_stats()
        runs_after = stats_after["total_runs"]
        
        assert runs_after > runs_before
        assert stats_after["total_time"] > stats_before["total_time"]

    def test_category_scores_in_stats(self):
        """测试统计中的类别分数"""
        system = BenchmarkSystem()
        
        system.run_all_benchmarks()
        
        stats = system.get_run_stats()
        
        assert "categories" in stats
        assert isinstance(stats["categories"], dict)


class TestBenchmarkSystemIntegration:
    """测试基准系统集成场景"""

    def test_standard_tasks_run(self):
        """测试标准任务运行"""
        system = BenchmarkSystem()
        
        standard_tasks = StandardBenchmarks.get_all_tasks()
        
        for task in standard_tasks:
            result = system.run_task(task)
            
            assert result.task_id == task.task_id
            assert result.task_name == task.name
            assert result.category == task.category

    def test_multiple_runs_compare(self):
        """测试多次运行对比"""
        system = BenchmarkSystem()
        
        results1 = system.run_all_benchmarks()
        stats1 = system.get_run_stats()
        
        results2 = system.run_all_benchmarks()
        stats2 = system.get_run_stats()
        
        assert stats2["total_runs"] > stats1["total_runs"]

    def test_result_serializability(self):
        """测试结果可序列化"""
        import json
        
        result = BenchmarkResult(
            task_id="serial_001",
            task_name="序列化测试",
            category=BenchmarkCategory.CODE_GENERATION,
            score=0.85,
            confidence=0.90,
            duration=2.5,
            start_time=datetime.now(),
            end_time=datetime.now(),
            success=True,
            metrics={"accuracy": 0.88, "speed": "fast"},
            error=None,
        )
        
        data = {
            "task_id": result.task_id,
            "task_name": result.task_name,
            "score": result.score,
            "confidence": result.confidence,
            "duration": result.duration,
            "success": result.success,
            "metrics": result.metrics,
        }
        
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_task_validation(self):
        """测试任务验证"""
        task = BenchmarkTask(
            task_id="valid_001",
            name="有效任务",
            category=BenchmarkCategory.PATTERN_RECOGNITION,
            description="测试",
        )
        
        assert task.task_id == "valid_001"
        assert task.name == "有效任务"
        assert task.category == BenchmarkCategory.PATTERN_RECOGNITION
        assert task.difficulty >= 1
        assert task.timeout > 0
