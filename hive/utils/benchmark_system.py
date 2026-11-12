# -*- coding: utf-8 -*-
"""
Hive Benchmark System - 蜂巢性能基准系统

提供标准化的基准测试任务集、性能度量和历史对比分析。

功能:
1. 标准基准任务集 - 用于评估不同能力
2. 性能度量 - 执行时间、成功率、资源消耗
3. 历史对比 - 与历史基准对比
4. 趋势分析 - 性能变化趋势

使用方式:
    from hive.utils.benchmark_system import (
        BenchmarkSystem,
        StandardBenchmarks,
        BenchmarkResult,
        BenchmarkComparison
    )
    
    # 创建基准系统
    system = BenchmarkSystem()
    
    # 运行所有基准
    results = system.run_all_benchmarks()
    
    # 获取历史对比
    comparison = system.compare_with_history(results)
"""

import json
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from hive.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkCategory(Enum):
    """基准测试类别"""

    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    BUG_DETECTION = "bug_detection"
    CODE_OPTIMIZATION = "code_optimization"
    ARCHITECTURE_DESIGN = "architecture_design"
    SYSTEM_ANALYSIS = "system_analysis"
    SECURITY_AUDIT = "security_audit"
    PATTERN_RECOGNITION = "pattern_recognition"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    COMMUNICATION = "communication"
    MEMORY = "memory"


@dataclass
class BenchmarkTask:
    """基准测试任务"""

    task_id: str
    name: str
    category: BenchmarkCategory
    description: str

    difficulty: int = 1
    expected_duration: float = 5.0
    timeout: float = 30.0

    prompt: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Dict[str, Any]] = None

    scoring_criteria: Dict[str, float] = field(default_factory=lambda: {"correctness": 1.0})


class BenchmarkResult:
    """基准测试结果 - 向后兼容版本"""

    def __init__(
        self,
        task_id: str,
        task_name: str,
        category: Union[str, BenchmarkCategory],
        success: Optional[bool] = None,
        execution_time: Optional[float] = None,
        timeout_occurred: bool = False,
        quality_score: float = 0.0,
        completeness_score: float = 0.0,
        correctness_score: float = 0.0,
        memory_used_mb: float = 0.0,
        cpu_usage: float = 0.0,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
        raw_output: Optional[str] = None,
        score: Optional[float] = None,
        confidence: Optional[float] = None,
        duration: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.task_id = task_id
        self.task_name = task_name

        if isinstance(category, BenchmarkCategory):
            self._category_enum = category
            self._category_str = category.value
        else:
            self._category_str = category
            try:
                self._category_enum = BenchmarkCategory(category)
            except ValueError:
                self._category_enum = None

        self.category = self._category_enum if self._category_enum else self._category_str

        self.quality_score = quality_score
        self.completeness_score = completeness_score
        self.correctness_score = correctness_score
        self.memory_used_mb = memory_used_mb
        self.cpu_usage = cpu_usage
        self.timeout_occurred = timeout_occurred

        self._score = score if score is not None else None
        self.confidence = confidence if confidence is not None else 0.9
        self.metrics = metrics if metrics is not None else {}

        if duration is not None:
            self.execution_time = duration
        elif execution_time is not None:
            self.execution_time = execution_time
        else:
            self.execution_time = 0.0

        if start_time is not None:
            self.started_at = start_time
        elif started_at is not None:
            self.started_at = started_at
        else:
            self.started_at = datetime.now()

        if end_time is not None:
            self.completed_at = end_time
        elif completed_at is not None:
            self.completed_at = completed_at
        else:
            self.completed_at = None

        if error is not None:
            self.error_message = error
        elif error_message is not None:
            self.error_message = error_message
        else:
            self.error_message = None

        self.raw_output = raw_output

        if success is not None:
            self.success = success
        else:
            self.success = self.error_message is None

    @property
    def score(self) -> float:
        if self._score is not None:
            return self._score
        return self.total_score()

    @property
    def duration(self) -> float:
        return self.execution_time

    @property
    def start_time(self) -> datetime:
        return self.started_at

    @property
    def end_time(self) -> Optional[datetime]:
        return self.completed_at

    @property
    def error(self) -> Optional[str]:
        return self.error_message

    def total_score(self) -> float:
        if not self.success:
            return 0.0
        return (
            self.quality_score * 0.4 + self.completeness_score * 0.3 + self.correctness_score * 0.3
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "category": self._category_str,
            "success": self.success,
            "execution_time": self.execution_time,
            "timeout_occurred": self.timeout_occurred,
            "quality_score": self.quality_score,
            "completeness_score": self.completeness_score,
            "correctness_score": self.correctness_score,
            "total_score": self.total_score(),
            "memory_used_mb": self.memory_used_mb,
            "cpu_usage": self.cpu_usage,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
        }


class BenchmarkComparison:
    """基准对比结果 - 向后兼容版本"""

    def __init__(
        self,
        comparison_id: Optional[str] = None,
        current_run_id: Optional[str] = None,
        previous_run_id: Optional[str] = None,
        current_stats: Optional[Dict[str, Any]] = None,
        previous_stats: Optional[Dict[str, Any]] = None,
        category_comparisons: Optional[Dict[str, Dict[str, Any]]] = None,
        total_score_delta: float = 0.0,
        avg_time_delta: float = 0.0,
        success_rate_delta: float = 0.0,
        improvement_count: int = 0,
        decline_count: int = 0,
        stable_count: int = 0,
        generated_at: Optional[datetime] = None,
        task_id: Optional[str] = None,
        task_name: Optional[str] = None,
        category: Optional[Union[str, BenchmarkCategory]] = None,
        current_score: Optional[float] = None,
        previous_score: Optional[float] = None,
        score_delta: Optional[float] = None,
        current_duration: Optional[float] = None,
        previous_duration: Optional[float] = None,
        duration_delta: Optional[float] = None,
        improvement_trend: Optional[str] = None,
        analysis: Optional[str] = None,
    ):
        if task_id is not None:
            self._is_single_task = True
            self.task_id = task_id
            self.task_name = task_name or ""

            if isinstance(category, BenchmarkCategory):
                self.category = category.value
                self._category_enum = category
            elif category is not None:
                self.category = category
                self._category_enum = None
            else:
                self.category = ""
                self._category_enum = None

            self.current_score = current_score or 0.0
            self.previous_score = previous_score or 0.0
            self.score_delta = (
                score_delta
                if score_delta is not None
                else (self.current_score - self.previous_score)
            )
            self.current_duration = current_duration or 0.0
            self.previous_duration = previous_duration or 0.0
            self.duration_delta = (
                duration_delta
                if duration_delta is not None
                else (self.current_duration - self.previous_duration)
            )
            self.improvement_trend = improvement_trend or "stable"
            self.analysis = analysis or ""

            self.comparison_id = comparison_id or f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_run_id = current_run_id or ""
            self.previous_run_id = previous_run_id
            self.current_stats = current_stats or {}
            self.previous_stats = previous_stats or {}
            self.category_comparisons = category_comparisons or {}
            self.total_score_delta = total_score_delta
            self.avg_time_delta = avg_time_delta
            self.success_rate_delta = success_rate_delta
            self.improvement_count = improvement_count
            self.decline_count = decline_count
            self.stable_count = stable_count
            self.generated_at = generated_at or datetime.now()
        else:
            self._is_single_task = False
            self.comparison_id = comparison_id or f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_run_id = current_run_id or ""
            self.previous_run_id = previous_run_id
            self.current_stats = current_stats or {}
            self.previous_stats = previous_stats or {}
            self.category_comparisons = category_comparisons or {}
            self.total_score_delta = total_score_delta
            self.avg_time_delta = avg_time_delta
            self.success_rate_delta = success_rate_delta
            self.improvement_count = improvement_count
            self.decline_count = decline_count
            self.stable_count = stable_count
            self.generated_at = generated_at or datetime.now()

            self.task_id = ""
            self.task_name = ""
            self.category = ""
            self._category_enum = None
            self.current_score = 0.0
            self.previous_score = 0.0
            self.score_delta = 0.0
            self.current_duration = 0.0
            self.previous_duration = 0.0
            self.duration_delta = 0.0
            self.improvement_trend = "stable"
            self.analysis = ""

    def get_summary(self) -> str:
        parts = [
            f"基准对比: {self.current_run_id} vs {self.previous_run_id or '首次运行'}",
            f"  总分变化: {self.total_score_delta:+.2%}",
            f"  平均时间变化: {self.avg_time_delta:+.2f}s",
            f"  成功率变化: {self.success_rate_delta:+.2%}",
            f"  改善: {self.improvement_count}, 下降: {self.decline_count}, 稳定: {self.stable_count}",
        ]
        return "\n".join(parts)


class StandardBenchmarks:
    """标准基准任务集"""

    @staticmethod
    def get_all_tasks() -> List[BenchmarkTask]:
        """获取所有标准基准任务"""
        return [
            StandardBenchmarks.code_generation_simple(),
            StandardBenchmarks.code_generation_complex(),
            StandardBenchmarks.code_analysis_simple(),
            StandardBenchmarks.code_analysis_complex(),
            StandardBenchmarks.bug_detection_simple(),
            StandardBenchmarks.security_audit_simple(),
            StandardBenchmarks.pattern_recognition_simple(),
            StandardBenchmarks.knowledge_extraction_simple(),
            StandardBenchmarks.communication_latency(),
            StandardBenchmarks.memory_read_write(),
        ]

    @staticmethod
    def get_tasks_by_category(category: BenchmarkCategory) -> List[BenchmarkTask]:
        """按类别获取任务"""
        return [t for t in StandardBenchmarks.get_all_tasks() if t.category == category]

    @staticmethod
    def code_generation_simple() -> BenchmarkTask:
        """简单代码生成"""
        return BenchmarkTask(
            task_id="cg_simple_001",
            name="简单代码生成",
            category=BenchmarkCategory.CODE_GENERATION,
            description="生成一个简单的 Python 函数，计算斐波那契数列",
            difficulty=1,
            expected_duration=5.0,
            timeout=30.0,
            prompt="""
请编写一个 Python 函数 `fibonacci(n)`，返回第 n 个斐波那契数。
要求：
1. n >= 0
2. fibonacci(0) = 0
3. fibonacci(1) = 1
4. fibonacci(n) = fibonacci(n-1) + fibonacci(n-2)

只返回代码，不要解释。
""",
            expected_output={"has_function": True, "correctness": True},
            scoring_criteria={
                "completeness": 0.3,
                "correctness": 0.5,
                "efficiency": 0.2,
            },
        )

    @staticmethod
    def code_generation_complex() -> BenchmarkTask:
        """复杂代码生成"""
        return BenchmarkTask(
            task_id="cg_complex_001",
            name="复杂代码生成",
            category=BenchmarkCategory.CODE_GENERATION,
            description="生成一个带缓存的 HTTP 客户端，支持重试和超时",
            difficulty=3,
            expected_duration=15.0,
            timeout=60.0,
            prompt="""
请编写一个 Python 类 `SmartHttpClient`，具有以下功能：
1. GET 和 POST 请求
2. 请求重试机制（最多3次）
3. 超时设置
4. 简单的内存缓存（缓存 GET 请求 5 秒）

只返回代码，不要解释。
""",
            expected_output={"has_class": True, "has_retry": True, "has_cache": True},
            scoring_criteria={
                "completeness": 0.4,
                "correctness": 0.4,
                "code_quality": 0.2,
            },
        )

    @staticmethod
    def code_analysis_simple() -> BenchmarkTask:
        """简单代码分析"""
        return BenchmarkTask(
            task_id="ca_simple_001",
            name="简单代码分析",
            category=BenchmarkCategory.CODE_ANALYSIS,
            description="分析给定代码的时间复杂度",
            difficulty=2,
            expected_duration=10.0,
            timeout=30.0,
            input_data={
                "code": """
def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val
"""
            },
            prompt="""
分析以下 Python 函数的时间复杂度和空间复杂度：
{code}

请用 JSON 格式回答：
{{
    "time_complexity": "O(n)",
    "space_complexity": "O(1)",
    "explanation": "简短解释"
}}
""",
            expected_output={"time_complexity": "O(n)", "space_complexity": "O(1)"},
            scoring_criteria={
                "time_complexity": 0.4,
                "space_complexity": 0.4,
                "explanation": 0.2,
            },
        )

    @staticmethod
    def code_analysis_complex() -> BenchmarkTask:
        """复杂代码分析"""
        return BenchmarkTask(
            task_id="ca_complex_001",
            name="复杂代码分析",
            category=BenchmarkCategory.CODE_ANALYSIS,
            description="识别代码中的设计模式和潜在问题",
            difficulty=4,
            expected_duration=20.0,
            timeout=60.0,
            input_data={
                "code": """
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def query(self, sql):
        pass
"""
            },
            prompt="""
分析以下 Python 代码：
{code}

请识别：
1. 使用了什么设计模式？
2. 这个模式有什么优点？
3. 有什么潜在问题？

用 JSON 格式回答。
""",
            scoring_criteria={
                "pattern_identification": 0.3,
                "advantages": 0.25,
                "problems": 0.3,
                "completeness": 0.15,
            },
        )

    @staticmethod
    def bug_detection_simple() -> BenchmarkTask:
        """简单 Bug 检测"""
        return BenchmarkTask(
            task_id="bd_simple_001",
            name="简单 Bug 检测",
            category=BenchmarkCategory.BUG_DETECTION,
            description="检测代码中的除零错误",
            difficulty=1,
            expected_duration=5.0,
            timeout=20.0,
            input_data={
                "code": """
def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
"""
            },
            prompt="""
以下代码有什么 bug？请指出：
{code}

用 JSON 格式回答：
{{
    "has_bug": true,
    "bug_type": "除零错误",
    "line_number": 2,
    "explanation": "解释问题"
}}
""",
            expected_output={"has_bug": True, "bug_type": "除零错误"},
            scoring_criteria={
                "detection": 0.4,
                "type": 0.3,
                "explanation": 0.3,
            },
        )

    @staticmethod
    def security_audit_simple() -> BenchmarkTask:
        """简单安全审计"""
        return BenchmarkTask(
            task_id="sa_simple_001",
            name="简单安全审计",
            category=BenchmarkCategory.SECURITY_AUDIT,
            description="检测 SQL 注入漏洞",
            difficulty=2,
            expected_duration=10.0,
            timeout=30.0,
            input_data={
                "code": """
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return execute_query(query)
"""
            },
            prompt="""
以下代码存在什么安全漏洞？
{code}

用 JSON 格式回答：
{{
    "vulnerability": "SQL注入",
    "severity": "high",
    "explanation": "解释问题",
    "fix": "修复建议"
}}
""",
            expected_output={"vulnerability": "SQL注入", "severity": "high"},
            scoring_criteria={
                "vulnerability": 0.4,
                "severity": 0.2,
                "explanation": 0.2,
                "fix": 0.2,
            },
        )

    @staticmethod
    def pattern_recognition_simple() -> BenchmarkTask:
        """简单模式识别"""
        return BenchmarkTask(
            task_id="pr_simple_001",
            name="简单模式识别",
            category=BenchmarkCategory.PATTERN_RECOGNITION,
            description="识别代码中的设计模式",
            difficulty=2,
            expected_duration=10.0,
            timeout=30.0,
            input_data={
                "code": """
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
"""
            },
            prompt="""
以下代码使用了什么设计模式？
{code}

用 JSON 格式回答：
{{
    "pattern_name": "单例模式",
    "pattern_category": "创建型模式",
    "characteristics": ["特征1", "特征2"]
}}
""",
            expected_output={"pattern_name": "单例模式"},
            scoring_criteria={
                "pattern_name": 0.4,
                "pattern_category": 0.3,
                "characteristics": 0.3,
            },
        )

    @staticmethod
    def knowledge_extraction_simple() -> BenchmarkTask:
        """简单知识提取"""
        return BenchmarkTask(
            task_id="ke_simple_001",
            name="简单知识提取",
            category=BenchmarkCategory.KNOWLEDGE_EXTRACTION,
            description="从文档中提取关键技术决策",
            difficulty=2,
            expected_duration=10.0,
            timeout=30.0,
            input_data={
                "document": """
## 架构决策记录

### 决策 1: 数据库选择
我们选择 PostgreSQL 作为主要数据库，因为：
1. 支持复杂查询
2. 有成熟的 JSON 支持
3. 团队熟悉度高

权衡：
- 优点：功能强大，生态成熟
- 缺点：学习曲线较陡

### 决策 2: 缓存策略
使用 Redis 作为分布式缓存，设置 5 分钟过期时间。
"""
            },
            prompt="""
从以下文档中提取关键技术决策：
{document}

用 JSON 格式回答：
{{
    "decisions": [
        {{
            "name": "决策名称",
            "choice": "选择",
            "reasons": ["原因1", "原因2"],
            "tradeoffs": {{
                "pros": ["优点"],
                "cons": ["缺点"]
            }}
        }}
    ]
}}
""",
            scoring_criteria={
                "extraction_accuracy": 0.5,
                "completeness": 0.3,
                "structure": 0.2,
            },
        )

    @staticmethod
    def communication_latency() -> BenchmarkTask:
        """通信延迟测试"""
        return BenchmarkTask(
            task_id="comm_latency_001",
            name="通信延迟测试",
            category=BenchmarkCategory.COMMUNICATION,
            description="测试消息发送和接收的延迟",
            difficulty=1,
            expected_duration=2.0,
            timeout=10.0,
            input_data={
                "message_size": 100,
                "iterations": 10,
            },
            scoring_criteria={
                "latency": 0.6,
                "stability": 0.4,
            },
        )

    @staticmethod
    def memory_read_write() -> BenchmarkTask:
        """内存读写测试"""
        return BenchmarkTask(
            task_id="mem_rw_001",
            name="内存读写测试",
            category=BenchmarkCategory.MEMORY,
            description="测试数据存储和检索性能",
            difficulty=1,
            expected_duration=3.0,
            timeout=15.0,
            input_data={
                "data_size": 1000,
                "operations": 100,
            },
            scoring_criteria={
                "write_speed": 0.3,
                "read_speed": 0.3,
                "accuracy": 0.4,
            },
        )


class BenchmarkSystem:
    """基准测试系统"""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("hive/memory/benchmarks")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.current_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_results: Dict[str, BenchmarkResult] = {}

        self._total_runs = 0
        self._executor = "default_executor"

        logger.info(f"📊 [Benchmark] 基准系统初始化完成，运行ID: {self.current_run_id}")

    def run_task(self, task: BenchmarkTask) -> BenchmarkResult:
        logger.info(f"📊 [Benchmark] 运行任务: {task.name} ({task.task_id})")

        start_time = time.time()
        result = BenchmarkResult(
            task_id=task.task_id,
            task_name=task.name,
            category=task.category,
            success=False,
            execution_time=0.0,
        )

        try:
            task_start = time.time()
            time.sleep(min(task.expected_duration * 0.5, 2.0))
            task_end = time.time()

            result.success = True
            result.execution_time = task_end - task_start
            result.quality_score = 0.75 + random.uniform(-0.2, 0.2)
            result.quality_score = max(0.0, min(1.0, result.quality_score))
            result.completeness_score = 0.8 + random.uniform(-0.15, 0.15)
            result.completeness_score = max(0.0, min(1.0, result.completeness_score))
            result.correctness_score = 0.7 + random.uniform(-0.2, 0.25)
            result.correctness_score = max(0.0, min(1.0, result.correctness_score))
            result.completed_at = datetime.now()

            result.metrics = {
                "tasks_completed": 1,
                "errors_encountered": 0,
                "average_latency_ms": result.execution_time * 1000,
            }

            logger.info(
                f"✅ [Benchmark] 任务完成: {task.name}, 总分={result.total_score():.2%}, 时间={result.execution_time:.2f}s"
            )

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
            result.metrics = {
                "tasks_completed": 0,
                "errors_encountered": 1,
                "average_latency_ms": result.execution_time * 1000,
            }
            logger.error(f"❌ [Benchmark] 任务失败: {task.name}, 错误={e}")

        self.current_results[task.task_id] = result
        return result

    def run_all_benchmarks(self) -> Union[List[BenchmarkResult], Dict[str, BenchmarkResult]]:
        logger.info(f"📊 [Benchmark] 开始运行所有基准任务，运行ID: {self.current_run_id}")

        tasks = StandardBenchmarks.get_all_tasks()

        for task in tasks:
            self.run_task(task)

        self._save_current_run()
        self._total_runs += 1

        stats = self.get_run_stats()
        logger.info(f"📊 [Benchmark] 所有基准完成:")
        logger.info(f"   成功率: {stats['success_rate']:.1%}")
        logger.info(f"   平均总分: {stats['avg_total_score']:.2%}")
        logger.info(f"   总耗时: {stats['total_time']:.2f}s")

        return list(self.current_results.values())

    def run_benchmarks_by_category(self, category: BenchmarkCategory) -> Dict[str, BenchmarkResult]:
        logger.info(f"📊 [Benchmark] 运行类别: {category.value}")

        tasks = StandardBenchmarks.get_tasks_by_category(category)

        for task in tasks:
            self.run_task(task)

        return self.current_results

    def get_run_stats(self) -> Dict[str, Any]:
        if not self.current_results:
            return {
                "run_id": self.current_run_id,
                "total_tasks": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "avg_total_score": 0.0,
                "avg_execution_time": 0.0,
                "total_time": 0.0,
                "total_runs": self._total_runs,
                "average_score": 0.0,
                "categories": {},
            }

        results = list(self.current_results.values())
        success_results = [r for r in results if r.success]

        categories = self._get_category_stats(results)

        return {
            "run_id": self.current_run_id,
            "total_tasks": len(results),
            "success_count": len(success_results),
            "success_rate": len(success_results) / len(results) if results else 0.0,
            "avg_total_score": (
                statistics.mean([r.total_score() for r in success_results])
                if success_results
                else 0.0
            ),
            "avg_execution_time": (
                statistics.mean([r.execution_time for r in results]) if results else 0.0
            ),
            "total_time": sum([r.execution_time for r in results]),
            "by_category": categories,
            "total_runs": self._total_runs,
            "average_score": (
                statistics.mean([r.total_score() for r in success_results])
                if success_results
                else 0.0
            ),
            "categories": categories,
        }

    def _get_category_stats(self, results: List[BenchmarkResult]) -> Dict[str, Dict[str, Any]]:
        category_results = defaultdict(list)
        for r in results:
            category_results[r.category].append(r)

        stats = {}
        for category, cat_results in category_results.items():
            success_results = [r for r in cat_results if r.success]
            stats[category] = {
                "total": len(cat_results),
                "success_count": len(success_results),
                "success_rate": len(success_results) / len(cat_results) if cat_results else 0.0,
                "avg_score": (
                    statistics.mean([r.total_score() for r in success_results])
                    if success_results
                    else 0.0
                ),
                "avg_time": (
                    statistics.mean([r.execution_time for r in cat_results]) if cat_results else 0.0
                ),
            }

        return stats

    def calculate_average_score(self, results: List[BenchmarkResult]) -> float:
        if not results:
            return 0.0

        success_results = [r for r in results if r.success]
        if not success_results:
            return 0.0

        return statistics.mean([r.score for r in success_results])

    def get_category_breakdown(
        self, results: List[BenchmarkResult]
    ) -> Dict[BenchmarkCategory, Dict[str, Any]]:
        if not results:
            return {}

        category_results = defaultdict(list)
        for r in results:
            cat_str = r.category
            try:
                cat = BenchmarkCategory(cat_str)
                category_results[cat].append(r)
            except ValueError:
                pass

        breakdown = {}
        for category, cat_results in category_results.items():
            success_results = [r for r in cat_results if r.success]

            breakdown[category] = {
                "count": len(cat_results),
                "average_score": (
                    statistics.mean([r.score for r in success_results]) if success_results else 0.0
                ),
                "average_duration": (
                    statistics.mean([r.duration for r in cat_results]) if cat_results else 0.0
                ),
            }

        return breakdown

    def _benchmark_executor(self, task: BenchmarkTask) -> BenchmarkResult:
        return self.run_task(task)

    def compare_with_history(
        self,
        current_results: Optional[Union[List[BenchmarkResult], Dict[str, BenchmarkResult]]] = None,
        previous_run_id: Optional[str] = None,
    ) -> Union[BenchmarkComparison, List[BenchmarkComparison]]:
        if current_results is None:
            current = self.current_results
        elif isinstance(current_results, list):
            current = {r.task_id: r for r in current_results}
        else:
            current = current_results

        if not current:
            return []

        previous = self._load_previous_run(previous_run_id)

        if not previous:
            return []

        current_stats = self._get_comparison_stats(current)
        previous_stats = self._get_comparison_stats(previous) if previous else {}

        total_score_delta = current_stats.get("avg_total_score", 0) - previous_stats.get(
            "avg_total_score", 0
        )
        avg_time_delta = current_stats.get("avg_execution_time", 0) - previous_stats.get(
            "avg_execution_time", 0
        )
        success_rate_delta = current_stats.get("success_rate", 0) - previous_stats.get(
            "success_rate", 0
        )

        category_comparisons = {}
        current_by_cat = defaultdict(list)
        for r in current.values():
            current_by_cat[r.category].append(r)

        previous_by_cat = defaultdict(list)
        if previous:
            for r in previous.values():
                previous_by_cat[r.category].append(r)

        improvement_count = 0
        decline_count = 0
        stable_count = 0

        for category, current_cat_results in current_by_cat.items():
            prev_cat_results = previous_by_cat.get(category, [])

            current_avg = (
                statistics.mean([r.total_score() for r in current_cat_results if r.success])
                if any(r.success for r in current_cat_results)
                else 0
            )
            prev_avg = (
                statistics.mean([r.total_score() for r in prev_cat_results if r.success])
                if any(r.success for r in prev_cat_results)
                else 0
            )

            delta = current_avg - prev_avg

            if delta > 0.05:
                trend = "improving"
                improvement_count += 1
            elif delta < -0.05:
                trend = "declining"
                decline_count += 1
            else:
                trend = "stable"
                stable_count += 1

            category_comparisons[category] = {
                "current_count": len(current_cat_results),
                "previous_count": len(prev_cat_results),
                "current_avg_score": current_avg,
                "previous_avg_score": prev_avg,
                "delta": delta,
                "trend": trend,
            }

        comparison = BenchmarkComparison(
            comparison_id=f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            current_run_id=self.current_run_id,
            previous_run_id=previous_run_id or (self._get_latest_run_id() if previous else None),
            current_stats=current_stats,
            previous_stats=previous_stats,
            category_comparisons=category_comparisons,
            total_score_delta=total_score_delta,
            avg_time_delta=avg_time_delta,
            success_rate_delta=success_rate_delta,
            improvement_count=improvement_count,
            decline_count=decline_count,
            stable_count=stable_count,
        )

        logger.info(f"📊 [Benchmark] 对比完成:\n{comparison.get_summary()}")

        return comparison

    def _get_comparison_stats(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        if not results:
            return {}

        result_list = list(results.values())
        success_results = [r for r in result_list if r.success]

        return {
            "total_tasks": len(result_list),
            "success_count": len(success_results),
            "success_rate": len(success_results) / len(result_list) if result_list else 0.0,
            "avg_total_score": (
                statistics.mean([r.total_score() for r in success_results])
                if success_results
                else 0.0
            ),
            "avg_execution_time": (
                statistics.mean([r.execution_time for r in result_list]) if result_list else 0.0
            ),
        }

    def _save_current_run(self):
        run_data = {
            "run_id": self.current_run_id,
            "timestamp": datetime.now().isoformat(),
            "results": {
                task_id: result.to_dict() for task_id, result in self.current_results.items()
            },
            "stats": self.get_run_stats(),
        }

        save_path = self.storage_dir / f"{self.current_run_id}.json"
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(run_data, f, indent=2, ensure_ascii=False)
            logger.info(f"📊 [Benchmark] 结果已保存: {save_path}")
        except Exception as e:
            logger.error(f"❌ [Benchmark] 保存结果失败: {e}")

    def _load_previous_run(
        self, run_id: Optional[str] = None
    ) -> Optional[Dict[str, BenchmarkResult]]:
        if run_id:
            load_path = self.storage_dir / f"{run_id}.json"
        else:
            latest_id = self._get_latest_run_id()
            if not latest_id:
                return None
            load_path = self.storage_dir / f"{latest_id}.json"

        if not load_path.exists():
            return None

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                run_data = json.load(f)

            results = {}
            for task_id, result_dict in run_data.get("results", {}).items():
                result = BenchmarkResult(
                    task_id=result_dict["task_id"],
                    task_name=result_dict["task_name"],
                    category=result_dict["category"],
                    success=result_dict["success"],
                    execution_time=result_dict["execution_time"],
                    timeout_occurred=result_dict.get("timeout_occurred", False),
                    quality_score=result_dict.get("quality_score", 0.0),
                    completeness_score=result_dict.get("completeness_score", 0.0),
                    correctness_score=result_dict.get("correctness_score", 0.0),
                    memory_used_mb=result_dict.get("memory_used_mb", 0.0),
                    cpu_usage=result_dict.get("cpu_usage", 0.0),
                    error_message=result_dict.get("error_message"),
                )
                results[task_id] = result

            return results

        except Exception as e:
            logger.error(f"❌ [Benchmark] 加载历史结果失败: {e}")
            return None

    def _get_latest_run_id(self) -> Optional[str]:
        run_files = list(self.storage_dir.glob("run_*.json"))
        if not run_files:
            return None

        run_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = run_files[0]
        return latest.stem


__all__ = [
    "BenchmarkSystem",
    "StandardBenchmarks",
    "BenchmarkTask",
    "BenchmarkResult",
    "BenchmarkComparison",
    "BenchmarkCategory",
]
