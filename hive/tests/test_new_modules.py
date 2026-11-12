# -*- coding: utf-8 -*-
"""测试新模块功能

验证Phase 1 Week 2创建的所有新模块。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import time


def test_logger():
    """测试日志系统"""
    print("\n=== Testing Logger ===")
    from hive.utils.logger import get_logger, setup_logging

    # 配置日志
    setup_logging(log_level="INFO", json_output=False)

    logger = get_logger(__name__)
    logger.info("Logger test started")
    logger.debug("Debug message", test_data="value")
    logger.warning("Warning message")
    logger.error("Error message", error_code=500)

    print("✓ Logger test passed")


def test_retry():
    """测试重试机制"""
    print("\n=== Testing Retry ===")
    from hive.utils.retry import RetryConfig, retry

    call_count = 0

    @retry(max_attempts=3, base_delay=0.1)
    def unstable_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Not yet")
        return "Success"

    result = unstable_function()
    assert result == "Success"
    assert call_count == 3

    print(f"✓ Retry test passed (called {call_count} times)")


def test_monitor():
    """测试资源监控"""
    print("\n=== Testing Resource Monitor ===")
    from hive.utils.monitor import ResourceMonitor, get_monitor

    monitor = get_monitor(interval=1.0)

    # 获取快照
    snapshot = monitor.get_snapshot()
    print(f"  CPU: {snapshot.cpu_percent}%")
    print(f"  Memory: {snapshot.memory_percent}%")
    print(f"  Disk: {snapshot.disk_percent}%")

    # 检查健康状态
    is_healthy, _, issues = monitor.check_health()
    print(f"  Healthy: {is_healthy}")
    if issues:
        print(f"  Issues: {issues}")

    # 获取系统信息
    sys_info = monitor.get_system_info()
    print(f"  Platform: {sys_info['platform']}")

    print("✓ Resource monitor test passed")


def test_metrics():
    """测试指标收集"""
    print("\n=== Testing Metrics ===")
    from hive.utils.metrics import MetricsCollector, Timer, get_metrics

    collector = get_metrics()

    # 测试计数器
    collector.increment("test_counter", value=1)
    collector.increment("test_counter", value=2)
    assert collector.get_counter("test_counter") == 3

    # 测试仪表盘
    collector.set_gauge("test_gauge", 42.5)
    assert collector.get_gauge("test_gauge") == 42.5

    # 测试直方图
    collector.observe("test_histogram", 1.5)
    collector.observe("test_histogram", 2.5)
    collector.observe("test_histogram", 3.5)
    stats = collector.get_histogram_stats("test_histogram")
    assert stats.count == 3
    assert stats.min == 1.5
    assert stats.max == 3.5

    # 测试计时器
    with Timer(collector, "test_timer"):
        time.sleep(0.1)

    timer_stats = collector.get_histogram_stats("test_timer")
    assert timer_stats is not None

    # 获取所有指标
    all_metrics = collector.get_all_metrics()
    print(f"  Counters: {len(all_metrics['counters'])}")
    print(f"  Gauges: {len(all_metrics['gauges'])}")
    print(f"  Histograms: {len(all_metrics['histograms'])}")

    print("✓ Metrics test passed")


def test_dna():
    """测试DNA结构"""
    print("\n=== Testing DNA ===")
    from hive.core.dna import DNA, BeeDNA, DNAPool

    # 测试基础DNA
    dna = DNA(
        species="test_species",
        mining_capability=0.8,
        learning_rate=0.7,
    )
    assert dna.validate() == True

    # 测试突变
    mutated = dna.mutate(mutation_rate=1.0)
    assert mutated.generation == 2

    # 测试交叉
    other = DNA(species="test_species", generation=1)
    child = dna.crossover(other)
    assert child.generation == 2

    # 测试BeeDNA
    bee_dna = BeeDNA(
        species="worker_bee",
        id="test_bee_001",
        role="worker",
    )
    bee_dna.update_performance(success=True, completion_time=1.0, error=False)
    fitness = bee_dna.calculate_fitness()
    assert fitness > 0

    # 测试DNA池
    pool = DNAPool()
    pool.add(dna)
    pool.add(mutated)
    assert pool.get_pool_size() == 2

    best = pool.get_best("test_species", count=1)
    assert len(best) > 0

    print("✓ DNA test passed")


def test_interfaces():
    """测试接口"""
    print("\n=== Testing Interfaces ===")
    from hive.core.interfaces import Task, TaskResult

    # 测试任务
    task = Task(
        id="task_001",
        type="test",
        priority=1,
        payload={"key": "value"},
    )

    # 测试任务结果
    result = TaskResult(
        task_id=task.id,
        success=True,
        result="completed",
        execution_time=0.5,
    )

    assert result.task_id == task.id
    assert result.success == True

    print("✓ Interfaces test passed")


def test_settings():
    """测试配置"""
    print("\n=== Testing Settings ===")
    from hive.config.settings import HiveSettings, settings

    # 测试默认值
    assert settings.ENVIRONMENT in ["development", "production", "testing"]
    assert settings.MAX_WORKERS >= 1

    # 测试辅助方法
    base_path = settings.get_path("BASE_DIR")
    assert base_path.exists() or base_path.parent.exists()

    print(f"  Environment: {settings.ENVIRONMENT}")
    print(f"  Max Workers: {settings.MAX_WORKERS}")
    print(f"  Log Level: {settings.LOG_LEVEL}")

    print("✓ Settings test passed")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("Running Phase 1 Week 2 Module Tests")
    print("=" * 50)

    try:
        test_settings()
        test_logger()
        test_retry()
        test_monitor()
        test_metrics()
        test_dna()
        test_interfaces()

        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
