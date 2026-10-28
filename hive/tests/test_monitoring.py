# -*- coding: utf-8 -*-
"""测试监控和指标功能"""

import pytest
import time

from hive.utils.monitor import ResourceMonitor, ResourceSnapshot, get_monitor
from hive.utils.metrics import (
    MetricsCollector,
    MetricPoint,
    MetricStats,
    Timer,
    get_metrics,
)


class TestResourceSnapshot:
    """测试资源快照"""

    def test_snapshot_creation(self):
        """测试快照创建"""
        snapshot = ResourceSnapshot(
            cpu_percent=50.0,
            cpu_count=8,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_total_gb=16.0,
            disk_percent=40.0,
            disk_used_gb=200.0,
            disk_total_gb=500.0,
            network_sent_mb=100.0,
            network_recv_mb=500.0,
            process_count=150,
            timestamp=time.time(),
        )
        assert snapshot.cpu_percent == 50.0
        assert snapshot.cpu_count == 8

    def test_is_healthy_all_good(self):
        """测试所有指标健康"""
        snapshot = ResourceSnapshot(
            cpu_percent=50.0,
            cpu_count=8,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_total_gb=16.0,
            disk_percent=40.0,
            disk_used_gb=200.0,
            disk_total_gb=500.0,
            network_sent_mb=100.0,
            network_recv_mb=500.0,
            process_count=150,
            timestamp=time.time(),
        )
        is_healthy, issues = snapshot.is_healthy()
        assert is_healthy is True
        assert len(issues) == 0

    def test_is_healthy_high_cpu(self):
        """测试 CPU 使用率过高"""
        snapshot = ResourceSnapshot(
            cpu_percent=95.0,
            cpu_count=8,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_total_gb=16.0,
            disk_percent=40.0,
            disk_used_gb=200.0,
            disk_total_gb=500.0,
            network_sent_mb=100.0,
            network_recv_mb=500.0,
            process_count=150,
            timestamp=time.time(),
        )
        is_healthy, issues = snapshot.is_healthy(cpu_threshold=90.0)
        assert is_healthy is False
        assert len(issues) > 0
        assert "CPU" in issues[0]


class TestResourceMonitor:
    """测试资源监控器"""

    def test_monitor_creation(self):
        """测试监控器创建"""
        monitor = ResourceMonitor(interval=5.0)
        assert monitor.interval == 5.0

    def test_get_snapshot(self):
        """测试获取快照"""
        monitor = ResourceMonitor()
        snapshot = monitor.get_snapshot()

        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.cpu_count >= 0
        assert snapshot.memory_percent >= 0
        assert snapshot.timestamp > 0

    def test_check_health(self):
        """测试健康检查"""
        monitor = ResourceMonitor()
        is_healthy, snapshot, issues = monitor.check_health()

        assert isinstance(is_healthy, bool)
        assert isinstance(snapshot, ResourceSnapshot)
        assert isinstance(issues, list)

    def test_get_system_info(self):
        """测试获取系统信息"""
        monitor = ResourceMonitor()
        info = monitor.get_system_info()

        assert isinstance(info, dict)
        assert "platform" in info
        assert "python_version" in info
        assert "hostname" in info

    def test_log_snapshot(self, caplog):
        """测试记录快照"""
        monitor = ResourceMonitor()
        snapshot = monitor.get_snapshot()

        # 不应抛出异常
        monitor.log_snapshot(snapshot)

    def test_get_monitor_singleton(self):
        """测试监控器单例"""
        monitor1 = get_monitor()
        monitor2 = get_monitor()

        assert monitor1 is monitor2


class TestMetricPoint:
    """测试指标点"""

    def test_point_creation(self):
        """测试指标点创建"""
        point = MetricPoint(
            value=42.0,
            timestamp=time.time(),
            labels={"status": "success"},
        )
        assert point.value == 42.0
        assert point.labels == {"status": "success"}


class TestMetricStats:
    """测试指标统计"""

    def test_stats_creation(self):
        """测试统计创建"""
        stats = MetricStats()
        assert stats.count == 0
        assert stats.sum == 0.0
        assert stats.min == float("inf")
        assert stats.max == float("-inf")

    def test_stats_update(self):
        """测试统计更新"""
        stats = MetricStats()
        stats.update(10.0)
        stats.update(20.0)
        stats.update(30.0)

        assert stats.count == 3
        assert stats.sum == 60.0
        assert stats.min == 10.0
        assert stats.max == 30.0
        assert stats.avg == 20.0


class TestMetricsCollector:
    """测试指标收集器"""

    def test_collector_creation(self):
        """测试收集器创建"""
        collector = MetricsCollector()
        assert collector._start_time > 0

    def test_increment_counter(self):
        """测试增加计数器"""
        collector = MetricsCollector()
        collector.increment("test_counter", 5.0)

        value = collector.get_counter("test_counter")
        assert value == 5.0

    def test_increment_with_labels(self):
        """测试带标签的计数器"""
        collector = MetricsCollector()
        collector.increment("requests", 1.0, labels={"endpoint": "/api"})

        value = collector.get_counter("requests", labels={"endpoint": "/api"})
        assert value == 1.0

    def test_set_gauge(self):
        """测试设置仪表盘"""
        collector = MetricsCollector()
        collector.set_gauge("temperature", 36.5)

        value = collector.get_gauge("temperature")
        assert value == 36.5

    def test_observe_histogram(self):
        """测试观察直方图"""
        collector = MetricsCollector()
        collector.observe("response_time", 100.0)
        collector.observe("response_time", 200.0)
        collector.observe("response_time", 150.0)

        stats = collector.get_histogram_stats("response_time")
        assert stats is not None
        assert stats.count == 3
        assert stats.min == 100.0
        assert stats.max == 200.0

    def test_get_all_metrics(self):
        """测试获取所有指标"""
        collector = MetricsCollector()
        collector.increment("counter1", 1.0)
        collector.set_gauge("gauge1", 42.0)
        collector.observe("histogram1", 10.0)

        metrics = collector.get_all_metrics()

        assert "counters" in metrics
        assert "gauges" in metrics
        assert "histograms" in metrics
        assert "uptime_seconds" in metrics
        assert metrics["counters"]["counter1"] == 1.0

    def test_reset_metrics(self):
        """测试重置指标"""
        collector = MetricsCollector()
        collector.increment("test", 10.0)

        collector.reset()

        value = collector.get_counter("test")
        assert value == 0.0

    def test_log_summary(self, caplog):
        """测试记录摘要"""
        collector = MetricsCollector()
        collector.increment("test", 1.0)

        # 不应抛出异常
        collector.log_summary()

    def test_get_metrics_singleton(self):
        """测试收集器单例"""
        collector1 = get_metrics()
        collector2 = get_metrics()

        assert collector1 is collector2


class TestTimer:
    """测试计时器"""

    def test_timer_context_manager(self):
        """测试计时器上下文管理器"""
        collector = MetricsCollector()
        timer = Timer(collector, "test_timer")

        with timer:
            time.sleep(0.01)

        stats = collector.get_histogram_stats("test_timer")
        assert stats is not None
        assert stats.count == 1
        assert stats.sum >= 0.01

    def test_timer_with_labels(self):
        """测试带标签的计时器"""
        collector = MetricsCollector()
        timer = Timer(collector, "test_timer", labels={"operation": "test"})

        with timer:
            time.sleep(0.01)

        stats = collector.get_histogram_stats("test_timer", labels={"operation": "test"})
        assert stats is not None
        assert stats.count == 1

    def test_timer_exception_handling(self):
        """测试计时器异常处理"""
        collector = MetricsCollector()
        timer = Timer(collector, "test_timer")

        try:
            with timer:
                raise ValueError("Test exception")
        except ValueError:
            pass

        # 即使发生异常，也应该记录时间
        stats = collector.get_histogram_stats("test_timer")
        assert stats is not None
