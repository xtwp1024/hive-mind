# -*- coding: utf-8 -*-
"""指标收集器

收集和跟踪系统级别的性能指标。
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

from hive.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """单个数据点"""

    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricStats:
    """指标统计信息"""

    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    avg: float = 0.0

    def update(self, value: float) -> None:
        """更新统计信息"""
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.avg = self.sum / self.count


class MetricsCollector:
    """指标收集器

    收集和跟踪各种系统指标，包括计数器、仪表盘、直方图等。
    """

    def __init__(self):
        """初始化指标收集器"""
        self._lock = threading.Lock()

        # 计数器：只增不减的值
        self._counters: dict[str, float] = defaultdict(float)

        # 仪表盘：可以上下波动的值
        self._gauges: dict[str, float] = {}

        # 直方图：记录值的分布（有界，最多保留1000个样本）
        self._histograms: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # 直方图统计信息
        self._histogram_stats: dict[str, MetricStats] = {}

        # 时间戳
        self._start_time = time.time()

    def increment(
        self, name: str, value: float = 1.0, labels: Optional[dict[str, str]] = None
    ) -> None:
        """增加计数器

        Args:
            name: 指标名称
            value: 增加的值（默认1.0）
            labels: 标签字典
        """
        full_name = self._make_full_name(name, labels)
        with self._lock:
            self._counters[full_name] += value

        logger.debug(
            "Counter incremented",
            metric=full_name,
            value=value,
            total=self._counters[full_name],
        )

    def set_gauge(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """设置仪表盘值

        Args:
            name: 指标名称
            value: 当前值
            labels: 标签字典
        """
        full_name = self._make_full_name(name, labels)
        with self._lock:
            self._gauges[full_name] = value

        logger.debug(
            "Gauge set",
            metric=full_name,
            value=value,
        )

    def observe(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """记录直方图观测值

        Args:
            name: 指标名称
            value: 观测值
            labels: 标签字典
        """
        full_name = self._make_full_name(name, labels)
        with self._lock:
            self._histograms[full_name].append(value)
            if full_name not in self._histogram_stats:
                self._histogram_stats[full_name] = MetricStats()
            self._histogram_stats[full_name].update(value)

        logger.debug(
            "Histogram observed",
            metric=full_name,
            value=value,
            count=len(self._histograms[full_name]),
        )

    def get_counter(self, name: str, labels: Optional[dict[str, str]] = None) -> float:
        """获取计数器值

        Args:
            name: 指标名称
            labels: 标签字典

        Returns:
            当前计数器值
        """
        full_name = self._make_full_name(name, labels)
        return self._counters.get(full_name, 0.0)

    def get_gauge(self, name: str, labels: Optional[dict[str, str]] = None) -> Optional[float]:
        """获取仪表盘值

        Args:
            name: 指标名称
            labels: 标签字典

        Returns:
            当前仪表盘值，如果不存在则返回None
        """
        full_name = self._make_full_name(name, labels)
        return self._gauges.get(full_name)

    def get_histogram_stats(
        self, name: str, labels: Optional[dict[str, str]] = None
    ) -> Optional[MetricStats]:
        """获取直方图统计信息

        Args:
            name: 指标名称
            labels: 标签字典

        Returns:
            统计信息对象，如果不存在则返回None
        """
        full_name = self._make_full_name(name, labels)
        return self._histogram_stats.get(full_name)

    def get_all_metrics(self) -> dict[str, Any]:
        """获取所有指标

        Returns:
            包含所有指标的字典
        """
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: {
                    "samples": list(values),
                    "stats": {
                        "count": stats.count,
                        "sum": stats.sum,
                        "min": stats.min,
                        "max": stats.max,
                        "avg": stats.avg,
                    },
                }
                for name, values, stats in [
                    (name, self._histograms[name], self._histogram_stats[name])
                    for name in self._histograms
                    if name in self._histogram_stats
                ]
            },
            "uptime_seconds": time.time() - self._start_time,
        }

    def reset(self) -> None:
        """重置所有指标"""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._histogram_stats.clear()
        self._start_time = time.time()

        logger.info("All metrics reset")

    def log_summary(self) -> None:
        """记录指标摘要到日志"""
        metrics = self.get_all_metrics()

        logger.info(
            "Metrics summary",
            uptime_seconds=f"{metrics['uptime_seconds']:.1f}",
            counters_count=len(metrics["counters"]),
            gauges_count=len(metrics["gauges"]),
            histograms_count=len(metrics["histograms"]),
        )

        if metrics["counters"]:
            logger.debug("Counters", counters=metrics["counters"])

        if metrics["gauges"]:
            logger.debug("Gauges", gauges=metrics["gauges"])

    @staticmethod
    def _make_full_name(name: str, labels: Optional[dict[str, str]]) -> str:
        """创建完整的指标名称（包含标签）"""
        if not labels:
            return name

        # 将标签按字母顺序排列以确保一致性
        label_strs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return f"{name}{{{','.join(label_strs)}}}"


# 上下文管理器用于计时
class Timer:
    """计时器上下文管理器

    用于测量代码块的执行时间并记录到直方图。
    """

    def __init__(
        self, collector: MetricsCollector, name: str, labels: Optional[dict[str, str]] = None
    ):
        """初始化计时器

        Args:
            collector: 指标收集器
            name: 指标名称
            labels: 标签字典
        """
        self.collector = collector
        self.name = name
        self.labels = labels
        self._start_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        """进入上下文"""
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出上下文"""
        if self._start_time is not None:
            duration = time.time() - self._start_time
            self.collector.observe(self.name, duration, self.labels)


# 全局指标收集器实例
_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """获取全局指标收集器实例（线程安全）"""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector


def export_prometheus(collector: Optional[MetricsCollector] = None) -> str:
    """导出Prometheus格式的指标

    Args:
        collector: 指标收集器实例，默认使用全局实例

    Returns:
        Prometheus格式的指标字符串
    """
    if collector is None:
        collector = get_metrics()

    metrics = collector.get_all_metrics()
    lines = []

    for name, value in metrics["counters"].items():
        safe_name = name.replace("{", "_").replace("}", "").replace(",", "_").replace("=", "_")
        lines.append(f"# TYPE {safe_name} counter")
        lines.append(f"{safe_name} {value}")

    for name, value in metrics["gauges"].items():
        safe_name = name.replace("{", "_").replace("}", "").replace(",", "_").replace("=", "_")
        lines.append(f"# TYPE {safe_name} gauge")
        lines.append(f"{safe_name} {value}")

    lines.append(f"# TYPE uptime_seconds gauge")
    lines.append(f"uptime_seconds {metrics['uptime_seconds']}")

    return "\n".join(lines)
