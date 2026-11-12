#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor Bridge - 进化系统与 Hive 监控系统集成

从监控系统获取运行指标，为进化决策提供数据支持。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Hive.Evolution.Monitor")


@dataclass
class SystemMetrics:
    """系统指标快照"""

    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    load_average: float = 0.0

    # 蜂巢指标
    active_bees: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    error_rate: float = 0.0
    avg_task_duration: float = 0.0

    # 进化指标
    generation: int = 0
    fitness_avg: float = 0.0
    fitness_best: float = 0.0
    mutations_count: int = 0

    def is_healthy(self) -> bool:
        """判断系统是否健康"""
        return (
            self.cpu_percent < 90
            and self.memory_percent < 85
            and self.error_rate < 0.3
            and self.load_average < 2.0
        )

    def stability_score(self) -> float:
        """计算稳定性分数 (0-1)"""
        scores = [
            1.0 - min(self.cpu_percent / 100, 1.0) * 0.3,
            1.0 - min(self.memory_percent / 100, 1.0) * 0.3,
            1.0 - min(self.error_rate, 1.0) * 0.4,
        ]
        return sum(scores) / len(scores)


class MonitorBridge:
    """
    监控桥接器

    连接 Hive 监控系统，获取实时指标用于进化决策。
    """

    def __init__(self):
        self._monitor = None
        self._metrics_history: List[SystemMetrics] = []
        self._max_history = 100

    @property
    def monitor(self):
        """延迟加载监控模块"""
        if self._monitor is None:
            try:
                from hive.utils.monitor import SystemMonitor

                self._monitor = SystemMonitor()
            except ImportError:
                logger.warning("无法导入 SystemMonitor，使用 psutil 降级")
                self._monitor = "fallback"
        return self._monitor

    def get_current_metrics(self) -> SystemMetrics:
        """获取当前系统指标"""
        metrics = SystemMetrics()

        try:
            import psutil

            # 系统资源
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.memory_percent = psutil.virtual_memory().percent
            metrics.disk_percent = psutil.disk_usage("/").percent

            # 负载
            try:
                load = psutil.getloadavg()
                metrics.load_average = load[0]  # 1分钟负载
            except (AttributeError, OSError):
                # Windows 不支持 getloadavg
                metrics.load_average = metrics.cpu_percent / 100 * 2

        except ImportError:
            logger.warning("psutil 不可用，使用默认值")

        # 尝试从 Hive 监控获取蜂巢指标
        if self.monitor and self.monitor != "fallback":
            try:
                hive_metrics = self.monitor.get_metrics()
                metrics.active_bees = hive_metrics.get("active_bees", 0)
                metrics.tasks_completed = hive_metrics.get("tasks_completed", 0)
                metrics.tasks_failed = hive_metrics.get("tasks_failed", 0)

                total_tasks = metrics.tasks_completed + metrics.tasks_failed
                if total_tasks > 0:
                    metrics.error_rate = metrics.tasks_failed / total_tasks

            except Exception as e:
                logger.debug(f"获取 Hive 指标失败: {e}")

        # 记录历史
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history:
            self._metrics_history.pop(0)

        return metrics

    def get_error_rate(self, window_minutes: int = 60) -> float:
        """获取指定时间窗口内的错误率"""
        cutoff = time.time() - (window_minutes * 60)
        recent = [m for m in self._metrics_history if m.timestamp > cutoff]

        if not recent:
            return 0.0

        return sum(m.error_rate for m in recent) / len(recent)

    def get_stability_trend(self, window_minutes: int = 30) -> str:
        """获取稳定性趋势"""
        cutoff = time.time() - (window_minutes * 60)
        recent = [m for m in self._metrics_history if m.timestamp > cutoff]

        if len(recent) < 2:
            return "unknown"

        # 比较前半段和后半段的稳定性
        mid = len(recent) // 2
        first_half = sum(m.stability_score() for m in recent[:mid]) / mid
        second_half = sum(m.stability_score() for m in recent[mid:]) / (len(recent) - mid)

        diff = second_half - first_half
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "degrading"
        else:
            return "stable"

    def should_backoff(self, load_threshold: float = 2.0) -> bool:
        """判断是否应该退避（系统负载过高）"""
        metrics = self.get_current_metrics()
        return (
            metrics.load_average > load_threshold
            or metrics.cpu_percent > 90
            or metrics.memory_percent > 90
        )

    def get_evolution_recommendations(self) -> Dict[str, Any]:
        """基于监控数据给出进化建议"""
        metrics = self.get_current_metrics()
        trend = self.get_stability_trend()

        recommendations = {
            "should_evolve": True,
            "priority": "normal",
            "suggested_strategy": "balanced",
            "reasons": [],
        }

        # 高负载时建议退避
        if self.should_backoff():
            recommendations["should_evolve"] = False
            recommendations["reasons"].append("系统负载过高，建议退避")
            return recommendations

        # 高错误率时优先修复
        if metrics.error_rate > 0.3:
            recommendations["priority"] = "high"
            recommendations["suggested_strategy"] = "repair-only"
            recommendations["reasons"].append(f"错误率较高 ({metrics.error_rate:.1%})，优先修复")

        # 稳定且健康时可以创新
        elif metrics.is_healthy() and trend == "stable":
            recommendations["suggested_strategy"] = "innovate"
            recommendations["reasons"].append("系统稳定，可以尝试创新")

        # 趋势恶化时加固
        elif trend == "degrading":
            recommendations["suggested_strategy"] = "harden"
            recommendations["reasons"].append("稳定性下降，建议加固")

        return recommendations


# 全局实例
_monitor_bridge: Optional[MonitorBridge] = None


def get_monitor_bridge() -> MonitorBridge:
    """获取全局监控桥接器"""
    global _monitor_bridge
    if _monitor_bridge is None:
        _monitor_bridge = MonitorBridge()
    return _monitor_bridge
