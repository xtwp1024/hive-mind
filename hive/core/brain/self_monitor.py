"""
SelfMonitor - 自我监控器
=========================

功能:
- 认知负载监控
- 注意力分配
- 置信度评估
- 健康检查

Usage:
    monitor = SelfMonitor()
    state = monitor.monitor()
    alert = await monitor.check_and_alert()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .working_memory import WorkingMemory

logger = logging.getLogger("Hive.Brain.SelfMonitor")


class HealthStatus(str, Enum):
    """健康状态"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class CognitiveMetrics:
    """
    认知指标

    Attributes:
        timestamp: 时间戳
        attention_load: 注意力负载 (0-1)
        memory_pressure: 记忆压力 (0-1)
        confidence: 全局置信度
        active_goals: 活跃目标数
        error_rate: 错误率
        throughput: 吞吐量 (thinks/分钟)
    """

    timestamp: float = field(default_factory=time.time)
    attention_load: float = 0.0
    memory_pressure: float = 0.0
    confidence: float = 0.5
    active_goals: int = 0
    error_rate: float = 0.0
    throughput: float = 0.0
    health_status: HealthStatus = HealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "attention_load": self.attention_load,
            "memory_pressure": self.memory_pressure,
            "confidence": self.confidence,
            "active_goals": self.active_goals,
            "error_rate": self.error_rate,
            "throughput": self.throughput,
            "health_status": self.health_status.value,
        }


@dataclass
class Alert:
    """告警"""

    id: str
    level: str  # info, warning, critical
    message: str
    metric: str
    value: Any
    threshold: Any
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


class SelfMonitor:
    """
    自我监控器

    监控大脑的认知状态，包括:
    1. 注意力负载
    2. 记忆压力
    3. 置信度
    4. 错误率
    5. 吞吐量

    当指标超出阈值时生成告警。
    """

    # 阈值配置
    THRESHOLDS = {
        "attention_load": {"warning": 0.7, "critical": 0.9},
        "memory_pressure": {"warning": 0.7, "critical": 0.9},
        "confidence": {"warning": 0.3, "critical": 0.2},  # 低置信度告警
        "error_rate": {"warning": 0.2, "critical": 0.4},
        "throughput": {"warning": 2.0, "critical": 0.5},  # 每分钟完成的认知周期数阈值
        "active_goals": {"warning": 8, "critical": 10},
    }

    def __init__(
        self,
        working_memory: Optional[WorkingMemory] = None,
        check_interval: float = 5.0,  # 检查间隔(秒)
    ):
        self.working_memory = working_memory
        self.check_interval = check_interval

        # 指标历史
        self._metrics_history: List[CognitiveMetrics] = []
        self._max_history = 100

        # 告警
        self._alerts: List[Alert] = []
        self._max_alerts = 50

        # 统计
        self._stats = {
            "checks": 0,
            "alerts_generated": 0,
            "critical_alerts": 0,
            "health_transitions": 0,
        }

        # 状态
        self._last_check_time = time.time()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._current_health = HealthStatus.HEALTHY

        # 回调
        self._alert_callbacks: List[callable] = []

        # 吞吐量跟踪
        self._cognitive_cycles = 0  # 完成的认知周期计数
        self._cycle_window_start = time.time()  # 当前窗口开始时间

    async def start(self):
        """启动监控"""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("🧠 SelfMonitor started")

    async def stop(self):
        """停止监控"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("🧠 SelfMonitor stopped")

    async def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                await self.check_and_alert()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"⚠️ Monitor loop error: {e}")

    async def check_and_alert(self) -> List[Alert]:
        """
        检查指标并生成告警

        Returns:
            List[Alert]: 新生成的告警列表
        """
        self._stats["checks"] += 1
        metrics = await self.monitor()

        # 保存历史
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history:
            self._metrics_history.pop(0)

        # 检查阈值
        new_alerts = []
        new_alerts.extend(self._check_threshold(metrics, "attention_load", metrics.attention_load))
        new_alerts.extend(
            self._check_threshold(metrics, "memory_pressure", metrics.memory_pressure)
        )
        new_alerts.extend(
            self._check_threshold(metrics, "confidence", metrics.confidence, inverted=True)
        )
        new_alerts.extend(self._check_threshold(metrics, "error_rate", metrics.error_rate))
        new_alerts.extend(self._check_threshold(metrics, "active_goals", metrics.active_goals))

        # 更新告警列表
        for alert in new_alerts:
            self._alerts.append(alert)
            self._stats["alerts_generated"] += 1
            if alert.level == "critical":
                self._stats["critical_alerts"] += 1

        # 限制告警数量
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts :]

        # 检查健康状态转换
        old_health = self._current_health
        if metrics.health_status != old_health:
            self._stats["health_transitions"] += 1
            self._current_health = metrics.health_status
            logger.warning(
                f"🧠 Health transition: {old_health.value} -> {metrics.health_status.value}"
            )

        # 触发回调
        for alert in new_alerts:
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.warning(f"⚠️ Alert callback error: {e}")

        return new_alerts

    def _check_threshold(
        self,
        metrics: CognitiveMetrics,
        metric_name: str,
        value: float,
        inverted: bool = False,
    ) -> List[Alert]:
        """检查指标是否超出阈值"""
        alerts = []
        threshold = self.THRESHOLDS.get(metric_name, {})

        if not threshold:
            return alerts

        level = None
        thresh_value = None

        if inverted:
            # 反向阈值（低值告警）
            if value <= threshold.get("critical", 0.2):
                level = "critical"
                thresh_value = threshold["critical"]
            elif value <= threshold.get("warning", 0.3):
                level = "warning"
                thresh_value = threshold["warning"]
        else:
            if value >= threshold.get("critical", 0.9):
                level = "critical"
                thresh_value = threshold["critical"]
            elif value >= threshold.get("warning", 0.7):
                level = "warning"
                thresh_value = threshold["warning"]

        if level:
            alert = Alert(
                id=f"alert_{len(self._alerts)}_{metric_name}",
                level=level,
                message=self._format_alert_message(metric_name, value, thresh_value, inverted),
                metric=metric_name,
                value=value,
                threshold=thresh_value,
            )
            alerts.append(alert)

        return alerts

    def _format_alert_message(
        self,
        metric: str,
        value: float,
        threshold: float,
        inverted: bool,
    ) -> str:
        """格式化告警消息"""
        metric_labels = {
            "attention_load": "注意力负载",
            "memory_pressure": "记忆压力",
            "confidence": "置信度",
            "error_rate": "错误率",
            "active_goals": "活跃目标数",
            "throughput": "吞吐量",
        }

        label = metric_labels.get(metric, metric)

        if inverted:
            return f"{label}过低: {value:.2f} (阈值: {threshold})"
        else:
            return f"{label}过高: {value:.2f} (阈值: {threshold})"

    async def monitor(self) -> CognitiveMetrics:
        """
        获取当前认知指标

        Returns:
            CognitiveMetrics: 当前指标
        """
        metrics = CognitiveMetrics()

        # 从工作记忆获取
        if self.working_memory:
            metrics.attention_load = self.working_memory.get_attention_load()
            metrics.memory_pressure = self.working_memory.get_memory_pressure()
            metrics.active_goals = self.working_memory.get_active_goal_count()

        # 计算吞吐量 (完成的认知周期/分钟)
        # 使用滑动窗口计算更准确的吞吐量
        self._cognitive_cycles += 1
        now = time.time()
        window_duration = now - self._cycle_window_start

        # 每分钟更新一次吞吐量计算
        if window_duration >= 60.0:
            metrics.throughput = self._cognitive_cycles / window_duration * 60.0
            self._cognitive_cycles = 0
            self._cycle_window_start = now
        else:
            # 估算吞吐量（基于历史数据）
            metrics.throughput = self._cognitive_cycles / max(window_duration, 1.0) * 60.0

        self._last_check_time = now

        # 计算错误率
        # (需要从HiveBrain获取，这里简化处理)
        metrics.error_rate = 0.0

        # 计算置信度 (综合指标)
        metrics.confidence = self._calculate_confidence(metrics)

        # 评估健康状态
        metrics.health_status = self._assess_health(metrics)

        return metrics

    def _calculate_confidence(self, metrics: CognitiveMetrics) -> float:
        """计算综合置信度"""
        # 基于注意力负载
        attention_factor = 1.0 - metrics.attention_load * 0.3

        # 基于记忆压力
        memory_factor = 1.0 - metrics.memory_pressure * 0.2

        # 基于错误率
        error_factor = 1.0 - metrics.error_rate * 0.5

        # 基于活跃目标
        goal_factor = 1.0 - min(metrics.active_goals / 10, 1.0) * 0.2

        confidence = attention_factor * memory_factor * error_factor * goal_factor
        return max(0.0, min(1.0, confidence))

    def _assess_health(self, metrics: CognitiveMetrics) -> HealthStatus:
        """评估健康状态"""
        # Critical check
        if (
            metrics.attention_load > 0.9
            or metrics.memory_pressure > 0.9
            or metrics.error_rate > 0.4
            or metrics.confidence < 0.2
        ):
            return HealthStatus.CRITICAL

        # Unhealthy check
        if (
            metrics.attention_load > 0.7
            or metrics.memory_pressure > 0.7
            or metrics.error_rate > 0.2
            or metrics.confidence < 0.3
        ):
            return HealthStatus.UNHEALTHY

        # Degraded check
        if (
            metrics.attention_load > 0.5
            or metrics.memory_pressure > 0.5
            or metrics.error_rate > 0.1
            or metrics.confidence < 0.5
        ):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def register_alert_callback(self, callback: callable):
        """注册告警回调"""
        self._alert_callbacks.append(callback)

    def get_current_metrics(self) -> Optional[CognitiveMetrics]:
        """获取最新指标"""
        return self._metrics_history[-1] if self._metrics_history else None

    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """获取最近告警"""
        return self._alerts[-count:]

    def get_critical_alerts(self) -> List[Alert]:
        """获取关键告警"""
        return [a for a in self._alerts if a.level == "critical"]

    def get_health_history(self, duration_seconds: float = 300) -> List[CognitiveMetrics]:
        """获取健康历史"""
        cutoff = time.time() - duration_seconds
        return [m for m in self._metrics_history if m.timestamp > cutoff]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "current_health": self._current_health.value,
            "metrics_in_history": len(self._metrics_history),
            "alerts_count": len(self._alerts),
            "callbacks_registered": len(self._alert_callbacks),
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict: 健康检查结果
        """
        metrics = await self.monitor()
        recent_alerts = self.get_recent_alerts(5)

        return {
            "healthy": metrics.health_status == HealthStatus.HEALTHY,
            "status": metrics.health_status.value,
            "metrics": metrics.to_dict(),
            "recent_alerts": [a.to_dict() for a in recent_alerts],
            "recommendations": self._generate_recommendations(metrics),
        }

    def _generate_recommendations(self, metrics: CognitiveMetrics) -> List[str]:
        """生成建议"""
        recommendations = []

        if metrics.attention_load > 0.7:
            recommendations.append("注意力负载较高，建议减少并行任务")

        if metrics.memory_pressure > 0.7:
            recommendations.append("记忆压力较大，建议清理不活跃的目标")

        if metrics.confidence < 0.5:
            recommendations.append("置信度较低，建议进行更多反思和学习")

        if metrics.error_rate > 0.2:
            recommendations.append("错误率较高，建议分析失败原因")

        if not recommendations:
            recommendations.append("系统运行正常")

        return recommendations
