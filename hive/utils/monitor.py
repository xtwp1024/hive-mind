# -*- coding: utf-8 -*-
"""资源监控器

使用psutil监控系统资源使用情况（CPU、内存、磁盘、网络）。
"""

import os
import platform
from dataclasses import dataclass
from typing import Optional
import threading

import psutil

from hive.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceSnapshot:
    """资源使用快照"""

    cpu_percent: float
    cpu_count: int
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    timestamp: float

    def is_healthy(
        self,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0,
    ) -> tuple[bool, list[str]]:
        """检查资源使用是否健康

        Args:
            cpu_threshold: CPU使用率阈值
            memory_threshold: 内存使用率阈值
            disk_threshold: 磁盘使用率阈值

        Returns:
            (是否健康, 问题列表)
        """
        issues = []

        if self.cpu_percent > cpu_threshold:
            issues.append(f"CPU使用率过高: {self.cpu_percent:.1f}%")

        if self.memory_percent > memory_threshold:
            issues.append(f"内存使用率过高: {self.memory_percent:.1f}%")

        if self.disk_percent > disk_threshold:
            issues.append(f"磁盘使用率过高: {self.disk_percent:.1f}%")

        return len(issues) == 0, issues


class ResourceMonitor:
    """系统资源监控器"""

    def __init__(self, interval: float = 5.0):
        """初始化资源监控器

        Args:
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self._last_network_stats = None

        # 初始化网络统计
        try:
            self._last_network_stats = psutil.net_io_counters()
        except Exception as e:
            logger.warning(
                "Failed to initialize network stats",
                error=str(e),
            )

    def get_snapshot(self) -> ResourceSnapshot:
        """获取当前资源使用快照

        Returns:
            ResourceSnapshot对象
        """
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count(logical=True) or 0

            # 内存使用
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)

            # 磁盘使用（优先 HIVE_HOME 所在盘 → Windows 用 D:\，其他用 /）
            hive_home = os.environ.get("HIVE_HOME", "")
            if hive_home and hive_home[0].isalpha():
                disk_path = hive_home[:2] + "\\"  # 例如 D:\
            elif platform.system() == "Windows":
                disk_path = "D:\\"
            else:
                disk_path = "/"
            disk = psutil.disk_usage(disk_path)
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)

            # 网络使用
            network_sent_mb = 0.0
            network_recv_mb = 0.0
            try:
                current_stats = psutil.net_io_counters()
                if self._last_network_stats:
                    sent_bytes = current_stats.bytes_sent - self._last_network_stats.bytes_sent
                    recv_bytes = current_stats.bytes_recv - self._last_network_stats.bytes_recv
                    network_sent_mb = sent_bytes / (1024**2)
                    network_recv_mb = recv_bytes / (1024**2)
                self._last_network_stats = current_stats
            except Exception as e:
                logger.warning(
                    "Failed to get network stats",
                    error=str(e),
                )

            # 进程数量
            process_count = len(psutil.pids())

            import time

            return ResourceSnapshot(
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                process_count=process_count,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(
                "Failed to get resource snapshot",
                error=str(e),
            )
            # 返回默认值
            import time

            return ResourceSnapshot(
                cpu_percent=0.0,
                cpu_count=0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                process_count=0,
                timestamp=time.time(),
            )

    def check_health(
        self,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0,
    ) -> tuple[bool, Optional[ResourceSnapshot], list[str]]:
        """检查系统健康状态

        Args:
            cpu_threshold: CPU使用率阈值
            memory_threshold: 内存使用率阈值
            disk_threshold: 磁盘使用率阈值

        Returns:
            (是否健康, 资源快照, 问题列表)
        """
        snapshot = self.get_snapshot()
        is_healthy, issues = snapshot.is_healthy(
            cpu_threshold=cpu_threshold,
            memory_threshold=memory_threshold,
            disk_threshold=disk_threshold,
        )

        if not is_healthy:
            logger.warning(
                "System health check failed",
                cpu_percent=snapshot.cpu_percent,
                memory_percent=snapshot.memory_percent,
                disk_percent=snapshot.disk_percent,
                issues=issues,
            )

        return is_healthy, snapshot, issues

    def get_system_info(self) -> dict:
        """获取系统信息

        Returns:
            系统信息字典
        """
        # 运行时冒烟验证（main.initialize_system）会读取这两个 key；
        # 以前缺失导致日志恒显 "CPU/Memory: unknown"，补齐真实值。
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True) or 0,
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }

    def log_snapshot(self, snapshot: Optional[ResourceSnapshot] = None) -> None:
        """记录资源快照到日志

        Args:
            snapshot: 资源快照，如果为None则获取新的
        """
        if snapshot is None:
            snapshot = self.get_snapshot()

        logger.info(
            "Resource snapshot",
            cpu=f"{snapshot.cpu_percent:.1f}%",
            memory=f"{snapshot.memory_percent:.1f}% ({snapshot.memory_used_gb:.2f}GB/{snapshot.memory_total_gb:.2f}GB)",
            disk=f"{snapshot.disk_percent:.1f}% ({snapshot.disk_used_gb:.2f}GB/{snapshot.disk_total_gb:.2f}GB)",
            network_sent=f"{snapshot.network_sent_mb:.2f}MB",
            network_recv=f"{snapshot.network_recv_mb:.2f}MB",
            processes=snapshot.process_count,
        )


# 全局监控器实例
_monitor: Optional[ResourceMonitor] = None
_monitor_lock = threading.Lock()


def get_monitor(interval: float = 5.0) -> ResourceMonitor:
    """获取全局资源监控器实例（线程安全）"""
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = ResourceMonitor(interval=interval)
    return _monitor
