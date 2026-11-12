# -*- coding: utf-8 -*-
import time
from pathlib import Path

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class PhantomBee(BaseBee):
    """幽灵蜂 - 隐身操作、后台监控、静默执行"""

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.stealth_mode = False
        self.monitoring_log = []

    def specialized_task(self, target_path):
        """执行幽灵任务"""
        logger.info(f"[PhantomBee-{self.id}] 幽灵: {target_path}")

        # 静默监控
        self._silent_monitor(target_path)

        # 隐身操作
        self._stealth_operation(target_path)

        logger.info(f"[PhantomBee-{self.id}] 幽灵任务完成")

    def _silent_monitor(self, target: str):
        """静默监控"""
        monitor_data = {
            "target": target,
            "timestamp": time.time(),
            "status": "monitored",
            "details": "Silent observation complete",
        }

        self.monitoring_log.append(monitor_data)

        # 不记录详细日志，保持隐身
        logger.debug(f"[PhantomBee-{self.id}] Monitoring: {target}")

    def _stealth_operation(self, target: str):
        """隐身操作"""
        # 模拟隐身操作
        operation_data = {
            "target": target,
            "operation": "stealth_scan",
            "result": "success",
            "detected": False,
        }

        self.monitoring_log.append(operation_data)

    def enable_stealth_mode(self):
        """启用隐身模式"""
        self.stealth_mode = True
        logger.warning(f"[PhantomBee-{self.id}] Stealth mode enabled")

    def disable_stealth_mode(self):
        """禁用隐身模式"""
        self.stealth_mode = False
        logger.info(f"[PhantomBee-{self.id}] Stealth mode disabled")

    def clear_traces(self):
        """清理痕迹"""
        self.monitoring_log.clear()
        logger.info(f"[PhantomBee-{self.id}] Traces cleared")

    def get_monitoring_log(self) -> list:
        """获取监控日志"""
        return self.monitoring_log.copy()

    def devour(self) -> dict:
        """消耗目标并返回幽灵报告"""
        target_path = self.dna.id

        try:
            self.specialized_task(target_path)

            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "stealth_mode": self.stealth_mode,
                "operations": len(self.monitoring_log),
            }
        except Exception as e:
            logger.error(f"[PhantomBee-{self.id}] 幽灵任务失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e),
            }
