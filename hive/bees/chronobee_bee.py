# -*- coding: utf-8 -*-
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class ChronoBee(BaseBee):
    """时间管理蜂 - 管理时间戳、调度和时间序列分析"""

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.timestamps = {}
        self.scheduled_tasks = deque()
        self.time_series = deque(maxlen=1000)

    def specialized_task(self, target_path):
        """执行时间管理任务"""
        logger.info(f"[ChronoBee-{self.id}] 管理时间: {target_path}")

        # 记录时间戳
        timestamp = time.time()
        self.timestamps[target_path] = {
            "recorded_at": datetime.now().isoformat(),
            "unix_time": timestamp,
        }

        # 添加到时间序列
        self.time_series.append(
            {"timestamp": timestamp, "event": "task_execution", "target": target_path}
        )

        # 计算时间间隔
        if len(self.time_series) >= 2:
            prev = self.time_series[-2]
            curr = self.time_series[-1]
            interval = curr["timestamp"] - prev["timestamp"]

            logger.info(f"[ChronoBee-{self.id}] 时间间隔: {interval:.4f}s")

        logger.info(f"[ChronoBee-{self.id}] 时间管理完成")

    def schedule_task(self, task: Dict, delay: float):
        """调度延迟任务"""
        execute_time = time.time() + delay
        self.scheduled_tasks.append({"task": task, "execute_at": execute_time})
        logger.info(f"[ChronoBee-{self.id}] 已调度任务，延迟: {delay}s")

    def analyze_time_series(self, window: int = 10) -> Dict:
        """分析时间序列"""
        if len(self.time_series) < 2:
            return {"status": "insufficient_data"}

        recent = list(self.time_series)[-window:]
        intervals = []

        for i in range(1, len(recent)):
            interval = recent[i]["timestamp"] - recent[i - 1]["timestamp"]
            intervals.append(interval)

        return {
            "count": len(intervals),
            "avg_interval": sum(intervals) / len(intervals),
            "min_interval": min(intervals),
            "max_interval": max(intervals),
        }

    def devour(self) -> dict:
        """消耗目标并返回时间报告"""
        target_path = self.dna.id

        try:
            self.specialized_task(target_path)

            analysis = self.analyze_time_series()

            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": f"{time.time() - self.timestamps[target_path]['unix_time']:.4f}s",
                "path": target_path,
                "timestamps": self.timestamps,
                "analysis": analysis,
            }
        except Exception as e:
            logger.error(f"[ChronoBee-{self.id}] 时间管理失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e),
            }
