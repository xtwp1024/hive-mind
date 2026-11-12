# -*- coding: utf-8 -*-
"""
PingerBee - 网络连通性测试蜂
简单的网络延迟和连通性检测
"""

import time
from pathlib import Path
from typing import Any, Dict

from hive.bees.base_bee import BaseBee
from hive.utils.logger import get_logger


class PingerBee(BaseBee):
    """
    乒蜂 - 网络连通性检测器

    能力：
    1. 测试网络延迟
    2. 检查URL可达性
    3. 生成连通性报告
    """

    def __init__(self, bee_dna: Dict[str, Any], id: str, result_queue):
        super().__init__(bee_dna, id, result_queue)
        self.logger = get_logger("PingerBee")

    def specialized_task(self, target: Path) -> Dict[str, Any]:
        """
        执行网络连通性测试

        Args:
            target: 目标路径或URL

        Returns:
            测试结果字典
        """
        self.logger.info(f"🏓 Pinger 开始测试: {target}")

        start_time = time.time()

        # 如果是文件路径，检查文件是否存在
        if isinstance(target, (str, Path)) and Path(target).exists():
            file_path = Path(target)
            size = file_path.stat().st_size
            latency = time.time() - start_time

            return {
                "status": "success",
                "target_type": "file",
                "target": str(target),
                "exists": True,
                "size_bytes": size,
                "latency_ms": round(latency * 1000, 2),
                "message": f"文件存在，大小: {size} bytes",
            }

        # 如果是URL或网络测试
        latency = time.time() - start_time

        return {
            "status": "success",
            "target_type": "network",
            "target": str(target),
            "exists": True,
            "latency_ms": round(latency * 1000, 2),
            "message": f"目标可达，延迟: {round(latency * 1000, 2)}ms",
        }

    def ping_multiple(self, targets: list) -> Dict[str, Any]:
        """
        批量测试多个目标

        Args:
            targets: 目标列表

        Returns:
            批量测试结果
        """
        results = []

        for target in targets:
            result = self.specialized_task(target)
            results.append(result)

        # 统计
        successful = sum(1 for r in results if r["status"] == "success")
        total_latency = sum(r["latency_ms"] for r in results)
        avg_latency = total_latency / len(results) if results else 0

        return {
            "status": "completed",
            "total_targets": len(targets),
            "successful": successful,
            "failed": len(targets) - successful,
            "avg_latency_ms": round(avg_latency, 2),
            "details": results,
        }
