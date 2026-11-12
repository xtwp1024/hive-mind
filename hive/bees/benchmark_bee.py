# -*- coding: utf-8 -*-
import cProfile
import io
import pstats
import time
from pathlib import Path

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkerBee(BaseBee):
    """基准测试蜂 - 测量和优化代码性能"""

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.benchmark_results = []

    def specialized_task(self, target_path):
        """执行基准测试任务"""
        logger.info(f"[BenchmarkerBee-{self.id}] 开始基准测试: {target_path}")

        target = Path(target_path)
        if not target.exists():
            logger.error(f"[BenchmarkerBee-{self.id}] 目标不存在: {target_path}")
            return

        results = {"target": str(target_path), "measurements": [], "report": {}}

        # 测量文件大小和读取时间
        start = time.time()
        file_size = target.stat().st_size
        content = target.read_text()
        read_time = time.time() - start

        results["measurements"].append(
            {"metric": "file_read_time", "value": read_time, "unit": "seconds"}
        )

        # 性能分析
        pr = cProfile.Profile()
        pr.enable()

        # 模拟处理
        time.sleep(0.1)

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()

        results["report"] = {
            "file_size": file_size,
            "read_time": read_time,
            "performance_profile": s.getvalue(),
        }

        self.benchmark_results.append(results)

        logger.info(f"[BenchmarkerBee-{self.id}] 基准测试完成")
        return results

    def devour(self) -> dict:
        """消耗目标并返回性能报告"""
        target_path = self.dna.id

        try:
            result = self.specialized_task(target_path)

            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": f"{result['measurements'][0]['value']:.4f}s",
                "path": target_path,
                "benchmark": result,
            }
        except Exception as e:
            logger.error(f"[BenchmarkerBee-{self.id}] 基准测试失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e),
            }
