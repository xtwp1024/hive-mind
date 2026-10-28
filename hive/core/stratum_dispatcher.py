# -*- coding: utf-8 -*-
"""
StratumDispatcher - 类 Stratum 矿池任务分发中心
设计灵感来源于 cazala/coin-hive 的 Stratum 协议矿池端。

矿池职责：
1. 向工蜂分发带难度的 PoWJob
2. 接收工蜂提交的 PoWSolution 并验证
3. 统计算力指标（accepted/rejected/hashes_per_second）
4. 触发事件钩子（on_job / on_accepted / on_rejected）
"""

import time
import uuid
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable
from hive.core.proof_of_work import PoWJob, PoWSolution, ProofOfWork

logger = logging.getLogger("Hive.StratumDispatcher")


@dataclass
class MinerStats:
    """单个矿工（工蜂）的统计信息（对标 coin-hive update 事件数据）"""
    bee_id:          str
    total_jobs:      int   = 0
    accepted_hashes: int   = 0
    rejected_hashes: int   = 0
    total_attempts:  int   = 0
    total_elapsed:   float = 0.0

    @property
    def hashes_per_second(self) -> float:
        return self.total_attempts / max(self.total_elapsed, 0.001)

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_hashes + self.rejected_hashes
        return self.accepted_hashes / total if total > 0 else 0.0


class StratumDispatcher:
    """
    类 Stratum 矿池 —— 任务分发与工作量验证中心。
    每个 PoWJob 对应一次工蜂任务，验证通过才算该工蜂真正完成了工作。
    """

    def __init__(
        self,
        pool_name:   str   = "HiveStratum",
        difficulty:  int   = 2,
        job_ttl:     float = 60.0,
    ):
        self.pool_name  = pool_name
        self.difficulty = difficulty
        self.job_ttl    = job_ttl

        self._active_jobs:  dict[str, PoWJob]     = {}  # job_id → job
        self._bee_jobs:     dict[str, str]         = {}  # bee_id → job_id
        self._stats:        dict[str, MinerStats]  = {}  # bee_id → stats
        self._lock = threading.Lock()

        # 事件钩子（对标 coin-hive miner.on(event, callback)）
        self._hooks: dict[str, list[Callable]] = {
            "job":      [],
            "accepted": [],
            "rejected": [],
            "error":    [],
        }

        logger.info(f"[Stratum:{self.pool_name}] 矿池已上线，难度={difficulty}")

    # ─────────────────────────────────────────── 事件系统 ──

    def on(self, event: str, callback: Callable) -> None:
        """注册事件钩子（对标 coin-hive miner.on('accepted', cb)）"""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _emit(self, event: str, data: Any) -> None:
        for cb in self._hooks.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"[Stratum] 事件钩子 {event} 异常: {e}")

    # ─────────────────────────────────────────── 核心流程 ──

    def push_job(self, bee_id: str, task_blob: str = "") -> PoWJob:
        """
        向工蜂分发任务（对标 Stratum 矿池推送 job）。
        每次调用生成一个唯一 PoWJob，包含任务内容摘要+难度。
        """
        if not task_blob:
            task_blob = f"task://{bee_id}/{uuid.uuid4().hex[:12]}"

        job = PoWJob(
            job_id     = uuid.uuid4().hex,
            blob       = task_blob,
            difficulty = self.difficulty,
            ttl_seconds= self.job_ttl,
        )
        with self._lock:
            self._active_jobs[job.job_id] = job
            self._bee_jobs[bee_id]        = job.job_id
            if bee_id not in self._stats:
                self._stats[bee_id] = MinerStats(bee_id=bee_id)
            self._stats[bee_id].total_jobs += 1

        logger.info(f"[Stratum:{self.pool_name}] 🎯 job → {bee_id[:12]} | blob={task_blob[:32]}")
        self._emit("job", {"bee_id": bee_id, "job": job})
        return job

    def submit(self, bee_id: str, solution: PoWSolution) -> str:
        """
        接收工蜂提交的 PoW 解（对标 Stratum submit）。
        返回 "accepted" 或 "rejected"（对标矿池验证结果）。
        """
        with self._lock:
            job = self._active_jobs.get(solution.job_id)
            stats = self._stats.get(bee_id)

        if not job:
            logger.warning(f"[Stratum] {bee_id} 提交了未知 job_id={solution.job_id[:8]}")
            self._emit("error", {"bee_id": bee_id, "reason": "unknown_job"})
            return "rejected"

        if job.is_expired():
            logger.warning(f"[Stratum] {bee_id} 提交了过期 job")
            self._emit("error", {"bee_id": bee_id, "reason": "expired_job"})
            return "rejected"

        valid = ProofOfWork.verify(job, solution)

        with self._lock:
            if stats:
                stats.total_attempts += solution.attempts
                stats.total_elapsed  += solution.elapsed
                if valid:
                    stats.accepted_hashes += 1
                else:
                    stats.rejected_hashes += 1
            # 一个 job 只验收一次
            self._active_jobs.pop(solution.job_id, None)

        if valid:
            logger.info(
                f"[Stratum:{self.pool_name}] ✅ accepted ← {bee_id[:12]} "
                f"attempts={solution.attempts} elapsed={solution.elapsed:.3f}s"
            )
            self._emit("accepted", {"bee_id": bee_id, "solution": solution})
            return "accepted"
        else:
            logger.warning(f"[Stratum:{self.pool_name}] ❌ rejected ← {bee_id[:12]}")
            self._emit("rejected", {"bee_id": bee_id, "solution": solution})
            return "rejected"

    def pool_stats(self) -> dict:
        """返回矿池整体统计（对标 coin-hive update 事件的全局视图）"""
        with self._lock:
            all_stats = list(self._stats.values())
        return {
            "pool":            self.pool_name,
            "total_miners":    len(all_stats),
            "total_accepted":  sum(s.accepted_hashes for s in all_stats),
            "total_rejected":  sum(s.rejected_hashes for s in all_stats),
            "total_attempts":  sum(s.total_attempts for s in all_stats),
            "miners":          {s.bee_id: {
                "hashes_per_second": round(s.hashes_per_second),
                "accepted":          s.accepted_hashes,
                "acceptance_rate":   f"{s.acceptance_rate:.1%}",
            } for s in all_stats},
        }
