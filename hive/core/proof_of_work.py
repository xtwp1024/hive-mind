# -*- coding: utf-8 -*-
"""
ProofOfWork - 工作量证明哈希器
设计灵感来源于 cazala/coin-hive 的 CryptoNight/Stratum PoW 机制。

核心思想：穷举 nonce，直到 SHA-256(blob + nonce) 满足目标难度（前 N 位为 0）。
这使得工蜂无法虚报任务完成——完成证明必须付出真实的 CPU 计算代价。
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("Hive.ProofOfWork")


@dataclass
class PoWJob:
    """一个挖矿任务（对标 Stratum 协议的 job）"""

    job_id: str
    blob: str  # 任务内容摘要（对标 mining blob）
    difficulty: int = 2  # 难度前缀零位数（2=前16位为0，轻量演示；生产可调高）
    issued_at: float = field(default_factory=time.time)
    ttl_seconds: float = 60.0

    @property
    def target_prefix(self) -> str:
        """目标哈希前缀（N 个十六进制 0）"""
        return "0" * self.difficulty

    def is_expired(self) -> bool:
        return time.time() > self.issued_at + self.ttl_seconds


@dataclass
class PoWSolution:
    """一个满足难度的解（对标 Stratum submit）"""

    job_id: str
    nonce: int
    hash_hex: str
    attempts: int  # 穷举了多少次才找到
    elapsed: float  # 计算耗时（秒）


class ProofOfWork:
    """
    工蜂端的 PoW 哈希计算器。
    接收 PoWJob，穷举 nonce 直到找到满足难度的哈希，返回 PoWSolution。
    """

    def __init__(self, throttle: float = 0.0, max_attempts: int = 10_000_000):
        """
        throttle: 0.0=全速, 0.5=50% 空闲（对标 coin-hive throttle）
        max_attempts: 最大穷举次数（防止无限死循环）
        """
        self.throttle = max(0.0, min(1.0, throttle))
        self.max_attempts = max_attempts
        self._total_hashes = 0

    def mine(self, job: PoWJob, start_nonce: int = 0) -> PoWSolution | None:
        """
        穷举 nonce，找到满足 job.difficulty 难度的哈希。
        对标 coin-hive 的 CryptoNight 内部哈希循环。
        """
        if job.is_expired():
            logger.warning(f"[PoW] job {job.job_id[:8]} 已过期，放弃")
            return None

        prefix = job.target_prefix
        t_start = time.time()
        nonce = start_nonce
        attempts = 0

        logger.info(f"[PoW] 开始挖矿 job={job.job_id[:8]}, difficulty={job.difficulty}")

        while attempts < self.max_attempts:
            # 节流：每 1000 次哈希后按比例休眠（对标 coin-hive throttle）
            if self.throttle > 0 and attempts % 1000 == 0 and attempts > 0:
                sleep_time = (1000 / max(1, attempts)) * self.throttle
                time.sleep(sleep_time)

            raw = f"{job.blob}:{nonce}".encode()
            h = hashlib.sha256(raw).hexdigest()
            attempts += 1
            self._total_hashes += 1

            if h.startswith(prefix):
                elapsed = time.time() - t_start
                logger.info(
                    f"[PoW] ✅ FOUND! job={job.job_id[:8]} "
                    f"nonce={nonce} hash={h[:16]}... "
                    f"attempts={attempts} elapsed={elapsed:.3f}s"
                )
                return PoWSolution(
                    job_id=job.job_id,
                    nonce=nonce,
                    hash_hex=h,
                    attempts=attempts,
                    elapsed=elapsed,
                )
            nonce += 1

        logger.warning(f"[PoW] job {job.job_id[:8]} 穷举 {attempts} 次未找到解，放弃")
        return None

    def hashes_per_second(self, window: float = 1.0) -> float:
        """简化版算力估算（总哈希数/秒）"""
        return self._total_hashes / max(window, 0.001)

    @staticmethod
    def verify(job: PoWJob, solution: PoWSolution) -> bool:
        """
        矿池端（StratumDispatcher）验证解的有效性。
        对标 Stratum 矿池的 submit 验证逻辑。
        """
        raw = f"{job.blob}:{solution.nonce}".encode()
        h = hashlib.sha256(raw).hexdigest()
        valid = h == solution.hash_hex and h.startswith(job.target_prefix)
        if not valid:
            logger.warning(f"[PoW] ❌ 验证失败 job={job.job_id[:8]}")
        return valid
