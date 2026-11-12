# -*- coding: utf-8 -*-
"""
BeePool - 声明式工蜂热池
设计灵感来源于 openshift/hive 的 ClusterPool + ClusterClaim 机制。

与传统命令式调度不同，BeePool 不关心"创建哪只蜂"，
只关心"我想要 N 只随时可用的工蜂"——Reconciler 负责维持这个现实。
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hive.core.dna import BeeDNA

logger = logging.getLogger("Hive.BeePool")


class BeeState(str, Enum):
    """工蜂生命周期状态（对标 openshift/hive ClusterDeployment 状态机）"""

    PROVISIONING = "provisioning"  # 孵化中（对标 Provisioning）
    HIBERNATED = "hibernated"  # 冬眠就绪，等待申领（对标 Hibernated）
    RUNNING = "running"  # 已被申领并激活（对标 Running/Claimed）
    DEPROVISIONING = "deprovisioning"  # 正在销毁（对标 Deprovisioning）


@dataclass
class PooledBee:
    """池中的一只工蜂条目（对标 ClusterDeployment）"""

    bee_id: str
    dna: BeeDNA
    state: BeeState = BeeState.PROVISIONING
    sync_config: dict[str, Any] = field(default_factory=dict)
    claimed_at: float | None = None
    created_at: float = field(default_factory=time.time)


class BeePool:
    """
    声明式工蜂热池。
    用户声明 desired_size=N，池将自动维持 N 只冬眠就绪的工蜂。
    通过 claim() 申领一只并自动触发后备补充。
    """

    def __init__(
        self,
        name: str,
        species: str,
        desired_size: int = 3,
        running_count: int = 1,
    ):
        self.name = name
        self.species = species
        self.desired_size = desired_size  # 期望总池大小（含冬眠）
        self.running_count = running_count  # 保持活跃（不冬眠）的最小数量
        self._pool: dict[str, PooledBee] = {}
        self._lock = threading.Lock()
        logger.info(
            f"[BeePool:{self.name}] 已声明，目标池大小={desired_size}，常驻活跃={running_count}"
        )

    # ---------------------------------------------------------- 内部操作 --

    def _provision(self, target_name: str = "pool-warmup") -> PooledBee:
        """孵化一只新工蜂进入池（对标 Hive 启动 install pod）"""
        bee_id = f"bee-{uuid.uuid4().hex[:8]}"
        dna = BeeDNA(species=self.species, target_name=target_name)
        entry = PooledBee(bee_id=bee_id, dna=dna, state=BeeState.PROVISIONING)
        with self._lock:
            self._pool[bee_id] = entry
        # 模拟孵化耗时后进入冬眠就绪（真实环境会异步等待工蜂完成初始化）
        time.sleep(0.05)
        with self._lock:
            self._pool[bee_id].state = BeeState.HIBERNATED
        logger.info(f"[BeePool:{self.name}] 孵化完成 → {bee_id} 进入冬眠待命")
        return entry

    def _hibernate(self, bee_id: str) -> None:
        """将工蜂置为冬眠（节省资源，对标 Cluster Hibernation）"""
        with self._lock:
            if bee_id in self._pool:
                self._pool[bee_id].state = BeeState.HIBERNATED
        logger.debug(f"[BeePool:{self.name}] {bee_id} 已冬眠")

    def _resume(self, bee_id: str) -> None:
        """唤醒冬眠工蜂（申领时触发，对标 Cluster Resume）"""
        with self._lock:
            if bee_id in self._pool:
                self._pool[bee_id].state = BeeState.RUNNING
        logger.info(f"[BeePool:{self.name}] {bee_id} 已唤醒 → 运行中")

    def _deprovision(self, bee_id: str) -> None:
        """销毁工蜂并从池中移除（对标 ClusterDeployment 删除）"""
        with self._lock:
            entry = self._pool.pop(bee_id, None)
        if entry:
            logger.info(f"[BeePool:{self.name}] {bee_id} 已销毁")

    # --------------------------------------------------------- 公开 API  --

    def claim(self, task_dna: BeeDNA | None = None, sync_config: dict | None = None) -> str | None:
        """
        申领一只冬眠就绪工蜂（对标 ClusterClaim）。
        - 优先从冬眠池中取，若无则返回 None（由 Reconciler 补充）。
        - 申领后自动唤醒，并注入 SyncSet 配置。
        返回被申领的 bee_id。
        """
        with self._lock:
            candidates = [b for b in self._pool.values() if b.state == BeeState.HIBERNATED]

            if not candidates:
                logger.warning(f"[BeePool:{self.name}] 无冬眠工蜂可申领，请等待 Reconciler 补充")
                return None

            target = candidates[0]
            # 注入 SyncSet 配置（对标 SyncSet 在 ClusterDeployment 就绪后自动同步）
            if sync_config:
                self._pool[target.bee_id].sync_config = sync_config
            if task_dna:
                self._pool[target.bee_id].dna = task_dna
            self._pool[target.bee_id].claimed_at = time.time()

        self._resume(target.bee_id)
        logger.info(f"[BeePool:{self.name}] ClusterClaim 已满足 → {target.bee_id} 交付")
        return target.bee_id

    def release(self, bee_id: str) -> None:
        """任务完成后销毁工蜂（申领过的集群不回池，对标 ClusterClaim 删除触发 Deprovision）"""
        self._deprovision(bee_id)

    def status(self) -> dict:
        """快照当前池的整体状态"""
        with self._lock:
            counts = {}
            for state in BeeState:
                counts[state.value] = sum(1 for b in self._pool.values() if b.state == state)
        return {
            "pool_name": self.name,
            "desired_size": self.desired_size,
            "running_count": self.running_count,
            **counts,
        }

    def warm_up(self) -> None:
        """预热池至 desired_size（初次启动调用，后续由 Reconciler 维护）"""
        logger.info(f"[BeePool:{self.name}] 开始预热至 {self.desired_size} 只...")
        threads = []
        for i in range(self.desired_size):
            t = threading.Thread(target=self._provision, args=(f"warmup-slot-{i}",), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        logger.info(f"[BeePool:{self.name}] 预热完成 → {self.status()}")
