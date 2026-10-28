# -*- coding: utf-8 -*-
"""
BeeReconciler - 协调控制循环
设计灵感来源于 openshift/hive 的 Reconcile Controller。

Reconciler 是"声明式哲学"的执行引擎：
它永不停歇地将现实（actual state）向期望（desired state）对齐。
池子里少了工蜂？补！死蜂占着坑？清！任务超期？驱逐！
"""

import time
import threading
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hive.core.cluster_pool import BeePool, BeeState

logger = logging.getLogger("Hive.BeeReconciler")


class BeeReconciler:
    """
    协调控制循环守护线程（对标 openshift/hive Hive Controllers）。
    监管一个 BeePool，持续维持其声明的期望状态。
    完全无状态：每次协调基于当下快照做决策，天然幂等。
    """

    def __init__(
        self,
        pool: "BeePool",
        interval_seconds: float = 5.0,
        max_bee_lifetime: float = 300.0,
    ):
        self.pool = pool
        self.interval = interval_seconds
        self.max_lifetime = max_bee_lifetime  # 工蜂最大存活时间（秒）
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------ 核心协调逻辑 --

    def _desired_state(self) -> dict:
        """期望状态：池中应保持 desired_size 只冬眠工蜂"""
        return {"hibernated": self.pool.desired_size}

    def _actual_state(self) -> dict:
        """实际状态：快照当前池"""
        from hive.core.cluster_pool import BeeState
        with self.pool._lock:
            hibernated = sum(1 for b in self.pool._pool.values() if b.state == BeeState.HIBERNATED)
            running    = sum(1 for b in self.pool._pool.values() if b.state == BeeState.RUNNING)
            all_bees   = list(self.pool._pool.values())
        return {"hibernated": hibernated, "running": running, "all_bees": all_bees}

    def _reconcile_once(self) -> None:
        """执行一次协调循环（对标 controller-runtime Reconcile 方法）"""
        from hive.core.cluster_pool import BeeState

        desired = self._desired_state()
        actual  = self._actual_state()

        # 1. 清除超期工蜂（对标 ClusterClaim.Spec.Lifetime 超时销毁）
        now = time.time()
        for bee in actual["all_bees"]:
            if bee.state == BeeState.RUNNING and bee.claimed_at:
                age = now - bee.claimed_at
                if age > self.max_lifetime:
                    logger.warning(f"[Reconciler:{self.pool.name}] {bee.bee_id} 存活超期 ({age:.0f}s)，强制驱逐")
                    self.pool._deprovision(bee.bee_id)

        # 2. 补充池（期望 > 实际冬眠数）
        gap = desired["hibernated"] - actual["hibernated"]
        if gap > 0:
            logger.info(f"[Reconciler:{self.pool.name}] 池缺口 {gap} 只，触发补充孵化...")
            threads = []
            for _ in range(gap):
                t = threading.Thread(
                    target=self.pool._provision,
                    args=(f"reconcile-refill",),
                    daemon=True,
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
        elif gap < 0:
            # 池过剩（例如手动添加了额外蜂），缩减冬眠工蜂
            excess = abs(gap)
            logger.info(f"[Reconciler:{self.pool.name}] 池过剩 {excess} 只，缩减...")
            with self.pool._lock:
                hibernated_bees = [b for b in self.pool._pool.values() if b.state == BeeState.HIBERNATED]
            for bee in hibernated_bees[:excess]:
                self.pool._deprovision(bee.bee_id)

    def _run_loop(self) -> None:
        """主协调循环（永不退出，直到 stop() 被调用）"""
        logger.info(f"[Reconciler:{self.pool.name}] 协调循环已启动，周期={self.interval}s")
        while not self._stop_event.is_set():
            try:
                self._reconcile_once()
            except Exception as e:
                logger.error(f"[Reconciler:{self.pool.name}] 协调异常（已自愈）: {e}")
            self._stop_event.wait(timeout=self.interval)
        logger.info(f"[Reconciler:{self.pool.name}] 协调循环已停止")

    # --------------------------------------------------------- 控制接口  --

    def start(self) -> None:
        """启动后台协调线程"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name=f"Reconciler-{self.pool.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """优雅停止协调线程"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def reconcile_now(self) -> None:
        """手动触发一次即时协调（用于测试/紧急修复）"""
        logger.info(f"[Reconciler:{self.pool.name}] 手动触发协调")
        self._reconcile_once()
