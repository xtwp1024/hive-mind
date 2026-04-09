# -*- coding: utf-8 -*-
"""
WorkerManager - Worker 进程管理器

专注负责 Worker 蜜蜂的生命周期管理。

Phase 3 P3: 自然选择机制 — 接入 SwarmEvolutionCoordinator 淘汰闭环
- 订阅 bee_culled ALERT 消息
- 收到淘汰警报时立即终止对应 Worker 进程
- 触发替换孵化（可选：自动补充被淘汰的蜂）
"""

import multiprocessing
import time
import logging
from typing import Any, Optional

from hive.core.dna import BeeDNA
from hive.core.bee_factory import BeeFactory
from hive.bees.base_bee import BaseBee
from hive.bees.async_base_bee import AsyncBaseBee

logger = logging.getLogger("Hive.WorkerManager")


class WorkerManager:
    """Worker 进程管理器"""

    def __init__(self, result_queue: multiprocessing.Queue):
        from hive.config.settings import get_settings
        self.result_queue = result_queue
        self.active_workers: dict[str, multiprocessing.Process] = {}
        self.active_async_workers: dict[str, 'AsyncBaseBee'] = {}  # Track async workers separately
        self.worker_futures: dict[str, dict[str, Any]] = {}
        self.bee_registry: dict[str, type[BaseBee]] = {}
        self.MAX_ACTIVE_WORKERS = get_settings().MAX_WORKERS
        # BeeFactory 处理 dna.role / dna.species -> Bee 类路由
        self._bee_factory = BeeFactory()

        # Phase 3 P3: 注册到蜂群通信，订阅淘汰警报
        self._swarm: Optional[Any] = None
        self._auto_respawn = True  # 被淘汰后是否自动补充
        self._register_swarm()

    def register_species(self, species_dict: dict[str, type[BaseBee]]) -> None:
        """注册蜜蜂种类（同时更新 WorkerManager 注册表和 BeeFactory）"""
        self.bee_registry = {k.lower(): v for k, v in species_dict.items()}
        self._bee_factory.register_from_species(species_dict)
        logger.info(f"[WorkerManager] 注册 {len(self.bee_registry)} 个蜂种")

    def spawn_worker(self, dna: BeeDNA) -> str | None:
        """启动 Worker 进程或异步 Worker

        Routing logic (via BeeFactory):
        - dna.role == "async": Creates AsyncBaseBee instance (runs in current process via asyncio)
        - Otherwise: Creates BaseBee as multiprocessing.Process (spawns new process)
        """
        if not self.has_available_slot():
            logger.warning(f"[WorkerManager] 无可用槽位 ({len(self.active_workers)}/{self.MAX_ACTIVE_WORKERS})")
            return None

        worker_id = f"Worker-{dna.id}"

        try:
            bee = self._bee_factory.create_bee(dna, worker_id, self.result_queue)
            bee_class = type(bee)

            # AsyncBaseBee doesn't inherit from multiprocessing.Process - it runs in main process
            if bee_class is AsyncBaseBee or issubclass(bee_class, AsyncBaseBee):
                self.active_async_workers[worker_id] = bee
                self._track_worker_start(worker_id, dna, bee_class)
                logger.info(f"[WorkerManager] 启动异步 Worker {worker_id} ({bee_class.__name__})")
            else:
                bee.start()
                self.active_workers[worker_id] = bee
                self._track_worker_start(worker_id, dna, bee_class)
                logger.info(f"[WorkerManager] 启动进程 Worker {worker_id} ({bee_class.__name__})")

            return worker_id
        except Exception as e:
            logger.error(f"[WorkerManager] 启动失败 {worker_id}: {e}")
            return None

    def _spawn_process_worker(self, dna: BeeDNA, worker_id: str, bee_class: type) -> str | None:
        """Spawn a multiprocessing.Process-based worker"""
        try:
            process = bee_class(dna, worker_id, self.result_queue)
            process.start()
            self.active_workers[worker_id] = process
            self._track_worker_start(worker_id, dna, bee_class)
            logger.info(f"[WorkerManager] 启动进程 Worker {worker_id} ({bee_class.__name__})")
            return worker_id
        except Exception as e:
            logger.error(f"[WorkerManager] 进程启动失败 {worker_id}: {e}")
            return None

    def _spawn_async_worker(self, dna: BeeDNA, worker_id: str, bee_class: type) -> str | None:
        """Spawn an asyncio-based async worker (runs in current process)

        AsyncBaseBee doesn't inherit from multiprocessing.Process - it runs
        its event loop in the current process and uses ProcessPoolExecutor
        for CPU-bound work.
        """
        try:
            # AsyncBaseBee runs in the main process via asyncio
            # We still track it for lifecycle management
            async_worker = bee_class(dna, worker_id, self.result_queue)
            self.active_async_workers[worker_id] = async_worker
            self._track_worker_start(worker_id, dna, bee_class)
            logger.info(f"[WorkerManager] 启动异步 Worker {worker_id} ({bee_class.__name__})")
            return worker_id
        except Exception as e:
            logger.error(f"[WorkerManager] 异步 Worker 启动失败 {worker_id}: {e}")
            return None

    def _determine_bee_class(self, dna: BeeDNA) -> type[BaseBee]:
        """确定蜜蜂类（委托给 BeeFactory 处理）

        Routing logic (BeeFactory):
        1. role == "async" -> AsyncBaseBee
        2. species -> BEE_SPECIES 查找 (case-insensitive)
        3. mutation_gene -> bee_registry 查找 (case-insensitive)
        4. role -> bee_registry 查找 (case-insensitive)
        5. fallback -> BaseBee
        """
        return self._bee_factory.determine_bee_class(dna)

    def _track_worker_start(self, worker_id: str, dna: BeeDNA, bee_class: type[BaseBee]) -> None:
        """跟踪 Worker 启动"""
        self.worker_futures[worker_id] = {
            "dna": dna,
            "bee_class": bee_class.__name__,
            "start_time": time.time(),
            "status": "running"
        }

    def has_available_slot(self) -> bool:
        """检查是否有可用槽位

        Note: AsyncBaseBee workers don't consume process slots since they run
        in the main process, but they do consume the MAX_ACTIVE_WORKERS limit
        for consistency with the queuing system.
        """
        total_workers = len(self.active_workers) + len(self.active_async_workers)
        return total_workers < self.MAX_ACTIVE_WORKERS

    def available_slots(self) -> int:
        """获取可用槽位数量"""
        total_workers = len(self.active_workers) + len(self.active_async_workers)
        return max(0, self.MAX_ACTIVE_WORKERS - total_workers)

    def cleanup_dead_workers(self) -> int:
        """清理死亡进程（并行 join，避免串行阻塞 n 秒）"""
        import threading
        # 快照进程引用，避免迭代过程中字典被修改导致 KeyError
        dead_items = [(wid, proc) for wid, proc in self.active_workers.items() if not proc.is_alive()]

        if not dead_items:
            return 0

        results = {}

        def _join_one(wid, proc, results_dict):
            """在独立线程中 join 单个进程"""
            proc.join(timeout=1)
            exit_code = proc.exitcode
            if exit_code is None:
                logger.warning(f"[WorkerManager] 强制终止 {wid}")
                try:
                    proc.terminate()
                    proc.join(timeout=1)
                except (OSError, PermissionError):
                    pass
                exit_code = proc.exitcode
            results_dict[wid] = exit_code

        threads = []
        for wid, proc in dead_items:
            t = threading.Thread(target=_join_one, args=(wid, proc, results))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # 在主线程中安全地更新状态和删除（持有 GIL，串行执行安全）
        for wid, _ in dead_items:
            self._update_worker_end_status(wid)
            self.active_workers.pop(wid, None)

        return len(dead_items)

    def _update_worker_end_status(self, worker_id: str) -> None:
        """更新 Worker 结束状态，并清理过期记录（保留最近1000条）"""
        if worker_id in self.worker_futures:
            self.worker_futures[worker_id]["status"] = "completed"
            self.worker_futures[worker_id]["end_time"] = time.time()
            duration = self.worker_futures[worker_id]["end_time"] - self.worker_futures[worker_id]["start_time"]
            logger.debug(f"[WorkerManager] Worker {worker_id} 已结束 ({duration:.2f}s)")

        # 防止 worker_futures 无界增长，保留最近 1000 条已完成记录
        completed = [k for k, v in self.worker_futures.items() if v.get("status") == "completed"]
        if len(completed) > 1000:
            for old_id in completed[:-1000]:
                del self.worker_futures[old_id]

    def shutdown(self, timeout: float = 30) -> int:
        """优雅关闭所有 Worker"""
        logger.info(f"[WorkerManager] 正在关闭 {len(self.active_workers)} 个 Worker...")

        terminated = 0
        per_worker_timeout = max(1.0, timeout / max(len(self.active_workers), 1))
        for wid, proc in list(self.active_workers.items()):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=per_worker_timeout)
                if proc.is_alive():
                    logger.warning(f"[WorkerManager] Worker {wid} 未响应 terminate，强制 kill")
                    proc.kill()
                    proc.join(timeout=2)
                terminated += 1

        self.active_workers.clear()
        return terminated

    # ===== Phase 3 P3: 自然选择机制 =====

    def _register_swarm(self):
        """Phase 3 P3: 注册到蜂群通信，订阅淘汰警报"""
        try:
            from hive.core.swarm_communication import get_swarm_communication, MessageType, SwarmMessage
            self._swarm = get_swarm_communication()
            self._swarm.register_bee("WorkerManager", self)
            self._swarm.subscribe(MessageType.ALERT, self._on_swarm_alert)
            logger.info("🔪 [WorkerManager] 已注册到蜂群网络，订阅淘汰警报")
        except ImportError:
            logger.warning("🔪 [WorkerManager] SwarmCommunication 未安装，自然选择机制不可用")

    def _on_swarm_alert(self, message: Any):
        """Phase 3 P3: 处理蜂群警报（淘汰通知）"""
        try:
            content = message.content if hasattr(message, 'content') else {}
            topic = content.get("topic", "")
            if topic == "bee_culled":
                bee_id = content.get("bee_id")
                bee_type = content.get("bee_type", "unknown")
                reason = content.get("reason", "unknown")
                logger.info(f"🔪 [WorkerManager] 收到淘汰警报: {bee_id} ({bee_type}) reason={reason}")
                terminated = self.terminate_worker(bee_id)
                if terminated:
                    logger.info(f"🔪 [WorkerManager] 已执行自然淘汰: {bee_id}")
                    # 可选：自动补充被淘汰的蜂
                    if self._auto_respawn and bee_type:
                        self._respawn_worker(bee_type)
                else:
                    logger.debug(f"🔪 [WorkerManager] {bee_id} 不在活动worker列表中")
        except Exception as e:
            logger.error(f"🔪 [WorkerManager] 处理淘汰警报失败: {e}")

    def terminate_worker(self, bee_id: str) -> bool:
        """
        Phase 3 P3: 根据 bee_id 终止对应 Worker

        Args:
            bee_id: 要终止的蜂 ID (即 WorkerManager 中的 worker_id)

        Returns:
            True 如果成功终止，False 如果未找到
        """
        # bee_id 就是 worker_id (格式: Worker-{dna.id} 或 DarwinBee.id)
        # 先在进程 workers 中查找
        if bee_id in self.active_workers:
            proc = self.active_workers[bee_id]
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2.0)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=1.0)
                    logger.info(f"🔪 [WorkerManager] 已终止进程 Worker: {bee_id}")
                else:
                    logger.debug(f"🔪 [WorkerManager] Worker {bee_id} 已经结束")
            except Exception as e:
                logger.error(f"🔪 [WorkerManager] 终止 {bee_id} 失败: {e}")
            finally:
                self.active_workers.pop(bee_id, None)
                self._update_worker_end_status(bee_id)
                return True

        # 再在异步 workers 中查找
        if bee_id in self.active_async_workers:
            del self.active_async_workers[bee_id]
            self._update_worker_end_status(bee_id)
            logger.info(f"🔪 [WorkerManager] 已移除异步 Worker: {bee_id}")
            return True

        return False

    def _respawn_worker(self, bee_type: str):
        """
        Phase 3 P3: 自动补充被淘汰的蜂

        Args:
            bee_type: 蜂种类型名 (如 "DarwinBee", "SentinelBee")
        """
        if not self.has_available_slot():
            logger.warning(f"🔪 [WorkerManager] 无可用槽位，无法补充 {bee_type}")
            return

        try:
            from hive.core.dna import BeeDNA
            dna = BeeDNA(id=f"{bee_type}_respawn_{int(time.time())}", role=bee_type.lower(), species=bee_type.lower())
            new_id = self.spawn_worker(dna)
            if new_id:
                logger.info(f"🔪 [WorkerManager] 已补充 {bee_type} -> {new_id}")
            else:
                logger.warning(f"🔪 [WorkerManager] 补充 {bee_type} 失败")
        except Exception as e:
            logger.error(f"🔪 [WorkerManager] 补充 Worker 异常: {e}")

    def get_status(self) -> dict[str, Any]:
        """获取管理器状态"""
        return {
            "active_workers": len(self.active_workers),
            "active_async_workers": len(self.active_async_workers),
            "worker_futures": len(self.worker_futures),
            "available_slots": self.available_slots()
        }
