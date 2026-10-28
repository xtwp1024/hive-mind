#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive Queen (蜂后) - 蜂巢核心统筹者

重构为门面模式，委托给专业组件处理。
"""

import logging
import multiprocessing
import os
import queue
from typing import Any, List, Dict, Optional

from hive.core.dna import BeeDNA
from hive.core.interfaces import QueenInterface, Task, TaskResult
from hive.core.deployer import SwarmDeployer
from hive.core.service_locator import BeeServiceLocator
from hive.utils.logger import get_logger

logger = get_logger("Hive.Queen")


class QueenBee(QueenInterface):
    """
    女王蜂（蜂群的神经网络中枢）。
    经过 alibaba/BeeHive 的超级解耦治疗，她现在不再自己大包大揽装配外挂件，
    而是一个极度轻量化的大脑：纯粹负责事件调度并通过 ServiceLocator 调用插件能力。

    注意：QueenBee 使用进程感知单例模式，确保每个进程只有一个女王蜂实例。
    """

    _instance: Optional['QueenBee'] = None
    _instance_pid: int = 0

    def __new__(cls, result_queue: multiprocessing.Queue):
        current_pid = os.getpid()
        # 如果当前进程已有实例，直接返回（进程隔离）
        if cls._instance is not None and cls._instance_pid == current_pid:
            return cls._instance
        # 否则创建新实例（可能是新进程或首次创建）
        instance = super().__new__(cls)
        cls._instance = instance
        cls._instance_pid = current_pid
        return instance

    def __init__(self, result_queue: multiprocessing.Queue):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        super().__init__(result_queue)

        self.result_queue = result_queue

        self.dna_queue = multiprocessing.Queue()

        # 将她核心依赖挂入解耦池，由于 WorkerManager 是紧耦合在队列上的，我们这里将它暂作核心件注册
        from hive.core.worker_manager import WorkerManager
        wm = WorkerManager(result_queue)
        BeeServiceLocator.register_service("WorkerManager", wm)

        # 大解耦手术：她身上再也不挂载那些从全球各大体系吸星而来的实体实例了！
        # 她只需要随时通过 BeeServiceLocator 拉取接口即可使用！
        self.deployer = SwarmDeployer()
        self.bee_registry = wm.bee_registry

        logger.info(f"👑 QueenBee 已诞生 (门面模式) - [模块级全解耦形态]")

    def deploy_swarm(self, colony_dna: list[BeeDNA]) -> None:
        """部署蜂群 (V1)"""
        if not colony_dna:
            logger.info("👑 无 DNA 指令，蜂群待命")
            return

        worker_manager = BeeServiceLocator.get_service("WorkerManager")
        if not worker_manager:
            logger.error("🛑 系统严重脱落！无法找到 worker_manager 模块！")
            return

        available = worker_manager.available_slots()
        if available <= 0:
            logger.warning(f"👑 蜂群容量已满，无法部署 {len(colony_dna)} 只工蜂！")
            return

        if len(colony_dna) > available:
            logger.warning(f"👑 仅能部署 {available}/{len(colony_dna)} 只工蜂，容量不足")

        to_deploy = colony_dna[:available]
        logger.info(f"👑 蜂后正在部署 {len(to_deploy)} 只工蜂...")

        spawned = 0
        for dna in to_deploy:
            if worker_manager.spawn_worker(dna):
                spawned += 1
        logger.info(f"👑 成功部署 {spawned}/{len(to_deploy)} 只工蜂")

    async def deploy_swarm_v2(self, colony_dna: list[BeeDNA]) -> list[dict[str, Any]]:
        """异步部署蜂群 (V2)"""
        return await self.deployer.deploy_async(colony_dna)

    def consume_honey(self) -> None:
        """吞噬融合 - 收集工蜂返回的结果"""
        # Fetch necessary detached services dynamically
        worker_manager = BeeServiceLocator.get_service("WorkerManager")
        if not worker_manager:
            logger.error("🛑 系统严重脱落！无法找到 worker_manager 模块！")
            return

        honey_processor = BeeServiceLocator.get_service("HoneyProcessor")
        if not honey_processor:
            from hive.core.honey_processor import HoneyProcessor
            honey_processor = HoneyProcessor()
            BeeServiceLocator.register_service("HoneyProcessor", honey_processor)

        payloads = []
        MAX_BATCH = 500  # 防止单次 consume 无限排空队列
        while len(payloads) < MAX_BATCH:
            try:
                payloads.append(self.result_queue.get_nowait())
            except queue.Empty:
                break

        if not payloads:
            return

        logger.info(f"==== 👑 女王蜂开始收割战果 (共有 {len(payloads)} 滴花蜜) ====")
        offspring = honey_processor.process_batch(payloads)

        # 路由逻辑：遍历原始 payloads，处理特殊信号（SPAWNED / MapReduce）
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            status = payload.get("status")
            worker_id = payload.get("source", "unknown_worker")

            if status == "SPAWNED":
                req = payload.get("spawn_request", {})
                child_meta = req.get("child_dna_metadata", {})
                target_species = req.get("target_species", "Standard")
                logger.info(f"🧬 [HiVA] 侦测到工蜂 {worker_id} 的超维裂殖动作！正在繁育其子代任务...")
                child_dna = BeeDNA(
                    species=target_species,
                    target_name=req.get("sub_task", "UNKNOWN EXTRAPOLATION"),
                    metadata=child_meta,
                )
                worker_manager.spawn_worker(child_dna)
                continue

            meta = payload.get("dna", {}).get("metadata", {}) if isinstance(payload.get("dna"), dict) else {}
            if "mr_job_id" in meta:
                mr_id = meta["mr_job_id"]
                mr_dispatcher = BeeServiceLocator.get_service("MapReduceDispatcher")
                if not mr_dispatcher:
                    from hive.core.mapreduce import MapReduceDispatcher
                    mr_dispatcher = MapReduceDispatcher(self)
                    BeeServiceLocator.register_service("MapReduceDispatcher", mr_dispatcher)
                reduce_result = mr_dispatcher.handle_map_result(mr_id, payload)
                if reduce_result:
                    if reduce_result.get("action_type") == "consensus":
                        from hive.core.consensus import ConsensusEngine
                        verdict = ConsensusEngine.process_node_payloads(reduce_result.get("raw_results", []))
                        logger.info(f"⚖️ 共识法庭最后裁决: {verdict['status']}")
                    else:
                        logger.info(f"🎉 大兵团战役 MR-{mr_id} 高阶报告已合魂生成！")
                continue

        # 进化繁育：对成功结果的 DNA 进行下一代繁殖
        if offspring:
            logger.info(f"🍯 蜂后吞噬了 {len(offspring)} 份花蜜，诞生 {len(offspring)} 只新工蜂")
            ev_engine = BeeServiceLocator.get_service("EvolutionEngine")
            if not ev_engine:
                from hive.core.evolution_engine import EvolutionEngine
                ev_engine = EvolutionEngine()
                BeeServiceLocator.register_service("EvolutionEngine", ev_engine)
            spawned = 0
            failed = 0
            for dna_dict in offspring:
                if isinstance(dna_dict, dict):
                    try:
                        child = ev_engine.reproduce(dna_dict)
                        if child:
                            if worker_manager.spawn_worker(child):
                                spawned += 1
                            else:
                                failed += 1
                    except Exception as e:
                        logger.warning(f"🐝 繁殖失败: {e}")
                        failed += 1
            if spawned > 0 or failed > 0:
                logger.info(f"🐝 成功诞生 {spawned} 只新工蜂，失败 {failed} 只")

        worker_manager.cleanup_dead_workers()

    def process_results_v2(self, results: list[dict[str, Any]]) -> list[BeeDNA]:
        """处理 V2 结果并生成下一代 DNA"""
        ev_engine = BeeServiceLocator.get_service("EvolutionEngine")
        if not ev_engine:
            from hive.core.evolution_engine import EvolutionEngine
            ev_engine = EvolutionEngine()
            BeeServiceLocator.register_service("EvolutionEngine", ev_engine)
        return ev_engine.evolve_batch(results)

    def trigger_immune_response(self, bad_honey: dict[str, Any]) -> None:
        """触发免疫响应"""
        from hive.core.service_locator import BeeServiceLocator
        hp = BeeServiceLocator.get_service("HoneyProcessor")
        if hp:
            hp.trigger_immune_response(bad_honey)

    async def coordinate_ministers(self, task: Task) -> TaskResult:
        """协调部长执行任务"""
        logger.info(f"👑 [{self.name}] 收到全局任务: {task.type}")
        if task.type == "workflow":
            return TaskResult(task_id=task.id, success=True, result={"status": "Dispatcher Handled"})
        return TaskResult(task_id=task.id, success=True, result={"status": "Acknowledged"})

    def evolve_species(self) -> bool:
        """进化物种"""
        logger.info(f"👑 [{self.name}] 正在执行物种进化/繁衍评估...")
        return True

    def get_hive_status(self) -> dict[str, Any]:
        """获取蜂巢状态"""
        wm = BeeServiceLocator.get_service("WorkerManager")
        return {
            "deployer": self.deployer.get_status(),
            "worker_manager": wm.get_status() if wm else {},
            "evolution": {},
            "processor": {}
        }

    def get_evolution_status(self) -> dict[str, Any]:
        """获取进化状态"""
        ev_engine = BeeServiceLocator.get_service("EvolutionEngine")
        if ev_engine and hasattr(ev_engine, "get_status"):
            return ev_engine.get_status()
        return {"status": "unavailable"}

    def get_colony_status(self) -> dict[str, Any]:
        """获取蜂群状态"""
        wm = BeeServiceLocator.get_service("WorkerManager")
        if wm:
            return wm.get_status()
        return {"active_workers": 0, "honey_collected": 0}

    def shutdown(self, wait: bool = True, timeout: float = 30) -> None:
        """优雅关闭"""
        logger.info("🛑 正在关闭蜂群...")
        worker_manager = BeeServiceLocator.get_service("WorkerManager")
        if worker_manager:
            worker_manager.shutdown(timeout=timeout)
        logger.info("✅ 所有工蜂已优雅关闭")
