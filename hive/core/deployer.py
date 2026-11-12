# -*- coding: utf-8 -*-
"""
SwarmDeployer - 任务分发器

专注负责 V1/V2 蜂群任务的创建和分发。
"""

import asyncio
import logging
from typing import Any

from hive.core.dispatcher import HiveDispatcher, TaskNode
from hive.core.dna import BeeDNA

logger = logging.getLogger("Hive.Deployer")


class SwarmDeployer:
    """蜂群任务分发器"""

    DEPLOY_INTERVAL = 0.1

    def __init__(self, dispatcher: HiveDispatcher | None = None):
        self.dispatcher = dispatcher or HiveDispatcher(actors=[])
        self.deployed_count = 0

    def deploy(self, dna_list: list[BeeDNA]) -> list[BeeDNA]:
        """V1 同步部署"""
        if not dna_list:
            logger.info("[Deployer] 无 DNA 指令，蜂群待命")
            return []

        logger.info(f"[Deployer] 同步部署 {len(dna_list)} 只工蜂...")
        self.deployed_count += len(dna_list)
        return dna_list

    async def deploy_async(self, dna_list: list[BeeDNA]) -> list[dict[str, Any]]:
        """V2 异步部署"""
        if not dna_list:
            return []

        if not self.dispatcher.actors:
            logger.warning(
                "[Deployer] 无已注册 Actor，异步部署将静默失败，请先调用 register_actor()"
            )
            return []

        logger.info(f"[Deployer] 异步调度 {len(dna_list)} 个任务...")

        task_nodes = [self._create_task_node(dna) for dna in dna_list]
        tasks = [self.dispatcher.execute_task(node) for node in task_nodes]

        await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for node in task_nodes:
            if node.result:
                results.append(node.result)
            elif node.error:
                results.append({"status": "FAILED", "error": node.error, "id": node.id})

        logger.info(f"[Deployer] 完成调度，收集到 {len(results)} 个结果")
        return results

    def _create_task_node(self, dna: BeeDNA) -> TaskNode:
        """创建任务节点"""
        caps = ["legacy_task_execution"]

        task_node = TaskNode(
            id=f"Task_{dna.id}",
            action="execute_dna",
            params={
                "dna_dict": dna.__dict__,
                "target_url": dna.target_url,
                "target_name": dna.target_name,
            },
            required_capabilities=caps,
        )

        task_node.required_capabilities.append(dna.role or "worker")
        return task_node

    def register_actor(self, actor: Any) -> None:
        """注册 Actor 到调度器"""
        with self.dispatcher._actors_lock:
            if actor not in self.dispatcher.actors:
                self.dispatcher.actors.append(actor)
                logger.info(f"[Deployer] 注册 Actor: {actor.name}")

    def get_status(self) -> dict[str, Any]:
        """获取分发器状态"""
        return {
            "deployed_count": self.deployed_count,
            "registered_actors": len(self.dispatcher.actors),
        }
