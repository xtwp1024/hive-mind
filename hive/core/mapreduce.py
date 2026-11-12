#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MapReduce Dispatcher (兵团级分发归并引擎)
灵感来源于大数据的 Apache Hive。
- Map 阶段: 将一个具有并列性的巨大任务或者集合切分为多个 BeeDNA 分散下发给多个工蜂同时解算。
- Reduce 阶段: 当前一组的多个工蜂完成并存回结果后，启动收尾逻辑，自动合并精华情报为一份统一战报。
"""

import logging
import threading
import uuid
from typing import Any, Dict, List

from hive.core.dna import BeeDNA

logger = logging.getLogger("Hive.MapReduce")


class MapReduceDispatcher:
    def __init__(self, queen_instance):
        self.queen = queen_instance
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit_map_job(
        self,
        base_task: str,
        sub_items: List[Any],
        target_species: str = "Standard",
        reduce_action="concat",
    ):
        """
        发起大兵团作业。例如: base_task='分析以下股票趋势', sub_items=['AAPL', 'GOOG', 'TSLA']
        它会生出 3 个相互平行的 DNA 丢给 Queen 孵化工蜂。
        """
        job_id = f"mrjob-{uuid.uuid4().hex[:6]}"
        total_items = len(sub_items)

        with self._lock:
            self.active_jobs[job_id] = {
                "base_task": base_task,
                "total_maps": total_items,
                "completed_maps": 0,
                "results": [],
                "reduce_action": reduce_action,
                "status": "MAPPING",
            }

        logger.info(
            f"🚀 [MapReduce] 新战役 {job_id} 打响！将切分为 {total_items} 条并行战线，调用 {target_species}。"
        )

        from hive.core.service_locator import BeeServiceLocator

        wm = BeeServiceLocator.get_service("WorkerManager")

        for item in sub_items:
            dna = BeeDNA(
                species=target_species,
                target_name=f"{base_task} => {item}",
                metadata={"mr_job_id": job_id, "item": item},
            )
            # 通过 ServiceLocator 立刻投料孵化
            if wm:
                wm.spawn_worker(dna)

        return job_id

    def submit_consensus_job(
        self, critical_task: str, redundant_nodes: int = 3, target_species: str = "Standard"
    ):
        """
        [Ethereum Hive 灵感] 发起防篡改/防幻觉的共识测算任务。
        系统将一模一样的高危任务直接发给复数个独立节点。
        """
        job_id = f"conjob-{uuid.uuid4().hex[:6]}"
        with self._lock:
            self.active_jobs[job_id] = {
                "base_task": critical_task,
                "total_maps": redundant_nodes,
                "completed_maps": 0,
                "results": [],
                "reduce_action": "consensus",
                "status": "CONSENSUS_RUNNING",
            }

        logger.warning(
            f"🛡️ [Consensus Network] 高级状态验证网已启动！已将 {critical_task[:20]}... 任务镜像分发给 {redundant_nodes} 个独立黑盒环境。"
        )

        from hive.core.service_locator import BeeServiceLocator

        wm = BeeServiceLocator.get_service("WorkerManager")

        for i in range(redundant_nodes):
            dna = BeeDNA(
                species=target_species,
                target_name=critical_task,
                metadata={"mr_job_id": job_id, "node_index": i},
            )
            if wm:
                wm.spawn_worker(dna)

        return job_id

    def handle_map_result(self, job_id: str, result_payload: dict):
        """由 QueenBee 的回收流水线回调，收集每次 Map 的结果"""
        with self._lock:
            if job_id not in self.active_jobs:
                return None

            job = self.active_jobs[job_id]
            if job.get("status") in ("REDUCING", "DONE"):
                return None

            job["results"].append(result_payload)
            job["completed_maps"] += 1

            logger.debug(
                f"📥 [MapReduce] {job_id} 进度: {job['completed_maps']} / {job['total_maps']}"
            )

            if job["completed_maps"] < job["total_maps"]:
                return None

            # 标记为 REDUCING 并取出 job 数据，锁外执行 reduce
            job["status"] = "REDUCING"
            job_snapshot = self.active_jobs.pop(job_id)

        # 锁外执行 reduce，避免长时间持锁阻塞其他结果回调
        return self._execute_reduce(job_id, job_snapshot)

    def _execute_reduce(self, job_id: str, job: dict):
        """触发 Reduce 收割合并步骤（在锁外执行）"""
        results = job["results"]
        action = job["reduce_action"]

        logger.info(
            f"🔗 [MapReduce] 战役 {job_id} 收网 (Reduce动作: {action})。总计归拢战报 {len(results)} 份。"
        )

        final_output = ""
        if action == "concat":
            # 将所有工蜂带回的信息素状态强行拼接
            str_list = [str(r.get("status", "")) for r in results]
            final_output = "\n=== 分割线 ===\n".join(str_list)
        elif action == "summarize":
            final_output = f"[系统摘要合成] 总计收回 {len(results)} 条结果。"
        elif action == "consensus":
            # 如果是共识测试任务，则不在这里处理拼凑，而是原样暴露出 raw_results 给 Queen 交由 Consensus引擎处决
            return {
                "status": "MR_COMPLETED",
                "mr_job_id": job_id,
                "action_type": "consensus",
                "base_task": job["base_task"],
                "raw_results": results,
            }
        else:
            final_output = f"[未定义 Reduce 操作] Collected {len(results)} items"

        return {
            "status": "MR_COMPLETED",
            "mr_job_id": job_id,
            "base_task": job["base_task"],
            "reduced_result": final_output,
        }
