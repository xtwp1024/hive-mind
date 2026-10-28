# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
from typing import Dict, List
import random

logger = get_logger(__name__)

class SwarmOrchestratorBee(BaseBee):
    """蜂群协调蜂 - 协调多个蜜蜂、任务分配、负载均衡"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.swarm_members = {}
        self.task_queue = []
        self.assignments = {}
    
    def specialized_task(self, target_path):
        """执行协调任务"""
        logger.info(f"[SwarmOrchestratorBee-{self.id}] 协调: {target_path}")
        
        # 分配任务
        assignment = self._assign_task(target_path)
        
        # 负载均衡
        balance = self._balance_load()
        
        # 进度跟踪
        progress = self._track_progress()
        
        logger.info(f"[SwarmOrchestratorBee-{self.id}] 协调完成")
    
    def _assign_task(self, task: str) -> Dict:
        """分配任务"""
        # 获取可用蜜蜂
        available_bees = list(self.swarm_members.keys())
        
        if not available_bees:
            logger.warning(f"[SwarmOrchestratorBee-{self.id}] 没有可用的蜜蜂")
            return {"status": "no_bees"}
        
        # 选择最优蜜蜂
        best_bee = random.choice(available_bees)
        
        assignment = {
            "task": task,
            "assigned_to": best_bee,
            "status": "assigned",
            "timestamp": "assign_timestamp"
        }
        
        self.assignments[task] = assignment
        return assignment
    
    def _balance_load(self) -> Dict:
        """负载均衡"""
        load_info = {
            "total_bees": len(self.swarm_members),
            "active_tasks": len(self.assignments),
            "load_distribution": {}
        }
        
        for bee_id in self.swarm_members:
            task_count = sum(1 for a in self.assignments.values() if a["assigned_to"] == bee_id)
            load_info["load_distribution"][bee_id] = task_count
        
        return load_info
    
    def _track_progress(self) -> Dict:
        """跟踪进度"""
        progress = {
            "total_tasks": len(self.task_queue),
            "completed_tasks": sum(1 for a in self.assignments.values() if a["status"] == "completed"),
            "in_progress_tasks": sum(1 for a in self.assignments.values() if a["status"] == "assigned"),
            "pending_tasks": len(self.task_queue) - len(self.assignments)
        }
        
        return progress
    
    def register_bee(self, bee_id: str, bee_type: str):
        """注册蜜蜂"""
        self.swarm_members[bee_id] = {
            "type": bee_type,
            "status": "idle",
            "tasks_completed": 0
        }
        logger.info(f"[SwarmOrchestratorBee-{self.id}] 注册蜜蜂: {bee_id} ({bee_type})")
    
    def add_task(self, task: str):
        """添加任务"""
        self.task_queue.append(task)
        logger.debug(f"[SwarmOrchestratorBee-{self.id}] 添加任务: {task}")
    
    def update_task_status(self, task: str, status: str):
        """更新任务状态"""
        if task in self.assignments:
            self.assignments[task]["status"] = status
    
    def devour(self) -> dict:
        """消耗目标并返回协调报告"""
        target_path = self.dna.id
        
        try:
            self.specialized_task(target_path)
            
            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "swarm_size": len(self.swarm_members),
                "tasks_assigned": len(self.assignments)
            }
        except Exception as e:
            logger.error(f"[SwarmOrchestratorBee-{self.id}] 协调失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e)
            }
