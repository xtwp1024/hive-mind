#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蜂巢预热系统 - 确保启动时所有蜂种都存在并在工作
"""

import time
import logging
from typing import List
from hive.core.dna import BeeDNA

logger = logging.getLogger("Hive.Warmup")


class HiveWarmup:
    """
    蜂巢预热器
    
    职责:
    1. generate_full_swarm: 为每种蜂种生成至少一个DNA
    2. create_idle_tasks: 创建待命任务，让蜂种"存在"但不需要联网
    """
    
    def __init__(self):
        from hive.bees import get_valid_species
        self.valid_species = get_valid_species()
        self.species_count = len(self.valid_species)
        logger.info(f"🔥 预热器加载了 {self.species_count} 种蜂种")
    
    def generate_full_swarm(self) -> List[BeeDNA]:
        """
        生成完整的蜂群DNA，确保每种蜂种至少有一个实例
        
        Returns:
            DNA列表，包含所有蜂种
        """
        dna_batch = []
        timestamp = int(time.time())
        
        for i, (species_name, bee_class) in enumerate(self.valid_species.items()):
            # 为每种蜂种生成一个DNA
            # 使用待命任务，让蜂种启动但不执行耗时的联网操作
            dna = BeeDNA(
                species="bee",
                role=species_name.lower(),  # 使用蜂种名称作为role
                id=f"Warmup_{species_name}_{timestamp}_{i}",
                target_name=f"Warmup_Task_{species_name}",
                target_url="",  # 空URL，表示待命状态
                mutation_gene=species_name.lower(),  # 强制使用该蜂种
                can_merge_back=False,  # 预热任务不需要导回
                metadata={
                    "mission": "warmup_idle",
                    "description": f"预热任务: {species_name} 蜂种就绪",
                    "idle_task": True
                }
            )
            dna_batch.append(dna)
        
        logger.info(f"🔥 [预热] 生成了 {len(dna_batch)} 个DNA (所有蜂种)")
        return dna_batch
    
    def create_active_swarm(self, capacity: int = 30) -> List[BeeDNA]:
        """
        创建活跃蜂群，混合预热任务和实际任务
        
        Args:
            capacity: 总容量
        
        Returns:
            DNA列表，包含预热任务和实际任务
        """
        # 1. 先确保所有蜂种都有预热任务
        warmup_dnas = self.generate_full_swarm()
        
        # 2. 计算剩余容量
        remaining_capacity = max(0, capacity - len(warmup_dnas))
        
        if remaining_capacity == 0:
            logger.info(f"🔥 [预热] 预热任务已占满全部容量 ({capacity})")
            return warmup_dnas
        
        # 3. 生成实际任务（如果有剩余容量）
        actual_dnas = []
        if remaining_capacity > 0:
            from hive.ministers.engineer import get_engineer
            engineer = get_engineer()
            actual_dnas = engineer.assign_tasks(capacity=remaining_capacity)
        
        # 4. 合并：预热任务在前，实际任务在后
        full_dna_batch = warmup_dnas + actual_dnas
        
        logger.info(f"🔥 [预热] 混合部署: {len(warmup_dnas)} 预热 + {len(actual_dnas)} 实际 = {len(full_dna_batch)} 总计")
        return full_dna_batch
    
    def get_species_status(self) -> dict:
        """获取蜂种状态"""
        return {
            "total_species": self.species_count,
            "species_names": list(self.valid_species.keys())
        }


# 单例
_warmup_instance = None
_warmup_lock = __import__('threading').Lock()

def get_warmup() -> HiveWarmup:
    """获取预热器单例"""
    global _warmup_instance
    if _warmup_instance is None:
        with _warmup_lock:
            if _warmup_instance is None:
                _warmup_instance = HiveWarmup()
    return _warmup_instance
