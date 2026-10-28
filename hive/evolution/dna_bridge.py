#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNA Bridge - 进化系统与 Hive DNA 系统的集成桥接

将 GEP 协议的进化成果应用到 Hive 的 DNA 系统中。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Hive.Evolution.DNABridge")


class DNABridge:
    """
    DNA 桥接器
    
    负责将进化引擎产生的基因和胶囊转化为 Hive DNA 系统可用的形式。
    """
    
    def __init__(self):
        self._dna_pool = None
        self._evolution_engine = None
    
    @property
    def dna_pool(self):
        """延迟加载 DNA 池"""
        if self._dna_pool is None:
            try:
                from hive.core.dna import DNAPool
                self._dna_pool = DNAPool()
            except ImportError:
                logger.warning("无法导入 DNAPool，DNA 集成功能受限")
        return self._dna_pool
    
    @property
    def evolution_engine(self):
        """延迟加载进化引擎"""
        if self._evolution_engine is None:
            try:
                from hive.core.evolution_engine import EvolutionEngine
                self._evolution_engine = EvolutionEngine()
            except ImportError:
                logger.warning("无法导入 EvolutionEngine")
        return self._evolution_engine
    
    def apply_gene_to_dna(self, gene: dict, target_species: str = "bee") -> bool:
        """
        将基因应用到 DNA 系统
        
        Args:
            gene: GEP 基因数据
            target_species: 目标物种
            
        Returns:
            是否成功应用
        """
        if not self.dna_pool:
            logger.error("DNA 池不可用")
            return False
        
        try:
            from hive.core.dna import DNA
            
            # 从基因中提取能力调整
            adjustments = gene.get("capability_adjustments", {})
            
            # 获取当前最佳 DNA
            best_dnas = self.dna_pool.get_best(target_species, count=1)
            
            if best_dnas:
                base_dna = best_dnas[0]
            else:
                # 创建新的基础 DNA
                base_dna = DNA(species=target_species)
            
            # 应用能力调整
            if "immune_response" in adjustments:
                base_dna.immune_response = min(1.0, base_dna.immune_response + adjustments["immune_response"])
            if "learning_rate" in adjustments:
                base_dna.learning_rate = min(1.0, base_dna.learning_rate + adjustments["learning_rate"])
            if "mutation_rate" in adjustments:
                base_dna.mutation_rate = max(0.01, min(0.5, adjustments["mutation_rate"]))
            
            # 添加基因元数据
            base_dna.metadata["applied_genes"] = base_dna.metadata.get("applied_genes", [])
            base_dna.metadata["applied_genes"].append(gene.get("id", "unknown"))
            
            # 注册到 DNA 池
            self.dna_pool.register(base_dna)
            
            logger.info(f"✅ 基因已应用到 {target_species} DNA")
            return True
            
        except Exception as e:
            logger.error(f"应用基因失败: {e}")
            return False
    
    def evolve_from_gep(self, genes: List[dict]) -> List:
        """
        基于 GEP 基因进行 DNA 进化
        
        Args:
            genes: GEP 基因列表
            
        Returns:
            新生成的 DNA 列表
        """
        if not self.evolution_engine:
            logger.error("进化引擎不可用")
            return []
        
        new_dnas = []
        
        for gene in genes:
            if gene.get("mutation_type") == "repair":
                # 修复型基因：提高稳定性
                result = self._apply_repair_gene(gene)
            elif gene.get("mutation_type") == "innovate":
                # 创新型基因：尝试新能力
                result = self._apply_innovation_gene(gene)
            else:
                result = None
            
            if result:
                new_dnas.append(result)
        
        return new_dnas
    
    def _apply_repair_gene(self, gene: dict):
        """应用修复型基因"""
        try:
            from hive.core.dna import BeeDNA
            
            # 修复型基因提高免疫响应和稳定性
            dna = BeeDNA(
                species="repaired",
                role="worker",
                id=f"repair-{gene.get('id', 'unknown')[:8]}",
                target_name=gene.get("source_crash", "general"),
            )
            dna.metadata["repair_gene"] = gene.get("id")
            dna.metadata["repair_target"] = gene.get("source_crash")
            
            return dna
        except Exception as e:
            logger.error(f"应用修复基因失败: {e}")
            return None
    
    def _apply_innovation_gene(self, gene: dict):
        """应用创新型基因"""
        try:
            from hive.core.dna import BeeDNA
            
            # 创新型基因创建新变种
            dna = BeeDNA(
                species="mutant",
                role=gene.get("new_role", "explorer"),
                id=f"innov-{gene.get('id', 'unknown')[:8]}",
                target_name=gene.get("target", "innovation"),
            )
            dna.metadata["innovation_gene"] = gene.get("id")
            dna.metadata["innovation_type"] = gene.get("innovation_type")
            
            return dna
        except Exception as e:
            logger.error(f"应用创新基因失败: {e}")
            return None
    
    def sync_fitness_to_gep(self, evolver) -> int:
        """
        将 DNA 适应度同步回 GEP 系统
        
        Args:
            evolver: Evolver 实例
            
        Returns:
            同步的基因数量
        """
        if not self.dna_pool:
            return 0
        
        synced = 0
        genes = evolver._read_json(evolver.genes_file)
        
        for gene_id, gene in genes.items():
            if gene_id.startswith("_"):
                continue
            
            # 查找应用了此基因的 DNA
            applied_dnas = []
            for dna in self.dna_pool._pool:
                if gene_id in dna.metadata.get("applied_genes", []):
                    applied_dnas.append(dna)
            
            if applied_dnas:
                # 计算平均适应度
                avg_fitness = sum(d.fitness for d in applied_dnas) / len(applied_dnas)
                gene["observed_fitness"] = avg_fitness
                gene["application_count"] = len(applied_dnas)
                synced += 1
        
        if synced > 0:
            evolver._write_json(evolver.genes_file, genes)
            logger.info(f"已同步 {synced} 个基因的适应度数据")
        
        return synced


# 全局桥接器实例
_bridge: Optional[DNABridge] = None
_bridge_lock = __import__('threading').Lock()


def get_bridge() -> DNABridge:
    """获取全局 DNA 桥接器"""
    global _bridge
    if _bridge is None:
        with _bridge_lock:
            if _bridge is None:
                _bridge = DNABridge()
    return _bridge
