# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
from typing import Dict, List, Set
import json

logger = get_logger(__name__)

class OntologyBee(BaseBee):
    """本体蜂 - 创建本体、定义关系、语义推理"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.ontology = {
            "entities": {},
            "relations": [],
            "axioms": []
        }
    
    def specialized_task(self, target_path):
        """执行本体任务"""
        logger.info(f"[OntologyBee-{self.id}] 本体: {target_path}")
        
        # 创建实体
        entity = self._create_entity(target_path)
        
        # 定义关系
        relations = self._define_relations(entity)
        
        logger.info(f"[OntologyBee-{self.id}] 本体构建完成")
    
    def _create_entity(self, context: str) -> Dict:
        """创建实体"""
        entity_id = f"entity_{hash(context)}"
        
        entity = {
            "id": entity_id,
            "type": "resource",
            "attributes": {
                "name": context,
                "context": "ontology"
            }
        }
        
        self.ontology["entities"][entity_id] = entity
        return entity
    
    def _define_relations(self, entity: Dict) -> List[Dict]:
        """定义关系"""
        relations = [
            {
                "from": entity["id"],
                "to": "root",
                "type": "belongs_to"
            },
            {
                "from": "root",
                "to": entity["id"],
                "type": "contains"
            }
        ]
        
        self.ontology["relations"].extend(relations)
        return relations
    
    def add_axiom(self, axiom: str):
        """添加公理"""
        self.ontology["axioms"].append(axiom)
        logger.debug(f"[OntologyBee-{self.id}] 添加公理: {axiom}")
    
    def infer_relations(self, entity_id: str) -> List[Dict]:
        """推理关系"""
        inferred = []
        
        for rel in self.ontology["relations"]:
            if rel["from"] == entity_id:
                inferred.append(rel)
        
        return inferred
    
    def build_knowledge_graph(self) -> Dict:
        """构建知识图谱"""
        return {
            "nodes": self.ontology["entities"],
            "edges": self.ontology["relations"],
            "axioms": self.ontology["axioms"]
        }
    
    def devour(self) -> dict:
        """消耗目标并返回本体报告"""
        target_path = self.dna.id
        
        try:
            self.specialized_task(target_path)
            
            graph = self.build_knowledge_graph()
            
            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "ontology": self.ontology,
                "graph": graph
            }
        except Exception as e:
            logger.error(f"[OntologyBee-{self.id}] 本体构建失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e)
            }
