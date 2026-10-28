# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
import random
from typing import List, Dict

logger = get_logger(__name__)

class DreamerBee(BaseBee):
    """梦想蜂 - 生成创意想法和头脑风暴"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.ideas = []
        self.concepts = [
            "自动化", "智能化", "协作", "进化", "优化",
            "创新", "集成", "扩展", "简化", "增强"
        ]
    
    def specialized_task(self, target_path):
        """执行创意生成任务"""
        logger.info(f"[DreamerBee-{self.id}] 梦想: {target_path}")
        
        # 生成创意想法
        idea = self._generate_idea(target_path)
        self.ideas.append(idea)
        
        logger.info(f"[DreamerBee-{self.id}] 创意: {idea}")
    
    def _generate_idea(self, context: str) -> Dict:
        """生成创意想法"""
        concept1 = random.choice(self.concepts)
        concept2 = random.choice(self.concepts)
        
        ideas = [
            f"将{concept1}与{concept2}结合",
            f"通过{concept1}来优化{context}",
            f"使用{concept2}增强{context}",
            f"实现{concept1}的{context}版本"
        ]
        
        return {
            "idea": random.choice(ideas),
            "context": context,
            "generated_at": "dream_sequence",
            "creativity_score": random.random()
        }
    
    def brainstorm(self, topic: str, count: int = 5) -> List[Dict]:
        """头脑风暴"""
        return [self._generate_idea(topic) for _ in range(count)]
    
    def evaluate_ideas(self) -> Dict:
        """评估想法"""
        if not self.ideas:
            return {"status": "no_ideas"}
        
        return {
            "total_ideas": len(self.ideas),
            "avg_creativity": sum(i["creativity_score"] for i in self.ideas) / len(self.ideas),
            "top_idea": max(self.ideas, key=lambda x: x["creativity_score"])
        }
    
    def devour(self) -> dict:
        """消耗目标并返回创意报告"""
        target_path = self.dna.id
        
        try:
            self.specialized_task(target_path)
            
            evaluation = self.evaluate_ideas()
            
            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "ideas": self.ideas,
                "evaluation": evaluation
            }
        except Exception as e:
            logger.error(f"[DreamerBee-{self.id}] 创意生成失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e)
            }
