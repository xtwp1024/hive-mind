# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
from typing import Dict, List
import re

logger = get_logger(__name__)

class EthicistBee(BaseBee):
    """伦理蜂 - 评估行为伦理、检测偏见"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.ethical_guidelines = {
            "privacy": "保护用户隐私",
            "fairness": "确保公平对待",
            "transparency": "保持透明度",
            "accountability": "承担责任"
        }
        self.biases_detected = []
    
    def specialized_task(self, target_path):
        """执行伦理评估任务"""
        logger.info(f"[EthicistBee-{self.id}] 伦理评估: {target_path}")
        
        # 评估目标路径
        evaluation = self._evaluate_ethics(target_path)
        
        # 检测偏见
        biases = self._detect_biases(target_path)
        
        if biases:
            self.biases_detected.extend(biases)
            logger.warning(f"[EthicistBee-{self.id}] 检测到偏见: {biases}")
    
    def _evaluate_ethics(self, content: str) -> Dict:
        """评估伦理合规性"""
        results = {}
        
        for principle, description in self.ethical_guidelines.items():
            # 简化的伦理检查
            compliant = self._check_compliance(content, principle)
            results[principle] = {
                "description": description,
                "compliant": compliant,
                "score": 1.0 if compliant else 0.0
            }
        
        return results
    
    def _check_compliance(self, content: str, principle: str) -> bool:
        """检查合规性（简化版）"""
        # 实际实现中应该有更复杂的逻辑
        return True
    
    def _detect_biases(self, content: str) -> List[str]:
        """检测偏见（简化版）"""
        biases = []
        
        # 检查敏感词
        sensitive_patterns = [
            r"better than",
            r"superior to",
            r"inferior to"
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                biases.append(f"潜在偏见: {pattern}")
        
        return biases
    
    def provide_recommendation(self, evaluation: Dict) -> str:
        """提供伦理建议"""
        non_compliant = [k for k, v in evaluation.items() if not v["compliant"]]
        
        if non_compliant:
            return f"需要改进: {', '.join(non_compliant)}"
        return "符合伦理标准"
    
    def devour(self) -> dict:
        """消耗目标并返回伦理报告"""
        target_path = self.dna.id
        
        try:
            self.specialized_task(target_path)
            
            evaluation = self._evaluate_ethics(target_path)
            
            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "evaluation": evaluation,
                "biases": self.biases_detected
            }
        except Exception as e:
            logger.error(f"[EthicistBee-{self.id}] 伦理评估失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e)
            }
