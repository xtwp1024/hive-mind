# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
from collections import deque
from typing import Dict, List
import hashlib

logger = get_logger(__name__)

class EchoBee(BaseBee):
    """回声蜂 - 存储、回放和聚合消息"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.messages = deque(maxlen=1000)
        self.message_hashes = set()
    
    def specialized_task(self, target_path):
        """执行消息存储任务"""
        logger.info(f"[EchoBee-{self.id}] 回声: {target_path}")
        
        # 创建消息
        message = {
            "content": target_path,
            "timestamp": "echo_timestamp",
            "source": self.dna.species
        }
        
        # 去重
        message_hash = self._compute_hash(message)
        if message_hash in self.message_hashes:
            logger.info(f"[EchoBee-{self.id}] 消息已存在，跳过")
            return
        
        # 存储消息
        self.messages.append(message)
        self.message_hashes.add(message_hash)
        
        logger.info(f"[EchoBee-{self.id}] 消息已存储，总数: {len(self.messages)}")
    
    def _compute_hash(self, message: Dict) -> str:
        """计算消息哈希"""
        content = str(message)
        return hashlib.md5(content.encode()).hexdigest()
    
    def replay_messages(self, count: int = 10) -> List[Dict]:
        """回放消息"""
        return list(self.messages)[-count:]
    
    def aggregate_messages(self, field: str = "source") -> Dict:
        """聚合消息"""
        result = {}
        for msg in self.messages:
            key = msg.get(field, "unknown")
            result[key] = result.get(key, 0) + 1
        return result
    
    def search_history(self, query: str) -> List[Dict]:
        """搜索历史"""
        return [
            msg for msg in self.messages
            if query.lower() in str(msg["content"]).lower()
        ]
    
    def devour(self) -> dict:
        """消耗目标并返回消息报告"""
        target_path = self.dna.id
        
        try:
            self.specialized_task(target_path)
            
            aggregation = self.aggregate_messages()
            
            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "total_messages": len(self.messages),
                "aggregation": aggregation
            }
        except Exception as e:
            logger.error(f"[EchoBee-{self.id}] 回声失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e)
            }
