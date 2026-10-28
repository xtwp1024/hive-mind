# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
import hashlib
import json
from pathlib import Path
import base64

logger = get_logger(__name__)

class CipherBee(BaseBee):
    """加密蜂 - 加密、解密、哈希和数字签名"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.keys = {}
        self.encrypted_data = {}
    
    def specialized_task(self, target_path):
        """执行加密任务"""
        logger.info(f"[CipherBee-{self.id}] 加密处理: {target_path}")
        
        target = Path(target_path)
        if not target.exists():
            logger.error(f"[CipherBee-{self.id}] 目标不存在: {target_path}")
            return
        
        # 读取内容
        content = target.read_text()
        
        # 计算哈希
        hash_value = self._compute_hash(content)
        
        # 生成密钥
        key = self._generate_key(target_path)
        self.keys[target_path] = key
        
        # 加密数据（简单base64编码模拟）
        encrypted = base64.b64encode(content.encode()).decode()
        self.encrypted_data[target_path] = encrypted
        
        logger.info(f"[CipherBee-{self.id}] 加密完成，哈希: {hash_value[:16]}...")
    
    def _compute_hash(self, data: str, algorithm: str = "sha256") -> str:
        """计算哈希值"""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode())
        return hash_obj.hexdigest()
    
    def _generate_key(self, seed: str) -> str:
        """生成密钥"""
        hash_obj = hashlib.sha256(seed.encode())
        return hash_obj.hexdigest()
    
    def decrypt_data(self, target_path: str) -> str:
        """解密数据"""
        if target_path not in self.encrypted_data:
            raise ValueError(f"未找到加密数据: {target_path}")
        
        encrypted = self.encrypted_data[target_path]
        decrypted = base64.b64decode(encrypted).decode()
        return decrypted
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """验证签名"""
        computed = self._compute_hash(data)
        return computed == signature
    
    def devour(self) -> dict:
        """消耗目标并返回加密报告"""
        target_path = self.dna.id
        
        try:
            self.specialized_task(target_path)
            
            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "key": self.keys.get(target_path, ""),
                "encrypted": self.encrypted_data.get(target_path, "")
            }
        except Exception as e:
            logger.error(f"[CipherBee-{self.id}] 加密失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e)
            }
