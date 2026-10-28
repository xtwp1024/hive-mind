# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
from pathlib import Path
from typing import Dict, List
import json

logger = get_logger(__name__)

class CuratorBee(BaseBee):
    """策展蜂 - 整理、分类和管理数据"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.catalog = {}
        self.metadata = {}
        self.categories = {}
    
    def specialized_task(self, target_path):
        """执行策展任务"""
        logger.info(f"[CuratorBee-{self.id}] 策展数据: {target_path}")
        
        target = Path(target_path)
        if not target.exists():
            logger.error(f"[CuratorBee-{self.id}] 目标不存在: {target_path}")
            return
        
        # 创建元数据
        metadata = self._create_metadata(target)
        self.metadata[str(target_path)] = metadata
        
        # 分类数据
        category = self._classify_file(target)
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(str(target_path))
        
        # 添加到目录
        self.catalog[str(target_path)] = {
            "category": category,
            "metadata": metadata,
            "cataloged_at": logger.handlers[0].baseFilename
        }
        
        logger.info(f"[CuratorBee-{self.id}] 策展完成，类别: {category}")
    
    def _create_metadata(self, file_path: Path) -> Dict:
        """创建文件元数据"""
        stat = file_path.stat()
        
        return {
            "name": file_path.name,
            "size": stat.st_size,
            "extension": file_path.suffix,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "is_file": file_path.is_file()
        }
    
    def _classify_file(self, file_path: Path) -> str:
        """分类文件"""
        ext = file_path.suffix.lower()
        
        if ext in ['.py', '.js', '.ts']:
            return "code"
        elif ext in ['.md', '.txt', '.rst']:
            return "documentation"
        elif ext in ['.json', '.yaml', '.yml', '.toml']:
            return "configuration"
        elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
            return "image"
        else:
            return "other"
    
    def get_catalog(self, category: str = None) -> List[Dict]:
        """获取目录"""
        if category:
            return [self.catalog.get(f, {}) for f in self.categories.get(category, [])]
        return list(self.catalog.values())
    
    def devour(self) -> dict:
        """消耗目标并返回策展报告"""
        target_path = self.dna.id
        
        try:
            self.specialized_task(target_path)
            
            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "metadata": self.metadata.get(target_path, {}),
                "catalog": self.catalog.get(target_path, {})
            }
        except Exception as e:
            logger.error(f"[CuratorBee-{self.id}] 策展失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e)
            }
