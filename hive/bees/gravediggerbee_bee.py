# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
from pathlib import Path
import shutil
from datetime import datetime, timedelta

logger = get_logger(__name__)

class GravediggerBee(BaseBee):
    """清道夫蜂 - 清理过期数据、归档、垃圾回收"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.archive_dir = Path("archive")
        self.cleaned_items = []
        self.archived_items = []
    
    def specialized_task(self, target_path):
        """执行清理任务"""
        logger.info(f"[GravediggerBee-{self.id}] 清理: {target_path}")
        
        target = Path(target_path)
        
        # 检查是否过期
        if self._is_expired(target):
            if target.is_file():
                self._cleanup_file(target)
            elif target.is_dir():
                self._archive_directory(target)
        else:
            logger.info(f"[GravediggerBee-{self.id}] 项目未过期，跳过")
    
    def _is_expired(self, path: Path, days: int = 30) -> bool:
        """检查是否过期"""
        if not path.exists():
            return False
        
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        return age > timedelta(days=days)
    
    def _cleanup_file(self, file_path: Path):
        """清理文件"""
        try:
            file_path.unlink()
            self.cleaned_items.append(str(file_path))
            logger.info(f"[GravediggerBee-{self.id}] 已清理: {file_path}")
        except Exception as e:
            logger.error(f"[GravediggerBee-{self.id}] 清理失败: {e}")
    
    def _archive_directory(self, dir_path: Path):
        """归档目录"""
        try:
            self.archive_dir.mkdir(exist_ok=True)
            archive_name = f"{dir_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            archive_path = self.archive_dir / archive_name
            
            shutil.copytree(dir_path, archive_path)
            self.archived_items.append(str(archive_path))
            
            logger.info(f"[GravediggerBee-{self.id}] 已归档: {archive_path}")
        except Exception as e:
            logger.error(f"[GravediggerBee-{self.id}] 归档失败: {e}")
    
    def cleanup_temp_files(self, temp_dir: Path = Path("temp")):
        """清理临时文件"""
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                self._cleanup_file(item)
    
    def devour(self) -> dict:
        """消耗目标并返回清理报告"""
        target_path = self.dna.id
        
        try:
            self.specialized_task(target_path)
            
            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "cleaned": len(self.cleaned_items),
                "archived": len(self.archived_items)
            }
        except Exception as e:
            logger.error(f"[GravediggerBee-{self.id}] 清理失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e)
            }
