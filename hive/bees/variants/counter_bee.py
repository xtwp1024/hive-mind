# -*- coding: utf-8 -*-
"""
CounterBee - 统计计数蜂
简单的文件和目录统计功能
"""

from pathlib import Path
from typing import Dict, Any
from hive.bees.base_bee import BaseBee
from hive.utils.logger import get_logger

class CounterBee(BaseBee):
    """
    计数器蜂 - 简单统计工具
    
    能力：
    1. 统计文件数量
    2. 统计目录数量
    3. 统计代码行数
    """
    
    def __init__(self, bee_dna: Dict[str, Any], id: str, result_queue):
        super().__init__(bee_dna, id, result_queue)
        self.logger = get_logger("CounterBee")
    
    def specialized_task(self, target: Path) -> Dict[str, Any]:
        """
        执行统计任务
        
        Args:
            target: 目标路径
            
        Returns:
            统计结果字典
        """
        self.logger.info(f"🔢 Counter 开始统计: {target}")
        
        if not target.exists():
            return {
                "status": "error",
                "message": f"目标不存在: {target}"
            }
        
        if target.is_file():
            return self._count_file(target)
        
        return self._count_directory(target)
    
    def _count_file(self, file_path: Path) -> Dict[str, Any]:
        """
        统计单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            统计结果
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            empty_lines = sum(1 for line in lines if not line.strip())
            code_lines = total_lines - empty_lines
            chars = sum(len(line) for line in lines)
            
            return {
                "status": "success",
                "target_type": "file",
                "target": str(file_path),
                "stats": {
                    "total_lines": total_lines,
                    "empty_lines": empty_lines,
                    "code_lines": code_lines,
                    "total_chars": chars,
                    "file_size": file_path.stat().st_size
                },
                "message": f"文件统计完成: {total_lines} 行"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"读取文件失败: {e}"
            }
    
    def _count_directory(self, directory: Path) -> Dict[str, Any]:
        """
        统计目录
        
        Args:
            directory: 目录路径
            
        Returns:
            统计结果
        """
        file_count = 0
        dir_count = 0
        total_lines = 0
        total_size = 0
        extensions = {}
        
        for item in directory.rglob("*"):
            if item.is_file():
                file_count += 1
                total_size += item.stat().st_size
                
                # 统计扩展名
                ext = item.suffix
                extensions[ext] = extensions.get(ext, 0) + 1
                
                # 尝试统计行数（仅文本文件）
                if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']:
                    try:
                        with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += len(f.readlines())
                    except (IOError, UnicodeDecodeError):
                        pass
                        
            elif item.is_dir():
                dir_count += 1
        
        return {
            "status": "success",
            "target_type": "directory",
            "target": str(directory),
            "stats": {
                "total_files": file_count,
                "total_directories": dir_count,
                "total_lines": total_lines,
                "total_size_bytes": total_size,
                "unique_extensions": len(extensions),
                "top_extensions": sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:5]
            },
            "message": f"目录统计完成: {file_count} 文件, {dir_count} 目录"
        }
