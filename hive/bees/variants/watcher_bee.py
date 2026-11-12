# -*- coding: utf-8 -*-
"""
WatcherBee - 文件变化监控蜂
监控文件和目录的变化
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from hive.bees.base_bee import BaseBee
from hive.utils.logger import get_logger


class WatcherBee(BaseBee):
    """
    观察者蜂 - 文件系统监控器

    能力：
    1. 监控文件变化
    2. 检测新增/删除/修改
    3. 生成变化报告
    """

    def __init__(self, bee_dna: Dict[str, Any], id: str, result_queue):
        super().__init__(bee_dna, id, result_queue)
        self.logger = get_logger("WatcherBee")
        self.file_cache = {}

    def specialized_task(self, target: Path) -> Dict[str, Any]:
        """
        执行文件监控任务

        Args:
            target: 目标路径

        Returns:
            监控结果字典
        """
        self.logger.info(f"👁️ Watcher 开始监控: {target}")

        if not target.exists():
            return {"status": "error", "message": f"目标不存在: {target}"}

        # 递归扫描目录
        if target.is_dir():
            changes = self._scan_directory(target)
            return {
                "status": "success",
                "target_type": "directory",
                "target": str(target),
                "changes_detected": changes,
                "message": f"发现 {len(changes)} 个变化",
            }

        # 单文件检查
        file_info = self._get_file_info(target)
        return {
            "status": "success",
            "target_type": "file",
            "target": str(target),
            "file_info": file_info,
            "message": f"文件检查完成",
        }

    def _scan_directory(self, directory: Path) -> Dict[str, Any]:
        """
        扫描目录并检测变化

        Args:
            directory: 目录路径

        Returns:
            变化报告
        """
        current_files = {}
        changes = {"added": [], "removed": [], "modified": [], "total_changes": 0}

        # 扫描当前目录
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                file_info = self._get_file_info(file_path)
                current_files[str(file_path)] = file_info

        # 检测新增文件
        for path in current_files:
            if path not in self.file_cache:
                changes["added"].append(
                    {
                        "path": path,
                        "size": current_files[path]["size"],
                        "modified_time": current_files[path]["modified_time"],
                    }
                )

        # 检测删除文件
        for path in self.file_cache:
            if path not in current_files:
                changes["removed"].append(
                    {"path": path, "last_known_size": self.file_cache[path]["size"]}
                )

        # 检测修改文件
        for path in current_files:
            if path in self.file_cache:
                if current_files[path]["size"] != self.file_cache[path]["size"]:
                    changes["modified"].append(
                        {
                            "path": path,
                            "old_size": self.file_cache[path]["size"],
                            "new_size": current_files[path]["size"],
                            "modified_time": current_files[path]["modified_time"],
                        }
                    )

        # 更新缓存
        self.file_cache = current_files
        changes["total_changes"] = (
            len(changes["added"]) + len(changes["removed"]) + len(changes["modified"])
        )

        return changes

    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        获取文件信息

        Args:
            file_path: 文件路径

        Returns:
            文件信息字典
        """
        stat = file_path.stat()
        return {
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
            "is_file": file_path.is_file(),
            "extension": file_path.suffix,
        }

    def get_file_count(self, directory: Path) -> Dict[str, int]:
        """
        统计目录中的文件数量

        Args:
            directory: 目录路径

        Returns:
            文件统计信息
        """
        if not directory.exists() or not directory.is_dir():
            return {"error": "无效的目录路径"}

        total_files = 0
        extensions = {}

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_files += 1
                ext = file_path.suffix
                extensions[ext] = extensions.get(ext, 0) + 1

        return {
            "total_files": total_files,
            "extensions": extensions,
            "unique_extensions": len(extensions),
        }
