# -*- coding: utf-8 -*-
"""
ValidatorBee - 验证蜂
简单的文件验证功能
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger


class ValidatorBee(BaseBee):
    """
    验证者蜂 - 文件验证工具

    能力：
    1. 验证文件完整性
    2. 计算文件哈希
    3. 检查文件存在性
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.logger = get_logger("ValidatorBee")

    def specialized_task(self, target: Path) -> Dict[str, Any]:
        """
        执行验证任务

        Args:
            target: 目标路径

        Returns:
            验证结果字典
        """
        self.logger.info(f"✓ Validator 开始验证: {target}")

        if not target.exists():
            return {"status": "error", "message": f"目标不存在: {target}", "exists": False}

        if target.is_file():
            return self._validate_file(target)

        return self._validate_directory(target)

    def _validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        验证单个文件

        Args:
            file_path: 文件路径

        Returns:
            验证结果
        """
        try:
            stat = file_path.stat()

            # 计算文件哈希
            file_hash = self._calculate_hash(file_path)

            return {
                "status": "success",
                "target_type": "file",
                "target": str(file_path),
                "exists": True,
                "size": stat.st_size,
                "modified_time": stat.st_mtime,
                "file_hash": file_hash,
                "readable": True,
                "writable": self._check_writable(file_path),
                "message": f"文件验证完成: {stat.st_size} bytes",
            }

        except Exception as e:
            return {
                "status": "error",
                "target": str(file_path),
                "exists": True,
                "error": str(e),
                "message": f"文件验证失败: {e}",
            }

    def _validate_directory(self, directory: Path) -> Dict[str, Any]:
        """
        验证目录

        Args:
            directory: 目录路径

        Returns:
            验证结果
        """
        try:
            file_count = 0
            dir_count = 0
            total_size = 0
            empty_dirs = []

            for item in directory.rglob("*"):
                if item.is_file():
                    file_count += 1
                    total_size += item.stat().st_size
                elif item.is_dir():
                    dir_count += 1
                    # 检查空目录
                    if not any(item.iterdir()):
                        empty_dirs.append(str(item))

            return {
                "status": "success",
                "target_type": "directory",
                "target": str(directory),
                "exists": True,
                "file_count": file_count,
                "dir_count": dir_count,
                "total_size": total_size,
                "empty_directories": len(empty_dirs),
                "readable": True,
                "writable": self._check_writable(directory),
                "message": f"目录验证完成: {file_count} 文件, {dir_count} 目录",
            }

        except Exception as e:
            return {
                "status": "error",
                "target": str(directory),
                "exists": True,
                "error": str(e),
                "message": f"目录验证失败: {e}",
            }

    def _calculate_hash(self, file_path: Path, algorithm: str = "md5") -> str:
        """
        计算文件哈希值

        Args:
            file_path: 文件路径
            algorithm: 哈希算法

        Returns:
            哈希值
        """
        hash_func = hashlib.new(algorithm)

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def _check_writable(self, path: Path) -> bool:
        """
        检查路径是否可写

        Args:
            path: 路径

        Returns:
            是否可写
        """
        try:
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except (OSError, PermissionError):
            return False

    def validate_multiple(self, paths: List[Path]) -> Dict[str, Any]:
        """
        批量验证多个路径

        Args:
            paths: 路径列表

        Returns:
            批量验证结果
        """
        results = []
        valid_count = 0
        invalid_count = 0

        for path in paths:
            result = self.specialized_task(path)
            results.append(result)

            if result["status"] == "success":
                valid_count += 1
            else:
                invalid_count += 1

        return {
            "status": "completed",
            "total_paths": len(paths),
            "valid": valid_count,
            "invalid": invalid_count,
            "details": results,
        }
