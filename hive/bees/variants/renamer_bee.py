# -*- coding: utf-8 -*-
"""
文件重命名蜜蜂 - 批量重命名文件
功能：批量重命名文件，支持模式匹配和替换
复杂度：低 (< 100行)
"""

import os
from pathlib import Path
from typing import List, Optional

from hive.bees.base_bee import BaseBee


class RenamerBee(BaseBee):
    """负责文件重命名操作的蜜蜂"""

    def __init__(self):
        super().__init__(name="renamer_bee", role="file_renamer", description="批量重命名文件")
        self.renamed_count = 0
        self.failed_count = 0
        self.renamed_files = []

    def rename_file(self, source: str, destination: str) -> bool:
        """
        重命名单个文件

        Args:
            source: 源文件路径
            destination: 目标文件路径

        Returns:
            bool: 是否成功
        """
        try:
            source_path = Path(source)
            dest_path = Path(destination)

            if not source_path.exists():
                print(f"❌ 源文件不存在: {source}")
                self.failed_count += 1
                return False

            if dest_path.exists():
                print(f"❌ 目标文件已存在: {destination}")
                self.failed_count += 1
                return False

            source_path.rename(dest_path)

            self.renamed_count += 1
            self.renamed_files.append({"old_name": str(source_path), "new_name": str(dest_path)})

            print(f"✅ 已重命名: {source_path.name} -> {dest_path.name}")
            return True

        except Exception as e:
            print(f"❌ 重命名失败: {source} -> {destination}")
            print(f"   错误: {str(e)}")
            self.failed_count += 1
            return False

    def rename_by_pattern(self, directory: str, pattern: str, old_text: str, new_text: str) -> dict:
        """
        按模式批量重命名文件

        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            old_text: 要替换的文本
            new_text: 替换为的文本

        Returns:
            dict: 重命名结果统计
        """
        try:
            dir_path = Path(directory)

            if not dir_path.exists() or not dir_path.is_dir():
                print(f"❌ 目录不存在: {directory}")
                self.failed_count += 1
                return {"total": 0, "renamed": 0, "failed": 0}

            files = list(dir_path.glob(pattern))

            if not files:
                print(f"❌ 没有找到匹配 {pattern} 的文件")
                return {"total": 0, "renamed": 0, "failed": 0}

            results = {"total": len(files), "renamed": 0, "failed": 0, "files": []}

            for file_path in files:
                old_name = file_path.name

                # 替换文本
                new_name = old_name.replace(old_text, new_text)

                # 如果没有变化，跳过
                if new_name == old_name:
                    continue

                new_path = file_path.parent / new_name

                if self.rename_file(str(file_path), str(new_path)):
                    results["renamed"] += 1
                    results["files"].append(
                        {"old_name": old_name, "new_name": new_name, "status": "success"}
                    )
                else:
                    results["failed"] += 1
                    results["files"].append(
                        {"old_name": old_name, "new_name": new_name, "status": "failed"}
                    )

            print(f"✅ 批量重命名完成: {results['renamed']}/{results['total']} 成功")
            return results

        except Exception as e:
            print(f"❌ 批量重命名失败: {directory}")
            print(f"   错误: {str(e)}")
            self.failed_count += 1
            return {"total": 0, "renamed": 0, "failed": 0}

    def add_prefix(self, directory: str, pattern: str, prefix: str) -> dict:
        """
        给文件添加前缀

        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            prefix: 要添加的前缀

        Returns:
            dict: 重命名结果统计
        """
        try:
            dir_path = Path(directory)

            if not dir_path.exists() or not dir_path.is_dir():
                print(f"❌ 目录不存在: {directory}")
                return {"total": 0, "renamed": 0, "failed": 0}

            files = list(dir_path.glob(pattern))

            if not files:
                print(f"❌ 没有找到匹配 {pattern} 的文件")
                return {"total": 0, "renamed": 0, "failed": 0}

            results = {"total": len(files), "renamed": 0, "failed": 0, "files": []}

            for file_path in files:
                old_name = file_path.name
                new_name = prefix + old_name
                new_path = file_path.parent / new_name

                if self.rename_file(str(file_path), str(new_path)):
                    results["renamed"] += 1
                    results["files"].append(
                        {"old_name": old_name, "new_name": new_name, "status": "success"}
                    )
                else:
                    results["failed"] += 1

            return results

        except Exception as e:
            print(f"❌ 添加前缀失败: {directory}")
            print(f"   错误: {str(e)}")
            return {"total": 0, "renamed": 0, "failed": 0}

    def get_statistics(self) -> dict:
        """
        获取重命名统计信息

        Returns:
            dict: 包含重命名次数、失败次数等统计
        """
        return {
            "renamed_count": self.renamed_count,
            "failed_count": self.failed_count,
            "success_rate": (
                self.renamed_count / (self.renamed_count + self.failed_count) * 100
                if (self.renamed_count + self.failed_count) > 0
                else 100
            ),
            "recent_renames": self.renamed_files[-5:] if self.renamed_files else [],
        }

    def execute(self, task: str, **kwargs) -> any:
        """
        执行重命名任务

        Args:
            task: 任务类型 ('rename_file', 'rename_by_pattern', 'add_prefix')
            **kwargs: 任务参数
        """
        if task == "rename_file":
            return self.rename_file(kwargs.get("source"), kwargs.get("destination"))

        elif task == "rename_by_pattern":
            return self.rename_by_pattern(
                kwargs.get("directory", "."),
                kwargs.get("pattern", "*"),
                kwargs.get("old_text", ""),
                kwargs.get("new_text", ""),
            )

        elif task == "add_prefix":
            return self.add_prefix(
                kwargs.get("directory", "."), kwargs.get("pattern", "*"), kwargs.get("prefix", "")
            )

        else:
            print(f"❌ 未知任务类型: {task}")
            return None
