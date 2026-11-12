# -*- coding: utf-8 -*-
"""
LoggerBee - 日志分析蜂
简单快速的日志文件分析
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger


class LoggerBee(BaseBee):
    """
    日志蜂 - 日志文件分析器

    能力：
    1. 解析日志文件
    2. 统计错误和警告
    3. 生成日志摘要
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.logger = get_logger("LoggerBee")

        # 常见日志模式
        self.error_pattern = re.compile(r"ERROR|Exception|Traceback|Failed", re.IGNORECASE)
        self.warning_pattern = re.compile(r"WARNING|WARN|Deprecated", re.IGNORECASE)
        self.info_pattern = re.compile(r"INFO|DEBUG", re.IGNORECASE)

    def specialized_task(self, target: Path) -> Dict[str, Any]:
        """
        执行日志分析任务

        Args:
            target: 目标路径（日志文件）

        Returns:
            分析结果字典
        """
        self.logger.info(f"📊 Logger 开始分析: {target}")

        if not target.exists() or not target.is_file():
            return {"status": "error", "message": f"目标不是有效的文件: {target}"}

        # 读取并分析日志
        try:
            with open(target, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            analysis = self._analyze_log_content(content, target)
            return {
                "status": "success",
                "target": str(target),
                "analysis": analysis,
                "message": f"分析完成: {analysis['total_lines']} 行",
            }

        except Exception as e:
            return {"status": "error", "message": f"读取日志文件失败: {e}"}

    def _analyze_log_content(self, content: str, file_path: Path) -> Dict[str, Any]:
        """
        分析日志内容

        Args:
            content: 日志内容
            file_path: 文件路径

        Returns:
            分析结果
        """
        lines = content.split("\n")
        total_lines = len(lines)

        # 统计各类日志
        error_count = 0
        warning_count = 0
        info_count = 0

        errors = []
        warnings = []

        for i, line in enumerate(lines, 1):
            if self.error_pattern.search(line):
                error_count += 1
                if error_count <= 10:  # 只记录前10个错误
                    errors.append({"line": i, "content": line.strip()[:200]})  # 限制长度

            elif self.warning_pattern.search(line):
                warning_count += 1
                if warning_count <= 10:  # 只记录前10个警告
                    warnings.append({"line": i, "content": line.strip()[:200]})

            elif self.info_pattern.search(line):
                info_count += 1

        # 计算文件大小
        file_size = file_path.stat().st_size

        return {
            "total_lines": total_lines,
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "other_count": total_lines - error_count - warning_count - info_count,
            "file_size_bytes": file_size,
            "error_samples": errors,
            "warning_samples": warnings,
            "error_rate": round(error_count / total_lines * 100, 2) if total_lines > 0 else 0,
            "warning_rate": round(warning_count / total_lines * 100, 2) if total_lines > 0 else 0,
        }

    def analyze_multiple_logs(self, log_paths: List[Path]) -> Dict[str, Any]:
        """
        批量分析多个日志文件

        Args:
            log_paths: 日志文件路径列表

        Returns:
            批量分析结果
        """
        results = []
        total_errors = 0
        total_warnings = 0

        for log_path in log_paths:
            result = self.specialized_task(log_path)
            results.append(result)

            if result["status"] == "success":
                total_errors += result["analysis"]["error_count"]
                total_warnings += result["analysis"]["warning_count"]

        return {
            "status": "completed",
            "total_files": len(log_paths),
            "successful_files": sum(1 for r in results if r["status"] == "success"),
            "failed_files": sum(1 for r in results if r["status"] == "error"),
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "details": results,
        }

    def find_errors_by_keyword(self, content: str, keyword: str) -> List[Dict[str, Any]]:
        """
        根据关键词查找错误

        Args:
            content: 日志内容
            keyword: 搜索关键词

        Returns:
            匹配的错误列表
        """
        lines = content.split("\n")
        matches = []

        for i, line in enumerate(lines, 1):
            if keyword.lower() in line.lower():
                matches.append({"line": i, "content": line.strip()[:300]})

        return matches[:20]  # 限制返回数量
