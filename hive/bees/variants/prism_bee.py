#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棱镜蜂 - PrismBee
多维度视角分析代码

变异率：14.00%
特殊能力：静态分析 + 动态分析 + 安全分析
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class PrismBee(BaseBee):
    """
    棱镜蜂 - 负责多维度代码分析

    特殊能力：
    - 静态代码分析
    - 动态行为分析
    - 安全漏洞检测
    - 依赖关系分析
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.analysis_results = {}

    def specialized_task(self, target_path):
        """
        执行多维度分析任务

        Args:
            target_path: 项目目录路径
        """
        logger.info(f"💎 PrismBee 开始任务: {self.dna.target_name}")

        try:
            # 1. 静态分析
            static_analysis = self._perform_static_analysis(target_path)

            # 2. 动态分析（基于代码模式推断）
            dynamic_analysis = self._perform_dynamic_analysis(target_path)

            # 3. 安全分析
            security_analysis = self._perform_security_analysis(target_path)

            # 4. 依赖分析
            dependency_analysis = self._perform_dependency_analysis(target_path)

            # 5. 综合评分
            overall_score = self._calculate_overall_score(
                static_analysis, dynamic_analysis, security_analysis
            )

            # 6. 保存分析报告
            self._save_prism_report(
                {
                    "project": self.dna.target_name,
                    "static_analysis": static_analysis,
                    "dynamic_analysis": dynamic_analysis,
                    "security_analysis": security_analysis,
                    "dependency_analysis": dependency_analysis,
                    "overall_score": overall_score,
                    "timestamp": __import__("time").ctime(),
                }
            )

            logger.info(f"✅ PrismBee 任务完成: {self.dna.target_name}")
            logger.info(f"   综合评分: {overall_score:.2f}/100")
            logger.info(f"   安全问题: {len(security_analysis['issues'])}")
            logger.info(f"   依赖项: {len(dependency_analysis['dependencies'])}")

            return {
                "status": "success",
                "overall_score": overall_score,
                "security_issues": len(security_analysis["issues"]),
                "dependencies": len(dependency_analysis["dependencies"]),
            }

        except Exception as e:
            logger.error(f"❌ PrismBee 任务失败: {e}")
            return {"status": "error", "error": str(e)}

    def _perform_static_analysis(self, target_path: Path) -> Dict[str, Any]:
        """
        执行静态分析

        Args:
            target_path: 项目目录路径

        Returns:
            静态分析结果
        """
        analysis = {
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "files_analyzed": 0,
            "language_distribution": {},
            "code_quality_metrics": {},
        }

        code_extensions = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".go": "Go",
        }

        for file_path in target_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in code_extensions:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()

                    analysis["files_analyzed"] += 1

                    # 统计代码行、注释行、空行
                    for line in lines:
                        analysis["code_lines"] += 1
                        stripped = line.strip()
                        if not stripped:
                            analysis["blank_lines"] += 1
                        elif (
                            stripped.startswith("#")
                            or stripped.startswith("//")
                            or stripped.startswith("/*")
                            or stripped.startswith("*")
                        ):
                            analysis["comment_lines"] += 1

                    # 统计语言分布
                    lang = code_extensions[file_path.suffix]
                    analysis["language_distribution"][lang] = (
                        analysis["language_distribution"].get(lang, 0) + 1
                    )

                except Exception as e:
                    logger.debug(f"静态分析失败 {file_path}: {e}")

        # 计算代码质量指标
        if analysis["code_lines"] > 0:
            analysis["code_quality_metrics"] = {
                "comment_ratio": round(analysis["comment_lines"] / analysis["code_lines"] * 100, 2),
                "blank_ratio": round(analysis["blank_lines"] / analysis["code_lines"] * 100, 2),
                "code_ratio": round(
                    (analysis["code_lines"] - analysis["comment_lines"] - analysis["blank_lines"])
                    / analysis["code_lines"]
                    * 100,
                    2,
                ),
            }

        return analysis

    def _perform_dynamic_analysis(self, target_path: Path) -> Dict[str, Any]:
        """
        执行动态分析（基于代码模式推断）

        Args:
            target_path: 项目目录路径

        Returns:
            动态分析结果
        """
        analysis = {
            "function_count": 0,
            "class_count": 0,
            "external_calls": 0,
            "file_operations": 0,
            "network_operations": 0,
            "database_operations": 0,
        }

        # Python特定分析
        for file_path in target_path.rglob("*.py"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # 统计函数
                analysis["function_count"] += len(re.findall(r"\bdef\s+\w+", content))

                # 统计类
                analysis["class_count"] += len(re.findall(r"\bclass\s+\w+", content))

                # 统计外部调用
                analysis["external_calls"] += len(re.findall(r"\w+\.\w+\(", content))

                # 统计文件操作
                analysis["file_operations"] += len(
                    re.findall(r"\b(open|read|write|close|file)\b", content)
                )

                # 统计网络操作
                analysis["network_operations"] += len(
                    re.findall(r"\b(requests|urllib|http|socket|tcp|udp)\b", content, re.IGNORECASE)
                )

                # 统计数据库操作
                analysis["database_operations"] += len(
                    re.findall(
                        r"\b(sql|query|select|insert|update|delete|mongo|redis)\b",
                        content,
                        re.IGNORECASE,
                    )
                )

            except Exception as e:
                logger.debug(f"动态分析失败 {file_path}: {e}")

        return analysis

    def _perform_security_analysis(self, target_path: Path) -> Dict[str, Any]:
        """
        执行安全分析

        Args:
            target_path: 项目目录路径

        Returns:
            安全分析结果
        """
        analysis = {"issues": [], "risk_score": 0, "recommendations": []}

        # 安全问题模式
        security_patterns = {
            "hardcoded_passwords": [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'passwd\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
            ],
            "sql_injection": [r"execute\s*\([^)]*\+%s", r"query\s*\([^)]*\+.*\)", r"select.*%s"],
            "command_injection": [
                r"\bsystem\s*\(",
                r"\bexec\s*\(",
                r"\bsubprocess\.(call|run|Popen).*shell\s*=\s*True",
            ],
            "insecure_random": [r"import\s+random", r"random\.(random|randint|choice)"],
        }

        for file_path in target_path.rglob("*.py"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                for issue_type, patterns in security_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[: match.start()].count("\n") + 1
                            analysis["issues"].append(
                                {
                                    "type": issue_type,
                                    "file": str(file_path.relative_to(target_path)),
                                    "line": line_num,
                                    "severity": (
                                        "high"
                                        if "password" in issue_type or "injection" in issue_type
                                        else "medium"
                                    ),
                                    "snippet": match.group()[:100],
                                }
                            )
                            analysis["risk_score"] += (
                                5 if "high" in analysis["issues"][-1]["severity"] else 2
                            )

            except Exception as e:
                logger.debug(f"安全分析失败 {file_path}: {e}")

        # 生成安全建议
        if len(analysis["issues"]) > 0:
            analysis["recommendations"].append(
                f"发现 {len(analysis['issues'])} 个潜在安全问题，建议进行安全审查"
            )
        else:
            analysis["recommendations"].append("未发现明显的安全问题")

        return analysis

    def _perform_dependency_analysis(self, target_path: Path) -> Dict[str, Any]:
        """
        执行依赖分析

        Args:
            target_path: 项目目录路径

        Returns:
            依赖分析结果
        """
        analysis = {"dependencies": [], "external_imports": [], "internal_modules": []}

        # 分析Python依赖
        for file_path in target_path.rglob("*.py"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # 提取import语句
                import_patterns = [r"^import\s+(\w+)", r"^from\s+(\w+)\s+import"]

                for pattern in import_patterns:
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        module = match.group(1)

                        # 排除标准库
                        standard_libs = {
                            "os",
                            "sys",
                            "re",
                            "json",
                            "time",
                            "datetime",
                            "pathlib",
                            "typing",
                            "collections",
                            "math",
                            "random",
                            "logging",
                            "hashlib",
                        }

                        if (
                            module not in standard_libs
                            and module not in analysis["external_imports"]
                        ):
                            # 检查是否是内部模块
                            is_internal = any(
                                (target_path / module / "__init__.py").exists()
                                or (target_path / f"{module}.py").exists()
                            )

                            if is_internal:
                                analysis["internal_modules"].append(module)
                            else:
                                analysis["external_imports"].append(module)

                # 检查requirements.txt等依赖文件
                dep_files = ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"]
                for dep_file in dep_files:
                    dep_path = target_path / dep_file
                    if dep_path.exists():
                        analysis["dependencies"].append(
                            {"file": dep_file, "size": dep_path.stat().st_size}
                        )

            except Exception as e:
                logger.debug(f"依赖分析失败 {file_path}: {e}")

        return analysis

    def _calculate_overall_score(
        self, static: Dict[str, Any], dynamic: Dict[str, Any], security: Dict[str, Any]
    ) -> float:
        """
        计算综合评分

        Args:
            static: 静态分析结果
            dynamic: 动态分析结果
            security: 安全分析结果

        Returns:
            综合评分 (0-100)
        """
        score = 100.0

        # 基于代码质量的扣分
        if "code_quality_metrics" in static:
            comment_ratio = static["code_quality_metrics"].get("comment_ratio", 0)
            if comment_ratio < 10:
                score -= 10  # 注释过少
            elif comment_ratio > 40:
                score -= 5  # 注释过多可能影响可读性

        # 基于安全的扣分
        security_penalty = min(len(security["issues"]) * 3, 30)
        score -= security_penalty

        # 基于复杂度的扣分
        if static["files_analyzed"] > 0:
            avg_functions = dynamic["function_count"] / static["files_analyzed"]
            if avg_functions > 20:
                score -= 10  # 函数密度过高

        return max(0.0, min(100.0, score))

    def _save_prism_report(self, report_data: Dict[str, Any]):
        """
        保存分析报告

        Args:
            report_data: 报告数据
        """
        try:
            reports_dir = Path("prism_reports")
            reports_dir.mkdir(exist_ok=True)

            report_filename = f"{self.dna.target_name.replace('/', '_')}_prism_report.json"
            report_path = reports_dir / report_filename

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📄 棱镜分析报告已保存: {report_path}")

        except Exception as e:
            logger.error(f"保存分析报告失败: {e}")
