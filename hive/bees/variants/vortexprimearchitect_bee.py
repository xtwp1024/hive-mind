# -*- coding: utf-8 -*-
"""
VortexPrimeArchitectBee - 混合变异蜂种
结合深度代码审计和LLM驱动的代码重构能力
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from hive.bees.base_bee import BaseBee
from hive.config.settings import settings
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class VortexPrimeArchitectBee(BaseBee):
    """
    混合变异：深度代码审计和优化 + 极致化：使用大脑重构代码

    特殊能力：杂交能力：代码复杂度分析 + 重构建议 + 性能极限优化 - LLM代码重构
    变异率：7.20%
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.report_dir = settings.REPORT_DIR / "vortexprimearchitect"
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 记录分析结果
        self.code_metrics = []
        self.refactoring_suggestions = []
        self.llm_refactoring_candidates = []
        self.architecture_insights = {}

    def specialized_task(self):
        """
        执行混合任务：
        1. 深度代码复杂度分析
        2. 生成重构建议
        3. 识别LLM重构候选代码
        4. 架构优化建议
        """
        logger.info(f"[{self.dna.target_name}] 开始执行混合任务：代码审计 + LLM重构")

        # 第一阶段：代码复杂度分析
        self._analyze_code_complexity()

        # 第二阶段：代码质量检测
        self._detect_code_quality_issues()

        # 第三阶段：识别LLM重构候选
        self._identify_llm_refactoring_candidates()

        # 第四阶段：架构分析
        self._analyze_architecture()

        # 第五阶段：生成重构建议
        self._generate_refactoring_suggestions()

        # 第六阶段：生成综合报告
        self._generate_comprehensive_report()

        logger.info(f"[{self.dna.target_name}] 混合任务完成")

    def _analyze_code_complexity(self):
        """分析代码复杂度"""
        logger.info(f"[{self.dna.target_name}] 分析代码复杂度...")

        from hive.config.settings import settings

        target_path = settings.BASE_DIR
        python_files = list(target_path.rglob("*.py"))

        for file_path in python_files:
            try:
                metrics = self._analyze_file_complexity(file_path)
                if metrics:
                    self.code_metrics.append(metrics)
                    logger.debug(
                        f"[{self.dna.target_name}] 分析文件: {file_path.relative_to(target_path)}"
                    )
            except Exception as e:
                logger.debug(f"[{self.dna.target_name}] 分析文件失败 {file_path}: {e}")

        logger.info(f"[{self.dna.target_name}] 共分析 {len(self.code_metrics)} 个文件")

    def _analyze_file_complexity(self, file_path: Path) -> dict:
        """分析单个文件的复杂度"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
        except (IOError, UnicodeDecodeError):
            return None

        # 基本指标
        total_lines = len(lines)
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        comment_lines = len([l for l in lines if l.strip().startswith("#")])
        blank_lines = len([l for l in lines if not l.strip()])

        # 函数和类分析
        functions = self._extract_functions(content)
        classes = self._extract_classes(content)

        # 复杂度指标
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(content)
        nesting_depth = self._calculate_max_nesting_depth(content)
        function_complexities = [
            self._calculate_cyclomatic_complexity(f["content"]) for f in functions
        ]

        return {
            "path": str(file_path.relative_to(settings.BASE_DIR)),
            "basic_metrics": {
                "total_lines": total_lines,
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "blank_lines": blank_lines,
                "comment_ratio": comment_lines / code_lines if code_lines > 0 else 0,
            },
            "structure": {
                "functions_count": len(functions),
                "classes_count": len(classes),
                "avg_function_length": (
                    sum(len(f["lines"]) for f in functions) / len(functions) if functions else 0
                ),
            },
            "complexity": {
                "cyclomatic_complexity": cyclomatic_complexity,
                "max_nesting_depth": nesting_depth,
                "avg_function_complexity": (
                    sum(function_complexities) / len(function_complexities)
                    if function_complexities
                    else 0
                ),
                "max_function_complexity": (
                    max(function_complexities) if function_complexities else 0
                ),
            },
            "functions": functions[:10],  # 只保留前10个函数的详细信息
            "classes": classes[:10],  # 只保留前10个类的详细信息
        }

    def _extract_functions(self, content: str) -> List[dict]:
        """提取函数信息"""
        functions = []
        pattern = r"def\s+(\w+)\s*\((.*?)\):"

        for match in re.finditer(pattern, content):
            func_name = match.group(1)
            func_start = match.end()

            # 找到函数体（简单的缩进检测）
            lines_after = content[func_start:].split("\n")
            func_lines = []
            for line in lines_after:
                if line.strip() and not line.startswith(" " * 4) and not line.startswith("\t"):
                    break
                func_lines.append(line)

            functions.append(
                {
                    "name": func_name,
                    "parameters": match.group(2),
                    "lines": func_lines,
                    "content": "\n".join(func_lines),
                    "length": len(func_lines),
                }
            )

        return functions

    def _extract_classes(self, content: str) -> List[dict]:
        """提取类信息"""
        classes = []
        pattern = r"class\s+(\w+)(?:\s*\([^)]*\))?:"

        for match in re.finditer(pattern, content):
            class_name = match.group(1)

            # 提取类中的方法
            class_methods = []
            method_pattern = r"def\s+(\w+)\s*\((.*?)\):"
            for method_match in re.finditer(
                method_pattern, content[match.end() : match.end() + 5000]
            ):
                class_methods.append(method_match.group(1))

            classes.append(
                {
                    "name": class_name,
                    "methods_count": len(class_methods),
                    "methods": class_methods[:10],
                }
            )

        return classes

    def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """计算圈复杂度"""
        # 简化的圈复杂度计算
        decision_keywords = ["if", "elif", "for", "while", "except", "and", "or"]
        complexity = 1  # 基础复杂度

        for keyword in decision_keywords:
            complexity += content.count(keyword)

        return complexity

    def _calculate_max_nesting_depth(self, content: str) -> int:
        """计算最大嵌套深度"""
        max_depth = 0
        current_depth = 0

        for line in content.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith(
                ("if ", "elif ", "for ", "while ", "def ", "class ", "try:", "except", "with ")
            ):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif not stripped or stripped.startswith("#"):
                continue
            elif stripped and not line.startswith(" " * 4) and not line.startswith("\t"):
                current_depth = 0

        return max_depth

    def _detect_code_quality_issues(self):
        """检测代码质量问题"""
        logger.info(f"[{self.dna.target_name}] 检测代码质量问题...")

        for metrics in self.code_metrics:
            issues = []

            # 检查圈复杂度
            if metrics["complexity"]["cyclomatic_complexity"] > 20:
                issues.append(
                    {
                        "type": "high_complexity",
                        "severity": "warning",
                        "description": f"圈复杂度过高: {metrics['complexity']['cyclomatic_complexity']}",
                        "suggestion": "考虑重构复杂函数",
                    }
                )

            # 检查嵌套深度
            if metrics["complexity"]["max_nesting_depth"] > 4:
                issues.append(
                    {
                        "type": "deep_nesting",
                        "severity": "warning",
                        "description": f"嵌套过深: {metrics['complexity']['max_nesting_depth']}层",
                        "suggestion": "使用卫语句或提取函数减少嵌套",
                    }
                )

            # 检查函数长度
            if metrics["structure"]["avg_function_length"] > 50:
                issues.append(
                    {
                        "type": "long_functions",
                        "severity": "info",
                        "description": f"平均函数过长: {metrics['structure']['avg_function_length']:.1f}行",
                        "suggestion": "考虑拆分长函数",
                    }
                )

            # 检查注释比例
            if metrics["basic_metrics"]["comment_ratio"] < 0.1:
                issues.append(
                    {
                        "type": "low_comments",
                        "severity": "info",
                        "description": f"注释比例过低: {metrics['basic_metrics']['comment_ratio']:.2%}",
                        "suggestion": "增加代码注释",
                    }
                )

            metrics["quality_issues"] = issues

    def _identify_llm_refactoring_candidates(self):
        """识别适合LLM重构的代码候选"""
        logger.info(f"[{self.dna.target_name}] 识别LLM重构候选...")

        for metrics in self.code_metrics:
            # 高复杂度文件
            if metrics["complexity"]["cyclomatic_complexity"] > 30:
                self.llm_refactoring_candidates.append(
                    {
                        "file": metrics["path"],
                        "reason": "high_complexity",
                        "complexity": metrics["complexity"]["cyclomatic_complexity"],
                        "priority": "high",
                        "suggestion": "使用LLM重构高复杂度代码",
                    }
                )

            # 长函数
            for func in metrics.get("functions", []):
                if func["length"] > 100:
                    self.llm_refactoring_candidates.append(
                        {
                            "file": metrics["path"],
                            "function": func["name"],
                            "reason": "long_function",
                            "length": func["length"],
                            "priority": "medium",
                            "suggestion": f'使用LLM拆分函数 {func["name"]}',
                        }
                    )

        logger.info(
            f"[{self.dna.target_name}] 识别 {len(self.llm_refactoring_candidates)} 个LLM重构候选"
        )

    def _analyze_architecture(self):
        """分析架构"""
        logger.info(f"[{self.dna.target_name}] 分析架构...")

        # 统计架构指标
        total_classes = sum(m["structure"]["classes_count"] for m in self.code_metrics)
        total_functions = sum(m["structure"]["functions_count"] for m in self.code_metrics)

        # 分析类依赖关系
        class_usage = {}
        for metrics in self.code_metrics:
            for cls in metrics.get("classes", []):
                class_name = cls["name"]
                if class_name not in class_usage:
                    class_usage[class_name] = {"files": [], "total_methods": 0}
                class_usage[class_name]["files"].append(metrics["path"])
                class_usage[class_name]["total_methods"] += cls["methods_count"]

        # 找出大型类
        large_classes = [
            {"name": name, "methods": data["total_methods"], "files": len(data["files"])}
            for name, data in class_usage.items()
            if data["total_methods"] > 20
        ]

        self.architecture_insights = {
            "overview": {
                "total_files": len(self.code_metrics),
                "total_classes": total_classes,
                "total_functions": total_functions,
                "avg_functions_per_file": (
                    total_functions / len(self.code_metrics) if self.code_metrics else 0
                ),
            },
            "class_analysis": {
                "total_unique_classes": len(class_usage),
                "large_classes": large_classes,
            },
            "complexity_distribution": {
                "low_complexity": len(
                    [m for m in self.code_metrics if m["complexity"]["cyclomatic_complexity"] < 10]
                ),
                "medium_complexity": len(
                    [
                        m
                        for m in self.code_metrics
                        if 10 <= m["complexity"]["cyclomatic_complexity"] < 20
                    ]
                ),
                "high_complexity": len(
                    [m for m in self.code_metrics if m["complexity"]["cyclomatic_complexity"] >= 20]
                ),
            },
        }

    def _generate_refactoring_suggestions(self):
        """生成重构建议"""
        logger.info(f"[{self.dna.target_name}] 生成重构建议...")

        # 基于分析结果生成建议
        suggestions = []

        # 复杂度优化建议
        high_complexity_files = [
            m["path"] for m in self.code_metrics if m["complexity"]["cyclomatic_complexity"] > 25
        ]
        if high_complexity_files:
            suggestions.append(
                {
                    "category": "complexity_optimization",
                    "priority": "high",
                    "files": high_complexity_files[:5],
                    "action": "减少圈复杂度",
                    "description": f"发现 {len(high_complexity_files)} 个高复杂度文件，建议使用LLM辅助重构",
                    "llm_prompt": "请分析以下代码并提供重构建议以降低圈复杂度：",
                }
            )

        # 函数拆分建议
        long_function_files = [
            m["path"]
            for m in self.code_metrics
            if any(f["length"] > 80 for f in m.get("functions", []))
        ]
        if long_function_files:
            suggestions.append(
                {
                    "category": "function_refactoring",
                    "priority": "medium",
                    "files": long_function_files[:5],
                    "action": "拆分长函数",
                    "description": f"发现包含长函数的文件，建议使用LLM辅助拆分",
                    "llm_prompt": "请识别并拆分以下文件中的长函数：",
                }
            )

        # 架构优化建议
        if self.architecture_insights["class_analysis"]["large_classes"]:
            suggestions.append(
                {
                    "category": "architecture_improvement",
                    "priority": "medium",
                    "action": "优化大型类",
                    "description": f"发现 {len(self.architecture_insights['class_analysis']['large_classes'])} 个大型类",
                    "llm_prompt": "请分析以下大型类并提供重构建议：",
                }
            )

        # 性能优化建议
        suggestions.append(
            {
                "category": "performance_optimization",
                "priority": "low",
                "action": "性能优化",
                "description": "建议使用LLM分析性能瓶颈并优化",
                "llm_prompt": "请分析代码并提供性能优化建议：",
            }
        )

        self.refactoring_suggestions = suggestions

    def _generate_comprehensive_report(self):
        """生成综合报告"""
        report = {
            "bee_type": "VortexPrimeArchitect",
            "target_name": self.dna.target_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "files_analyzed": len(self.code_metrics),
                "total_functions": self.architecture_insights["overview"]["total_functions"],
                "total_classes": self.architecture_insights["overview"]["total_classes"],
                "high_complexity_files": len(
                    [m for m in self.code_metrics if m["complexity"]["cyclomatic_complexity"] > 20]
                ),
                "llm_refactoring_candidates": len(self.llm_refactoring_candidates),
            },
            "code_metrics": self.code_metrics,
            "llm_refactoring_candidates": self.llm_refactoring_candidates,
            "architecture_insights": self.architecture_insights,
            "refactoring_suggestions": self.refactoring_suggestions,
            "recommendations": self._generate_recommendations(),
        }

        # 保存报告
        report_file = (
            self.report_dir
            / f"{self.dna.target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"[{self.dna.target_name}] 综合报告已生成: {report_file}")

    def _generate_recommendations(self) -> list:
        """生成改进建议"""
        recommendations = []

        # 高优先级建议
        if self.architecture_insights["complexity_distribution"]["high_complexity"] > 0:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "complexity",
                    "action": "降低代码复杂度",
                    "description": f"有 {self.architecture_insights['complexity_distribution']['high_complexity']} 个文件复杂度过高，建议优先重构",
                }
            )

        # LLM重构建议
        if len(self.llm_refactoring_candidates) > 0:
            high_priority = [c for c in self.llm_refactoring_candidates if c["priority"] == "high"]
            if high_priority:
                recommendations.append(
                    {
                        "priority": "high",
                        "category": "llm_refactoring",
                        "action": "使用LLM重构高优先级代码",
                        "description": f"发现 {len(high_priority)} 个高优先级LLM重构候选，建议使用LLM辅助重构",
                    }
                )

        # 架构建议
        if self.architecture_insights["class_analysis"]["large_classes"]:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "architecture",
                    "action": "优化类设计",
                    "description": f"发现大型类，建议应用单一职责原则拆分",
                }
            )

        # 持续改进建议
        recommendations.append(
            {
                "priority": "low",
                "category": "continuous_improvement",
                "action": "建立代码审查流程",
                "description": "建议建立自动化代码质量检查流程，持续监控代码质量",
            }
        )

        return recommendations
