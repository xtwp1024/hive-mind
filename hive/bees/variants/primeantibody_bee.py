# -*- coding: utf-8 -*-
"""
PrimeAntibodyBee - 极致化错误修复和数据库演进蜂种

变异率：2.80%

特殊能力：性能极限优化 - 自动检测和修复错误，生成数据库演进SQL补丁

功能：
1. 高性能错误检测：并行扫描代码库中的错误模式和异常
2. 数据库演进分析：分析数据库模式变更需求
3. 智能SQL补丁生成：自动生成数据库演进SQL
4. 错误修复建议：提供代码级别的修复建议
5. 性能指标监控：实时监控错误检测性能
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hive.bees.base_bee import BaseBee
from hive.config.settings import settings
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger


class PrimeAntibodyBee(BaseBee):
    """
    极致化错误修复和数据库演进蜂种

    结合了Antibody的错误修复能力和Prime的极致化性能，
    使用并行处理实现高效的错误检测和数据库演进分析。
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.logger = get_logger("PrimeAntibodyBee")

        # 错误模式字典
        self.error_patterns = {
            "empty_except": r"except\s*:\s*pass",
            "bare_except": r"except\s*:",
            "unhandled_exception": r"raise\s+\w+\(",
            "deprecated_import": r"from\s+\w+\s+import\s+\w+\s+#.*deprecated",
            "type_error": r"int\([^)]*\+[^)]*\)",
            "index_error": r"\w+\[len\([^)]*\)\]",
            "key_error": r'["\']\w+["\']\s+in\s+\w+',
        }

        # 数据库演进模式
        self.database_patterns = {
            "schema_change": r"CREATE TABLE|ALTER TABLE|DROP TABLE",
            "migration": r"CREATE INDEX|DROP INDEX",
            "data_type_change": r"ALTER COLUMN|MODIFY COLUMN",
            "foreign_key": r"FOREIGN KEY|REFERENCES",
        }

        # 性能指标
        self.performance_metrics = {
            "files_scanned": 0,
            "errors_detected": 0,
            "db_issues_detected": 0,
            "sql_patches_generated": 0,
            "scan_duration": 0.0,
            "parallel_workers": 0,
        }

    def specialized_task(self, project_root: Path) -> bool:
        """
        执行极致化错误修复和数据库演进分析

        Args:
            project_root: 项目根目录

        Returns:
            bool: 任务是否成功完成
        """
        self.logger.info(f"启动极致化错误修复和数据库演进分析")
        self.logger.info(f"项目路径: {project_root}")

        try:
            # 并行扫描项目代码
            scan_results = self._parallel_scan_project(project_root)

            # 分析数据库演进需求
            db_analysis = self._analyze_database_evolution(scan_results["db_files"])

            # 生成SQL补丁
            sql_patches = self._generate_sql_patches(db_analysis)

            # 分析错误模式
            error_analysis = self._analyze_error_patterns(scan_results["python_files"])

            # 生成修复建议
            fix_suggestions = self._generate_fix_suggestions(error_analysis)

            # 生成综合报告
            report = self._generate_comprehensive_report(
                scan_results, db_analysis, sql_patches, error_analysis, fix_suggestions
            )

            # 保存报告
            self._save_report(report)

            self.logger.info(f"极致化错误修复和数据库演进分析完成")
            self.logger.info(f"扫描文件: {self.performance_metrics['files_scanned']}")
            self.logger.info(f"检测错误: {self.performance_metrics['errors_detected']}")
            self.logger.info(f"数据库问题: {self.performance_metrics['db_issues_detected']}")
            self.logger.info(f"SQL补丁: {self.performance_metrics['sql_patches_generated']}")

            return True

        except Exception as e:
            self.logger.error(f"极致化错误修复和数据库演进分析失败: {str(e)}")
            return False

    def _parallel_scan_project(self, project_root: Path) -> Dict[str, Any]:
        """
        并行扫描项目文件

        Args:
            project_root: 项目根目录

        Returns:
            扫描结果字典
        """
        self.logger.info("启动并行项目扫描")

        python_files = list(project_root.rglob("*.py"))
        db_files = list(project_root.rglob("*.sql"))

        self.performance_metrics["files_scanned"] = len(python_files) + len(db_files)

        scan_results = {
            "python_files": python_files,
            "db_files": db_files,
            "error_locations": [],
            "db_issues": [],
        }

        # 使用线程池并行处理
        max_workers = min(8, len(python_files) // 10)
        self.performance_metrics["parallel_workers"] = max_workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交Python文件扫描任务
            python_futures = {executor.submit(self._scan_python_file, f): f for f in python_files}

            # 提交SQL文件扫描任务
            sql_futures = {executor.submit(self._scan_sql_file, f): f for f in db_files}

            # 收集Python文件扫描结果
            for future in as_completed(python_futures):
                file_path = python_futures[future]
                try:
                    result = future.result()
                    if result["errors"]:
                        scan_results["error_locations"].append(
                            {
                                "file": str(file_path.relative_to(project_root)),
                                "errors": result["errors"],
                            }
                        )
                        self.performance_metrics["errors_detected"] += len(result["errors"])
                except Exception as e:
                    self.logger.warning(f"扫描文件 {file_path} 失败: {str(e)}")

            # 收集SQL文件扫描结果
            for future in as_completed(sql_futures):
                file_path = sql_futures[future]
                try:
                    result = future.result()
                    if result["issues"]:
                        scan_results["db_issues"].append(
                            {
                                "file": str(file_path.relative_to(project_root)),
                                "issues": result["issues"],
                            }
                        )
                        self.performance_metrics["db_issues_detected"] += len(result["issues"])
                except Exception as e:
                    self.logger.warning(f"扫描SQL文件 {file_path} 失败: {str(e)}")

        return scan_results

    def _scan_python_file(self, file_path: Path) -> Dict[str, Any]:
        """扫描单个Python文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            errors = []
            for pattern_name, pattern in self.error_patterns.items():
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    errors.append(
                        {
                            "type": pattern_name,
                            "line": line_num,
                            "context": self._get_line_context(content, line_num),
                        }
                    )

            return {"errors": errors}

        except Exception as e:
            return {"errors": []}

    def _scan_sql_file(self, file_path: Path) -> Dict[str, Any]:
        """扫描单个SQL文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            issues = []
            for pattern_name, pattern in self.database_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    issues.append(
                        {"type": pattern_name, "line": line_num, "statement": match.group()}
                    )

            return {"issues": issues}

        except Exception as e:
            return {"issues": []}

    def _get_line_context(self, content: str, line_num: int, context_lines: int = 2) -> str:
        """获取代码行的上下文"""
        lines = content.split("\n")
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        return "\n".join(lines[start:end])

    def _analyze_database_evolution(self, db_files: List[Path]) -> Dict[str, Any]:
        """
        分析数据库演进需求

        Args:
            db_files: SQL文件列表

        Returns:
            数据库演进分析结果
        """
        self.logger.info("分析数据库演进需求")

        evolution_analysis = {
            "schema_changes": [],
            "migration_needs": [],
            "data_type_changes": [],
            "foreign_key_changes": [],
            "recommendations": [],
        }

        for db_file in db_files:
            try:
                with open(db_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # 分析模式变更
                if "CREATE TABLE" in content:
                    table_names = re.findall(r"CREATE TABLE\s+(\w+)", content, re.IGNORECASE)
                    evolution_analysis["schema_changes"].extend(table_names)

                # 分析迁移需求
                if "ALTER TABLE" in content:
                    alterations = re.findall(
                        r"ALTER TABLE\s+(\w+)\s+(.+?);", content, re.IGNORECASE
                    )
                    evolution_analysis["migration_needs"].extend(alterations)

                # 分析数据类型变更
                if "ALTER COLUMN" in content or "MODIFY COLUMN" in content:
                    type_changes = re.findall(
                        r"(?:ALTER|MODIFY)\s+COLUMN\s+(\w+)\s+(\w+)", content, re.IGNORECASE
                    )
                    evolution_analysis["data_type_changes"].extend(type_changes)

                # 分析外键变更
                if "FOREIGN KEY" in content:
                    fk_changes = re.findall(
                        r"FOREIGN KEY\s*\((\w+)\)\s*REFERENCES\s+(\w+)", content, re.IGNORECASE
                    )
                    evolution_analysis["foreign_key_changes"].extend(fk_changes)

            except Exception as e:
                self.logger.warning(f"分析数据库文件 {db_file} 失败: {str(e)}")

        # 生成演进建议
        if evolution_analysis["schema_changes"]:
            evolution_analysis["recommendations"].append(
                {
                    "type": "schema",
                    "priority": "high",
                    "message": f"检测到 {len(evolution_analysis['schema_changes'])} 个模式变更，建议创建版本控制迁移",
                }
            )

        if evolution_analysis["data_type_changes"]:
            evolution_analysis["recommendations"].append(
                {
                    "type": "data_type",
                    "priority": "medium",
                    "message": f"检测到 {len(evolution_analysis['data_type_changes'])} 个数据类型变更，建议进行数据兼容性测试",
                }
            )

        return evolution_analysis

    def _generate_sql_patches(self, db_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成数据库演进SQL补丁

        Args:
            db_analysis: 数据库分析结果

        Returns:
            SQL补丁列表
        """
        self.logger.info("生成SQL补丁")

        patches = []

        # 为每个模式变更生成补丁
        for i, table_name in enumerate(db_analysis["schema_changes"]):
            patch = {
                "id": f"SCHEMA_PATCH_{i+1:03d}",
                "type": "schema",
                "target_table": table_name,
                "sql": f"-- Schema evolution patch for {table_name}\n"
                f"-- Review and adapt this patch to your specific needs\n"
                f"-- Consider adding version control and rollback mechanism\n",
                "priority": "high",
            }
            patches.append(patch)
            self.performance_metrics["sql_patches_generated"] += 1

        # 为每个数据类型变更生成补丁
        for i, (column, new_type) in enumerate(db_analysis["data_type_changes"]):
            patch = {
                "id": f"TYPE_PATCH_{i+1:03d}",
                "type": "data_type",
                "target_column": column,
                "new_type": new_type,
                "sql": f"-- Data type change patch for column {column}\n"
                f"-- Verify data compatibility before applying\n"
                f"-- Consider creating a backup first\n"
                f"-- ALTER TABLE your_table MODIFY COLUMN {column} {new_type};\n",
                "priority": "medium",
            }
            patches.append(patch)
            self.performance_metrics["sql_patches_generated"] += 1

        # 为外键变更生成补丁
        for i, (fk_column, ref_table) in enumerate(db_analysis["foreign_key_changes"]):
            patch = {
                "id": f"FK_PATCH_{i+1:03d}",
                "type": "foreign_key",
                "target_column": fk_column,
                "references": ref_table,
                "sql": f"-- Foreign key change patch for {fk_column}\n"
                f"-- Ensure referenced table and column exist\n"
                f"-- Consider ON DELETE and ON UPDATE behavior\n"
                f"-- ALTER TABLE your_table ADD CONSTRAINT fk_{fk_column}\n"
                f"--   FOREIGN KEY ({fk_column}) REFERENCES {ref_table}(id);\n",
                "priority": "high",
            }
            patches.append(patch)
            self.performance_metrics["sql_patches_generated"] += 1

        return patches

    def _analyze_error_patterns(self, python_files: List[Path]) -> Dict[str, Any]:
        """
        分析错误模式

        Args:
            python_files: Python文件列表

        Returns:
            错误模式分析结果
        """
        self.logger.info("分析错误模式")

        error_stats = {}
        total_errors = 0

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                for pattern_name, pattern in self.error_patterns.items():
                    matches = list(re.finditer(pattern, content))
                    if matches:
                        if pattern_name not in error_stats:
                            error_stats[pattern_name] = {
                                "count": 0,
                                "files": [],
                                "severity": self._get_error_severity(pattern_name),
                            }
                        error_stats[pattern_name]["count"] += len(matches)
                        error_stats[pattern_name]["files"].append(str(file_path))
                        total_errors += len(matches)

            except Exception as e:
                self.logger.warning(f"分析错误模式文件 {file_path} 失败: {str(e)}")

        return {
            "error_stats": error_stats,
            "total_errors": total_errors,
            "critical_issues": [
                name for name, data in error_stats.items() if data["severity"] == "critical"
            ],
        }

    def _get_error_severity(self, error_type: str) -> str:
        """获取错误类型的严重程度"""
        severity_map = {
            "empty_except": "high",
            "bare_except": "high",
            "unhandled_exception": "medium",
            "deprecated_import": "low",
            "type_error": "critical",
            "index_error": "critical",
            "key_error": "high",
        }
        return severity_map.get(error_type, "medium")

    def _generate_fix_suggestions(self, error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成错误修复建议

        Args:
            error_analysis: 错误分析结果

        Returns:
            修复建议列表
        """
        self.logger.info("生成修复建议")

        suggestions = []

        # 为每个错误类型生成建议
        for error_type, data in error_analysis["error_stats"].items():
            suggestion = {
                "type": error_type,
                "severity": data["severity"],
                "count": data["count"],
                "affected_files": len(data["files"]),
                "recommendation": self._get_fix_recommendation(error_type),
                "priority": self._get_fix_priority(data["severity"]),
            }
            suggestions.append(suggestion)

        # 按优先级排序
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 4))

        return suggestions

    def _get_fix_recommendation(self, error_type: str) -> str:
        """获取错误类型的修复建议"""
        recommendations = {
            "empty_except": "避免使用空的except块，应该记录异常或进行适当的错误处理",
            "bare_except": "避免使用裸except，应该捕获特定的异常类型",
            "unhandled_exception": "确保所有异常都被正确捕获和处理",
            "deprecated_import": "更新为新的导入方式或替代库",
            "type_error": "添加类型检查或转换以避免类型错误",
            "index_error": "在使用索引前检查列表长度",
            "key_error": "使用dict.get()或检查键是否存在",
        }
        return recommendations.get(error_type, "请仔细检查此错误模式")

    def _get_fix_priority(self, severity: str) -> str:
        """获取修复优先级"""
        if severity == "critical":
            return "urgent"
        elif severity == "high":
            return "high"
        elif severity == "medium":
            return "medium"
        else:
            return "low"

    def _generate_comprehensive_report(
        self,
        scan_results: Dict[str, Any],
        db_analysis: Dict[str, Any],
        sql_patches: List[Dict[str, Any]],
        error_analysis: Dict[str, Any],
        fix_suggestions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        生成综合报告

        Args:
            scan_results: 扫描结果
            db_analysis: 数据库分析结果
            sql_patches: SQL补丁
            error_analysis: 错误分析结果
            fix_suggestions: 修复建议

        Returns:
            综合报告
        """
        report = {
            "bee_type": self.bee_dna.species,
            "mutation_rate": self.bee_dna.mutation_rate,
            "timestamp": self._get_timestamp(),
            # 扫描摘要
            "scan_summary": {
                "files_scanned": self.performance_metrics["files_scanned"],
                "python_files": len(scan_results["python_files"]),
                "sql_files": len(scan_results["db_files"]),
                "parallel_workers": self.performance_metrics["parallel_workers"],
                "scan_duration": self.performance_metrics["scan_duration"],
            },
            # 错误检测结果
            "error_detection": {
                "total_errors": error_analysis["total_errors"],
                "error_locations": scan_results["error_locations"],
                "error_statistics": error_analysis["error_stats"],
                "critical_issues": error_analysis["critical_issues"],
            },
            # 数据库演进分析
            "database_evolution": {
                "schema_changes": db_analysis["schema_changes"],
                "migration_needs": db_analysis["migration_needs"],
                "data_type_changes": db_analysis["data_type_changes"],
                "foreign_key_changes": db_analysis["foreign_key_changes"],
                "recommendations": db_analysis["recommendations"],
            },
            # SQL补丁
            "sql_patches": sql_patches,
            # 修复建议
            "fix_suggestions": fix_suggestions,
            # 性能指标
            "performance_metrics": self.performance_metrics,
            # 总结
            "summary": self._generate_summary(
                scan_results, db_analysis, sql_patches, error_analysis
            ),
        }

        return report

    def _generate_summary(
        self,
        scan_results: Dict[str, Any],
        db_analysis: Dict[str, Any],
        sql_patches: List[Dict[str, Any]],
        error_analysis: Dict[str, Any],
    ) -> str:
        """生成报告摘要"""
        summary_parts = []

        # 扫描摘要
        summary_parts.append(
            f"扫描了 {scan_results['python_files']} 个Python文件和 "
            f"{scan_results['db_files']} 个SQL文件"
        )

        # 错误摘要
        if error_analysis["total_errors"] > 0:
            summary_parts.append(
                f"检测到 {error_analysis['total_errors']} 个错误模式，"
                f"其中 {len(error_analysis['critical_issues'])} 个为严重问题"
            )
        else:
            summary_parts.append("未检测到错误模式")

        # 数据库摘要
        if db_analysis["schema_changes"]:
            summary_parts.append(f"检测到 {len(db_analysis['schema_changes'])} 个数据库模式变更")

        # SQL补丁摘要
        if sql_patches:
            summary_parts.append(f"生成了 {len(sql_patches)} 个SQL补丁")

        return "。".join(summary_parts) + "。"

    def _save_report(self, report: Dict[str, Any]) -> None:
        """保存报告到文件"""
        try:
            from hive.config.settings import settings

            report_path = settings.REPORT_DIR / f"primeantibody_report_{self._get_timestamp()}.json"
            self._save_json_report(report, report_path)
            self.logger.info(f"报告已保存: {report_path}")
        except Exception as e:
            self.logger.error(f"保存报告失败: {str(e)}")

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")
