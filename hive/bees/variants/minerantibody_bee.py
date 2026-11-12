# -*- coding: utf-8 -*-
"""
MinerAntibodyBee - 混合变异蜂种

特殊能力：杂交能力 - 扫描JSON/CSV/权重文件 + 生成SQL补丁
变异率：11.50%

结合了Miner的资产提取能力和Antibody的错误修复能力，
既能提取和分类项目的非代码资产，又能分析和生成数据库修复补丁。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from hive.bees.base_bee import BaseBee
from hive.config.settings import settings
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class MinerAntibodyBee(BaseBee):
    """
    混合变异蜂种 - 资产提取 + 数据库错误修复
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)

        # 初始化报告目录
        self.report_dir = settings.REPORT_DIR / "minerantibody"
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 提取的资产
        self.extracted_assets: List[Dict[str, Any]] = []

        # 数据库问题
        self.database_issues: List[Dict[str, Any]] = []

        # 生成的SQL补丁
        self.sql_patches: List[Dict[str, Any]] = []

        # 资产类型定义
        self.asset_patterns = {
            "json_config": [".json"],
            "csv_data": [".csv"],
            "weight_file": [".pth", ".pt", ".h5", ".hdf5", ".ckpt", ".pkl", ".npy"],
            "data_file": [".parquet", ".feather", ".arrow", ".db", ".sqlite"],
        }

        # 数据库问题模式
        self.db_issue_patterns = {
            "missing_primary_key": r"CREATE TABLE\s+(\w+)\s*\([^)]*\)(?!.*PRIMARY KEY)",
            "missing_index": r"CREATE TABLE\s+(\w+)\s*\([^)]*\)(?!.*INDEX)",
            "missing_foreign_key": [
                r"CREATE TABLE\s+(\w+)\s*\([^)]*(?:user_id|owner_id|ref_id)[^)]*\)(?!.*FOREIGN KEY)",
            ],
            "varchar_without_length": r"CREATE TABLE\s+\w+\s*\([^)]*VARCHAR\s*\)",
            "text_column": r"CREATE TABLE\s+\w+\s*\([^)]*TEXT\s*[^)]*\)",
            "missing_timestamp": r"CREATE TABLE\s+(\w+)\s*\([^)]*\)(?!.*(?:created_at|updated_at|timestamp))",
            "not_null_missing": r"CREATE TABLE\s+\w+\s*\([^)]*\w+\s+(?:INT|VARCHAR|TEXT|FLOAT)\s*(?!.*NOT NULL)",
        }

        # 扫描统计
        self.scan_stats = {
            "assets": {
                "total_assets": 0,
                "by_type": {},
            },
            "database": {
                "total_issues": 0,
                "by_type": {},
            },
            "patches": {
                "total_patches": 0,
            },
        }

        logger.info(f"⚕️ MinerAntibodyBee 初始化完成 | 变异率: 11.50%")

    def specialized_task(self, target_path: Path):
        """
        执行混合任务：资产提取 + 数据库错误修复
        """
        logger.info(f"⚕️ MinerAntibodyBee 开始执行混合任务: {target_path}")

        try:
            # 目标目录
            project_root = target_path

            # 执行资产提取
            logger.info("⛏️ 执行资产提取...")
            self._extract_assets(project_root)

            # 执行数据库问题分析
            logger.info("🔬 执行数据库问题分析...")
            self._analyze_database_issues(project_root)

            # 生成SQL补丁
            logger.info("💊 生成SQL补丁...")
            self._generate_sql_patches()

            # 生成综合报告
            self._generate_comprehensive_report()

            logger.info(
                f"✅ MinerAntibodyBee 任务完成 | 资产: {len(self.extracted_assets)} | 问题: {len(self.database_issues)} | 补丁: {len(self.sql_patches)}"
            )

        except Exception as e:
            logger.error(f"❌ MinerAntibodyBee 任务失败: {e}", exc_info=True)
            raise

    def _extract_assets(self, project_root: Path):
        """
        提取项目资产
        """
        for asset_type, extensions in self.asset_patterns.items():
            for ext in extensions:
                files = list(project_root.rglob(f"*{ext}"))

                for file_path in files:
                    try:
                        asset_info = self._analyze_asset(file_path, asset_type)
                        if asset_info:
                            self.extracted_assets.append(asset_info)
                            self.scan_stats["assets"]["total_assets"] += 1

                            # 统计资产类型
                            if asset_type not in self.scan_stats["assets"]["by_type"]:
                                self.scan_stats["assets"]["by_type"][asset_type] = 0
                            self.scan_stats["assets"]["by_type"][asset_type] += 1

                    except Exception as e:
                        logger.warning(f"分析资产失败 {file_path}: {e}")

        logger.info(f"⛏️ 资产提取完成 | 提取资产: {self.scan_stats['assets']['total_assets']}")

    def _analyze_asset(self, file_path: Path, asset_type: str) -> Dict[str, Any]:
        """
        分析单个资产文件
        """
        file_size = file_path.stat().st_size
        relative_path = str(file_path.relative_to(file_path.parent.parent.parent))

        asset_info = {
            "type": asset_type,
            "path": relative_path,
            "size_bytes": file_size,
            "size_human": self._human_readable_size(file_size),
            "extension": file_path.suffix,
        }

        # 根据资产类型进行特定分析
        if asset_type == "json_config":
            asset_info.update(self._analyze_json_file(file_path))
        elif asset_type == "csv_data":
            asset_info.update(self._analyze_csv_file(file_path))
        elif asset_type == "weight_file":
            asset_info["is_model_weight"] = True
            asset_info["estimated_size_category"] = self._categorize_weight_size(file_size)
        elif asset_type == "data_file":
            asset_info["is_database"] = file_path.suffix in [".db", ".sqlite"]

        return asset_info

    def _analyze_json_file(self, file_path: Path) -> Dict[str, Any]:
        """
        分析JSON配置文件
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "is_valid_json": True,
                "key_count": len(self._flatten_keys(data)),
                "has_nested_structure": (
                    any(isinstance(v, (dict, list)) for v in data.values())
                    if isinstance(data, dict)
                    else False
                ),
                "top_level_keys": list(data.keys()) if isinstance(data, dict) else [],
            }
        except Exception as e:
            return {
                "is_valid_json": False,
                "error": str(e),
            }

    def _analyze_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """
        分析CSV数据文件
        """
        try:
            import csv

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                sample_rows = [row for _, row in zip(range(5), reader)]

            return {
                "has_headers": len(headers) > 0,
                "column_count": len(headers),
                "sample_row_count": len(sample_rows),
            }
        except Exception as e:
            return {
                "has_headers": False,
                "error": str(e),
            }

    def _categorize_weight_size(self, size_bytes: int) -> str:
        """
        分类模型权重大小
        """
        mb = size_bytes / (1024 * 1024)
        if mb < 1:
            return "small"
        elif mb < 100:
            return "medium"
        else:
            return "large"

    def _flatten_keys(self, data: Any, parent_key: str = "") -> List[str]:
        """
        展平JSON键
        """
        items = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                items.append(new_key)
                if isinstance(v, (dict, list)):
                    items.extend(self._flatten_keys(v, new_key))
        return items

    def _human_readable_size(self, size_bytes: int) -> str:
        """
        转换为人类可读的文件大小
        """
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def _analyze_database_issues(self, project_root: Path):
        """
        分析数据库问题
        """
        # 扫描SQL文件和包含SQL的Python文件
        sql_files = list(project_root.rglob("*.sql"))
        python_files = list(project_root.rglob("*.py"))

        # 分析SQL文件
        for sql_file in sql_files:
            try:
                self._analyze_sql_file(sql_file)
            except Exception as e:
                logger.warning(f"分析SQL文件失败 {sql_file}: {e}")

        # 分析Python文件中的SQL语句
        for py_file in python_files:
            try:
                self._analyze_python_for_sql(py_file)
            except Exception as e:
                logger.debug(f"分析Python文件SQL失败 {py_file}: {e}")

        logger.info(
            f"🔬 数据库问题分析完成 | 发现问题: {self.scan_stats['database']['total_issues']}"
        )

    def _analyze_sql_file(self, file_path: Path):
        """
        分析SQL文件
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            file_issues = []

            # 检查每种数据库问题模式
            for issue_type, patterns in self.db_issue_patterns.items():
                if isinstance(patterns, list):
                    for pattern in patterns:
                        self._check_pattern(
                            content, lines, file_path, issue_type, pattern, file_issues
                        )
                else:
                    self._check_pattern(
                        content, lines, file_path, issue_type, patterns, file_issues
                    )

            if file_issues:
                self.database_issues.extend(file_issues)
                self.scan_stats["database"]["total_issues"] += len(file_issues)

                # 统计问题类型
                for issue in file_issues:
                    issue_type = issue["type"]
                    if issue_type not in self.scan_stats["database"]["by_type"]:
                        self.scan_stats["database"]["by_type"][issue_type] = 0
                    self.scan_stats["database"]["by_type"][issue_type] += 1

        except Exception as e:
            logger.warning(f"分析SQL文件时出错 {file_path}: {e}")

    def _analyze_python_for_sql(self, file_path: Path):
        """
        分析Python文件中的SQL语句
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            # 查找包含CREATE TABLE的SQL语句
            sql_pattern = r"CREATE TABLE\s+[\w\s\(\),]+\);?"
            sql_matches = re.finditer(sql_pattern, content, re.IGNORECASE | re.MULTILINE)

            file_issues = []

            for match in sql_matches:
                sql_statement = match.group(0)

                # 分析SQL语句
                for issue_type, patterns in self.db_issue_patterns.items():
                    if isinstance(patterns, list):
                        for pattern in patterns:
                            if re.search(pattern, sql_statement, re.IGNORECASE):
                                line_num = content[: match.start()].count("\n") + 1
                                issue = {
                                    "type": issue_type,
                                    "file": str(
                                        file_path.relative_to(file_path.parent.parent.parent)
                                    ),
                                    "line": line_num,
                                    "sql_statement": sql_statement[:200],  # 限制长度
                                    "severity": "MEDIUM",
                                }
                                file_issues.append(issue)
                    else:
                        if re.search(patterns, sql_statement, re.IGNORECASE):
                            line_num = content[: match.start()].count("\n") + 1
                            issue = {
                                "type": issue_type,
                                "file": str(file_path.relative_to(file_path.parent.parent.parent)),
                                "line": line_num,
                                "sql_statement": sql_statement[:200],
                                "severity": "MEDIUM",
                            }
                            file_issues.append(issue)

            if file_issues:
                self.database_issues.extend(file_issues)
                self.scan_stats["database"]["total_issues"] += len(file_issues)

        except Exception as e:
            logger.debug(f"分析Python文件SQL时出错 {file_path}: {e}")

    def _check_pattern(
        self,
        content: str,
        lines: List[str],
        file_path: Path,
        issue_type: str,
        pattern: str,
        file_issues: List[Dict[str, Any]],
    ):
        """
        检查特定的数据库问题模式
        """
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            line_num = content[: match.start()].count("\n") + 1
            line_content = lines[line_num - 1].strip()

            issue = {
                "type": issue_type,
                "file": str(file_path.relative_to(file_path.parent.parent.parent)),
                "line": line_num,
                "content": line_content[:200],
                "severity": self._get_issue_severity(issue_type),
            }
            file_issues.append(issue)

    def _get_issue_severity(self, issue_type: str) -> str:
        """
        根据问题类型确定严重程度
        """
        high_severity = ["missing_primary_key", "missing_foreign_key"]
        medium_severity = ["missing_index", "varchar_without_length", "missing_timestamp"]

        if issue_type in high_severity:
            return "HIGH"
        elif issue_type in medium_severity:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_sql_patches(self):
        """
        生成SQL补丁
        """
        # 按文件分组问题
        issues_by_file = {}
        for issue in self.database_issues:
            file = issue["file"]
            if file not in issues_by_file:
                issues_by_file[file] = []
            issues_by_file[file].append(issue)

        # 为每个文件生成补丁
        for file, issues in issues_by_file.items():
            patch = {
                "target_file": file,
                "issue_count": len(issues),
                "patch_statements": [],
            }

            # 根据问题类型生成修复语句
            for issue in issues:
                statements = self._generate_fix_statements(issue)
                patch["patch_statements"].extend(statements)

            if patch["patch_statements"]:
                self.sql_patches.append(patch)
                self.scan_stats["patches"]["total_patches"] += len(patch["patch_statements"])

        logger.info(
            f"💊 SQL补丁生成完成 | 补丁数: {len(self.sql_patches)} | 语句数: {self.scan_stats['patches']['total_patches']}"
        )

    def _generate_fix_statements(self, issue: Dict[str, Any]) -> List[str]:
        """
        根据问题生成修复语句
        """
        statements = []
        issue_type = issue["type"]

        if issue_type == "missing_primary_key":
            statements.append("-- 添加主键")
            statements.append("ALTER TABLE table_name ADD COLUMN id SERIAL PRIMARY KEY;")

        elif issue_type == "missing_index":
            statements.append("-- 添加索引")
            statements.append("CREATE INDEX idx_table_column ON table_name(column_name);")

        elif issue_type == "missing_foreign_key":
            statements.append("-- 添加外键约束")
            statements.append(
                "ALTER TABLE table_name ADD CONSTRAINT fk_name FOREIGN KEY (column_name) REFERENCES referenced_table(id);"
            )

        elif issue_type == "varchar_without_length":
            statements.append("-- 修改VARCHAR列，添加长度限制")
            statements.append("ALTER TABLE table_name ALTER COLUMN column_name TYPE VARCHAR(255);")

        elif issue_type == "missing_timestamp":
            statements.append("-- 添加时间戳列")
            statements.append(
                "ALTER TABLE table_name ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"
            )
            statements.append(
                "ALTER TABLE table_name ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"
            )

        elif issue_type == "not_null_missing":
            statements.append("-- 添加NOT NULL约束")
            statements.append("ALTER TABLE table_name ALTER COLUMN column_name SET NOT NULL;")

        return statements

    def _generate_comprehensive_report(self):
        """
        生成综合报告
        """
        logger.info("📝 生成综合报告...")

        # 按严重程度分组数据库问题
        by_severity = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for issue in self.database_issues:
            by_severity[issue["severity"]].append(issue)

        # 按类型分组资产
        assets_by_type = {}
        for asset in self.extracted_assets:
            asset_type = asset["type"]
            if asset_type not in assets_by_type:
                assets_by_type[asset_type] = []
            assets_by_type[asset_type].append(asset)

        # 生成建议
        recommendations = self._generate_recommendations()

        # 构建报告
        report = {
            "scan_metadata": {
                "bee_type": "MinerAntibodyBee",
                "mutation_rate": "11.50%",
                "scan_date": self._get_current_timestamp(),
            },
            "asset_summary": {
                "total_assets_found": self.scan_stats["assets"]["total_assets"],
                "assets_by_type": self.scan_stats["assets"]["by_type"],
                "total_size_bytes": sum(asset["size_bytes"] for asset in self.extracted_assets),
            },
            "database_summary": {
                "total_issues": self.scan_stats["database"]["total_issues"],
                "issues_by_type": self.scan_stats["database"]["by_type"],
                "high_severity_issues": len(by_severity["HIGH"]),
                "medium_severity_issues": len(by_severity["MEDIUM"]),
            },
            "patches_summary": {
                "total_patches": self.scan_stats["patches"]["total_patches"],
                "files_with_patches": len(self.sql_patches),
            },
            "extracted_assets": {
                "by_type": {
                    asset_type: assets[:10] for asset_type, assets in assets_by_type.items()
                },
            },
            "database_issues": {
                "high_severity": by_severity["HIGH"][:10],
                "medium_severity": by_severity["MEDIUM"][:10],
            },
            "sql_patches": self.sql_patches[:10],
            "recommendations": recommendations,
            "summary": {
                "has_assets": len(self.extracted_assets) > 0,
                "has_database_issues": len(self.database_issues) > 0,
                "has_patches": len(self.sql_patches) > 0,
                "action_required": len(by_severity["HIGH"]) > 0,
            },
        }

        # 保存报告
        report_path = self.report_dir / f"comprehensive_report_{self._get_current_timestamp()}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 综合报告已生成: {report_path}")

    def _generate_recommendations(self) -> List[str]:
        """
        生成建议
        """
        recommendations = []

        # 资产建议
        if self.scan_stats["assets"]["total_assets"] > 0:
            recommendations.append(
                f"发现 {self.scan_stats['assets']['total_assets']} 个资产文件，建议建立资产清单"
            )

        weight_count = self.scan_stats["assets"]["by_type"].get("weight_file", 0)
        if weight_count > 0:
            recommendations.append(f"发现 {weight_count} 个模型权重文件，建议做好版本管理")

        # 数据库建议
        if self.scan_stats["database"]["total_issues"] > 0:
            recommendations.append(
                f"发现 {self.scan_stats['database']['total_issues']} 个数据库问题，建议逐一修复"
            )

        high_count = len([i for i in self.database_issues if i["severity"] == "HIGH"])
        if high_count > 0:
            recommendations.append(f"存在 {high_count} 个高风险数据库问题，需要立即处理")

        # 补丁建议
        if self.scan_stats["patches"]["total_patches"] > 0:
            recommendations.append(
                f"生成了 {self.scan_stats['patches']['total_patches']} 条SQL补丁，建议在测试环境验证后应用"
            )

        return recommendations

    def _get_current_timestamp(self) -> str:
        """
        获取当前时间戳
        """
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")
