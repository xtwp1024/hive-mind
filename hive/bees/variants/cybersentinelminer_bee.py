# -*- coding: utf-8 -*-
"""
CyberSentinelMinerBee - 混合变异蜂种

特殊能力：杂交能力 - 模式匹配检测eval/exec + 扫描JSON/CSV/权重文件
变异率：10.70%

结合了CyberSentinel的安全扫描能力和CyberMiner的资产提取能力，
既能检测代码中的有毒模式，又能提取和分类项目的非代码资产。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from hive.bees.base_bee import BaseBee
from hive.config.settings import settings
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class CyberSentinelMinerBee(BaseBee):
    """
    混合变异蜂种 - 安全扫描 + 资产提取
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)

        # 初始化报告目录
        self.report_dir = settings.REPORT_DIR / "cybersentinelminer"
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 检测到的安全问题
        self.security_issues: List[Dict[str, Any]] = []

        # 提取的资产
        self.extracted_assets: List[Dict[str, Any]] = []

        # 有毒模式定义
        self.toxic_patterns = {
            "eval_exec": [
                r"\beval\s*\(",
                r"\bexec\s*\(",
                r"\b__import__\s*\(",
            ],
            "compile_exec": [
                r"\bcompile\s*\(",
            ],
            "pickle_unsafe": [
                r"\bpickle\.load\s*\(",
                r"\bpickle\.loads\s*\(",
            ],
            "subprocess_shell": [
                r"\bsubprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True",
                r"\bos\.system\s*\(",
            ],
            "hardcoded_secrets": [
                r'password\s*=\s*["\']([^"\']+)["\']',
                r'api_key\s*=\s*["\']([^"\']+)["\']',
                r'secret\s*=\s*["\']([^"\']+)["\']',
            ],
            "weak_crypto": [
                r"\bmd5\s*\(",
                r"\bsha1\s*\(",
            ],
        }

        # 资产类型定义
        self.asset_patterns = {
            "json_config": [".json"],
            "csv_data": [".csv"],
            "weight_file": [".pth", ".pt", ".h5", ".hdf5", ".ckpt", ".pkl", ".npy"],
            "data_file": [".parquet", ".feather", ".arrow", ".db", ".sqlite"],
        }

        # 扫描统计
        self.scan_stats = {
            "security": {
                "total_files": 0,
                "files_with_issues": 0,
                "total_issues": 0,
            },
            "assets": {
                "total_assets": 0,
                "by_type": {},
            },
        }

        logger.info(f"🔐 CyberSentinelMinerBee 初始化完成 | 变异率: 10.70%")

    def specialized_task(self):
        """
        执行混合任务：安全扫描 + 资产提取
        """
        logger.info("🔐 CyberSentinelMinerBee 开始执行混合任务")

        try:
            # 获取项目根目录
            from hive.config.settings import settings

            project_root = settings.BASE_DIR

            # 执行安全扫描
            logger.info("🛡️ 执行安全扫描...")
            self._scan_security_issues(project_root)

            # 执行资产提取
            logger.info("⛏️ 执行资产提取...")
            self._extract_assets(project_root)

            # 生成综合报告
            self._generate_comprehensive_report()

            logger.info(
                f"✅ CyberSentinelMinerBee 任务完成 | 安全问题: {len(self.security_issues)} | 资产: {len(self.extracted_assets)}"
            )

        except Exception as e:
            logger.error(f"❌ CyberSentinelMinerBee 任务失败: {e}", exc_info=True)
            raise

    def _scan_security_issues(self, project_root: Path):
        """
        扫描安全问题
        """
        python_files = list(project_root.rglob("*.py"))

        for py_file in python_files:
            try:
                self._scan_file_for_security(py_file)
            except Exception as e:
                logger.warning(f"安全扫描失败 {py_file}: {e}")

        logger.info(
            f"🛡️ 安全扫描完成 | 扫描文件: {self.scan_stats['security']['total_files']} | 发现问题: {self.scan_stats['security']['total_issues']}"
        )

    def _scan_file_for_security(self, file_path: Path):
        """
        扫描单个文件的安全问题
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            self.scan_stats["security"]["total_files"] += 1
            file_issues = []

            # 检查有毒模式
            for issue_type, patterns in self.toxic_patterns.items():
                for pattern in patterns:
                    matches = list(re.finditer(pattern, content))
                    if matches:
                        for match in matches:
                            line_num = content[: match.start()].count("\n") + 1
                            line_content = lines[line_num - 1].strip()

                            issue = {
                                "type": issue_type,
                                "file": str(file_path.relative_to(file_path.parent.parent.parent)),
                                "line": line_num,
                                "content": line_content,
                                "severity": self._get_severity(issue_type),
                            }

                            self.security_issues.append(issue)
                            file_issues.append(issue)

            if file_issues:
                self.scan_stats["security"]["files_with_issues"] += 1
                self.scan_stats["security"]["total_issues"] += len(file_issues)

        except Exception as e:
            logger.warning(f"扫描文件安全时出错 {file_path}: {e}")

    def _get_severity(self, issue_type: str) -> str:
        """
        根据问题类型确定严重程度
        """
        high_severity = [
            "eval_exec",
            "compile_exec",
            "pickle_unsafe",
            "subprocess_shell",
            "hardcoded_secrets",
        ]

        if issue_type in high_severity:
            return "HIGH"
        else:
            return "MEDIUM"

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

        return asset_info

    def _analyze_json_file(self, file_path: Path) -> Dict[str, Any]:
        """
        分析JSON配置文件
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 分析JSON结构
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
                # 读取前几行
                reader = csv.reader(f)
                headers = next(reader, [])
                sample_rows = [row for _, row in zip(range(5), reader)]

            return {
                "has_headers": len(headers) > 0,
                "column_count": len(headers),
                "sample_row_count": len(sample_rows),
                "estimated_rows": "unknown",  # 需要完整扫描才能确定
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

    def _generate_comprehensive_report(self):
        """
        生成综合报告
        """
        logger.info("📝 生成综合报告...")

        # 按严重程度分组安全问题
        by_severity = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for issue in self.security_issues:
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
                "bee_type": "CyberSentinelMinerBee",
                "mutation_rate": "10.70%",
                "scan_date": self._get_current_timestamp(),
            },
            "security_summary": {
                "total_files_scanned": self.scan_stats["security"]["total_files"],
                "files_with_issues": self.scan_stats["security"]["files_with_issues"],
                "total_security_issues": self.scan_stats["security"]["total_issues"],
                "high_severity_issues": len(by_severity["HIGH"]),
                "medium_severity_issues": len(by_severity["MEDIUM"]),
            },
            "asset_summary": {
                "total_assets_found": self.scan_stats["assets"]["total_assets"],
                "assets_by_type": self.scan_stats["assets"]["by_type"],
            },
            "security_issues": {
                "high_severity": by_severity["HIGH"][:10],
                "medium_severity": by_severity["MEDIUM"][:10],
            },
            "extracted_assets": {
                "by_type": {
                    asset_type: assets[:10]  # 限制数量
                    for asset_type, assets in assets_by_type.items()
                },
                "total_size_bytes": sum(asset["size_bytes"] for asset in self.extracted_assets),
            },
            "recommendations": recommendations,
            "summary": {
                "overall_status": "GOOD" if len(by_severity["HIGH"]) == 0 else "ATTENTION_REQUIRED",
                "has_security_risks": len(by_severity["HIGH"]) > 0,
                "has_valuable_assets": self.scan_stats["assets"]["total_assets"] > 0,
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

        # 安全建议
        high_count = len([i for i in self.security_issues if i["severity"] == "HIGH"])
        if high_count > 0:
            recommendations.append(f"发现 {high_count} 个高风险安全问题，需要立即处理")

        if self.scan_stats["security"]["total_issues"] > 0:
            recommendations.append(
                f"共发现 {self.scan_stats['security']['total_issues']} 个安全问题，建议逐步修复"
            )

        # 资产建议
        if self.scan_stats["assets"]["total_assets"] > 0:
            recommendations.append(
                f"发现 {self.scan_stats['assets']['total_assets']} 个资产文件，建议建立资产清单"
            )

        weight_count = self.scan_stats["assets"]["by_type"].get("weight_file", 0)
        if weight_count > 0:
            recommendations.append(f"发现 {weight_count} 个模型权重文件，建议做好版本管理")

        config_count = self.scan_stats["assets"]["by_type"].get("json_config", 0)
        if config_count > 0:
            recommendations.append(f"发现 {config_count} 个配置文件，建议文档化配置项")

        return recommendations

    def _get_current_timestamp(self) -> str:
        """
        获取当前时间戳
        """
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")
