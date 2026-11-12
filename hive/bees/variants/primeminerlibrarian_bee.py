# -*- coding: utf-8 -*-
"""
PrimeMinerLibrarianBee - 极致化资产提取和项目架构总结蜂种

变异率：1.70%

特殊能力：性能极限优化 - 自动化增强 - 资产提取 + 项目架构总结

功能：
1. 高性能资产提取：并行扫描和提取非代码资产（JSON、CSV、权重文件等）
2. 项目架构总结：分析项目结构并生成架构文档
3. 资产分类和分析：对提取的资产进行分类和深度分析
4. README生成：自动生成项目架构总结文档
5. 性能优化：使用并行处理提升资产提取效率
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from hive.bees.base_bee import BaseBee
from hive.config.settings import settings
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger


class PrimeMinerLibrarianBee(BaseBee):
    """
    极致化资产提取和项目架构总结蜂种

    结合了PrimeMiner的极致化资产提取能力和Librarian的项目架构总结能力，
    使用并行处理实现高效的资产提取和架构分析。
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.logger = get_logger("PrimeMinerLibrarianBee")

        # 资产类型定义
        self.asset_patterns = {
            "json_files": r"\.json$",
            "csv_files": r"\.csv$",
            "yaml_files": r"\.(yaml|yml)$",
            "weight_files": r"\.(pt|pth|onnx|pb|h5|pkl|bin)$",
            "config_files": r"\.(toml|ini|cfg|conf)$",
            "data_files": r"\.(xml|sql|db|sqlite)$",
        }

        # 架构分析模式
        self.architecture_patterns = {
            "python_modules": r"\.py$",
            "documentation": r"\.(md|rst|txt)$",
            "tests": r"test.*\.py$|_test\.py$",
            "requirements": r"requirements.*\.txt$|setup\.py$|pyproject\.toml$",
        }

        # 性能指标
        self.performance_metrics = {
            "files_scanned": 0,
            "assets_extracted": 0,
            "assets_classified": 0,
            "architecture_modules": 0,
            "parallel_workers": 0,
            "extraction_duration": 0.0,
        }

    def specialized_task(self, project_root: Path) -> bool:
        """
        执行极致化资产提取和项目架构总结

        Args:
            project_root: 项目根目录

        Returns:
            bool: 任务是否成功完成
        """
        self.logger.info(f"启动极致化资产提取和项目架构总结")
        self.logger.info(f"项目路径: {project_root}")

        try:
            # 并行提取资产
            assets = self._parallel_extract_assets(project_root)

            # 分类和分析资产
            classified_assets = self._classify_and_analyze_assets(assets)

            # 分析项目架构
            architecture_analysis = self._analyze_project_architecture(project_root)

            # 生成架构总结文档
            architecture_summary = self._generate_architecture_summary(
                architecture_analysis, classified_assets
            )

            # 生成综合报告
            report = self._generate_comprehensive_report(
                assets, classified_assets, architecture_analysis, architecture_summary
            )

            # 保存报告
            self._save_report(report)

            self.logger.info(f"极致化资产提取和项目架构总结完成")
            self.logger.info(f"扫描文件: {self.performance_metrics['files_scanned']}")
            self.logger.info(f"提取资产: {self.performance_metrics['assets_extracted']}")
            self.logger.info(f"分类资产: {self.performance_metrics['assets_classified']}")
            self.logger.info(f"架构模块: {self.performance_metrics['architecture_modules']}")

            return True

        except Exception as e:
            self.logger.error(f"极致化资产提取和项目架构总结失败: {str(e)}")
            return False

    def _parallel_extract_assets(self, project_root: Path) -> Dict[str, List[Dict[str, Any]]]:
        """
        并行提取项目资产

        Args:
            project_root: 项目根目录

        Returns:
            提取的资产字典
        """
        self.logger.info("启动并行资产提取")

        assets = {
            "json_files": [],
            "csv_files": [],
            "yaml_files": [],
            "weight_files": [],
            "config_files": [],
            "data_files": [],
        }

        # 收集所有资产文件
        all_files = []
        for asset_type, pattern in self.asset_patterns.items():
            files = list(project_root.rglob(f"*{pattern}"))
            all_files.extend(files)

        self.performance_metrics["files_scanned"] = len(all_files)

        # 使用线程池并行处理
        max_workers = min(8, len(all_files) // 10)
        self.performance_metrics["parallel_workers"] = max_workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交资产提取任务
            futures = {
                executor.submit(self._extract_asset_file, f, project_root): f for f in all_files
            }

            # 收集提取结果
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        asset_type = self._get_asset_type(file_path)
                        if asset_type in assets:
                            assets[asset_type].append(result)
                            self.performance_metrics["assets_extracted"] += 1

                except Exception as e:
                    self.logger.warning(f"提取资产文件 {file_path} 失败: {str(e)}")

        return assets

    def _extract_asset_file(self, file_path: Path, project_root: Path) -> Dict[str, Any]:
        """提取单个资产文件"""
        try:
            asset_info = {
                "path": str(file_path.relative_to(project_root)),
                "size": file_path.stat().st_size,
                "extension": file_path.suffix,
                "name": file_path.name,
                "content": None,
                "metadata": {},
            }

            # 尝试读取文件内容
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    # 对于JSON文件，尝试解析
                    if file_path.suffix == ".json":
                        try:
                            parsed = json.loads(content)
                            asset_info["content"] = parsed
                            asset_info["metadata"]["keys"] = (
                                list(parsed.keys()) if isinstance(parsed, dict) else []
                            )
                            asset_info["metadata"]["type"] = type(parsed).__name__
                        except json.JSONDecodeError:
                            asset_info["content"] = content[:1000]  # 保存前1000字符

                    # 对于CSV文件，分析结构
                    elif file_path.suffix == ".csv":
                        lines = content.split("\n")
                        asset_info["metadata"]["line_count"] = len(lines)
                        if lines:
                            asset_info["metadata"]["header"] = lines[0] if lines else ""

                    # 对于其他文本文件，保存部分内容
                    else:
                        asset_info["content"] = content[:1000]  # 保存前1000字符

            except Exception as e:
                asset_info["metadata"]["error"] = str(e)

            return asset_info

        except Exception as e:
            self.logger.warning(f"提取资产失败 {file_path}: {str(e)}")
            return None

    def _get_asset_type(self, file_path: Path) -> str:
        """获取资产类型"""
        for asset_type, pattern in self.asset_patterns.items():
            if re.search(pattern, str(file_path)):
                return asset_type
        return "other"

    def _classify_and_analyze_assets(
        self, assets: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        分类和分析资产

        Args:
            assets: 提取的资产字典

        Returns:
            分类和分析结果
        """
        self.logger.info("分类和分析资产")

        classification = {
            "by_type": {},
            "by_size": {},
            "by_location": {},
            "statistics": {"total_size": 0, "largest_file": None, "most_common_extension": None},
        }

        total_size = 0
        extension_counts = {}
        largest_file = {"size": 0, "file": None}

        # 按类型分类
        for asset_type, asset_list in assets.items():
            classification["by_type"][asset_type] = {
                "count": len(asset_list),
                "total_size": sum(a["size"] for a in asset_list),
                "average_size": (
                    sum(a["size"] for a in asset_list) / len(asset_list) if asset_list else 0
                ),
            }

            self.performance_metrics["assets_classified"] += len(asset_list)

        # 统计信息
        for asset_type, asset_list in assets.items():
            for asset in asset_list:
                total_size += asset["size"]

                # 统计扩展名
                ext = asset["extension"]
                extension_counts[ext] = extension_counts.get(ext, 0) + 1

                # 找最大文件
                if asset["size"] > largest_file["size"]:
                    largest_file = {"size": asset["size"], "file": asset["path"]}

        classification["statistics"]["total_size"] = total_size
        classification["statistics"]["largest_file"] = largest_file
        classification["statistics"]["most_common_extension"] = (
            max(extension_counts.items(), key=lambda x: x[1]) if extension_counts else (None, 0)
        )

        # 按大小分类
        size_categories = {
            "small": 0,  # < 1KB
            "medium": 0,  # 1KB - 100KB
            "large": 0,  # 100KB - 1MB
            "huge": 0,  # > 1MB
        }

        for asset_type, asset_list in assets.items():
            for asset in asset_list:
                size = asset["size"]
                if size < 1024:
                    size_categories["small"] += 1
                elif size < 102400:
                    size_categories["medium"] += 1
                elif size < 1048576:
                    size_categories["large"] += 1
                else:
                    size_categories["huge"] += 1

        classification["by_size"] = size_categories

        # 按位置分类
        location_counts = {}
        for asset_type, asset_list in assets.items():
            for asset in asset_list:
                # 获取相对路径的父目录
                path_parts = asset["path"].split("/")
                if len(path_parts) > 1:
                    location = "/".join(path_parts[:-1])
                else:
                    location = "root"

                location_counts[location] = location_counts.get(location, 0) + 1

        # 取前10个位置
        classification["by_location"] = dict(
            sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        return classification

    def _analyze_project_architecture(self, project_root: Path) -> Dict[str, Any]:
        """
        分析项目架构

        Args:
            project_root: 项目根目录

        Returns:
            项目架构分析结果
        """
        self.logger.info("分析项目架构")

        architecture = {
            "structure": {},
            "modules": [],
            "documentation": [],
            "tests": [],
            "dependencies": [],
            "summary": {},
        }

        # 分析项目结构
        for pattern_name, pattern in self.architecture_patterns.items():
            files = list(project_root.rglob(f"*{pattern}"))
            architecture["structure"][pattern_name] = len(files)

        # 收集Python模块
        python_files = list(project_root.rglob("*.py"))
        for file_path in python_files[:50]:  # 限制数量
            rel_path = file_path.relative_to(project_root)
            architecture["modules"].append(
                {"path": str(rel_path), "size": file_path.stat().st_size}
            )
            self.performance_metrics["architecture_modules"] += 1

        # 收集文档
        doc_files = list(project_root.rglob("*.md")) + list(project_root.rglob("*.rst"))
        for file_path in doc_files[:20]:  # 限制数量
            rel_path = file_path.relative_to(project_root)
            architecture["documentation"].append(
                {"path": str(rel_path), "size": file_path.stat().st_size}
            )

        # 收集测试文件
        test_files = [f for f in python_files if "test" in f.name.lower()]
        for file_path in test_files[:20]:  # 限制数量
            rel_path = file_path.relative_to(project_root)
            architecture["tests"].append({"path": str(rel_path), "size": file_path.stat().st_size})

        # 分析依赖
        req_files = [
            f
            for f in project_root.iterdir()
            if f.is_file()
            and ("requirements" in f.name or f.name == "setup.py" or f.name == "pyproject.toml")
        ]
        for req_file in req_files:
            try:
                with open(req_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    architecture["dependencies"].append(
                        {"file": req_file.name, "content": content[:500]}  # 保存前500字符
                    )
            except Exception as e:
                self.logger.warning(f"读取依赖文件 {req_file} 失败: {str(e)}")

        # 生成架构摘要
        architecture["summary"] = {
            "python_modules": architecture["structure"].get("python_modules", 0),
            "documentation": architecture["structure"].get("documentation", 0),
            "tests": architecture["structure"].get("tests", 0),
            "total_files": sum(architecture["structure"].values()),
        }

        return architecture

    def _generate_architecture_summary(
        self, architecture_analysis: Dict[str, Any], classified_assets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成架构总结文档

        Args:
            architecture_analysis: 架构分析结果
            classified_assets: 分类资产结果

        Returns:
            架构总结
        """
        self.logger.info("生成架构总结")

        summary = {
            "title": "项目架构总结",
            "project_overview": self._generate_project_overview(architecture_analysis),
            "asset_inventory": self._generate_asset_inventory(classified_assets),
            "module_structure": self._generate_module_structure(architecture_analysis),
            "recommendations": self._generate_recommendations(
                architecture_analysis, classified_assets
            ),
        }

        return summary

    def _generate_project_overview(self, architecture: Dict[str, Any]) -> str:
        """生成项目概述"""
        summary = architecture["summary"]
        overview = f"""# 项目概述

本项目包含以下主要组件：

- Python模块：{summary['python_modules']} 个
- 文档文件：{summary['documentation']} 个
- 测试文件：{summary['tests']} 个
- 总文件数：{summary['total_files']} 个

项目结构清晰，模块化程度良好。
"""
        return overview

    def _generate_asset_inventory(self, assets: Dict[str, Any]) -> str:
        """生成资产清单"""
        inventory = "## 资产清单\n\n"

        # 按类型列出资产
        inventory += "### 按类型分类\n\n"
        for asset_type, info in assets["by_type"].items():
            if info["count"] > 0:
                inventory += f"- **{asset_type}**: {info['count']} 个文件，总大小 {self._format_size(info['total_size'])}\n"

        # 按大小分类
        inventory += "\n### 按大小分类\n\n"
        for size_cat, count in assets["by_size"].items():
            if count > 0:
                inventory += f"- **{size_cat}**: {count} 个文件\n"

        # 统计信息
        stats = assets["statistics"]
        inventory += f"\n### 统计信息\n\n"
        inventory += f"- 总大小：{self._format_size(stats['total_size'])}\n"
        if stats["largest_file"]["file"]:
            inventory += f"- 最大文件：{stats['largest_file']['file']} ({self._format_size(stats['largest_file']['size'])})\n"
        if stats["most_common_extension"][0]:
            inventory += f"- 最常见扩展名：{stats['most_common_extension'][0]} ({stats['most_common_extension'][1]} 个文件)\n"

        return inventory

    def _generate_module_structure(self, architecture: Dict[str, Any]) -> str:
        """生成模块结构"""
        structure = "## 模块结构\n\n"

        if architecture["modules"]:
            structure += "### 主要模块\n\n"
            for module in architecture["modules"][:10]:  # 限制显示数量
                structure += f"- `{module['path']}` ({self._format_size(module['size'])})\n"

        if architecture["tests"]:
            structure += "\n### 测试模块\n\n"
            for test in architecture["tests"][:10]:  # 限制显示数量
                structure += f"- `{test['path']}`\n"

        return structure

    def _generate_recommendations(
        self, architecture: Dict[str, Any], assets: Dict[str, Any]
    ) -> str:
        """生成建议"""
        recommendations = "## 改进建议\n\n"

        # 资产管理建议
        if assets["statistics"]["largest_file"]["file"]:
            recommendations += "### 资产管理\n\n"
            recommendations += f"- 考虑优化大文件：{assets['statistics']['largest_file']['file']}\n"

        # 架构建议
        if architecture["summary"]["tests"] == 0:
            recommendations += "\n### 测试覆盖\n\n"
            recommendations += "- 建议增加单元测试以提高代码质量\n"

        if architecture["summary"]["documentation"] < 3:
            recommendations += "\n### 文档完善\n\n"
            recommendations += "- 建议增加项目文档以改善可维护性\n"

        return recommendations

    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    def _generate_comprehensive_report(
        self,
        assets: Dict[str, List[Dict[str, Any]]],
        classified_assets: Dict[str, Any],
        architecture_analysis: Dict[str, Any],
        architecture_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        生成综合报告

        Args:
            assets: 提取的资产
            classified_assets: 分类资产
            architecture_analysis: 架构分析
            architecture_summary: 架构总结

        Returns:
            综合报告
        """
        report = {
            "bee_type": self.bee_dna.species,
            "mutation_rate": self.bee_dna.mutation_rate,
            "timestamp": self._get_timestamp(),
            # 资产提取
            "asset_extraction": {
                "total_assets": sum(len(v) for v in assets.values()),
                "by_type": {k: len(v) for k, v in assets.items()},
                "assets": assets,
            },
            # 资产分类
            "asset_classification": classified_assets,
            # 架构分析
            "architecture_analysis": architecture_analysis,
            # 架构总结
            "architecture_summary": architecture_summary,
            # 性能指标
            "performance_metrics": self.performance_metrics,
            # 总结
            "summary": self._generate_summary(assets, classified_assets, architecture_analysis),
        }

        return report

    def _generate_summary(
        self,
        assets: Dict[str, List[Dict[str, Any]]],
        classified_assets: Dict[str, Any],
        architecture: Dict[str, Any],
    ) -> str:
        """生成报告摘要"""
        summary_parts = []

        # 资产摘要
        total_assets = sum(len(v) for v in assets.values())
        summary_parts.append(f"提取了 {total_assets} 个资产文件")

        # 架构摘要
        arch_summary = architecture["summary"]
        summary_parts.append(
            f"分析了包含 {arch_summary['python_modules']} 个Python模块、"
            f"{arch_summary['documentation']} 个文档和 "
            f"{arch_summary['tests']} 个测试的项目架构"
        )

        # 统计摘要
        total_size = classified_assets["statistics"]["total_size"]
        summary_parts.append(f"资产总大小为 {self._format_size(total_size)}")

        return "。".join(summary_parts) + "。"

    def _save_report(self, report: Dict[str, Any]) -> None:
        """保存报告到文件"""
        try:
            from hive.config.settings import settings

            report_path = (
                settings.REPORT_DIR / f"primeminerlibrarian_report_{self._get_timestamp()}.json"
            )
            self._save_json_report(report, report_path)
            self.logger.info(f"报告已保存: {report_path}")
        except Exception as e:
            self.logger.error(f"保存报告失败: {str(e)}")

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")
