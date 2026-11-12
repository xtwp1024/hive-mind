# -*- coding: utf-8 -*-
"""
PrimeMinerBee - 极致化资产提取蜂种

特殊能力：性能极限优化 - 扫描JSON/CSV/权重文件
变异率：3.15%

极致化的资产提取蜂种，专注于高性能地扫描和提取项目的非代码资产，
使用并行处理和优化的算法来加速资产发现和分析。
"""

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

from hive.bees.base_bee import BaseBee
from hive.config.settings import settings
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class PrimeMinerBee(BaseBee):
    """
    极致化资产提取蜂种 - 高性能资产扫描和提取
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)

        # 初始化报告目录
        self.report_dir = settings.REPORT_DIR / "primeminer"
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 提取的资产
        self.extracted_assets: List[Dict[str, Any]] = []

        # 并行工作线程数
        self.max_workers = 4

        # 资产类型定义
        self.asset_patterns = {
            "json_config": [".json"],
            "csv_data": [".csv"],
            "weight_file": [
                ".pth",
                ".pt",
                ".h5",
                ".hdf5",
                ".ckpt",
                ".pkl",
                ".npy",
                ".bin",
                ".safetensors",
            ],
            "data_file": [".parquet", ".feather", ".arrow", ".db", ".sqlite", ".sqlite3"],
            "image_asset": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico"],
            "font_asset": [".ttf", ".otf", ".woff", ".woff2"],
            "media_asset": [".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac"],
            "archive_file": [".zip", ".tar", ".gz", ".7z", ".rar"],
        }

        # 扫描统计
        self.scan_stats = {
            "total_assets": 0,
            "by_type": {},
            "total_size_bytes": 0,
            "scan_time": 0.0,
            "files_processed": 0,
        }

        logger.info(f"💎 PrimeMinerBee 初始化完成 | 变异率: 3.15% | 并行线程: {self.max_workers}")

    def specialized_task(self):
        """
        执行极致化资产提取任务
        """
        logger.info("💎 PrimeMinerBee 开始执行高性能资产提取")

        try:
            # 获取项目根目录
            from hive.config.settings import settings

            project_root = settings.BASE_DIR

            # 并行提取资产
            self._parallel_extract_assets(project_root)

            # 分析资产
            self._analyze_assets()

            # 生成资产报告
            self._generate_asset_report()

            logger.info(
                f"✅ PrimeMinerBee 提取完成 | 资产数: {len(self.extracted_assets)} | 用时: {self.scan_stats['scan_time']:.2f}s"
            )

        except Exception as e:
            logger.error(f"❌ PrimeMinerBee 提取失败: {e}", exc_info=True)
            raise

    def _parallel_extract_assets(self, project_root: Path):
        """
        并行提取项目资产
        """
        import time

        start_time = time.time()

        logger.info(f"💎 开始并行提取资产 (线程数: {self.max_workers})...")

        # 收集所有资产文件
        all_files = []
        for asset_type, extensions in self.asset_patterns.items():
            for ext in extensions:
                files = list(project_root.rglob(f"*{ext}"))
                for file_path in files:
                    all_files.append((file_path, asset_type))

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self._analyze_asset, file_path, asset_type): (file_path, asset_type)
                for file_path, asset_type in all_files
            }

            # 收集结果
            for future in as_completed(future_to_file):
                file_path, asset_type = future_to_file[future]
                try:
                    asset_info = future.result()
                    if asset_info:
                        self.extracted_assets.append(asset_info)
                        self.scan_stats["total_assets"] += 1
                        self.scan_stats["total_size_bytes"] += asset_info["size_bytes"]
                        self.scan_stats["files_processed"] += 1

                        # 统计资产类型
                        if asset_type not in self.scan_stats["by_type"]:
                            self.scan_stats["by_type"][asset_type] = 0
                        self.scan_stats["by_type"][asset_type] += 1

                except Exception as e:
                    logger.warning(f"分析资产失败 {file_path}: {e}")

        # 计算扫描时间
        self.scan_stats["scan_time"] = time.time() - start_time

        logger.info(
            f"✅ 并行提取完成 | 提取资产: {self.scan_stats['total_assets']} | 用时: {self.scan_stats['scan_time']:.2f}s"
        )

    def _analyze_asset(self, file_path: Path, asset_type: str) -> Dict[str, Any]:
        """
        分析单个资产文件（线程安全）
        """
        try:
            file_size = file_path.stat().st_size
            relative_path = str(file_path.relative_to(file_path.parent.parent.parent))

            asset_info = {
                "type": asset_type,
                "path": relative_path,
                "size_bytes": file_size,
                "size_human": self._human_readable_size(file_size),
                "extension": file_path.suffix,
                "filename": file_path.name,
                "directory": str(file_path.parent),
            }

            # 根据资产类型进行特定分析
            if asset_type == "json_config":
                asset_info.update(self._analyze_json_file(file_path))
            elif asset_type == "csv_data":
                asset_info.update(self._analyze_csv_file(file_path))
            elif asset_type == "weight_file":
                asset_info.update(self._analyze_weight_file(file_path, file_size))
            elif asset_type == "data_file":
                asset_info["is_database"] = file_path.suffix in [".db", ".sqlite", ".sqlite3"]
                asset_info["is_dataset"] = file_path.suffix in [".parquet", ".feather", ".arrow"]
            elif asset_type == "image_asset":
                asset_info.update(self._analyze_image_file(file_path))
            elif asset_type == "archive_file":
                asset_info["is_archive"] = True

            return asset_info

        except Exception as e:
            logger.debug(f"分析资产时出错 {file_path}: {e}")
            return None

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
                "data_type": (
                    "object"
                    if isinstance(data, dict)
                    else "array" if isinstance(data, list) else "primitive"
                ),
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
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                # 读取前几行
                reader = csv.reader(f)
                headers = next(reader, [])
                sample_rows = [row for _, row in zip(range(5), reader)]

            # 分析列类型
            column_types = {}
            if sample_rows:
                for i, header in enumerate(headers):
                    values = [row[i] if i < len(row) else "" for row in sample_rows]
                    column_types[header] = self._infer_column_type(values)

            return {
                "has_headers": len(headers) > 0,
                "column_count": len(headers),
                "sample_row_count": len(sample_rows),
                "column_types": column_types,
            }
        except Exception as e:
            return {
                "has_headers": False,
                "error": str(e),
            }

    def _infer_column_type(self, values: List[str]) -> str:
        """
        推断列数据类型
        """
        try:
            # 尝试转换为数字
            numeric_count = sum(
                1 for v in values if v and v.replace(".", "").replace("-", "").isdigit()
            )
            if numeric_count > len(values) * 0.8:
                return "numeric"

            # 尝试转换为布尔
            bool_count = sum(
                1 for v in values if v.lower() in ["true", "false", "yes", "no", "1", "0"]
            )
            if bool_count > len(values) * 0.8:
                return "boolean"

            return "string"
        except (KeyError, TypeError, ValueError):
            return "unknown"

    def _analyze_weight_file(self, file_path: Path, file_size: int) -> Dict[str, Any]:
        """
        分析模型权重文件
        """
        mb = file_size / (1024 * 1024)

        if mb < 1:
            size_category = "small"
        elif mb < 100:
            size_category = "medium"
        elif mb < 1000:
            size_category = "large"
        else:
            size_category = "extra_large"

        return {
            "is_model_weight": True,
            "size_category": size_category,
            "is_safetensors": file_path.suffix == ".safetensors",
        }

    def _analyze_image_file(self, file_path: Path) -> Dict[str, Any]:
        """
        分析图像文件
        """
        return {
            "is_image": True,
            "image_format": file_path.suffix[1:].upper(),
        }

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
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def _analyze_assets(self):
        """
        分析提取的资产
        """
        logger.info("📊 分析提取的资产...")

        # 按类型分组
        assets_by_type = {}
        for asset in self.extracted_assets:
            asset_type = asset["type"]
            if asset_type not in assets_by_type:
                assets_by_type[asset_type] = []
            assets_by_type[asset_type].append(asset)

        # 找出最大的文件
        largest_files = sorted(self.extracted_assets, key=lambda x: x["size_bytes"], reverse=True)[
            :10
        ]

        # 计算性能指标
        files_per_second = (
            self.scan_stats["files_processed"] / self.scan_stats["scan_time"]
            if self.scan_stats["scan_time"] > 0
            else 0
        )

        logger.info(f"📊 资产类型分布: {json.dumps(self.scan_stats['by_type'], indent=2)}")
        logger.info(f"📊 处理速度: {files_per_second:.2f} 文件/秒")

    def _generate_asset_report(self):
        """
        生成资产报告
        """
        logger.info("📝 生成资产报告...")

        # 按类型分组
        assets_by_type = {}
        for asset in self.extracted_assets:
            asset_type = asset["type"]
            if asset_type not in assets_by_type:
                assets_by_type[asset_type] = []
            assets_by_type[asset_type].append(asset)

        # 按大小排序
        for asset_type in assets_by_type:
            assets_by_type[asset_type].sort(key=lambda x: x["size_bytes"], reverse=True)

        # 找出最大的文件
        largest_files = sorted(self.extracted_assets, key=lambda x: x["size_bytes"], reverse=True)[
            :20
        ]

        # 计算性能指标
        files_per_second = (
            self.scan_stats["files_processed"] / self.scan_stats["scan_time"]
            if self.scan_stats["scan_time"] > 0
            else 0
        )

        # 生成建议
        recommendations = self._generate_recommendations()

        # 构建报告
        report = {
            "scan_metadata": {
                "bee_type": "PrimeMinerBee",
                "mutation_rate": "3.15%",
                "scan_date": self._get_current_timestamp(),
                "parallel_workers": self.max_workers,
            },
            "performance_metrics": {
                "files_processed": self.scan_stats["files_processed"],
                "scan_time_seconds": round(self.scan_stats["scan_time"], 2),
                "files_per_second": round(files_per_second, 2),
                "parallel_efficiency": (
                    "HIGH"
                    if files_per_second > 50
                    else "MEDIUM" if files_per_second > 20 else "LOW"
                ),
            },
            "asset_summary": {
                "total_assets": self.scan_stats["total_assets"],
                "total_size_bytes": self.scan_stats["total_size_bytes"],
                "total_size_human": self._human_readable_size(self.scan_stats["total_size_bytes"]),
                "assets_by_type": self.scan_stats["by_type"],
            },
            "assets_by_type": {
                asset_type: assets[:15] for asset_type, assets in assets_by_type.items()  # 限制数量
            },
            "largest_files": [
                {
                    "path": asset["path"],
                    "size_bytes": asset["size_bytes"],
                    "size_human": asset["size_human"],
                    "type": asset["type"],
                }
                for asset in largest_files
            ],
            "recommendations": recommendations,
            "summary": {
                "assets_found": self.scan_stats["total_assets"],
                "total_size": self._human_readable_size(self.scan_stats["total_size_bytes"]),
                "has_weights": self.scan_stats["by_type"].get("weight_file", 0) > 0,
                "has_config": self.scan_stats["by_type"].get("json_config", 0) > 0,
                "has_data": self.scan_stats["by_type"].get("csv_data", 0) > 0
                or self.scan_stats["by_type"].get("data_file", 0) > 0,
                "performance_rating": (
                    "EXCELLENT"
                    if files_per_second > 50
                    else "GOOD" if files_per_second > 20 else "AVERAGE"
                ),
            },
        }

        # 保存报告
        report_path = self.report_dir / f"asset_report_{self._get_current_timestamp()}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 资产报告已生成: {report_path}")

    def _generate_recommendations(self) -> List[str]:
        """
        生成建议
        """
        recommendations = []

        if self.scan_stats["total_assets"] > 0:
            recommendations.append(
                f"发现 {self.scan_stats['total_assets']} 个资产文件，建议建立资产清单"
            )

        weight_count = self.scan_stats["by_type"].get("weight_file", 0)
        if weight_count > 0:
            recommendations.append(f"发现 {weight_count} 个模型权重文件，建议做好版本管理和备份")

        config_count = self.scan_stats["by_type"].get("json_config", 0)
        if config_count > 0:
            recommendations.append(f"发现 {config_count} 个配置文件，建议文档化配置项和敏感信息")

        data_count = self.scan_stats["by_type"].get("csv_data", 0) + self.scan_stats["by_type"].get(
            "data_file", 0
        )
        if data_count > 0:
            recommendations.append(f"发现 {data_count} 个数据文件，建议检查数据完整性和隐私性")

        total_size_mb = self.scan_stats["total_size_bytes"] / (1024 * 1024)
        if total_size_mb > 1000:
            recommendations.append(f"总资产大小超过1GB，建议考虑使用云存储或CDN")

        return recommendations

    def _get_current_timestamp(self) -> str:
        """
        获取当前时间戳
        """
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")
