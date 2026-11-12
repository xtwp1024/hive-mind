#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
赛博矿工蜂 - CyberMinerBee
赛博化：提取非代码资产

变异率：3.60%
特殊能力：自动化增强 - 扫描JSON/CSV/权重文件
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class CyberMinerBee(BaseBee):
    """
    赛博矿工蜂 - 负责提取非代码资产（JSON、CSV、权重文件等）

    特殊能力：
    - 自动化扫描JSON配置文件
    - 提取CSV数据文件
    - 识别模型权重文件
    - 生成资产清单
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.mined_assets = []
        self.asset_types = {
            "json": [".json"],
            "csv": [".csv"],
            "yaml": [".yaml", ".yml"],
            "weights": [".h5", ".hdf5", ".pkl", ".pickle", ".pt", ".pth", ".onnx", ".safetensors"],
            "config": [".toml", ".ini", ".cfg", ".conf"],
            "data": [".npy", ".npz", ".mat", ".data", ".bin"],
        }

    def specialized_task(self, target_path):
        """
        执行赛博化资产提取任务

        Args:
            target_path: 项目目录路径
        """
        logger.info(f"🤖 CyberMinerBee 开始任务: {self.dna.target_name}")

        try:
            # 1. 扫描并分类所有非代码资产
            assets = self._scan_and_classify_assets(target_path)

            # 2. 提取JSON配置
            json_configs = self._extract_json_configs(target_path)

            # 3. 分析CSV数据
            csv_analysis = self._analyze_csv_files(target_path)

            # 4. 识别模型权重文件
            weight_files = self._identify_weight_files(target_path)

            # 5. 生成赛博资产报告
            cyber_report = self._generate_cyber_report(
                {
                    "project": self.dna.target_name,
                    "total_assets": len(assets),
                    "asset_breakdown": self._breakdown_assets(assets),
                    "json_configs": json_configs,
                    "csv_analysis": csv_analysis,
                    "weight_files": weight_files,
                    "timestamp": __import__("time").ctime(),
                }
            )

            # 6. 保存资产清单
            self._save_asset_manifest(cyber_report)

            logger.info(f"✅ CyberMinerBee 任务完成: {self.dna.target_name}")
            logger.info(f"   发现资产总数: {len(assets)}")
            logger.info(f"   JSON配置: {len(json_configs)}")
            logger.info(f"   CSV文件: {len(csv_analysis)}")
            logger.info(f"   权重文件: {len(weight_files)}")

            return {
                "status": "success",
                "assets_found": len(assets),
                "json_configs": len(json_configs),
                "csv_files": len(csv_analysis),
                "weight_files": len(weight_files),
            }

        except Exception as e:
            logger.error(f"❌ CyberMinerBee 任务失败: {e}")
            return {"status": "error", "error": str(e)}

    def _scan_and_classify_assets(self, target_path: Path) -> List[Dict[str, Any]]:
        """
        扫描并分类所有非代码资产

        Args:
            target_path: 项目目录路径

        Returns:
            资产列表
        """
        assets = []

        # 排除的目录
        exclude_dirs = {".git", "__pycache__", "node_modules", "venv", ".env", "dist", "build"}

        for file_path in target_path.rglob("*"):
            if not file_path.is_file():
                continue

            # 跳过排除目录
            if any(exclude_dir in str(file_path) for exclude_dir in exclude_dirs):
                continue

            # 检查文件扩展名
            file_ext = file_path.suffix.lower()

            # 分类资产类型
            asset_type = None
            for type_name, extensions in self.asset_types.items():
                if file_ext in extensions:
                    asset_type = type_name
                    break

            if asset_type:
                try:
                    file_size = file_path.stat().st_size
                    assets.append(
                        {
                            "path": str(file_path.relative_to(target_path)),
                            "type": asset_type,
                            "extension": file_ext,
                            "size": file_size,
                            "modified": __import__("time").ctime(file_path.stat().st_mtime),
                        }
                    )
                except Exception as e:
                    logger.debug(f"扫描资产失败 {file_path}: {e}")

        return assets

    def _extract_json_configs(self, target_path: Path) -> List[Dict[str, Any]]:
        """
        提取JSON配置文件的内容

        Args:
            target_path: 项目目录路径

        Returns:
            JSON配置列表
        """
        json_configs = []

        for file_path in target_path.rglob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                json_configs.append(
                    {
                        "path": str(file_path.relative_to(target_path)),
                        "keys_count": len(self._get_all_keys(config)),
                        "size": file_path.stat().st_size,
                        "structure": self._analyze_json_structure(config),
                    }
                )

            except Exception as e:
                logger.debug(f"读取JSON失败 {file_path}: {e}")

        return json_configs

    def _get_all_keys(self, obj: Any, parent_key: str = "") -> List[str]:
        """
        递归获取JSON对象的所有键

        Args:
            obj: JSON对象
            parent_key: 父键名

        Returns:
            键列表
        """
        keys = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                keys.append(full_key)
                keys.extend(self._get_all_keys(value, full_key))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                keys.extend(self._get_all_keys(item, f"{parent_key}[{i}]"))

        return keys

    def _analyze_json_structure(self, obj: Any) -> str:
        """
        分析JSON结构

        Args:
            obj: JSON对象

        Returns:
            结构描述
        """
        if isinstance(obj, dict):
            return f"Dict({len(obj)} keys)"
        elif isinstance(obj, list):
            return f"List({len(obj)} items)"
        elif isinstance(obj, str):
            return f"String(len={len(obj)})"
        elif isinstance(obj, (int, float)):
            return f"Number({type(obj).__name__})"
        elif isinstance(obj, bool):
            return "Boolean"
        else:
            return type(obj).__name__

    def _analyze_csv_files(self, target_path: Path) -> List[Dict[str, Any]]:
        """
        分析CSV数据文件

        Args:
            target_path: 项目目录路径

        Returns:
            CSV分析结果列表
        """
        csv_analysis = []

        for file_path in target_path.rglob("*.csv"):
            try:
                import csv

                with open(file_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                    if rows:
                        headers = rows[0] if len(rows) > 0 else []
                        data_rows = rows[1:] if len(rows) > 1 else []

                        csv_analysis.append(
                            {
                                "path": str(file_path.relative_to(target_path)),
                                "total_rows": len(rows),
                                "data_rows": len(data_rows),
                                "columns": len(headers),
                                "headers": headers,
                                "size": file_path.stat().st_size,
                            }
                        )

            except Exception as e:
                logger.debug(f"分析CSV失败 {file_path}: {e}")

        return csv_analysis

    def _identify_weight_files(self, target_path: Path) -> List[Dict[str, Any]]:
        """
        识别模型权重文件

        Args:
            target_path: 项目目录路径

        Returns:
            权重文件列表
        """
        weight_files = []
        weight_extensions = self.asset_types["weights"]

        for file_path in target_path.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() in weight_extensions:
                try:
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    weight_files.append(
                        {
                            "path": str(file_path.relative_to(target_path)),
                            "extension": file_path.suffix,
                            "size_mb": round(file_size_mb, 2),
                            "size_category": self._categorize_weight_size(file_size_mb),
                        }
                    )
                except Exception as e:
                    logger.debug(f"识别权重文件失败 {file_path}: {e}")

        return weight_files

    def _categorize_weight_size(self, size_mb: float) -> str:
        """
        分类权重文件大小

        Args:
            size_mb: 文件大小（MB）

        Returns:
            大小类别
        """
        if size_mb < 1:
            return "small"
        elif size_mb < 10:
            return "medium"
        elif size_mb < 100:
            return "large"
        else:
            return "xlarge"

    def _breakdown_assets(self, assets: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        按类型统计资产

        Args:
            assets: 资产列表

        Returns:
            类型统计
        """
        breakdown = {}

        for asset in assets:
            asset_type = asset["type"]
            breakdown[asset_type] = breakdown.get(asset_type, 0) + 1

        return breakdown

    def _generate_cyber_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成赛博资产报告

        Args:
            data: 原始数据

        Returns:
            格式化的报告
        """
        report = {
            **data,
            "summary": {
                "total_assets": data["total_assets"],
                "asset_types": len(data["asset_breakdown"]),
                "config_files": len(data["json_configs"]),
                "data_files": len(data["csv_analysis"]),
                "model_weights": len(data["weight_files"]),
            },
            "recommendations": self._generate_recommendations(data),
        }

        return report

    def _generate_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """
        生成资产处理建议

        Args:
            data: 资产数据

        Returns:
            建议列表
        """
        recommendations = []

        # 配置文件建议
        if len(data["json_configs"]) > 0:
            recommendations.append(
                f"发现 {len(data['json_configs'])} 个JSON配置文件，建议进行配置合并和版本控制"
            )

        # 数据文件建议
        if len(data["csv_analysis"]) > 0:
            total_rows = sum(csv["data_rows"] for csv in data["csv_analysis"])
            recommendations.append(
                f"发现 {len(data['csv_analysis'])} 个CSV文件，共 {total_rows} 行数据，建议建立数据索引"
            )

        # 模型权重建议
        large_weights = [
            w for w in data["weight_files"] if w["size_category"] in ["large", "xlarge"]
        ]
        if large_weights:
            recommendations.append(
                f"发现 {len(large_weights)} 个大型权重文件，建议使用模型压缩技术"
            )

        if not recommendations:
            recommendations.append("项目资产管理良好，无需特殊处理")

        return recommendations

    def _save_asset_manifest(self, report: Dict[str, Any]):
        """
        保存资产清单

        Args:
            report: 赛博报告
        """
        try:
            manifest_dir = Path("cyber_manifests")
            manifest_dir.mkdir(exist_ok=True)

            manifest_filename = f"{self.dna.target_name.replace('/', '_')}_cyber_manifest.json"
            manifest_path = manifest_dir / manifest_filename

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"📄 赛博资产清单已保存: {manifest_path}")

        except Exception as e:
            logger.error(f"保存资产清单失败: {e}")
