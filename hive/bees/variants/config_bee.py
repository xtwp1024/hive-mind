# -*- coding: utf-8 -*-
"""
ConfigBee - 配置文件分析蜂
简单快速的配置文件解析和验证
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger


class ConfigBee(BaseBee):
    """
    配置蜂 - 配置文件分析器

    能力：
    1. 解析JSON/YAML配置
    2. 验证配置完整性
    3. 检测配置问题
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.logger = get_logger("ConfigBee")

    def specialized_task(self, target: Path) -> Dict[str, Any]:
        """
        执行配置分析任务

        Args:
            target: 目标路径（配置文件）

        Returns:
            分析结果字典
        """
        self.logger.info(f"⚙️ Config 开始分析: {target}")

        if not target.exists() or not target.is_file():
            return {"status": "error", "message": f"目标不是有效的文件: {target}"}

        # 判断文件类型
        suffix = target.suffix.lower()

        if suffix == ".json":
            return self._analyze_json_config(target)
        elif suffix in [".yaml", ".yml"]:
            return self._analyze_yaml_config(target)
        else:
            return self._analyze_text_config(target)

    def _analyze_json_config(self, file_path: Path) -> Dict[str, Any]:
        """
        分析JSON配置文件

        Args:
            file_path: JSON文件路径

        Returns:
            分析结果
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            analysis = self._analyze_config_data(data, file_path)

            return {
                "status": "success",
                "target": str(file_path),
                "config_type": "JSON",
                "analysis": analysis,
                "message": "JSON配置解析成功",
            }

        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "target": str(file_path),
                "config_type": "JSON",
                "error": f"JSON解析错误: {e}",
                "message": "JSON格式错误",
            }
        except Exception as e:
            return {
                "status": "error",
                "target": str(file_path),
                "config_type": "JSON",
                "error": str(e),
                "message": "读取配置失败",
            }

    def _analyze_yaml_config(self, file_path: Path) -> Dict[str, Any]:
        """
        分析YAML配置文件

        Args:
            file_path: YAML文件路径

        Returns:
            分析结果
        """
        try:
            import yaml

            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            analysis = self._analyze_config_data(data, file_path)

            return {
                "status": "success",
                "target": str(file_path),
                "config_type": "YAML",
                "analysis": analysis,
                "message": "YAML配置解析成功",
            }

        except ImportError:
            return {
                "status": "error",
                "target": str(file_path),
                "config_type": "YAML",
                "error": "未安装PyYAML库",
                "message": "缺少YAML解析器",
            }
        except Exception as e:
            return {
                "status": "error",
                "target": str(file_path),
                "config_type": "YAML",
                "error": str(e),
                "message": "YAML解析失败",
            }

    def _analyze_text_config(self, file_path: Path) -> Dict[str, Any]:
        """
        分析文本配置文件（如.env、ini等）

        Args:
            file_path: 文本配置文件路径

        Returns:
            分析结果
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 简单分析
            total_lines = len(lines)
            empty_lines = sum(1 for line in lines if not line.strip())
            comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
            config_lines = total_lines - empty_lines - comment_lines

            # 提取键值对（简单格式：KEY=VALUE）
            config_pairs = []
            for line in lines:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    config_pairs.append(
                        {
                            "key": key.strip(),
                            "value": value.strip(),
                            "has_value": bool(value.strip()),
                        }
                    )

            return {
                "status": "success",
                "target": str(file_path),
                "config_type": "TEXT",
                "analysis": {
                    "total_lines": total_lines,
                    "empty_lines": empty_lines,
                    "comment_lines": comment_lines,
                    "config_lines": config_lines,
                    "config_pairs": len(config_pairs),
                    "empty_values": sum(1 for p in config_pairs if not p["has_value"]),
                    "sample_pairs": config_pairs[:10],
                },
                "message": f"文本配置分析完成: {config_lines} 行配置",
            }

        except Exception as e:
            return {
                "status": "error",
                "target": str(file_path),
                "config_type": "TEXT",
                "error": str(e),
                "message": "读取配置失败",
            }

    def _analyze_config_data(self, data: Any, file_path: Path) -> Dict[str, Any]:
        """
        分析配置数据结构

        Args:
            data: 配置数据
            file_path: 文件路径

        Returns:
            分析结果
        """
        if not isinstance(data, dict):
            return {"error": "配置根节点不是字典", "type": str(type(data))}

        # 统计配置项
        total_keys = len(data)
        nested_dicts = sum(1 for v in data.values() if isinstance(v, dict))
        lists = sum(1 for v in data.values() if isinstance(v, list))
        strings = sum(1 for v in data.values() if isinstance(v, str))
        numbers = sum(1 for v in data.values() if isinstance(v, (int, float)))
        booleans = sum(1 for v in data.values() if isinstance(v, bool))
        nulls = sum(1 for v in data.values() if v is None)

        # 检测空值
        empty_values = [k for k, v in data.items() if v is None or v == ""]
        empty_lists = [k for k, v in data.items() if isinstance(v, list) and len(v) == 0]

        return {
            "total_keys": total_keys,
            "nested_dicts": nested_dicts,
            "lists": lists,
            "strings": strings,
            "numbers": numbers,
            "booleans": booleans,
            "nulls": nulls,
            "empty_values": empty_values,
            "empty_lists": empty_lists,
            "config_size": file_path.stat().st_size,
        }

    def validate_config(self, file_path: Path, required_keys: List[str]) -> Dict[str, Any]:
        """
        验证配置是否包含必需的键

        Args:
            file_path: 配置文件路径
            required_keys: 必需的键列表

        Returns:
            验证结果
        """
        result = self.specialized_task(file_path)

        if result["status"] != "success":
            return result

        if result["config_type"] == "TEXT":
            return {"status": "error", "message": "文本配置不支持键验证"}

        # 对于JSON/YAML，检查必需键
        # 这里简化处理，实际需要解析配置数据
        return {
            "status": "success",
            "message": "配置验证功能待实现",
            "required_keys": required_keys,
        }
