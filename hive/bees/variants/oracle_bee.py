#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预言家蜂 - OracleBee
预测性分析，预见未来问题

变异率：12.00%
特殊能力：时间序列预测 + 异常检测
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class OracleBee(BaseBee):
    """
    预言家蜂 - 负责预测性分析和异常检测

    特殊能力：
    - 分析项目发展趋势
    - 预测潜在问题
    - 识别异常模式
    - 生成预测报告
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.predictions = []
        self.anomalies = []

    def specialized_task(self, target_path):
        """
        执行预测性分析任务

        Args:
            target_path: 项目目录路径
        """
        logger.info(f"🔮 OracleBee 开始任务: {self.dna.target_name}")

        try:
            # 1. 分析项目元数据
            metadata_analysis = self._analyze_metadata(target_path)

            # 2. 检测异常模式
            anomalies = self._detect_anomalies(target_path)

            # 3. 预测潜在问题
            predictions = self._make_predictions(target_path, metadata_analysis)

            # 4. 生成风险评估
            risk_assessment = self._assess_risks(anomalies, predictions)

            # 5. 保存预言报告
            self._save_oracle_report(
                {
                    "project": self.dna.target_name,
                    "metadata_analysis": metadata_analysis,
                    "anomalies_detected": anomalies,
                    "predictions": predictions,
                    "risk_assessment": risk_assessment,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info(f"✅ OracleBee 任务完成: {self.dna.target_name}")
            logger.info(f"   检测异常: {len(anomalies)}")
            logger.info(f"   预测问题: {len(predictions)}")
            logger.info(f"   风险等级: {risk_assessment.get('overall_risk', 'unknown')}")

            return {
                "status": "success",
                "anomalies": len(anomalies),
                "predictions": len(predictions),
                "risk_level": risk_assessment.get("overall_risk", "unknown"),
            }

        except Exception as e:
            logger.error(f"❌ OracleBee 任务失败: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_metadata(self, target_path: Path) -> Dict[str, Any]:
        """
        分析项目元数据

        Args:
            target_path: 项目目录路径

        Returns:
            元数据分析结果
        """
        metadata = {
            "file_count": 0,
            "directory_count": 0,
            "total_size": 0,
            "file_types": {},
            "last_modified": None,
        }

        try:
            for item in target_path.rglob("*"):
                if item.is_file():
                    metadata["file_count"] += 1
                    metadata["total_size"] += item.stat().st_size

                    # 统计文件类型
                    ext = item.suffix.lower()
                    metadata["file_types"][ext] = metadata["file_types"].get(ext, 0) + 1

                    # 记录最后修改时间
                    mod_time = item.stat().st_mtime
                    if metadata["last_modified"] is None or mod_time > metadata["last_modified"]:
                        metadata["last_modified"] = mod_time

                elif item.is_dir():
                    metadata["directory_count"] += 1

        except Exception as e:
            logger.debug(f"分析元数据失败: {e}")

        return metadata

    def _detect_anomalies(self, target_path: Path) -> List[Dict[str, Any]]:
        """
        检测异常模式

        Args:
            target_path: 项目目录路径

        Returns:
            异常列表
        """
        anomalies = []

        # 检测1: 空目录
        empty_dirs = []
        for item in target_path.rglob("*"):
            if item.is_dir():
                try:
                    if not any(item.iterdir()):
                        empty_dirs.append(str(item.relative_to(target_path)))
                except (OSError, PermissionError):
                    pass

        if len(empty_dirs) > 5:
            anomalies.append(
                {
                    "type": "empty_directories",
                    "severity": "low",
                    "count": len(empty_dirs),
                    "description": f"发现 {len(empty_dirs)} 个空目录",
                }
            )

        # 检测2: 孤立的文件
        isolated_files = []
        for item in target_path.rglob("*"):
            if item.is_file():
                parent = item.parent
                siblings = list(parent.glob("*"))
                if len(siblings) == 1:
                    isolated_files.append(str(item.relative_to(target_path)))

        if len(isolated_files) > 0:
            anomalies.append(
                {
                    "type": "isolated_files",
                    "severity": "medium",
                    "count": len(isolated_files),
                    "description": f"发现 {len(isolated_files)} 个孤立文件",
                }
            )

        # 检测3: 深层嵌套
        deep_files = []
        for item in target_path.rglob("*"):
            if item.is_file():
                depth = len(item.relative_to(target_path).parts) - 1
                if depth > 6:
                    deep_files.append({"path": str(item.relative_to(target_path)), "depth": depth})

        if len(deep_files) > 0:
            anomalies.append(
                {
                    "type": "deep_nesting",
                    "severity": "medium",
                    "count": len(deep_files),
                    "description": f"发现 {len(deep_files)} 个深层嵌套文件",
                }
            )

        return anomalies

    def _make_predictions(
        self, target_path: Path, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        预测潜在问题

        Args:
            target_path: 项目目录路径
            metadata: 元数据分析结果

        Returns:
            预测列表
        """
        predictions = []

        # 预测1: 项目规模增长
        file_count = metadata["file_count"]
        if file_count < 50:
            predictions.append(
                {
                    "type": "growth_prediction",
                    "likelihood": "high",
                    "description": "项目规模较小，预计6个月内文件数量将增长2-3倍",
                    "recommendation": "建议提前规划模块化架构",
                }
            )

        # 预测2: 维护复杂度
        avg_files_per_dir = file_count / max(metadata["directory_count"], 1)
        if avg_files_per_dir > 20:
            predictions.append(
                {
                    "type": "maintenance_complexity",
                    "likelihood": "medium",
                    "description": f"平均每个目录包含 {avg_files_per_dir:.1f} 个文件，维护复杂度较高",
                    "recommendation": "建议重构目录结构，按功能模块组织",
                }
            )

        # 预测3: 代码分散风险
        dominant_type = max(metadata["file_types"].items(), key=lambda x: x[1], default=(None, 0))
        if dominant_type[0] and dominant_type[1] / file_count < 0.5:
            predictions.append(
                {
                    "type": "code_fragmentation",
                    "likelihood": "low",
                    "description": f"代码文件类型分散，主要类型占比不足50%",
                    "recommendation": "考虑统一技术栈或明确各类型的用途",
                }
            )

        # 预测4: 文档缺失风险
        doc_files = metadata["file_types"].get(".md", 0)
        if doc_files < 5:
            predictions.append(
                {
                    "type": "documentation_risk",
                    "likelihood": "high",
                    "description": "文档文件数量较少，可能影响项目可维护性",
                    "recommendation": "建议增加README和API文档",
                }
            )

        return predictions

    def _assess_risks(self, anomalies: List[Dict], predictions: List[Dict]) -> Dict[str, Any]:
        """
        评估风险

        Args:
            anomalies: 异常列表
            predictions: 预测列表

        Returns:
            风险评估结果
        """
        risk_score = 0

        # 基于异常的得分
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                risk_score += 10
            elif anomaly["severity"] == "medium":
                risk_score += 5
            else:
                risk_score += 2

        # 基于预测的得分
        for prediction in predictions:
            if prediction["likelihood"] == "high":
                risk_score += 8
            elif prediction["likelihood"] == "medium":
                risk_score += 4

        # 确定风险等级
        if risk_score > 30:
            overall_risk = "high"
        elif risk_score > 15:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        return {
            "risk_score": risk_score,
            "overall_risk": overall_risk,
            "risk_factors": len(anomalies) + len(predictions),
            "recommendation": self._get_risk_recommendation(overall_risk),
        }

    def _get_risk_recommendation(self, risk_level: str) -> str:
        """
        获取风险建议

        Args:
            risk_level: 风险等级

        Returns:
            建议文本
        """
        if risk_level == "high":
            return "建议立即进行代码审查和架构优化"
        elif risk_level == "medium":
            return "建议制定改进计划并逐步实施"
        else:
            return "项目状态良好，继续保持当前实践"

    def _save_oracle_report(self, report_data: Dict[str, Any]):
        """
        保存预言报告

        Args:
            report_data: 报告数据
        """
        try:
            reports_dir = Path("oracle_reports")
            reports_dir.mkdir(exist_ok=True)

            report_filename = f"{self.dna.target_name.replace('/', '_')}_oracle_report.json"
            report_path = reports_dir / report_filename

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📄 预言报告已保存: {report_path}")

        except Exception as e:
            logger.error(f"保存预言报告失败: {e}")
