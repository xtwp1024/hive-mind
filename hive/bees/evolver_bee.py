#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import re
import traceback
from pathlib import Path

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.gep.protocol import GEPManager
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class EvolverBee(BaseBee):
    """
    EvolverBee (进化者蜂)

    这是我们吸纳 autogame-17/evolver 项目后专门编写的核心智能体。
    它负责：从系统运行的历史遗迹中（日志、Morgue/死因证明）吸收经验，
    进行自我审查，制定补丁方案，并保存为标准 GEP (Genome Evolution Protocol) 基因。

    它是基于路线B：Python 原生重写 Evolver 哲学的结晶。
    """

    species_name = "evolver"
    description = "扫描系统报错日志，针对脆弱点进行自动修复大模型推理，产出 GEP 经验资产"

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.gep = GEPManager()

    def specialized_task(self, target_path) -> dict:
        """主入口"""
        mode = self.dna.metadata.get("evolve_mode", "balanced")
        logger.info(f"🧬 [EvolverBee] 正在启动能力进化引擎引擎... (策略: {mode})")

        # 1. 扫描死忙证明 (Morgue) 或最近日志
        crash_reports = self._scan_morgue()

        if not crash_reports:
            logger.info("✔️ [EvolverBee] 未发现亟需修复的致命异常事件，系统健康。")
            return {"status": "SUCCESS", "info": "Healthy, no critical fixes needed"}

        # 2. 从崩溃中选择一个典型特征尝试诊断与拟合基因 (模拟与LLM对话)
        report = crash_reports[0]

        try:
            # 申请人类介入（安全控制）: 告诉统筹者发现漏洞准备进行修补模拟
            if self.dna.metadata.get("require_human_audit", True):
                self.request_human_intervention(
                    reason=f"Evolver系统准备为异常生成GEP基因补丁。异常摘要: {report.get('error_type')}",
                    risk_level="MEDIUM",
                    context={"report": report},
                )
        except InterruptedError:
            return {"status": "AWAITING_HUMAN"}

        # 3. 提取基因 (GEP)
        gene_id = self._synthesize_gene_from_crash(report)

        return {
            "status": "SUCCESS",
            "mode": mode,
            "generated_gene": gene_id,
            "handled_crash": report.get("file_name", "unknown"),
        }

    def _scan_morgue(self) -> list:
        """扫描蜂群系统 morgue（停尸房/崩溃证书簿）收集经验"""
        from hive.config.settings import get_settings

        morgue_dir = get_settings().BASE_DIR / "hive" / "morgue"
        reports = []
        if morgue_dir.exists():
            import glob
            import json

            for cert_file in glob.glob(str(morgue_dir / "*.json")):
                try:
                    with open(cert_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # 我们只处理还没被进化过的报告
                        if not data.get("evolved_at"):
                            data["file_name"] = Path(cert_file).name
                            data["full_path"] = cert_file
                            reports.append(data)
                except Exception:
                    pass

        return sorted(reports, key=lambda x: x.get("timestamp", 0), reverse=True)

    def _synthesize_gene_from_crash(self, crash_report: dict) -> str:
        """从崩溃报告中推理出 GEP 补丁知识 (模拟大模型抽象)"""
        # 在真实的框架中，这里应该将错误栈通过模型客户端(如 Langchain)进行 Prompt 构造并获取修补意见
        # 我们现在编写完整的模拟逻辑框架

        # 拟提取的大模型反馈:
        pseudo_insight = f"对 {crash_report.get('bee_type', 'Unknown')} 的容错性升级"
        pseudo_patch = "def safe_execute(): pass  # auto-generated patch snippet"

        gene_id = f"gene-{crash_report.get('error_type', 'GeneralError')[-8:]}"

        gene_payload = {
            "description": pseudo_insight,
            "source_crash": crash_report.get("id", "none"),
            "mutation_type": "repair",
            "patch": pseudo_patch,
            "validation_cmd": ["python", "-m", "pytest", "tests/"],  # 验证指令
        }

        # 保存到 GEP 系统
        self.gep.save_gene(gene_id, gene_payload)

        # 标记死因证明已被消化
        try:
            with open(crash_report["full_path"], "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["evolved_at"] = self.gep.events_file.stat().st_mtime
                data["gene_reference"] = gene_id
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            logger.error(f"无法更新已消化的死僵证书: {e}")

        return gene_id
