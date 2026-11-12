#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GEP (Genome Evolution Protocol) 协议层
在 Python 中复刻 Evolver 的核心存储系统：管理 genes.json, capsules.json, events.jsonl
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Hive.GEP")


class GEPManager:
    def __init__(self, data_dir: str = None):
        if not data_dir:
            from hive.config.settings import get_settings

            data_dir = os.path.join(get_settings().DATA_DIR, "gep")

        self.data_dir = Path(data_dir)
        self.genes_file = self.data_dir / "genes.json"
        self.capsules_file = self.data_dir / "capsules.json"
        self.events_file = self.data_dir / "events.jsonl"

        self._ensure_files()

    def _ensure_files(self):
        """确保 GEP 协议要求的底层存储存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.genes_file.exists():
            self._write_json(self.genes_file, {})
        if not self.capsules_file.exists():
            self._write_json(self.capsules_file, {})
        if not self.events_file.exists():
            self.events_file.touch()

    def _read_json(self, p: Path) -> dict:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_json(self, p: Path, data: dict):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record_event(self, event_type: str, details: dict):
        """记录每一次进化、崩溃或审查事件到 jsonl 日志流中"""
        event = {
            "id": f"evt-{uuid.uuid4().hex[:8]}",
            "timestamp": time.time(),
            "type": event_type,
            "details": details,
        }
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def get_gene(self, gene_id: str) -> Optional[dict]:
        genes = self._read_json(self.genes_file)
        return genes.get(gene_id)

    def save_gene(self, gene_id: str, payload: dict):
        """保存一个能力基因 (一段修复补丁或演进知识)"""
        genes = self._read_json(self.genes_file)
        payload["updated_at"] = time.time()
        genes[gene_id] = payload
        self._write_json(self.genes_file, genes)
        logger.info(f"🧬 [GEP] 基因保存成功: {gene_id}")
        self.record_event("GENE_SAVED", {"gene_id": gene_id})

    def save_capsule(self, capsule_id: str, payload: dict):
        """保存胶囊 (复合能力或流程集合)"""
        capsules = self._read_json(self.capsules_file)
        payload["updated_at"] = time.time()
        capsules[capsule_id] = payload
        self._write_json(self.capsules_file, capsules)
        logger.info(f"💊 [GEP] 胶囊保存成功: {capsule_id}")
        self.record_event("CAPSULE_SAVED", {"capsule_id": capsule_id})

    def analyze_recent_failures(self, hours: int = 24) -> List[dict]:
        """从 events 库中简单获取近期失败事件（为进化引擎提供线索）"""
        cutoff = time.time() - (hours * 3600)
        failures = []
        try:
            with open(self.events_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    evt = json.loads(line)
                    if evt.get("timestamp", 0) > cutoff and evt.get("type") in [
                        "PROCESS_CRASH",
                        "TASK_FAILED",
                    ]:
                        failures.append(evt)
        except Exception as e:
            logger.error(f"分析失败日志异常: {e}")
        return failures
