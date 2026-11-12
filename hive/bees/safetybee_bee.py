# -*- coding: utf-8 -*-
import re
from pathlib import Path
from typing import Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class SafetyBee(BaseBee):
    """安全蜂 - 检测安全威胁、验证输入、安全审计"""

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.security_log = []
        self.threats_detected = []

    def specialized_task(self, target_path):
        """执行安全任务"""
        logger.info(f"[SafetyBee-{self.id}] 安全: {target_path}")

        target = Path(target_path)
        if not target.exists():
            logger.error(f"[SafetyBee-{self.id}] 目标不存在: {target_path}")
            return

        # 安全扫描
        threats = self._scan_for_threats(target)

        # 验证输入
        validation = self._validate_input(target)

        # 安全审计
        audit = self._security_audit(target)

        if threats:
            self.threats_detected.extend(threats)
            logger.warning(f"[SafetyBee-{self.id}] 检测到威胁: {len(threats)}")

    def _scan_for_threats(self, file_path: Path) -> List[Dict]:
        """扫描威胁"""
        threats = []
        content = file_path.read_text()

        # 检查敏感模式
        threat_patterns = [
            (r"password\s*=", "明文密码"),
            (r"api_key\s*=", "API密钥泄露"),
            (r"eval\(", "代码注入风险"),
            (r"exec\(", "代码执行风险"),
        ]

        for pattern, threat_type in threat_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                threats.append({"type": threat_type, "count": len(matches), "severity": "high"})

        return threats

    def _validate_input(self, file_path: Path) -> Dict:
        """验证输入"""
        validation = {"valid": True, "checks": []}

        # 检查文件大小
        size = file_path.stat().st_size
        size_ok = size < 10 * 1024 * 1024  # 10MB
        validation["checks"].append({"check": "file_size", "passed": size_ok, "value": size})

        if not size_ok:
            validation["valid"] = False

        return validation

    def _security_audit(self, file_path: Path) -> Dict:
        """安全审计"""
        audit = {"file": str(file_path), "timestamp": "audit_timestamp", "score": 1.0, "issues": []}

        # 模拟审计
        if self.threats_detected:
            audit["score"] = 0.5
            audit["issues"] = [t["type"] for t in self.threats_detected]

        return audit

    def verify_input(self, data: str, patterns: List[str]) -> bool:
        """验证输入"""
        for pattern in patterns:
            if not re.search(pattern, data):
                return False
        return True

    def scan_vulnerabilities(self, code: str) -> List[Dict]:
        """扫描漏洞"""
        # 简化的漏洞扫描
        vulnerabilities = []

        if "import os" in code and "system(" in code:
            vulnerabilities.append({"type": "command_injection", "severity": "high"})

        return vulnerabilities

    def devour(self) -> dict:
        """消耗目标并返回安全报告"""
        target_path = self.dna.id

        try:
            self.specialized_task(target_path)

            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "threats": len(self.threats_detected),
                "log_entries": len(self.security_log),
            }
        except Exception as e:
            logger.error(f"[SafetyBee-{self.id}] 安全检查失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e),
            }
