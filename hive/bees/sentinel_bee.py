#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel Bee - Unified Security Scanner
Scans codebase for toxic patterns, security risks, and vulnerabilities.
Consolidates SentinelBee and CyberSentinelBee.
"""

import re
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.config.settings import get_settings
from hive.core.swarm_communication import get_swarm_communication, MessageType, SwarmMessage

logger = logging.getLogger("Hive.Sentinel")

class SentinelBee(BaseBee):
    """
    Sentinel Bee - Unified Security Scanner.
    Detects: Code Injection, Command Injection, Weak Crypto, Hardcoded Secrets, etc.
    """
    
    def __init__(self, dna: BeeDNA, id: str, result_queue: Any):
        super().__init__(dna, id, result_queue)
        self.settings = get_settings()
        self.report_dir = self.settings.REPORT_DIR / "sentinel"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Consolidate patterns from Sentinel and CyberSentinel
        self.toxic_patterns = {
            'code_injection': [
                { 'pattern': r'\beval\s*\(', 'severity': 'HIGH', 'description': 'Usage of eval() can execute arbitrary code.' },
                { 'pattern': r'\bexec\s*\(', 'severity': 'HIGH', 'description': 'Usage of exec() can execute arbitrary code.' },
                { 'pattern': r'\bcompile\s*\(', 'severity': 'MEDIUM', 'description': 'Dynamic compilation can be risky.' },
                { 'pattern': r'\b__import__\s*\(', 'severity': 'HIGH', 'description': 'Dynamic import can be risky.' },
            ],
            'command_injection': [
                { 'pattern': r'\bos\.system\s*\(', 'severity': 'HIGH', 'description': 'Executing system commands is risky.' },
                { 'pattern': r'\bos\.popen\s*\(', 'severity': 'HIGH', 'description': 'Executing system commands is risky.' },
                { 'pattern': r'\bsubprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True', 'severity': 'HIGH', 'description': 'Shell=True enables command injection.' },
            ],
            'deserialization': [
                { 'pattern': r'\bpickle\.load(s)?\s*\(', 'severity': 'HIGH', 'description': 'Unsafe deserialization (pickle).' },
                { 'pattern': r'\bmarshal\.load(s)?\s*\(', 'severity': 'HIGH', 'description': 'Unsafe deserialization (marshal).' },
                { 'pattern': r'\byaml\.load\s*\(', 'severity': 'MEDIUM', 'description': 'Unsafe YAML loading (use safe_load).' },
            ],
            'secrets': [
                { 'pattern': r'password\s*=\s*["\']([^"\']+)["\']', 'severity': 'HIGH', 'description': 'Possible hardcoded password.' },
                { 'pattern': r'api_key\s*=\s*["\']([^"\']+)["\']', 'severity': 'HIGH', 'description': 'Possible hardcoded API key.' },
                { 'pattern': r'secret\s*=\s*["\']([^"\']+)["\']', 'severity': 'HIGH', 'description': 'Possible hardcoded secret.' },
                { 'pattern': r'token\s*=\s*["\']([^"\']+)["\']', 'severity': 'HIGH', 'description': 'Possible hardcoded token.' },
            ],
            'weak_crypto': [
                { 'pattern': r'\bmd5\s*\(', 'severity': 'MEDIUM', 'description': 'MD5 is weak. Use SHA256.' },
                { 'pattern': r'\bsha1\s*\(', 'severity': 'MEDIUM', 'description': 'SHA1 is weak. Use SHA256.' },
                { 'pattern': r'\bDES\s*\(', 'severity': 'HIGH', 'description': 'DES is deprecated.' },
            ],
            'sql_injection': [
                { 'pattern': r'["\'].*?["\']\s*\+\s*["\'].*?["\']\s*\.\s*format', 'severity': 'MEDIUM', 'description': 'Potential SQL injection via format.' },
                { 'pattern': r'execute\s*\(\s*f["\']', 'severity': 'HIGH', 'description': 'Potential SQL injection via f-string.' },
            ],
            'network': [
                { 'pattern': r'http://', 'severity': 'MEDIUM', 'description': 'Unencrypted HTTP Usage.' },
                { 'pattern': r'0\.0\.0\.0', 'severity': 'LOW', 'description': 'Binding to all interfaces.' },
            ],
            'debug': [
                { 'pattern': r'\bprint\s*\(', 'severity': 'LOW', 'description': 'Leftover debug print.' },
                { 'pattern': r'\bpdb\.set_trace', 'severity': 'LOW', 'description': 'Debugger breakdown.' },
            ],
        }
        
        self.scan_results = {
            "issues": [],
            "stats": {"high": 0, "medium": 0, "low": 0, "files_scanned": 0}
        }

        # Phase 2: 接入蜂群通信系统
        self._swarm = get_swarm_communication()
        self._swarm.register_bee(self.id, self)
        self._swarm.subscribe(MessageType.TASK_REQUEST, self._on_task_request)
        logger.info(f"🛡️ [Sentinel] 已注册到蜂群网络，等待任务请求...")

    def _on_task_request(self, message: SwarmMessage):
        """Phase 2: 处理来自其他蜂的任务请求"""
        content = message.content
        if content.get("action") != "security_scan":
            return

        task_id = content.get("task_id")
        target_str = content.get("target")
        requester = content.get("requester", "unknown")

        if not target_str or not task_id:
            return

        logger.info(f"🛡️ [Sentinel] 收到安全扫描请求 (task={task_id}, from={requester})")

        try:
            target_path = Path(target_str)
            if not target_path.is_absolute():
                target_path = Path.cwd() / target_path

            # 执行扫描
            scan_result = self._scan_codebase(target_path)

            # 通过蜂群网络返回结果（点对点发给请求者）
            result_msg = SwarmMessage(
                id=f"knowledge_{task_id}",
                type=MessageType.KNOWLEDGE,
                from_bee=self.id,
                to_bee=requester,
                content={
                    "topic": "security_scan_result",
                    "task_id": task_id,
                    "result": {
                        "status": "success",
                        "stats": scan_result["stats"],
                        "issues": scan_result["issues"][:10],  # 最多10个问题
                    }
                },
                priority=7,
            )
            self._swarm.send_message(result_msg)
            logger.info(f"✅ [Sentinel] 安全扫描结果已发送给 {requester}")

        except Exception as e:
            logger.error(f"❌ [Sentinel] 安全扫描失败: {e}")
            error_msg = SwarmMessage(
                id=f"knowledge_{task_id}",
                type=MessageType.KNOWLEDGE,
                from_bee=self.id,
                to_bee=requester,
                content={
                    "topic": "security_scan_result",
                    "task_id": task_id,
                    "result": {"status": "error", "message": str(e)}
                },
                priority=7,
            )
            self._swarm.send_message(error_msg)

    def specialized_task(self, target_path: Path):
        """Execute security scan."""
        logger.info(f"🛡️ SentinelBee Scanning: {self.dna.target_name}")
        
        try:
            self.scan_results = self._scan_codebase(target_path)
            
            # Generate Report
            self._generate_report(target_path)
            
            stats = self.scan_results["stats"]
            logger.info(f"✅ Scan Complete: {stats['files_scanned']} files. Issues: H{stats['high']} M{stats['medium']} L{stats['low']}")
            
            return {
                "status": "success",
                "stats": stats,
                "report_location": str(self.report_dir)
            }
            
        except Exception as e:
            logger.error(f"❌ Scan Failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def _scan_codebase(self, target_path: Path) -> Dict:
        """Scan all relevant files."""
        issues = []
        files_count = 0
        extensions = {'.py', '.js', '.ts', '.go', '.java', '.c', '.cpp', '.php', '.rb', '.sh', '.yaml', '.json', '.xml'}
        exclude_dirs = {'.git', '__pycache__', 'node_modules', 'venv', '.env', 'dist', 'build'}

        for file_path in target_path.rglob('*'):
            if not file_path.is_file(): continue
            if any(p in exclude_dirs for p in file_path.parts): continue
            if file_path.suffix not in extensions: continue

            files_count += 1
            file_issues = self._scan_file(file_path, target_path)
            issues.extend(file_issues)

        # Calculate stats
        stats = {
            "high": len([i for i in issues if i['severity'] == 'HIGH']),
            "medium": len([i for i in issues if i['severity'] == 'MEDIUM']),
            "low": len([i for i in issues if i['severity'] == 'LOW']),
            "files_scanned": files_count,
            "total_issues": len(issues)
        }
        
        return {"issues": issues, "stats": stats}

    def _scan_file(self, file_path: Path, root_path: Path) -> List[Dict]:
        """Scan a single file."""
        found = []
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
            
            for line_idx, line in enumerate(lines):
                for cat, patterns in self.toxic_patterns.items():
                    for pat in patterns:
                        if re.search(pat['pattern'], line):
                            found.append({
                                "file": str(file_path.relative_to(root_path)),
                                "line": line_idx + 1,
                                "content": line.strip()[:100],
                                "category": cat,
                                "severity": pat['severity'],
                                "description": pat['description']
                            })
        except Exception as e:
            logger.debug(f"Could not read {file_path}: {e}")
            
        return found

    def _generate_report(self, target_path: Path):
        """Generate JSON report."""
        report = {
            "target": self.dna.target_name,
            "timestamp": datetime.now().isoformat(),
            "bee_id": self.id,
            "stats": self.scan_results["stats"],
            "issues": self.scan_results["issues"],
            "recommendations": self._generate_recommendations()
        }
        
        filename = f"{self.dna.target_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(self.report_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

    def _generate_recommendations(self) -> List[str]:
        """Generate friendly advice."""
        stats = self.scan_results["stats"]
        recs = []
        
        if stats["high"] > 0:
            recs.append("CRITICAL: High severity issues found. Manual review required immediately.")
            recs.append("Check for remote code execution (RCE) vulnerabilities (eval/exec).")
            
        if stats["medium"] > 0:
            recs.append("Review use of weak cryptography and sql injection risks.")
            
        if any(i['category'] == 'secrets' for i in self.scan_results["issues"]):
            recs.append("Rotate any potentially exposed API keys or secrets detected.")
            
        return recs
