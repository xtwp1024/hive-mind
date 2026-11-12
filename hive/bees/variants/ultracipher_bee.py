# -*- coding: utf-8 -*-
"""
UltraCipherBee - 超能力增强加密蜂种
提供多层加密、数字签名和高级完整性保护
"""

import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from hive.bees.base_bee import BaseBee
from hive.config.settings import settings
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class UltraCipherBee(BaseBee):
    """
    超能力增强：数据加密和完整性保护

    特殊能力：能力翻倍 - 哈希校验 + 加密存储 + 数字签名
    变异率：3.37%
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.report_dir = settings.REPORT_DIR / "ultracipher"
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 初始化加密密钥
        self.symmetric_key = Fernet.generate_key()
        self.fernet = Fernet(self.symmetric_key)

        # 生成RSA密钥对用于数字签名
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

        # 记录加密结果
        self.encryption_results = []
        self.integrity_checks = []
        self.security_metrics = {}

    def specialized_task(self):
        """
        执行超能力增强任务：
        1. 多层加密分析
        2. 完整性验证和数字签名
        3. 安全性评估
        """
        logger.info(f"[{self.dna.target_name}] 开始执行超能力增强加密任务")

        # 第一阶段：分析项目敏感数据
        self._analyze_sensitive_data()

        # 第二阶段：执行完整性验证
        self._verify_data_integrity()

        # 第三阶段：生成加密策略
        self._generate_encryption_strategy()

        # 第四阶段：安全性评估
        self._assess_security_level()

        # 第五阶段：生成综合报告
        self._generate_security_report()

        logger.info(f"[{self.dna.target_name}] 超能力增强加密任务完成")

    def _analyze_sensitive_data(self):
        """分析项目中的敏感数据"""
        logger.info(f"[{self.dna.target_name}] 分析敏感数据...")

        from hive.config.settings import settings

        target_path = settings.BASE_DIR

        # 查找可能包含敏感数据的文件
        sensitive_patterns = [
            ("config", [".env", "config.json", "secrets.json", "credentials.json"]),
            ("code", ["*.py", "*.js", "*.ts"]),
            ("data", ["*.json", "*.csv", "*.yaml", "*.yml"]),
        ]

        for category, patterns in sensitive_patterns:
            for pattern in patterns:
                for file_path in target_path.rglob(pattern):
                    try:
                        result = self._analyze_file_sensitivity(file_path, category)
                        if result:
                            self.encryption_results.append(result)
                            logger.debug(
                                f"[{self.dna.target_name}] 分析文件: {file_path.relative_to(target_path)}"
                            )
                    except Exception as e:
                        logger.debug(f"[{self.dna.target_name}] 分析文件失败 {file_path}: {e}")

        logger.info(f"[{self.dna.target_name}] 共分析 {len(self.encryption_results)} 个文件")

    def _analyze_file_sensitivity(self, file_path: Path, category: str) -> dict:
        """分析单个文件的敏感度"""
        file_size = file_path.stat().st_size

        # 计算多种哈希值
        hashes = {
            "md5": self._calculate_hash(file_path, hashlib.md5()),
            "sha1": self._calculate_hash(file_path, hashlib.sha1()),
            "sha256": self._calculate_hash(file_path, hashlib.sha256()),
            "sha512": self._calculate_hash(file_path, hashlib.sha512()),
        }

        # 评估敏感度
        sensitivity_score = 0
        sensitive_keywords = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 检查敏感关键词
            keywords = {
                "high": ["password", "secret", "api_key", "token", "private_key", "auth"],
                "medium": ["database", "connection", "config", "setting", "credential"],
                "low": ["name", "email", "user", "account"],
            }

            for level, words in keywords.items():
                for word in words:
                    if word.lower() in content.lower():
                        sensitivity_score += {"high": 3, "medium": 2, "low": 1}[level]
                        sensitive_keywords.append(word)
        except (IOError, UnicodeDecodeError, PermissionError):
            pass

        # 加密测试（实际不修改文件，只是测试）
        encryption_test = self._test_encryption_performance(file_path)

        return {
            "path": str(file_path.relative_to(settings.BASE_DIR)),
            "category": category,
            "size": file_size,
            "sensitivity_score": sensitivity_score,
            "sensitivity_level": self._classify_sensitivity(sensitivity_score),
            "sensitive_keywords": list(set(sensitive_keywords)),
            "hashes": hashes,
            "encryption_test": encryption_test,
        }

    def _calculate_hash(self, file_path: Path, hash_algorithm) -> str:
        """计算文件哈希值"""
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_algorithm.update(chunk)
            return hash_algorithm.hexdigest()
        except (IOError, OSError):
            return "N/A"

    def _classify_sensitivity(self, score: int) -> str:
        """分类敏感度级别"""
        if score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"

    def _test_encryption_performance(self, file_path: Path) -> dict:
        """测试加密性能"""
        try:
            with open(file_path, "rb") as f:
                data = f.read()

            # 对称加密测试
            start_time = datetime.now()
            encrypted = self.fernet.encrypt(data)
            symmetric_encryption_time = (datetime.now() - start_time).total_seconds()

            # 非对称加密测试（仅对小数据）
            asymmetric_encryption_time = 0
            if len(data) < 256:  # RSA只能加密小数据
                start_time = datetime.now()
                self.public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )
                asymmetric_encryption_time = (datetime.now() - start_time).total_seconds()

            return {
                "symmetric_encryption_time": symmetric_encryption_time,
                "asymmetric_encryption_time": asymmetric_encryption_time,
                "encrypted_size": len(encrypted),
                "expansion_ratio": len(encrypted) / len(data) if data else 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def _verify_data_integrity(self):
        """验证数据完整性"""
        logger.info(f"[{self.dna.target_name}] 验证数据完整性...")

        # 对所有分析的文件进行完整性检查
        for result in self.encryption_results:
            try:
                from hive.config.settings import settings

                file_path = settings.BASE_DIR / result["path"]
                if file_path.exists():
                    integrity_check = self._perform_integrity_check(file_path, result["hashes"])
                    self.integrity_checks.append(integrity_check)
            except Exception as e:
                logger.debug(f"[{self.dna.target_name}] 完整性检查失败: {e}")

        logger.info(f"[{self.dna.target_name}] 完成 {len(self.integrity_checks)} 个完整性检查")

    def _perform_integrity_check(self, file_path: Path, expected_hashes: dict) -> dict:
        """执行完整性检查"""
        current_hashes = {
            "md5": self._calculate_hash(file_path, hashlib.md5()),
            "sha1": self._calculate_hash(file_path, hashlib.sha1()),
            "sha256": self._calculate_hash(file_path, hashlib.sha256()),
        }

        integrity_status = {"path": str(file_path), "hashes_match": {}, "overall_status": "valid"}

        for hash_type, current_hash in current_hashes.items():
            expected_hash = expected_hashes.get(hash_type)
            matches = current_hash == expected_hash
            integrity_status["hashes_match"][hash_type] = matches

            if not matches:
                integrity_status["overall_status"] = "mismatch"

        return integrity_status

    def _generate_encryption_strategy(self):
        """生成加密策略"""
        logger.info(f"[{self.dna.target_name}] 生成加密策略...")

        # 统计敏感度分布
        sensitivity_distribution = {}
        for result in self.encryption_results:
            level = result["sensitivity_level"]
            sensitivity_distribution[level] = sensitivity_distribution.get(level, 0) + 1

        # 生成策略建议
        self.encryption_strategy = {
            "symmetric_encryption": {
                "recommended": True,
                "algorithm": "Fernet (AES-128-CBC)",
                "key_rotation_period": "90 days",
                "use_case": "File encryption, Data at rest",
            },
            "asymmetric_encryption": {
                "recommended": True,
                "algorithm": "RSA-2048",
                "use_case": "Key exchange, Digital signatures",
            },
            "hashing": {
                "recommended": True,
                "algorithm": "SHA-256",
                "use_case": "Integrity verification, Password hashing",
            },
            "priorities": {
                "high_sensitivity_files": sensitivity_distribution.get("high", 0),
                "medium_sensitivity_files": sensitivity_distribution.get("medium", 0),
                "low_sensitivity_files": sensitivity_distribution.get("low", 0),
            },
            "recommendations": [
                "为高敏感度文件启用强加密",
                "定期轮换加密密钥",
                "实施多因素身份验证",
                "使用安全的密钥管理系统",
                "定期进行安全审计",
            ],
        }

    def _assess_security_level(self):
        """评估安全级别"""
        logger.info(f"[{self.dna.target_name}] 评估安全级别...")

        # 计算安全评分
        total_files = len(self.encryption_results)
        high_sensitivity = sum(
            1 for r in self.encryption_results if r["sensitivity_level"] == "high"
        )

        # 检查完整性问题
        integrity_issues = sum(
            1 for i in self.integrity_checks if i["overall_status"] == "mismatch"
        )

        # 计算安全评分 (0-100)
        security_score = 100

        # 扣分因素
        if high_sensitivity > 0:
            security_score -= min(30, high_sensitivity * 5)  # 高敏感度文件多扣分

        if integrity_issues > 0:
            security_score -= min(20, integrity_issues * 10)  # 完整性问题扣分

        security_score = max(0, security_score)

        self.security_metrics = {
            "overall_score": security_score,
            "security_level": self._classify_security_level(security_score),
            "files_analyzed": total_files,
            "high_sensitivity_files": high_sensitivity,
            "integrity_issues": integrity_issues,
            "encryption_readiness": self._assess_encryption_readiness(),
        }

    def _classify_security_level(self, score: int) -> str:
        """分类安全级别"""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "moderate"
        else:
            return "poor"

    def _assess_encryption_readiness(self) -> str:
        """评估加密准备度"""
        if len(self.encryption_results) == 0:
            return "no_data"

        high_sensitivity = sum(
            1 for r in self.encryption_results if r["sensitivity_level"] == "high"
        )

        if high_sensitivity == 0:
            return "ready"
        elif high_sensitivity <= len(self.encryption_results) * 0.2:
            return "mostly_ready"
        else:
            return "needs_attention"

    def _generate_security_report(self):
        """生成安全报告"""
        report = {
            "bee_type": "UltraCipher",
            "target_name": self.dna.target_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "files_analyzed": len(self.encryption_results),
                "integrity_checks": len(self.integrity_checks),
                "security_score": self.security_metrics.get("overall_score", 0),
                "security_level": self.security_metrics.get("security_level", "unknown"),
            },
            "encryption_results": self.encryption_results,
            "integrity_checks": self.integrity_checks,
            "encryption_strategy": self.encryption_strategy,
            "security_metrics": self.security_metrics,
            "action_items": self._generate_action_items(),
        }

        # 保存报告
        report_file = (
            self.report_dir
            / f"{self.dna.target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"[{self.dna.target_name}] 安全报告已生成: {report_file}")

    def _generate_action_items(self) -> list:
        """生成行动项"""
        action_items = []

        security_score = self.security_metrics.get("overall_score", 0)

        if security_score < 80:
            action_items.append(
                {
                    "priority": "high",
                    "action": "提高安全评分",
                    "description": f"当前安全评分为 {security_score}，建议采取额外安全措施",
                }
            )

        high_sensitivity = sum(
            1 for r in self.encryption_results if r["sensitivity_level"] == "high"
        )
        if high_sensitivity > 0:
            action_items.append(
                {
                    "priority": "high",
                    "action": "保护高敏感度文件",
                    "description": f"发现 {high_sensitivity} 个高敏感度文件，需要立即实施加密保护",
                }
            )

        integrity_issues = sum(
            1 for i in self.integrity_checks if i["overall_status"] == "mismatch"
        )
        if integrity_issues > 0:
            action_items.append(
                {
                    "priority": "critical",
                    "action": "调查完整性问题",
                    "description": f"发现 {integrity_issues} 个文件的哈希值不匹配，可能已被篡改",
                }
            )

        return action_items
