#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加密蜂 - CipherBee
数据加密和完整性保护

变异率：15.00%
特殊能力：哈希校验 + 加密存储
"""

import hashlib
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

try:
    from hive.utils.brain_interface import BrainInterface, get_brain
except ImportError:
    BrainInterface = None
    get_brain = None

logger = get_logger(__name__)

# --- Adopted from Awesome-Redteam ---
# ref: D:\Inbound\Awesome-Redteam
REDTEAM_PROMPT_TEMPLATE = """
# System
Role: You are a white-hat security specialist (CipherBee).
Goal: Identify potential security vulnerabilities in the target code and output a valid PoC (Proof of Concept) or remediation plan.
Context: Using the methodologies from the RedTeam-OffensiveSecurity knowledge base.

# Target File: {filename}
```python
{code}
```

# Task
1. Analyze the code for vulnerability type: [SQL Injection, XSS, RCE, Deserialization, Hardcoded Credentials]
2. If a vulnerability exists, write a PoC script (in Python/Bash).
3. If no vulnerability, provide a security reinforcement plan.
"""

REDTEAM_TOOL_REGISTRY = {
    "sqlmap": "sql injection detection",
    "hydra": "brute force login",
    "john": "hash cracking",
    "nmap": "port scanning"
}
# -----------------------------------


class CipherBee(BaseBee):
    """
    加密蜂 - 负责数据加密和完整性保护
    
    特殊能力：
    - 计算文件哈希值进行完整性校验
    - 加密敏感数据
    - 生成数字签名
    """
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.hash_cache = {}
        self.encryption_reports = []
        self._brain: Optional[BrainInterface] = None

    def _get_brain(self) -> Optional[BrainInterface]:
        """懒加载 BrainInterface 单例"""
        if self._brain is None and BrainInterface is not None and get_brain is not None:
            self._brain = get_brain()
        return self._brain

    def _call_llm_security(
        self,
        file_path: str,
        code_content: str
    ) -> Optional[str]:
        """
        调用 Brain LLM 获取安全漏洞分析

        Args:
            file_path: 目标文件路径
            code_content: 代码内容

        Returns:
            LLM 生成的漏洞分析报告，失败返回 None
        """
        brain = self._get_brain()
        if brain is None:
            logger.warning("🔐 CipherBee: BrainInterface 不可用，跳过 LLM 安全分析")
            return None

        prompt = REDTEAM_PROMPT_TEMPLATE.format(
            filename=Path(file_path).name,
            code=code_content[:3000]  # 限制上下文长度
        )

        system_prompt = """你是一位白帽安全专家（CipherBee），负责代码安全审计。
你的任务是：
1. 分析代码中的安全漏洞（SQL注入、XSS、RCE、反序列化、硬编码凭证等）
2. 如果发现漏洞，输出 PoC 脚本（Python/Bash）
3. 如果无漏洞，提供安全加固建议

请用中文回答，包含：
- 漏洞类型和严重程度（高/中/低）
- 具体位置（文件和行号）
- PoC 代码示例（如适用）
- 修复建议

工具可用：sqlmap、hydra、john、nmap"""

        try:
            result = brain.consult(prompt, system_prompt)
            if result:
                logger.info(f"🔐 CipherBee LLM 安全分析完成: {Path(file_path).name}")
                return result
            else:
                logger.warning(f"🔐 CipherBee LLM 返回空结果: {file_path}")
                return None
        except Exception as e:
            logger.error(f"🔐 CipherBee LLM 调用失败: {e}")
            return None

    def _analyze_file_security(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        对单个文件执行 LLM 安全分析

        Returns:
            包含 vulnerability_analysis 的字典，失败返回 None
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()

            # 只对有意义的文件调用 LLM（> 15 行）
            if len(code_content.split('\n')) < 15:
                return None

            # 快速静态检查：是否有可疑模式
            suspicious_patterns = [
                'eval(', 'exec(', 'os.system', 'subprocess',
                'request', 'input(', 'hardcode', 'password',
                'api_key', 'token', 'secret', 'jwt', 'auth'
            ]
            has_suspicious_code = any(
                p.lower() in code_content.lower() for p in suspicious_patterns
            )

            if not has_suspicious_code:
                return None  # 没有可疑模式，不需要深度分析

            llm_result = self._call_llm_security(str(file_path), code_content)

            if llm_result:
                return {
                    "file": str(file_path),
                    "vulnerability_analysis": llm_result,
                    "suspicious_patterns_found": has_suspicious_code
                }
        except Exception as e:
            logger.debug(f"🔐 文件安全分析失败 {file_path}: {e}")

        return None
    
    def specialized_task(self, target_path):
        """
        执行加密和完整性保护任务
        
        Args:
            target_path: 项目目录路径
        """
        logger.info(f"🔐 CipherBee 开始任务: {self.dna.target_name}")
        
        try:
            # 1. 计算项目文件的哈希值
            hash_results = self._calculate_project_hashes(target_path)
            
            # 2. 生成完整性报告
            integrity_report = self._generate_integrity_report(hash_results)
            
            # 3. 识别敏感文件
            sensitive_files = self._identify_sensitive_files(target_path)
            
            # 4. 生成加密建议
            encryption_recommendations = self._generate_encryption_recommendations(
                sensitive_files, hash_results
            )

            # 5. ===== Phase 0 Live Link: 调用 LLM 进行 RedTeam 安全分析 =====
            llm_security_results: List[Dict[str, Any]] = []
            if sensitive_files and BrainInterface is not None:
                logger.info(f"🔐 CipherBee 开始 LLM 安全分析 ({len(sensitive_files)} 个敏感文件)...")

                target = Path(target_path)
                files_to_analyze: List[Path] = []
                seen_paths: set = set()

                for sf in sensitive_files:
                    file_str = sf.get('path', '')
                    if file_str and file_str not in seen_paths:
                        p = Path(file_str)
                        if not p.is_absolute():
                            p = target / p
                        if p.exists():
                            files_to_analyze.append(p)
                            seen_paths.add(file_str)

                # 限制 LLM 调用次数
                max_analyze_files = 5
                for file_path in files_to_analyze[:max_analyze_files]:
                    logger.info(f"🔐 正在安全分析: {file_path.name}")
                    result = self._analyze_file_security(file_path)
                    if result:
                        llm_security_results.append(result)
                    time.sleep(1)  # 避免频率限制

                logger.info(f"🔐 CipherBee LLM 安全分析完成: {len(llm_security_results)} 个文件")

            # 6. 保存结果
            self._save_cipher_report(target_path, {
                "project": self.dna.target_name,
                "hash_results": hash_results,
                "integrity_report": integrity_report,
                "sensitive_files": sensitive_files,
                "encryption_recommendations": encryption_recommendations,
                "llm_security_results": llm_security_results,  # Phase 0 Live Link 新增
                "timestamp": __import__('time').ctime()
            })

            logger.info(f"✅ CipherBee 任务完成: {self.dna.target_name}")
            logger.info(f"   已计算 {len(hash_results)} 个文件的哈希值")
            logger.info(f"   发现 {len(sensitive_files)} 个潜在敏感文件")
            logger.info(f"   LLM RedTeam 安全分析: {len(llm_security_results)} 个文件")

            return {
                "status": "success",
                "files_hashed": len(hash_results),
                "sensitive_files_found": len(sensitive_files),
                "integrity_score": integrity_report["integrity_score"],
                "llm_security_files": len(llm_security_results)
            }
            
        except Exception as e:
            logger.error(f"❌ CipherBee 任务失败: {e}")
            return {"status": "error", "error": str(e)}

    def generate_redteam_payload(self, file_path: str) -> dict:
        """
        生成 RedTeam 渗透测试 Payload
        
        Args:
            file_path: 目标文件路径
            
        Returns:
            dict: 渗透测试载荷
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            prompt = REDTEAM_PROMPT_TEMPLATE.format(
                filename=Path(file_path).name,
                code=content[:2000]  # Avoid context overflow
            )
            
            return {
                "target_file": file_path,
                "prompt": prompt,
                "tools_available": REDTEAM_TOOL_REGISTRY,
                "model_suggestion": "gpt-4-turbo"
            }
        except Exception as e:
            logger.error(f"RedTeam Payload 生成失败: {e}")
            return {}
    
    def _calculate_project_hashes(self, target_path: Path) -> Dict[str, str]:
        """
        计算项目中所有代码文件的哈希值
        
        Args:
            target_path: 项目目录路径
            
        Returns:
            文件路径到哈希值的映射
        """
        hash_results = {}
        
        # 定义需要计算哈希的文件类型
        code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb']
        
        for file_path in target_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in code_extensions:
                try:
                    file_hash = self._calculate_file_hash(file_path)
                    relative_path = str(file_path.relative_to(target_path))
                    hash_results[relative_path] = file_hash
                    
                    logger.debug(f"哈希计算: {relative_path} -> {file_hash[:16]}...")
                    
                except Exception as e:
                    logger.debug(f"哈希计算失败 {file_path}: {e}")
        
        return hash_results
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """
        计算单个文件的哈希值
        
        Args:
            file_path: 文件路径
            algorithm: 哈希算法 (sha256, md5, sha1)
            
        Returns:
            哈希值字符串
        """
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            # 分块读取大文件
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _generate_integrity_report(self, hash_results: Dict[str, str]) -> Dict[str, Any]:
        """
        生成完整性报告
        
        Args:
            hash_results: 哈希计算结果
            
        Returns:
            完整性报告字典
        """
        report = {
            "total_files": len(hash_results),
            "total_size_bytes": 0,
            "hash_distribution": {},
            "integrity_score": 0.0
        }
        
        # 统计哈希分布
        for file_path, file_hash in hash_results.items():
            hash_prefix = file_hash[:8]
            report["hash_distribution"][hash_prefix] = \
                report["hash_distribution"].get(hash_prefix, 0) + 1
        
        # 计算完整性得分（基于唯一哈希数量）
        unique_hashes = len(set(hash_results.values()))
        report["integrity_score"] = unique_hashes / max(len(hash_results), 1) * 100
        
        return report
    
    def _identify_sensitive_files(self, target_path: Path) -> List[Dict[str, Any]]:
        """
        识别潜在的敏感文件
        
        Args:
            target_path: 项目目录路径
            
        Returns:
            敏感文件列表
        """
        sensitive_files = []
        
        # 敏感文件模式
        sensitive_patterns = [
            'password', 'secret', 'key', 'token', 'credential', 
            'auth', 'api_key', 'private', 'config'
        ]
        
        for file_path in target_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            filename = file_path.name.lower()
            relative_path = str(file_path.relative_to(target_path)).lower()
            
            # 检查文件名和路径
            for pattern in sensitive_patterns:
                if pattern in filename or pattern in relative_path:
                    try:
                        file_size = file_path.stat().st_size
                        sensitive_files.append({
                            "path": str(file_path.relative_to(target_path)),
                            "pattern": pattern,
                            "size": file_size,
                            "severity": "high" if "key" in pattern or "secret" in pattern else "medium"
                        })
                        break
                    except Exception as e:
                        logger.debug(f"检查敏感文件失败 {file_path}: {e}")
        
        return sensitive_files
    
    def _generate_encryption_recommendations(
        self, 
        sensitive_files: List[Dict[str, Any]], 
        hash_results: Dict[str, str]
    ) -> List[str]:
        """
        生成加密建议
        
        Args:
            sensitive_files: 敏感文件列表
            hash_results: 哈希计算结果
            
        Returns:
            加密建议列表
        """
        recommendations = []
        
        # 基于敏感文件的建议
        high_severity_files = [f for f in sensitive_files if f["severity"] == "high"]
        if high_severity_files:
            recommendations.append(
                f"发现 {len(high_severity_files)} 个高优先级敏感文件，建议使用强加密保护"
            )
        
        # 基于文件大小的建议
        large_sensitive_files = [f for f in sensitive_files if f["size"] > 10240]  # > 10KB
        if large_sensitive_files:
            recommendations.append(
                f"发现 {len(large_sensitive_files)} 个大型敏感文件，建议使用流式加密"
            )
        
        # 基于哈希分布的建议
        unique_hashes = len(set(hash_results.values()))
        if unique_hashes < len(hash_results) * 0.9:  # 如果有重复文件
            recommendations.append(
                "检测到重复文件，建议在加密前进行去重以减少加密开销"
            )
        
        if not recommendations:
            recommendations.append("项目安全性良好，未发现紧急加密需求")
        
        return recommendations
    
    def _save_cipher_report(self, target_path: Path, report_data: Dict):
        """
        保存加密报告
        
        Args:
            target_path: 项目目录路径
            report_data: 报告数据
        """
        try:
            # 在当前工作目录创建报告目录
            reports_dir = Path("cipher_reports")
            reports_dir.mkdir(exist_ok=True)
            
            # 生成报告文件名
            report_filename = f"{self.dna.target_name.replace('/', '_')}_cipher_report.json"
            report_path = reports_dir / report_filename
            
            # 保存报告
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 加密报告已保存: {report_path}")
            
        except Exception as e:
            logger.error(f"保存加密报告失败: {e}")
