# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, Any, List
import time
import multiprocessing

from hive.ministers import BaseMinister
from hive.core.dna import BeeDNA
from hive.config.settings import get_settings

logger = logging.getLogger("Hive.Doctor")

class DoctorBee(BaseMinister):
    """
    DoctorBee (医疗官/医生蜂) - Minister Level
    
    职责:
    1. 自我诊断 (Self-Diagnosis): 扫描项目依赖、版本和健康状况。
    2. 代码维护 (Code Maintenance): 自动修复简单的集成问题。
    3. 免疫增强 (Immunity): 与 QueenBee 协作处理系统级错误。
    
    源自: synthesized_assets/agentic_framework.py
    """
    
    def __init__(self, dna: BeeDNA, id: str, result_queue: multiprocessing.Queue):
        super().__init__(dna, id, result_queue)
        self.settings = get_settings()
        # 修正: 使用当前工作目录作为根目录，因为 settings.BASE_DIR 可能被配置为 Inbound
        self.project_root = Path.cwd() 
        
        logger.info(f"🚑 医疗官 {self.id} 已就位。随时准备治疗。")

    def diagnose(self) -> Dict[str, Any]:
        """
        运行全面诊断
        """
        report = {
            "status": "HEALTHY",
            "issues": [],
            "timestamp": time.time()
        }
        
        logger.info("🩺 开始全面系统诊断...")
        
        # 1. 依赖检查
        deps_report = self._check_dependencies()
        report["dependency_check"] = deps_report
        if deps_report["missing"]:
            report["status"] = "SICK"
            report["issues"].append(f"Missing dependencies: {deps_report['missing']}")

        # 2. 空间检查
        disk_report = self._check_disk_space()
        report["disk_check"] = disk_report
        
        # 3. 核心文件完整性
        integrity_report = self._check_core_integrity()
        report["integrity_check"] = integrity_report
        if integrity_report["missing_files"]:
            report["status"] = "CRITICAL"
            report["issues"].append(f"Missing core files: {integrity_report['missing_files']}")
            logger.error(f"❌ 完整性校验失败! 缺失文件: {integrity_report['missing_files']} (Root: {self.project_root})")

        logger.info(f"🩺 诊断完成。状态: {report['status']}")
        return report

    def _check_dependencies(self) -> Dict[str, Any]:
        """简单模拟 DependencyValidatorBee"""
        missing = []
        try:
            import chromadb
        except ImportError: missing.append("chromadb")
        
        try:
            import requests
        except ImportError: missing.append("requests")
            
        return {"missing": missing, "checked": ["chromadb", "requests", "pydantic"]}

    def _check_disk_space(self) -> Dict[str, Any]:
        """检查剩余空间"""
        import shutil
        total, used, free = shutil.disk_usage(self.project_root)
        return {
            "free_gb": round(free / (1024**3), 2),
            "status": "OK" if free > 1024**3 else "LOW_SPACE"
        }

    def _check_core_integrity(self) -> Dict[str, Any]:
        """检查关键文件是否存在"""
        core_files = [
            "hive_swarm.py",
            "hive/core/queen.py",
            "hive/core/king.py",
            "hive/ministers/engineer.py"
        ]
        missing = []
        for f in core_files:
            if not (self.project_root / f).exists():
                missing.append(f)
        return {"missing_files": missing}

    def treat(self, diagnosis: Dict[str, Any]):
        """
        根据诊断结果进行治疗
        """
        if diagnosis["status"] == "HEALTHY":
            logger.info("✅ 系统健康，无需治疗。")
            return

        logger.info("💉 开始尝试治疗程序...")
        
        # 处理依赖缺失
        if diagnosis.get("dependency_check", {}).get("missing"):
            import subprocess
            import sys
            for pkg in diagnosis["dependency_check"]["missing"]:
                logger.info(f"💊 正在处方: pip install {pkg}")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                    logger.info(f"✅ {pkg} 治愈成功")
                except Exception as e:
                    logger.error(f"❌ {pkg} 治疗失败: {e}")

        # 其他治疗逻辑...
