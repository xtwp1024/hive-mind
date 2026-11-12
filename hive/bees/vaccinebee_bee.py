# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
from typing import Dict, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class VaccineBee(BaseBee):
    """疫苗蜂 - 检测病毒、清理恶意代码、免疫增强"""

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.virus_signatures = {}
        self.quarantine_zone = {}
        self.immunization_records = {}

    def specialized_task(self, target_path):
        """执行疫苗任务"""
        logger.info(f"[VaccineBee-{self.id}] 疫苗: {target_path}")

        target = Path(target_path)
        if not target.exists():
            logger.error(f"[VaccineBee-{self.id}] 目标不存在: {target_path}")
            return

        # 检测病毒
        viruses = self._detect_virus(target)

        if viruses:
            # 清理恶意代码
            self._cleanup_malware(target, viruses)
        else:
            # 免疫增强
            self._enhance_immunity(target)

        logger.info(f"[VaccineBee-{self.id}] 疫苗处理完成")

    def _detect_virus(self, file_path: Path) -> List[Dict]:
        """检测病毒"""
        content = file_path.read_text()
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # 检查已知病毒签名
        viruses = []

        suspicious_patterns = ["malicious_code", "eval(base64", "system(", "exec("]

        for pattern in suspicious_patterns:
            if pattern in content:
                virus_id = f"virus_{hashlib.md5(pattern.encode()).hexdigest()[:8]}"
                viruses.append({"id": virus_id, "type": pattern, "severity": "high"})

        return viruses

    def _cleanup_malware(self, file_path: Path, viruses: List[Dict]):
        """清理恶意代码"""
        content = file_path.read_text()

        for virus in viruses:
            # 从内容中移除恶意代码
            content = content.replace(virus["type"], "[MALWARE REMOVED]")

        # 备份原文件
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        file_path.rename(backup_path)

        # 写入清理后的内容
        file_path.write_text(content)

        # 记录隔离
        self.quarantine_zone[str(file_path)] = {
            "original": str(backup_path),
            "viruses": viruses,
            "cleaned": True,
        }

        logger.info(f"[VaccineBee-{self.id}] 已清理: {file_path}")

    def _enhance_immunity(self, file_path: Path):
        """增强免疫"""
        # 创建免疫记录
        immunity_id = hashlib.md5(str(file_path).encode()).hexdigest()[:8]

        self.immunization_records[str(file_path)] = {
            "immunity_id": immunity_id,
            "file_hash": hashlib.md5(file_path.read_text().encode()).hexdigest(),
            "scan_date": "vaccine_timestamp",
            "status": "immune",
        }

        logger.info(f"[VaccineBee-{self.id}] 免疫增强: {file_path}")

    def scan_system(self, directory: Path) -> Dict:
        """扫描系统"""
        scan_results = {
            "directory": str(directory),
            "files_scanned": 0,
            "viruses_found": 0,
            "infected_files": [],
        }

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                scan_results["files_scanned"] += 1
                viruses = self._detect_virus(file_path)

                if viruses:
                    scan_results["viruses_found"] += len(viruses)
                    scan_results["infected_files"].append(str(file_path))

        return scan_results

    def restore_backup(self, file_path: Path) -> bool:
        """恢复备份"""
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")

        if backup_path.exists():
            import shutil

            shutil.copy2(backup_path, file_path)
            logger.info(f"[VaccineBee-{self.id}] 已恢复: {file_path}")
            return True

        return False

    def devour(self) -> dict:
        """消耗目标并返回疫苗报告"""
        target_path = self.dna.id

        try:
            self.specialized_task(target_path)

            return {
                "name": self.dna.id,
                "status": "COMPLETED",
                "duration": "0.1s",
                "path": target_path,
                "quarantined": len(self.quarantine_zone),
                "immunized": len(self.immunization_records),
            }
        except Exception as e:
            logger.error(f"[VaccineBee-{self.id}] 疫苗处理失败: {e}")
            return {
                "name": self.dna.id,
                "status": "FAILED",
                "duration": "0s",
                "path": target_path,
                "error": str(e),
            }
