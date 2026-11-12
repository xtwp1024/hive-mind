#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
免疫归档工具

提供独立的免疫记忆归档功能，避免循环导入问题。
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("Hive.ImmuneArchive")


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used safely in a filename.

    Args:
        name: The string to sanitize

    Returns:
        A safe filename string
    """
    # Remove any characters that are not alphanumeric, underscore, hyphen, or dot
    sanitized = re.sub(r"[^\w\-.]", "_", name)
    # Limit length to prevent overly long filenames
    return sanitized[:100]


def archive_failure(death_cert: Dict[str, Any]) -> bool:
    """
    归档失败记录到免疫记忆库

    Args:
        death_cert: 死亡证明数据

    Returns:
        是否归档成功
    """
    if not death_cert:
        return False

    try:
        from hive.config.settings import get_settings

        settings = get_settings()

        archive_dir = settings.BASE_DIR / "hive" / "immune_memory"
        archive_dir.mkdir(parents=True, exist_ok=True)

        target_name = death_cert.get("target_name", "unknown")
        # Security: Sanitize target_name to prevent path traversal via filename
        safe_target = _sanitize_filename(target_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = archive_dir / f"immune_{safe_target}_{timestamp}.json"

        archive_data = {
            "timestamp": datetime.now().isoformat(),
            "target_name": target_name,
            "error_type": death_cert.get("error_type", "unknown"),
            "error_message": death_cert.get("error_message", ""),
            "dna_snapshot": death_cert.get("dna_snapshot", {}),
            "bee_type": death_cert.get("bee_type", "unknown"),
        }

        with open(archive_file, "w", encoding="utf-8") as f:
            json.dump(archive_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"[ImmuneArchive] 归档失败记录: {archive_file}")
        return True

    except Exception as e:
        logger.error(f"[ImmuneArchive] 归档失败: {e}")
        return False


def get_immune_warnings(target_name: str) -> list:
    """
    获取目标的免疫警告

    Args:
        target_name: 目标名称

    Returns:
        警告列表
    """
    warnings = []

    try:
        from hive.config.settings import get_settings

        settings = get_settings()

        archive_dir = settings.BASE_DIR / "hive" / "immune_memory"
        if not archive_dir.exists():
            return warnings

        for archive_file in archive_dir.glob(f"immune_{target_name}_*.json"):
            try:
                with open(archive_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                warnings.append(
                    {
                        "error_type": data.get("error_type", "unknown"),
                        "error_message": data.get("error_message", "")[:100],
                        "timestamp": data.get("timestamp", ""),
                    }
                )
            except (json.JSONDecodeError, IOError):
                continue

        return warnings[-5:]

    except Exception as e:
        logger.debug(f"[ImmuneArchive] 获取免疫警告失败: {e}")
        return warnings
