#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事件流写入器 — 给前端动画用的实时数据管道"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_events_file: Path | None = None
_lock = threading.Lock()


def init_events_file() -> Path:
    """初始化事件文件，清空旧内容"""
    global _events_file
    home = os.getenv("HIVE_HOME", str(Path.cwd() / "D:/HIVE"))
    data_dir = Path(home) / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _events_file = data_dir / "events.jsonl"
    _events_file.write_text("")  # 清空
    return _events_file


def emit(event_type: str, **data) -> None:
    """发一条事件到事件流"""
    if _events_file is None:
        return  # 未初始化，静默跳过
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "data": data,
    }
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with _lock:
        try:
            with open(_events_file, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except Exception:
            pass  # 前端断了不影响主循环
