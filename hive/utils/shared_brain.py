# -*- coding: utf-8 -*-
"""
共享大脑事件日志写入器

将结构化事件写入 JSONL，并同步生成索引（JSONL）。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SharedBrainEventWriter:
    """共享大脑事件日志写入器（线程安全）。"""

    events_dir: Path
    enabled: bool = True
    source: str = "hive"

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        if self.enabled:
            self.events_dir.mkdir(parents=True, exist_ok=True)

    def _get_date_key(self, timestamp: str | None = None) -> str:
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        return datetime.now().strftime("%Y-%m-%d")

    def _event_path(self, date_key: str) -> Path:
        return self.events_dir / f"{date_key}.jsonl"

    def _index_path(self, date_key: str) -> Path:
        return self.events_dir / f"{date_key}.index.jsonl"

    def write_event(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return

        timestamp = event.get("timestamp") or event.get("time") or event.get("ts")
        date_key = self._get_date_key(timestamp)
        event_path = self._event_path(date_key)
        index_path = self._index_path(date_key)

        payload = {
            "source": self.source,
            "timestamp": timestamp or datetime.now().isoformat(),
            "level": event.get("level"),
            "logger": event.get("logger"),
            "event": event.get("event"),
            "message": event.get("message"),
            "data": event,
        }

        line = json.dumps(payload, ensure_ascii=False)
        line_bytes = (line + "\n").encode("utf-8")

        with self._lock:
            event_path.parent.mkdir(parents=True, exist_ok=True)
            if event_path.exists():
                offset = event_path.stat().st_size
            else:
                offset = 0

            with event_path.open("ab") as f:
                f.write(line_bytes)

            index_record = {
                "offset": offset,
                "length": len(line_bytes),
                "timestamp": payload["timestamp"],
                "level": payload.get("level"),
                "logger": payload.get("logger"),
                "event": payload.get("event"),
            }
            index_line = json.dumps(index_record, ensure_ascii=False)
            with index_path.open("a", encoding="utf-8") as idx_f:
                idx_f.write(index_line + "\n")


class SharedBrainLogProcessor:
    """structlog 处理器：将日志写入共享大脑事件日志。"""

    def __init__(self, writer: SharedBrainEventWriter) -> None:
        self._writer = writer
        self._logger = logging.getLogger("Hive.SharedBrainLogProcessor")

    def __call__(self, logger: Any, method_name: str, event_dict: dict) -> dict:
        try:
            self._writer.write_event(event_dict)
        except Exception as e:
            # 事件日志失败不应影响主流程，但应记录以便调试
            self._logger.debug(f"Shared brain event write failed (non-critical): {e}")
        return event_dict
