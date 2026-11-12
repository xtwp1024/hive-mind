#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事件流写入器 — 给前端动画用的实时数据管道

P1-7：根锚点判据已收敛到 hive.paths（单一真源）。本模块保留
``PROJECT_ROOT`` 与 ``resolve_hive_home`` 两个名字作为兼容薄壳，
供历史导入与 P1-1 回归注入使用；判断逻辑一行也不在这里。
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from hive.paths import (
    PROJECT_ROOT,
    resolve_events_file,
)
from hive.paths import resolve_hive_home as _resolve_hive_home

_events_file: Path | None = None
_lock = threading.Lock()


def resolve_hive_home() -> Path:
    """兼容薄壳：委托给 hive.paths 的唯一实现。

    传入本模块的 PROJECT_ROOT 作为基准根，使 P1-1 回归里的
    ``monkeypatch.setattr(es, "PROJECT_ROOT", tmp_path)`` 仍可注入隔离；
    HIVE_HOME 的 strip / expanduser / 回退判据全在 hive.paths 一份。
    """
    return _resolve_hive_home(PROJECT_ROOT)


def init_events_file() -> Path:
    """初始化事件文件，清空旧内容"""
    global _events_file
    _events_file = resolve_events_file(PROJECT_ROOT)
    _events_file.parent.mkdir(parents=True, exist_ok=True)
    _events_file.write_text("", encoding="utf-8")  # 清空
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
