#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事件流写入器 — 给前端动画用的实时数据管道"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

# 项目根锚点：本文件位于 <root>/hive/utils/，上溯两级即仓库根。
# 与读取方 web_dashboard.py 的 BASE_DIR (Path(__file__).parent) 对齐，
# 保证写入方与读取方始终落在同一个 <root>/Data/events.jsonl。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_events_file: Path | None = None
_lock = threading.Lock()


def resolve_hive_home() -> Path:
    """解析事件流根目录：单一环境变量读取 + 明确的跨平台默认值。

    旧实现把「cwd 拼一个带盘符的常量」当作 os.getenv 的默认值，有两处隐患：
      1. 那个带盘符的常量在 Windows 是绝对路径，``/`` 会把 cwd 整个丢掉
         （落点恒为 D 盘）；在 POSIX 则是相对路径，会拼出带冒号的怪目录。
         同一份默认值跨平台落点不同。
      2. 默认值依赖 cwd，而读取方用的是文件所在目录。从别处启动进程时
         写入方与读取方分家，前端实时模式读不到事件。

    改为：HIVE_HOME 非空则用它（只读取一次），否则回退项目根；不拼 cwd、
    不硬编码盘符。空串 / 纯空白按「未设置」处理，避免 ``Path("")`` 落到 cwd。
    """
    home = os.environ.get("HIVE_HOME", "").strip()
    return Path(home).expanduser() if home else PROJECT_ROOT


def init_events_file() -> Path:
    """初始化事件文件，清空旧内容"""
    global _events_file
    data_dir = resolve_hive_home() / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _events_file = data_dir / "events.jsonl"
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
