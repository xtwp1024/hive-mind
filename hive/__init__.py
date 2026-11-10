#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive V2 包初始化
Antigravity 蜂巢系统的重构版本
"""

# ============================================
# 🚨 关键：强制所有第三方库缓存落 D 盘
# 必须在任何 chromadb/transformers/torch 等 import 之前执行
# 否则模型会被下载到 C:\Users\...\AppData\Local\...
# ============================================
import os as _os
from pathlib import Path as _Path

from .paths import resolve_hive_home as _resolve_hive_home

def _force_cache_to_drive() -> None:
    """将 AI 相关缓存目录重定向到 hive 根（不写死盘符、不拼 cwd）"""
    # 根锚点统一由 hive.paths.resolve_hive_home 解析（P1-7 单一真源）。
    # 这里只保留 .env 预加载：它把 HIVE_HOME 写进 os.environ，随后被唯一实现读到。
    if not _os.environ.get("HIVE_HOME"):
        # 尝试从 .env 手动加载 HIVE_HOME / BASE_DIR
        _env_file = _Path(_os.getcwd()) / ".env"
        if _env_file.exists():
            for _line in _env_file.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip()
                if _key in ("HIVE_HOME", "BASE_DIR") and _val and _key not in _os.environ:
                    _os.environ[_key] = _val

    base = _resolve_hive_home()
    cache_root = base / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # ChromaDB 专用缓存（含 ONNX 模型）
    _os.environ.setdefault("CHROMA_HOME", str(cache_root / "chroma"))
    # HuggingFace / transformers
    _os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface"))
    _os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    # PyTorch hub
    _os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    # 通用 XDG（Linux/macOS 也兼容）
    _os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    # 遥测关闭（不写 C 盘隐藏目录）
    _os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

    # ChromaDB 额外的 ONNX 模型缓存路径（这个最关键）
    _os.environ.setdefault("CHROMA_ONNX_CACHE", str(cache_root / "chroma" / "onnx"))

_force_cache_to_drive()
del _force_cache_to_drive  # 释放辅助符号

__version__ = "2.0.0"
__author__ = "Antigravity Team"

try:
    print(f"🏗️ Hive V2 Package Initialized (v{__version__})")
except (UnicodeEncodeError, UnicodeDecodeError):
    print(f"Hive V2 Package Initialized (v{__version__})")
