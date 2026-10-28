#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory Box (智能体物理海马体 / 记忆匣子引擎)
设计灵感来源于著名 Flutter 极速键值库 Isar/Hive。
它通过将数据强行装箱 (Box) 并放置于磁盘与内存夹层，赋予系统“前世记忆”能力，让任何 Bee 过目不忘。
基于轻量级、无包依赖的原则，内部由标准库 shelve / jsonline 作为底层驱动。
"""

import os
import atexit
import shelve
import logging
import threading
from pathlib import Path
from typing import Any, Optional, Dict
from hive.config.settings import get_settings

logger = logging.getLogger("Hive.MemoryBox")

class MemoryBox:
    _open_boxes: Dict[str, 'MemoryBox'] = {}
    _lock = threading.Lock()

    def __init__(self, box_name: str):
        self.box_name = box_name
        self._db_path = get_settings().BASE_DIR / ".hive_memory"

        # 确保物理文件夹存在
        if not self._db_path.exists():
            self._db_path.mkdir(parents=True, exist_ok=True)

        self._shelf_file = str(self._db_path / f"box_{box_name}.db")
        self._shelf = shelve.open(self._shelf_file, writeback=False)
        logger.debug(f"🗃️ [Memory Box] 记忆匣 '{box_name}' 生理连结已接通。")

        # 注册退出时自动安全闭匣
        atexit.register(self.close)

    @classmethod
    def open_box(cls, box_name: str) -> 'MemoryBox':
        """极速并安全的开启一个箱子，采用了进程隔离单例控制

        box_name 应包含 PID（由调用方传入），确保每个子进程使用独立的 shelve 文件，
        避免多进程并发访问同一文件导致数据库损坏。

        注意：多进程环境下，MemoryBox应在fork后打开，不应共享父进程的缓存实例。
        使用进程特定缓存替代类级共享字典，彻底避免fork后的状态污染。
        """
        import os
        current_pid = os.getpid()
        # 使用进程特定的缓存key，避免fork后状态污染
        process_cache_key = f"_boxes_pid{current_pid}"
        process_cache = getattr(cls, process_cache_key, None)
        if process_cache is None:
            process_cache = {}
            setattr(cls, process_cache_key, process_cache)

        # 如果box_name包含其他进程的PID（_pid存在但不是当前PID），说明是跨进程传入的，绕过缓存直接创建
        import re
        pid_pattern = re.search(r'_pid(\d+)', box_name)
        if pid_pattern:
            embedded_pid = int(pid_pattern.group(1))
            if embedded_pid != current_pid:
                # 跨进程调用，强制创建新实例（使用原box_name，不添加新PID后缀）
                return MemoryBox(box_name)

        if box_name not in process_cache:
            process_cache[box_name] = MemoryBox(box_name)
        return process_cache[box_name]

    def put(self, key: str, value: Any):
        """塞入记忆碎片 (O(1) 速度)"""
        if not isinstance(key, str):
            key = str(key)
        self._shelf[key] = value
        self._shelf.sync()  # 刷入磁盘以保证永续不丢

    def get(self, key: str, default: Any = None) -> Any:
        """极速抽提历史记忆"""
        return self._shelf.get(key, default)

    def delete(self, key: str):
        """抹除特定记忆"""
        if key in self._shelf:
            del self._shelf[key]
            self._shelf.sync()

    def clear(self):
        """脑起搏器：强制格式化当前记忆匣"""
        self._shelf.clear()
        self._shelf.sync()

    def close(self):
        """断开连接"""
        import os
        current_pid = os.getpid()
        process_cache_key = f"_boxes_pid{current_pid}"
        process_cache = getattr(MemoryBox, process_cache_key, None)
        try:
            if hasattr(self, '_shelf') and self._shelf is not None:
                self._shelf.close()
                self._shelf = None
            if process_cache and self.box_name in process_cache:
                del process_cache[self.box_name]
            logger.debug(f"🗃️ [Memory Box] '{self.box_name}' 已安全封闭。")
        except Exception as e:
            logger.debug(f"🗃️ [Memory Box] 关闭时出现异常: {e}")
            
    def __del__(self):
        """析构时确保资源释放"""
        self.close()
            
    def get_all_keys(self) -> list:
        return list(self._shelf.keys())
