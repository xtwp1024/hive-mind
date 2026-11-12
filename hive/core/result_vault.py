# -*- coding: utf-8 -*-
"""
ResultVault - 任务结果版本链仓库
设计灵感来源于 Hive2Hive 的 FileManager（add/update/recover + FileObserver）。

核心保证：每次写操作不覆盖旧值，而是追加新版本节点形成不可篡改的历史链。
任何历史版本都可以通过 recover() 精确回溯，冲突时触发 IVersionSelector 策略。
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("Hive.ResultVault")


@dataclass
class ResultVersion:
    """一个版本节点（对标 Hive2Hive 中的 IFileVersion）"""

    version_idx: int
    value: Any
    author: str  # 写入者 agent ID
    timestamp: float = field(default_factory=time.time)
    note: str = ""  # 版本说明（可选）

    def __repr__(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"v{self.version_idx}[{self.author}@{ts}]"


class VersionConflictError(Exception):
    """同一 key 在写入时检测到潜在冲突（对标 Hive2Hive ConflictManagement）"""

    pass


class ResultVault:
    """
    任务结果版本化仓库。
    - add():    首次写入（创建 v0）
    - update(): 追加新版本（不覆盖，只追加）
    - get():    获取最新版本值
    - recover():版本回溯
    - versions():枚举全部历史
    - watch():  FileObserver 变更监听
    """

    def __init__(self, vault_name: str = "DefaultVault"):
        self.vault_name = vault_name
        self._store: dict[str, list[ResultVersion]] = {}
        self._watchers: dict[str, list[Callable]] = {}
        self._lock = threading.RLock()
        logger.info(f"[ResultVault:{self.vault_name}] 已初始化")

    # ────────────────────────────────── 写操作 (FileManager) ──

    def add(self, key: str, value: Any, author: str, note: str = "") -> ResultVersion:
        """
        首次写入结果（对标 fileManager.add(file)）。
        若 key 已存在则抛出错误（用 update() 追加新版本）。
        """
        with self._lock:
            if key in self._store:
                raise ValueError(f"key={key!r} 已存在，请用 update() 追加版本")
            v = ResultVersion(version_idx=0, value=value, author=author, note=note)
            self._store[key] = [v]

        logger.info(f"[ResultVault] ✅ add({key!r}) → {v} 写入成功")
        self._notify(key, v)
        return v

    def update(self, key: str, value: Any, author: str, note: str = "") -> ResultVersion:
        """
        追加新版本（对标 fileManager.update(file)）。
        旧版本保留完整，不可覆盖删除。
        """
        with self._lock:
            if key not in self._store:
                raise KeyError(f"key={key!r} 不存在，请先用 add() 创建")
            idx = len(self._store[key])
            v = ResultVersion(version_idx=idx, value=value, author=author, note=note)
            self._store[key].append(v)

        logger.info(f"[ResultVault] ✅ update({key!r}) → {v}")
        self._notify(key, v)
        return v

    # ────────────────────────────────── 读操作 ──

    def get(self, key: str) -> Any:
        """获取最新版本值（对标普通文件读取）"""
        with self._lock:
            chain = self._store.get(key)
        if not chain:
            raise KeyError(f"key={key!r} 不存在")
        return chain[-1].value

    def recover(self, key: str, version_idx: int) -> ResultVersion:
        """
        回滚到指定历史版本（对标 fileManager.recover(file, versionSelector)）。
        不删除历史，而是把目标版本内容追加为新版本（保留完整历史链）。
        """
        with self._lock:
            chain = self._store.get(key, [])
        if not chain:
            raise KeyError(f"key={key!r} 不存在")
        if version_idx >= len(chain):
            raise IndexError(f"version {version_idx} 超出范围（共 {len(chain)} 版）")

        target = chain[version_idx]
        recovered = self.update(
            key=key,
            value=target.value,
            author=f"recover(←v{version_idx})",
            note=f"回溯自 {target}",
        )
        logger.info(f"[ResultVault] ⏪ recover({key!r}, v{version_idx}) → 已作为 {recovered} 追加")
        return recovered

    def versions(self, key: str) -> list[ResultVersion]:
        """列举所有历史版本（对标 IVersionSelector 可用列表）"""
        with self._lock:
            return list(self._store.get(key, []))

    def delete(self, key: str, author: str) -> None:
        """软删除：移除 key（对标 fileManager.delete()）"""
        with self._lock:
            if key not in self._store:
                raise KeyError(f"key={key!r} 不存在")
            del self._store[key]
        logger.info(f"[ResultVault] 🗑️  delete({key!r}) by {author}")

    # ────────────────────────────────── FileObserver ──

    def watch(self, key: str, callback: Callable[[str, ResultVersion], None]) -> None:
        """注册变更监听（对标 H2HFileObserver + H2HFileObserverListener）"""
        with self._lock:
            self._watchers.setdefault(key, []).append(callback)
        logger.debug(f"[ResultVault] 🔔 watch({key!r}) 已注册")

    def _notify(self, key: str, version: ResultVersion) -> None:
        """触发所有该 key 的监听回调"""
        with self._lock:
            cbs = list(self._watchers.get(key, []))
        for cb in cbs:
            try:
                cb(key, version)
            except Exception as e:
                logger.error(f"[ResultVault] watcher 异常: {e}")

    def vault_summary(self) -> dict:
        with self._lock:
            return {k: len(v) for k, v in self._store.items()}
