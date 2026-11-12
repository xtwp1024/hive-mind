# -*- coding: utf-8 -*-
"""
Conflict Resolution (冲突解决策略)

Phase 5: 分布式网络 - 知识同步增强

核心职责:
1. 知识版本冲突检测 - 检测同时更新导致的冲突
2. 自动冲突解决 - 基于规则的自动合并
3. 手动冲突标记 - 需要人工干预的冲突
4. 冲突日志记录 - 完整的冲突历史
5. 回滚机制 - 恢复到冲突前状态

冲突类型:
┌─────────────────────────────────────────────────────────────┐
│  冲突类型              │  严重程度  │  解决策略               │
├─────────────────────────────────────────────────────────────┤
│  版本冲突 (Version)   │  中       │  时间戳优先/版本号优先  │
│  内容冲突 (Content)   │  高       │  自动合并/标记待审核    │
│  来源冲突 (Source)    │  中       │  来源优先级              │
│  依赖冲突 (Dependency)│  高       │  递归解析                │
└─────────────────────────────────────────────────────────────┘

解决策略:
┌─────────────────────────────────────────────────────────────┐
│  策略名称              │  描述                               │
├─────────────────────────────────────────────────────────────┤
│  LAST_WRITE_WINS       │  最后写入优先 (时间戳)             │
│  HIGHER_VERSION        │  高版本优先                         │
│  MERGE_AUTO            │  尝试自动合并                       │
│  SOURCE_PRIORITY       │  来源优先级 (可信节点优先)          │
│  VOTE_RESOLUTION       │  投票决议 (多节点投票)             │
│  MANUAL_REVIEW         │  标记待人工审核                     │
└─────────────────────────────────────────────────────────────┘
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("Hive.ConflictResolution")


class ConflictType(str, Enum):
    """冲突类型"""

    VERSION = "version"
    CONTENT = "content"
    SOURCE = "source"
    DEPENDENCY = "dependency"
    METADATA = "metadata"
    UNKNOWN = "unknown"


class ConflictSeverity(str, Enum):
    """冲突严重程度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResolutionStrategy(str, Enum):
    """解决策略"""

    LAST_WRITE_WINS = "last_write_wins"
    HIGHER_VERSION = "higher_version"
    MERGE_AUTO = "merge_auto"
    SOURCE_PRIORITY = "source_priority"
    VOTE_RESOLUTION = "vote_resolution"
    MANUAL_REVIEW = "manual_review"


class ResolutionStatus(str, Enum):
    """解决状态"""

    DETECTED = "detected"
    AUTO_RESOLVED = "auto_resolved"
    PENDING_REVIEW = "pending_review"
    RESOLVED = "resolved"
    REVERTED = "reverted"


@dataclass
class KnowledgeVersion:
    """知识版本"""

    entry_id: str
    version: int
    content_hash: str

    source_node: str
    source_timestamp: float

    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "version": self.version,
            "content_hash": self.content_hash,
            "source_node": self.source_node,
            "source_timestamp": self.source_timestamp,
            "content": self.content,
            "metadata": self.metadata,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeVersion":
        return cls(
            entry_id=data["entry_id"],
            version=data.get("version", 1),
            content_hash=data.get("content_hash", ""),
            source_node=data.get("source_node", ""),
            source_timestamp=data.get("source_timestamp", time.time()),
            content=data.get("content"),
            metadata=data.get("metadata", {}),
            dependencies=data.get("dependencies", []),
        )


@dataclass
class ConflictRecord:
    """冲突记录"""

    conflict_id: str

    conflict_type: ConflictType
    severity: ConflictSeverity

    entry_id: str

    local_version: Optional[KnowledgeVersion] = None
    remote_version: Optional[KnowledgeVersion] = None

    detected_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None

    resolution_strategy: Optional[ResolutionStrategy] = None
    resolution_status: ResolutionStatus = ResolutionStatus.DETECTED

    resolved_by: Optional[str] = None
    resolved_version: Optional[KnowledgeVersion] = None

    conflict_details: Dict[str, Any] = field(default_factory=dict)
    resolution_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "entry_id": self.entry_id,
            "local_version": self.local_version.to_dict() if self.local_version else None,
            "remote_version": self.remote_version.to_dict() if self.remote_version else None,
            "detected_at": self.detected_at,
            "resolved_at": self.resolved_at,
            "resolution_strategy": (
                self.resolution_strategy.value if self.resolution_strategy else None
            ),
            "resolution_status": self.resolution_status.value,
            "resolved_by": self.resolved_by,
            "resolved_version": self.resolved_version.to_dict() if self.resolved_version else None,
            "conflict_details": self.conflict_details,
            "resolution_details": self.resolution_details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConflictRecord":
        local_v = None
        if data.get("local_version"):
            local_v = KnowledgeVersion.from_dict(data["local_version"])

        remote_v = None
        if data.get("remote_version"):
            remote_v = KnowledgeVersion.from_dict(data["remote_version"])

        resolved_v = None
        if data.get("resolved_version"):
            resolved_v = KnowledgeVersion.from_dict(data["resolved_version"])

        return cls(
            conflict_id=data["conflict_id"],
            conflict_type=ConflictType(data.get("conflict_type", "unknown")),
            severity=ConflictSeverity(data.get("severity", "medium")),
            entry_id=data["entry_id"],
            local_version=local_v,
            remote_version=remote_v,
            detected_at=data.get("detected_at", time.time()),
            resolved_at=data.get("resolved_at"),
            resolution_strategy=(
                ResolutionStrategy(data["resolution_strategy"])
                if data.get("resolution_strategy")
                else None
            ),
            resolution_status=ResolutionStatus(data.get("resolution_status", "detected")),
            resolved_by=data.get("resolved_by"),
            resolved_version=resolved_v,
            conflict_details=data.get("conflict_details", {}),
            resolution_details=data.get("resolution_details", {}),
        )


class ConflictDetector:
    """
    冲突检测器

    检测知识更新时的各种冲突。
    """

    def __init__(self):
        self._hash_cache: Dict[str, str] = {}
        logger.info("🔍 [ConflictDetector] 初始化")

    def detect_conflict(
        self,
        local_version: KnowledgeVersion,
        remote_version: KnowledgeVersion,
    ) -> Optional[Tuple[ConflictType, ConflictSeverity, Dict[str, Any]]]:
        """
        检测两个版本之间的冲突

        Args:
            local_version: 本地版本
            remote_version: 远程版本

        Returns:
            (ConflictType, ConflictSeverity, details) 或 None
        """
        if local_version.entry_id != remote_version.entry_id:
            return None

        if local_version.content_hash == remote_version.content_hash:
            return None

        if self._is_version_conflict(local_version, remote_version):
            return (
                ConflictType.VERSION,
                ConflictSeverity.MEDIUM,
                self._analyze_version_conflict(local_version, remote_version),
            )

        if self._is_content_conflict(local_version, remote_version):
            return (
                ConflictType.CONTENT,
                ConflictSeverity.HIGH,
                self._analyze_content_conflict(local_version, remote_version),
            )

        if self._is_dependency_conflict(local_version, remote_version):
            return (
                ConflictType.DEPENDENCY,
                ConflictSeverity.HIGH,
                self._analyze_dependency_conflict(local_version, remote_version),
            )

        if self._is_source_conflict(local_version, remote_version):
            return (
                ConflictType.SOURCE,
                ConflictSeverity.MEDIUM,
                self._analyze_source_conflict(local_version, remote_version),
            )

        return None

    def _is_version_conflict(
        self,
        local: KnowledgeVersion,
        remote: KnowledgeVersion,
    ) -> bool:
        """检测版本冲突"""
        if local.version == remote.version:
            return local.content_hash != remote.content_hash
        return False

    def _analyze_version_conflict(
        self,
        local: KnowledgeVersion,
        remote: KnowledgeVersion,
    ) -> Dict[str, Any]:
        """分析版本冲突详情"""
        return {
            "same_version": local.version == remote.version,
            "local_version": local.version,
            "remote_version": remote.version,
            "local_source": local.source_node,
            "remote_source": remote.source_node,
            "time_diff": abs(local.source_timestamp - remote.source_timestamp),
            "local_hash": local.content_hash,
            "remote_hash": remote.content_hash,
        }

    def _is_content_conflict(
        self,
        local: KnowledgeVersion,
        remote: KnowledgeVersion,
    ) -> bool:
        """检测内容冲突"""
        if remote.version > local.version:
            return False

        if local.content_hash == remote.content_hash:
            return False

        if self._can_auto_merge(local.content, remote.content):
            return False

        return True

    def _analyze_content_conflict(
        self,
        local: KnowledgeVersion,
        remote: KnowledgeVersion,
    ) -> Dict[str, Any]:
        """分析内容冲突详情"""
        local_type = type(local.content).__name__
        remote_type = type(remote.content).__name__

        can_merge = self._can_auto_merge(local.content, remote.content)
        diff_size = self._estimate_diff_size(local.content, remote.content)

        return {
            "local_content_type": local_type,
            "remote_content_type": remote_type,
            "can_auto_merge": can_merge,
            "estimated_diff_size": diff_size,
            "local_is_dict": isinstance(local.content, dict),
            "remote_is_dict": isinstance(remote.content, dict),
            "local_is_list": isinstance(local.content, list),
            "remote_is_list": isinstance(remote.content, list),
        }

    def _is_dependency_conflict(
        self,
        local: KnowledgeVersion,
        remote: KnowledgeVersion,
    ) -> bool:
        """检测依赖冲突"""
        if not local.dependencies or not remote.dependencies:
            return False

        local_set = set(local.dependencies)
        remote_set = set(remote.dependencies)

        if local_set != remote_set:
            return True

        return False

    def _analyze_dependency_conflict(
        self,
        local: KnowledgeVersion,
        remote: KnowledgeVersion,
    ) -> Dict[str, Any]:
        """分析依赖冲突详情"""
        local_set = set(local.dependencies)
        remote_set = set(remote.dependencies)

        return {
            "local_dependencies": local.dependencies,
            "remote_dependencies": remote.dependencies,
            "added": list(remote_set - local_set),
            "removed": list(local_set - remote_set),
            "common": list(local_set & remote_set),
        }

    def _is_source_conflict(
        self,
        local: KnowledgeVersion,
        remote: KnowledgeVersion,
    ) -> bool:
        """检测来源冲突"""
        if local.source_node == remote.source_node:
            return False

        if remote.version > local.version:
            return False

        return True

    def _analyze_source_conflict(
        self,
        local: KnowledgeVersion,
        remote: KnowledgeVersion,
    ) -> Dict[str, Any]:
        """分析来源冲突详情"""
        return {
            "local_source": local.source_node,
            "remote_source": remote.source_node,
            "sources_differ": local.source_node != remote.source_node,
            "version_diff": remote.version - local.version,
        }

    def _can_auto_merge(self, local_content: Any, remote_content: Any) -> bool:
        """判断是否可以自动合并"""
        if local_content is None or remote_content is None:
            return True

        if isinstance(local_content, dict) and isinstance(remote_content, dict):
            return True

        if isinstance(local_content, list) and isinstance(remote_content, list):
            return True

        if isinstance(local_content, str) and isinstance(remote_content, str):
            return len(local_content) < 500 and len(remote_content) < 500

        return False

    def _estimate_diff_size(self, local_content: Any, remote_content: Any) -> int:
        """估计差异大小"""
        try:
            local_str = json.dumps(local_content, sort_keys=True)
            remote_str = json.dumps(remote_content, sort_keys=True)
            return sum(1 for a, b in zip(local_str, remote_str) if a != b)
        except Exception:
            return 1000


class ConflictResolver:
    """
    冲突解决器

    基于策略自动解决冲突。
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.HIGHER_VERSION,
        source_priorities: Optional[Dict[str, int]] = None,
    ):
        self.default_strategy = default_strategy
        self.source_priorities = source_priorities or {}

        self._merge_callbacks: Dict[str, Callable] = {}
        self._lock = threading.RLock()

        logger.info(f"⚖️ [ConflictResolver] 初始化，默认策略: {default_strategy.value}")

    def resolve(
        self,
        conflict: ConflictRecord,
        strategy: Optional[ResolutionStrategy] = None,
    ) -> ConflictRecord:
        """
        解决冲突

        Args:
            conflict: 冲突记录
            strategy: 解决策略 (默认使用 default_strategy)

        Returns:
            ConflictRecord - 更新后的冲突记录
        """
        if strategy is None:
            strategy = self.default_strategy

        conflict.resolution_strategy = strategy

        if conflict.local_version is None or conflict.remote_version is None:
            conflict.resolution_status = ResolutionStatus.PENDING_REVIEW
            conflict.resolution_details = {"reason": "缺少必要版本信息"}
            return conflict

        if strategy == ResolutionStrategy.LAST_WRITE_WINS:
            result = self._resolve_last_write_wins(conflict)
        elif strategy == ResolutionStrategy.HIGHER_VERSION:
            result = self._resolve_higher_version(conflict)
        elif strategy == ResolutionStrategy.MERGE_AUTO:
            result = self._resolve_merge_auto(conflict)
        elif strategy == ResolutionStrategy.SOURCE_PRIORITY:
            result = self._resolve_source_priority(conflict)
        elif strategy == ResolutionStrategy.MANUAL_REVIEW:
            result = self._mark_for_review(conflict)
        else:
            result = self._mark_for_review(conflict)

        return result

    def _resolve_last_write_wins(self, conflict: ConflictRecord) -> ConflictRecord:
        """最后写入优先策略"""
        local_ts = conflict.local_version.source_timestamp
        remote_ts = conflict.remote_version.source_timestamp

        if remote_ts > local_ts:
            conflict.resolved_version = deepcopy(conflict.remote_version)
            conflict.resolution_details = {
                "strategy": "last_write_wins",
                "winner": "remote",
                "local_timestamp": local_ts,
                "remote_timestamp": remote_ts,
                "time_diff": remote_ts - local_ts,
            }
        else:
            conflict.resolved_version = deepcopy(conflict.local_version)
            conflict.resolution_details = {
                "strategy": "last_write_wins",
                "winner": "local",
                "local_timestamp": local_ts,
                "remote_timestamp": remote_ts,
                "time_diff": local_ts - remote_ts,
            }

        conflict.resolution_status = ResolutionStatus.AUTO_RESOLVED
        conflict.resolved_at = time.time()
        conflict.resolved_by = "auto:last_write_wins"

        return conflict

    def _resolve_higher_version(self, conflict: ConflictRecord) -> ConflictRecord:
        """高版本优先策略"""
        local_ver = conflict.local_version.version
        remote_ver = conflict.remote_version.version

        if remote_ver > local_ver:
            conflict.resolved_version = deepcopy(conflict.remote_version)
            conflict.resolution_details = {
                "strategy": "higher_version",
                "winner": "remote",
                "local_version": local_ver,
                "remote_version": remote_ver,
            }
        elif remote_ver < local_ver:
            conflict.resolved_version = deepcopy(conflict.local_version)
            conflict.resolution_details = {
                "strategy": "higher_version",
                "winner": "local",
                "local_version": local_ver,
                "remote_version": remote_ver,
            }
        else:
            return self._resolve_last_write_wins(conflict)

        conflict.resolution_status = ResolutionStatus.AUTO_RESOLVED
        conflict.resolved_at = time.time()
        conflict.resolved_by = "auto:higher_version"

        return conflict

    def _resolve_merge_auto(self, conflict: ConflictRecord) -> ConflictRecord:
        """自动合并策略"""
        local = conflict.local_version
        remote = conflict.remote_version

        merged_content = self._merge_content(local.content, remote.content)

        if merged_content is None:
            return self._mark_for_review(conflict)

        new_hash = self._compute_hash(merged_content)

        conflict.resolved_version = KnowledgeVersion(
            entry_id=local.entry_id,
            version=max(local.version, remote.version) + 1,
            content_hash=new_hash,
            source_node="merged",
            source_timestamp=time.time(),
            content=merged_content,
            metadata={
                "merged_from": {
                    "local": {"source": local.source_node, "version": local.version},
                    "remote": {"source": remote.source_node, "version": remote.version},
                }
            },
            dependencies=list(set(local.dependencies) | set(remote.dependencies)),
        )

        conflict.resolution_details = {
            "strategy": "merge_auto",
            "merged_version": conflict.resolved_version.version,
            "merged_hash": new_hash,
        }

        conflict.resolution_status = ResolutionStatus.AUTO_RESOLVED
        conflict.resolved_at = time.time()
        conflict.resolved_by = "auto:merge_auto"

        return conflict

    def _resolve_source_priority(self, conflict: ConflictRecord) -> ConflictRecord:
        """来源优先级策略"""
        local_source = conflict.local_version.source_node
        remote_source = conflict.remote_version.source_node

        local_prio = self.source_priorities.get(local_source, 0)
        remote_prio = self.source_priorities.get(remote_source, 0)

        if remote_prio > local_prio:
            conflict.resolved_version = deepcopy(conflict.remote_version)
            winner = "remote"
        elif remote_prio < local_prio:
            conflict.resolved_version = deepcopy(conflict.local_version)
            winner = "local"
        else:
            return self._resolve_last_write_wins(conflict)

        conflict.resolution_details = {
            "strategy": "source_priority",
            "winner": winner,
            "local_priority": local_prio,
            "remote_priority": remote_prio,
        }

        conflict.resolution_status = ResolutionStatus.AUTO_RESOLVED
        conflict.resolved_at = time.time()
        conflict.resolved_by = "auto:source_priority"

        return conflict

    def _mark_for_review(self, conflict: ConflictRecord) -> ConflictRecord:
        """标记为待人工审核"""
        conflict.resolution_status = ResolutionStatus.PENDING_REVIEW
        conflict.resolution_details = {
            "reason": "需要人工审核",
            "auto_merge_possible": self._can_auto_merge(
                conflict.local_version.content if conflict.local_version else None,
                conflict.remote_version.content if conflict.remote_version else None,
            ),
        }
        return conflict

    def _merge_content(self, local: Any, remote: Any) -> Optional[Any]:
        """合并内容"""
        if local is None:
            return remote
        if remote is None:
            return local

        if isinstance(local, dict) and isinstance(remote, dict):
            return self._merge_dicts(local, remote)

        if isinstance(local, list) and isinstance(remote, list):
            return self._merge_lists(local, remote)

        if isinstance(local, str) and isinstance(remote, str):
            if local == remote:
                return local
            return None

        if type(local) == type(remote):
            return local if time.time() % 2 == 0 else remote

        return None

    def _merge_dicts(self, local: Dict, remote: Dict) -> Dict:
        """合并字典"""
        result = {}

        all_keys = set(local.keys()) | set(remote.keys())

        for key in all_keys:
            local_val = local.get(key)
            remote_val = remote.get(key)

            if key not in local:
                result[key] = remote_val
            elif key not in remote:
                result[key] = local_val
            elif local_val == remote_val:
                result[key] = local_val
            elif isinstance(local_val, dict) and isinstance(remote_val, dict):
                result[key] = self._merge_dicts(local_val, remote_val)
            elif isinstance(local_val, list) and isinstance(remote_val, list):
                result[key] = self._merge_lists(local_val, remote_val)
            else:
                result[f"{key}_local"] = local_val
                result[f"{key}_remote"] = remote_val
                result[f"{key}_merged_at"] = time.time()

        return result

    def _merge_lists(self, local: List, remote: List) -> List:
        """合并列表 (去重)"""
        seen = set()
        result = []

        for item in local + remote:
            try:
                item_hash = json.dumps(item, sort_keys=True)
                if item_hash not in seen:
                    seen.add(item_hash)
                    result.append(item)
            except Exception:
                if item not in result:
                    result.append(item)

        return result

    def _can_auto_merge(self, local: Any, remote: Any) -> bool:
        """判断是否可以自动合并"""
        if local is None or remote is None:
            return True

        if isinstance(local, dict) and isinstance(remote, dict):
            return True

        if isinstance(local, list) and isinstance(remote, list):
            return True

        return False

    def _compute_hash(self, content: Any) -> str:
        """计算内容哈希"""
        try:
            normalized = json.dumps(content, sort_keys=True, ensure_ascii=True)
            return hashlib.sha256(normalized.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(content).encode()).hexdigest()[:16]

    def set_source_priority(self, source: str, priority: int) -> None:
        """设置来源优先级"""
        with self._lock:
            self.source_priorities[source] = priority
        logger.info(f"⚖️ [ConflictResolver] 设置来源优先级: {source} = {priority}")

    def register_merge_callback(
        self,
        content_type: str,
        callback: Callable,
    ) -> None:
        """注册自定义合并回调"""
        with self._lock:
            self._merge_callbacks[content_type] = callback
        logger.info(f"⚖️ [ConflictResolver] 注册合并回调: {content_type}")


class ConflictLog:
    """
    冲突日志

    记录所有冲突历史，支持查询和统计。
    """

    def __init__(self, max_logs: int = 10000):
        self.max_logs = max_logs
        self._logs: deque = deque(maxlen=max_logs)
        self._by_entry: Dict[str, List[str]] = defaultdict(list)
        self._by_status: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()

        logger.info(f"📝 [ConflictLog] 初始化，最大日志数: {max_logs}")

    def log_conflict(self, conflict: ConflictRecord) -> None:
        """记录冲突"""
        with self._lock:
            self._logs.append(conflict)
            self._by_entry[conflict.entry_id].append(conflict.conflict_id)
            self._by_status[conflict.resolution_status.value].append(conflict.conflict_id)

        logger.info(
            f"📝 [ConflictLog] 记录冲突: {conflict.conflict_id} "
            f"type={conflict.conflict_type.value} severity={conflict.severity.value}"
        )

    def update_conflict(self, conflict_id: str, **updates) -> bool:
        """更新冲突记录"""
        with self._lock:
            for conflict in self._logs:
                if conflict.conflict_id == conflict_id:
                    for key, value in updates.items():
                        if hasattr(conflict, key):
                            setattr(conflict, key, value)

                    old_status = conflict.resolution_status.value
                    if "resolution_status" in updates:
                        if conflict_id in self._by_status.get(old_status, []):
                            self._by_status[old_status].remove(conflict_id)
                        new_status = updates["resolution_status"].value
                        self._by_status[new_status].append(conflict_id)

                    return True
            return False

    def get_conflict(self, conflict_id: str) -> Optional[ConflictRecord]:
        """获取冲突记录"""
        with self._lock:
            for conflict in self._logs:
                if conflict.conflict_id == conflict_id:
                    return conflict
            return None

    def get_by_entry(self, entry_id: str) -> List[ConflictRecord]:
        """获取某个条目的所有冲突"""
        with self._lock:
            conflict_ids = self._by_entry.get(entry_id, [])
            return [c for c in self._logs if c.conflict_id in conflict_ids]

    def get_by_status(self, status: ResolutionStatus) -> List[ConflictRecord]:
        """获取特定状态的冲突"""
        with self._lock:
            conflict_ids = self._by_status.get(status.value, [])
            return [c for c in self._logs if c.conflict_id in conflict_ids]

    def get_pending_review(self) -> List[ConflictRecord]:
        """获取待审核的冲突"""
        return self.get_by_status(ResolutionStatus.PENDING_REVIEW)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            by_type = defaultdict(int)
            by_severity = defaultdict(int)
            by_status = defaultdict(int)

            for c in self._logs:
                by_type[c.conflict_type.value] += 1
                by_severity[c.severity.value] += 1
                by_status[c.resolution_status.value] += 1

            return {
                "total_conflicts": len(self._logs),
                "by_type": dict(by_type),
                "by_severity": dict(by_severity),
                "by_status": dict(by_status),
                "pending_review": len(self.get_pending_review()),
                "max_logs": self.max_logs,
            }


class ConflictResolutionEngine:
    """
    冲突解决引擎

    整合检测器、解决器、日志的统一接口。
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.HIGHER_VERSION,
        max_logs: int = 10000,
        auto_resolve: bool = True,
    ):
        self.detector = ConflictDetector()
        self.resolver = ConflictResolver(default_strategy)
        self.logger = ConflictLog(max_logs)

        self.auto_resolve = auto_resolve
        self._lock = threading.RLock()

        logger.info(
            f"🔧 [ConflictResolutionEngine] 初始化，自动解决: {auto_resolve}, 默认策略: {default_strategy.value}"
        )

    def detect_and_resolve(
        self,
        local_version: KnowledgeVersion,
        remote_version: KnowledgeVersion,
        strategy: Optional[ResolutionStrategy] = None,
    ) -> Tuple[Optional[ConflictRecord], Optional[KnowledgeVersion]]:
        """
        检测并解决冲突

        Args:
            local_version: 本地版本
            remote_version: 远程版本
            strategy: 解决策略

        Returns:
            (ConflictRecord|None, resolved_version|None)
            - 如果没有冲突: (None, remote_version)
            - 如果有冲突: (ConflictRecord, resolved_version|None)
        """
        conflict_info = self.detector.detect_conflict(local_version, remote_version)

        if conflict_info is None:
            return None, remote_version

        conflict_type, severity, details = conflict_info

        conflict = ConflictRecord(
            conflict_id=f"conflict_{uuid.uuid4().hex[:12]}",
            conflict_type=conflict_type,
            severity=severity,
            entry_id=local_version.entry_id,
            local_version=deepcopy(local_version),
            remote_version=deepcopy(remote_version),
            conflict_details=details,
        )

        self.logger.log_conflict(conflict)

        logger.warning(
            f"🔧 [ConflictEngine] 检测到冲突: {conflict.conflict_id} "
            f"type={conflict_type.value} severity={severity.value}"
        )

        if self.auto_resolve:
            conflict = self.resolver.resolve(conflict, strategy)
            self.logger.update_conflict(
                conflict.conflict_id,
                resolution_status=conflict.resolution_status,
                resolution_strategy=conflict.resolution_strategy,
                resolved_version=conflict.resolved_version,
                resolved_at=conflict.resolved_at,
                resolved_by=conflict.resolved_by,
                resolution_details=conflict.resolution_details,
            )

            if conflict.resolution_status == ResolutionStatus.AUTO_RESOLVED:
                logger.info(
                    f"🔧 [ConflictEngine] 冲突自动解决: {conflict.conflict_id} "
                    f"strategy={conflict.resolution_strategy.value if conflict.resolution_strategy else 'N/A'}"
                )
            elif conflict.resolution_status == ResolutionStatus.PENDING_REVIEW:
                logger.warning(f"🔧 [ConflictEngine] 冲突待人工审核: {conflict.conflict_id}")

        return conflict, conflict.resolved_version

    def resolve_manually(
        self,
        conflict_id: str,
        chosen_version: KnowledgeVersion,
        resolved_by: str = "manual",
    ) -> bool:
        """
        手动解决冲突

        Args:
            conflict_id: 冲突ID
            chosen_version: 选择的版本
            resolved_by: 解决者标识

        Returns:
            bool - 是否成功
        """
        conflict = self.logger.get_conflict(conflict_id)
        if not conflict:
            return False

        conflict.resolved_version = chosen_version
        conflict.resolution_status = ResolutionStatus.RESOLVED
        conflict.resolved_at = time.time()
        conflict.resolved_by = resolved_by
        conflict.resolution_strategy = ResolutionStrategy.MANUAL_REVIEW
        conflict.resolution_details = {"manual_resolution": True}

        return self.logger.update_conflict(
            conflict_id,
            resolved_version=chosen_version,
            resolution_status=ResolutionStatus.RESOLVED,
            resolution_strategy=ResolutionStrategy.MANUAL_REVIEW,
            resolved_at=time.time(),
            resolved_by=resolved_by,
            resolution_details={"manual_resolution": True},
        )

    def set_source_priority(self, source: str, priority: int) -> None:
        """设置来源优先级"""
        self.resolver.set_source_priority(source, priority)

    def get_pending_conflicts(self) -> List[ConflictRecord]:
        """获取待审核的冲突"""
        return self.logger.get_pending_review()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        log_stats = self.logger.get_stats()

        return {
            "auto_resolve": self.auto_resolve,
            "default_strategy": self.resolver.default_strategy.value,
            "source_priorities": dict(self.resolver.source_priorities),
            **log_stats,
        }


_conflict_engine: Optional[ConflictResolutionEngine] = None
_conflict_lock = threading.Lock()


def get_conflict_engine() -> ConflictResolutionEngine:
    """获取冲突解决引擎单例"""
    global _conflict_engine
    if _conflict_engine is None:
        with _conflict_lock:
            if _conflict_engine is None:
                _conflict_engine = ConflictResolutionEngine()
    return _conflict_engine


def init_conflict_engine(**kwargs) -> ConflictResolutionEngine:
    """初始化冲突解决引擎"""
    global _conflict_engine
    with _conflict_lock:
        if _conflict_engine is not None:
            return _conflict_engine
        _conflict_engine = ConflictResolutionEngine(**kwargs)
    return _conflict_engine
