# -*- coding: utf-8 -*-
"""
Knowledge Sync (知识同步)

Phase 5 P2: 全球知识共享
任意蜂群学到的知识可被其他蜂群使用。

核心职责:
1. 增量同步 — 只同步新增知识，避免重复传输
2. 知识去重 — 基于内容hash去重，避免存储冗余
3. 优先级同步 — 重要的知识优先同步
4. 知识版本管理 — 知识的创建/更新/失效
5. 联邦知识路由 — 根据知识类型路由到合适的节点

与 HiveFederation 集成:
  - 监听知识广播，自动接收同步
  - 主动从其他节点拉取缺失知识
  - 推送本节点新知识到联邦

使用方式:
    sync = KnowledgeSync(federation)
    sync.start()
    sync.push_knowledge({"type": "pattern", "content": "...", "priority": "high"})
    result = sync.pull_knowledge(filter={"type": "pattern"})
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("Hive.KnowledgeSync")


# ===== 知识条目 =====

@dataclass
class KnowledgeEntry:
    """知识条目"""
    id: str
    type: str           # "pattern", "strategy", "code", "insight", "lesson"
    content: Any        # 知识内容 (dict/list/str)
    content_hash: str   # 内容摘要
    priority: str = "normal"  # "critical", "high", "normal", "low"
    tags: list[str] = field(default_factory=list)
    source_node: str = ""      # 来源节点
    source_timestamp: float = 0  # 来源节点创建时间
    local_timestamp: float = 0   # 本地接收时间
    version: int = 1             # 版本号
    refs: int = 0                # 被引用次数
    失效: bool = False            # 是否已失效

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "content_hash": self.content_hash,
            "priority": self.priority,
            "tags": self.tags,
            "source_node": self.source_node,
            "source_timestamp": self.source_timestamp,
            "local_timestamp": self.local_timestamp,
            "version": self.version,
            "refs": self.refs,
            "失效": self.失效,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeEntry":
        return cls(
            id=data["id"],
            type=data["type"],
            content=data["content"],
            content_hash=data["content_hash"],
            priority=data.get("priority", "normal"),
            tags=data.get("tags", []),
            source_node=data.get("source_node", ""),
            source_timestamp=data.get("source_timestamp", 0),
            local_timestamp=data.get("local_timestamp", time.time()),
            version=data.get("version", 1),
            refs=data.get("refs", 0),
            失效=data.get("失效", False),
        )

    @classmethod
    def compute_hash(cls, content: Any) -> str:
        """计算内容hash"""
        normalized = json.dumps(content, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# ===== 知识同步 =====

class KnowledgeSync:
    """
    知识同步器

    Phase 5 P2: 全球知识共享
    """

    # 知识类型
    TYPE_PATTERN = "pattern"       # 代码模式
    TYPE_STRATEGY = "strategy"    # 策略
    TYPE_CODE = "code"            # 代码片段
    TYPE_INSIGHT = "insight"     # 洞察
    TYPE_LESSON = "lesson"       # 经验教训

    # 优先级
    PRIORITY_CRITICAL = "critical"
    PRIORITY_HIGH = "high"
    PRIORITY_NORMAL = "normal"
    PRIORITY_LOW = "low"

    def __init__(
        self,
        federation,  # HiveFederation 实例
        sync_interval: float = 30.0,  # 同步间隔 (秒)
        max_knowledge: int = 10000,    # 最大知识条数
        push_on_receive: bool = True,   # 收到知识后是否立即转发
    ):
        """
        初始化知识同步器

        Args:
            federation: HiveFederation 实例
            sync_interval: 定时同步间隔
            max_knowledge: 最大知识条数
            push_on_receive: 收到知识后是否立即转发给其他节点
        """
        self._federation = federation
        self._sync_interval = sync_interval
        self._max_knowledge = max_knowledge
        self._push_on_receive = push_on_receive

        # 知识库
        self._knowledge: dict[str, KnowledgeEntry] = {}
        self._kb_lock = threading.RLock()

        # 索引 (加速查询)
        self._by_type: dict[str, set[str]] = {}      # type -> entry ids
        self._by_hash: dict[str, str] = {}          # content_hash -> entry id
        self._by_tag: dict[str, set[str]] = {}      # tag -> entry ids

        # 同步状态
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._last_sync: float = 0

        # 已同步的hash集合 (用于去重)
        self._synced_hashes: set[str] = set()

        logger.info(f"📚 [KnowledgeSync] 知识同步器初始化，max_knowledge={max_knowledge}")

    # ===== 生命周期 =====

    def start(self):
        """启动知识同步"""
        if self._running:
            return
        self._running = True

        # Phase 5 P2: 注册联邦回调
        self._federation.set_knowledge_callback(self._on_knowledge_received)

        # 启动定时同步
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="knowledge-sync"
        )
        self._sync_thread.start()

        logger.info(f"📚 [KnowledgeSync] 知识同步器已启动")

    def stop(self):
        """停止知识同步"""
        if not self._running:
            return
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=3.0)
        logger.info(f"📚 [KnowledgeSync] 知识同步器已停止")

    def _sync_loop(self):
        """定时同步循环"""
        while self._running:
            time.sleep(self._sync_interval)
            try:
                self._do_full_sync()
            except Exception as e:
                logger.error(f"📚 [KnowledgeSync] 同步循环异常: {e}")

    def _do_full_sync(self):
        """执行一次完整同步"""
        nodes = self._federation.get_all_nodes()
        if not nodes:
            return

        self._last_sync = time.time()

        # 收集本节点知识摘要
        my_hashes = set(self._by_hash.keys())

        for node in nodes:
            if node.node_id == self._federation.node_id:
                continue

            try:
                # 发送本节点的知识hash列表 (让对方判断缺少哪些)
                missing = self._request_missing_hashes(node, my_hashes)
                if missing:
                    self._pull_knowledge(node, missing)
            except Exception as e:
                logger.debug(f"📚 [KnowledgeSync] 与 {node.node_id} 同步失败: {e}")

        # 推送本节点的新知识
        self._push_new_knowledge(nodes)

    def _request_missing_hashes(self, node, my_hashes: set) -> list[str]:
        """向节点请求缺失的知识hash"""
        # 简化: 直接请求对方的所有知识
        # 生产环境应该传 my_hashes 让对方计算差量
        try:
            result = self._federation._send_and_receive(
                node,
                "knowledge_sync_request",
                {"node_hashes": list(my_hashes)}
            )
            if result:
                return result.get("missing_hashes", [])
        except Exception:
            pass
        return []

    def _pull_knowledge(self, node, hashes: list[str]):
        """从节点拉取指定知识"""
        try:
            result = self._federation._send_and_receive(
                node,
                "knowledge_pull",
                {"hashes": hashes}
            )
            if result:
                entries = result.get("entries", [])
                for entry_dict in entries:
                    self._receive_entry(entry_dict)
        except Exception as e:
            logger.debug(f"📚 [KnowledgeSync] 拉取知识失败: {e}")

    def _push_new_knowledge(self, nodes: list):
        """推送本节点新知识到其他节点"""
        for node in nodes:
            if node.node_id == self._federation.node_id:
                continue

            # 获取未同步的知识
            new_entries = [
                e for e in self._knowledge.values()
                if e.content_hash not in self._synced_hashes
            ]

            if not new_entries:
                continue

            # 按优先级排序
            new_entries.sort(key=lambda e: (
                ["low", "normal", "high", "critical"].index(e.priority)
            ), reverse=True)

            # 分批推送 (避免单次过大)
            batch = []
            batch_size = 0
            max_batch = 50  # 最多50条

            for entry in new_entries:
                if len(batch) >= max_batch:
                    break
                entry_size = len(json.dumps(entry.to_dict()))
                if batch_size + entry_size > 100_000:  # 每次最多100KB
                    break
                batch.append(entry.to_dict())
                batch_size += entry_size

            if batch:
                try:
                    self._federation._send_to_node(
                        node,
                        "knowledge_push",
                        {"entries": batch}
                    )
                    for entry_dict in batch:
                        self._synced_hashes.add(entry_dict["content_hash"])
                except Exception as e:
                    logger.debug(f"📚 [KnowledgeSync] 推送知识失败: {e}")

    def _on_knowledge_received(self, msg_type: str, data: Any):
        """Phase 5 P2: 处理接收到的知识"""
        if msg_type == "push":
            # 收到结构化知识条目
            self._receive_entry(data)
        elif msg_type == "broadcast":
            # 收到广播知识 (转换格式)
            knowledge = data.get("knowledge", {})
            if isinstance(knowledge, dict):
                entry = KnowledgeEntry(
                    id=knowledge.get("id", str(uuid.uuid4())),
                    type=knowledge.get("type", "insight"),
                    content=knowledge.get("content"),
                    content_hash=knowledge.get("content_hash") or KnowledgeEntry.compute_hash(knowledge.get("content", "")),
                    priority=knowledge.get("priority", "normal"),
                    tags=knowledge.get("tags", []),
                    source_node=data.get("origin", ""),
                    source_timestamp=data.get("timestamp", time.time()),
                )
                self._receive_entry(entry.to_dict())

    # ===== 知识操作 =====

    def push_knowledge(
        self,
        content: Any,
        knowledge_type: str = "insight",
        priority: str = "normal",
        tags: Optional[list[str]] = None,
        allow_duplicate: bool = False,
    ) -> Optional[str]:
        """
        推送知识到联邦

        Args:
            content: 知识内容
            knowledge_type: 知识类型 (pattern/strategy/code/insight/lesson)
            priority: 优先级 (critical/high/normal/low)
            tags: 标签列表
            allow_duplicate: 是否允许重复

        Returns:
            知识ID，失败返回 None
        """
        content_hash = KnowledgeEntry.compute_hash(content)

        with self._kb_lock:
            # 检查重复
            if content_hash in self._by_hash and not allow_duplicate:
                existing_id = self._by_hash[content_hash]
                logger.debug(f"📚 [KnowledgeSync] 知识已存在: {existing_id}")
                return None

            # 创建条目
            entry = KnowledgeEntry(
                id=str(uuid.uuid4()),
                type=knowledge_type,
                content=content,
                content_hash=content_hash,
                priority=priority,
                tags=tags or [],
                source_node=self._federation.node_id,
                source_timestamp=time.time(),
                local_timestamp=time.time(),
            )

            self._add_entry(entry)
            self._synced_hashes.add(content_hash)

        # 广播到联邦
        if self._running:
            self._broadcast_entry(entry)

        logger.info(f"📚 [KnowledgeSync] 知识已推送: {entry.id} type={knowledge_type} priority={priority}")
        return entry.id

    def _add_entry(self, entry: KnowledgeEntry):
        """添加知识条目到本地库"""
        with self._kb_lock:
            self._knowledge[entry.id] = entry
            self._by_hash[entry.content_hash] = entry.id

            # 索引
            if entry.type not in self._by_type:
                self._by_type[entry.type] = set()
            self._by_type[entry.type].add(entry.id)

            for tag in entry.tags:
                if tag not in self._by_tag:
                    self._by_tag[tag] = set()
                self._by_tag[tag].add(entry.id)

            # 清理超量知识
            self._prune()

    def _prune(self):
        """清理超量知识 (保留高价值知识)"""
        while len(self._knowledge) > self._max_knowledge:
            # 删除最低优先级的最早条目
            candidates = [
                (eid, e) for eid, e in self._knowledge.items()
                if e.失效 and e.refs == 0
            ]
            if not candidates:
                candidates = [
                    (eid, e) for eid, e in self._knowledge.items()
                    if e.priority == "low"
                ]
            if not candidates:
                candidates = list(self._knowledge.items())

            # 按 (优先级索引, 时间) 排序
            priority_order = ["critical", "high", "normal", "low"]
            candidates.sort(key=lambda x: (
                priority_order.index(x[1].priority),
                x[1].local_timestamp
            ))

            remove_id = candidates[0][0]
            self._remove_entry(remove_id)

    def _remove_entry(self, entry_id: str):
        """移除知识条目"""
        entry = self._knowledge.pop(entry_id, None)
        if not entry:
            return

        self._by_hash.pop(entry.content_hash, None)

        if entry.type in self._by_type:
            self._by_type[entry.type].discard(entry_id)

        for tag in entry.tags:
            if tag in self._by_tag:
                self._by_tag[tag].discard(entry_id)

    def _receive_entry(self, entry_dict: dict):
        """接收知识条目"""
        content_hash = entry_dict.get("content_hash")
        if not content_hash:
            return

        with self._kb_lock:
            # 去重
            if content_hash in self._by_hash:
                # 已存在，更新版本
                existing_id = self._by_hash[content_hash]
                existing = self._knowledge.get(existing_id)
                if existing and existing.source_timestamp < entry_dict.get("source_timestamp", 0):
                    # 来源更新，替换
                    new_entry = KnowledgeEntry.from_dict(entry_dict)
                    self._remove_entry(existing_id)
                    self._add_entry(new_entry)
                    logger.info(f"📚 [KnowledgeSync] 知识更新: {existing_id}")
                return

            # 新增
            entry = KnowledgeEntry.from_dict(entry_dict)
            self._add_entry(entry)
            logger.debug(f"📚 [KnowledgeSync] 接收知识: {entry.id}")

    def _broadcast_entry(self, entry: KnowledgeEntry):
        """广播知识条目到联邦"""
        nodes = self._federation.get_all_nodes()
        for node in nodes:
            if node.node_id == self._federation.node_id:
                continue
            self._federation._send_to_node(
                node,
                "knowledge_push",
                {"entries": [entry.to_dict()]}
            )

    # ===== 查询 =====

    def query(
        self,
        knowledge_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        min_priority: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 50,
    ) -> list[KnowledgeEntry]:
        """
        查询知识

        Args:
            knowledge_type: 按类型过滤
            tags: 按标签过滤 (AND)
            min_priority: 最低优先级
            keyword: 关键词 (模糊匹配 content)
            limit: 返回条数限制

        Returns:
            匹配的知识条目列表
        """
        with self._kb_lock:
            # 确定候选集
            candidates: set[str] | None = None

            if knowledge_type:
                ct = self._by_type.get(knowledge_type, set())
                candidates = ct if candidates is None else candidates & ct

            if tags:
                for tag in tags:
                    tg = self._by_tag.get(tag, set())
                    candidates = tg if candidates is None else candidates & tg

            if candidates is None:
                candidates = set(self._knowledge.keys())

            # 过滤
            priority_order = ["critical", "high", "normal", "low"]
            min_prio_idx = priority_order.index(min_priority) if min_priority else 0

            results = []
            for eid in candidates:
                entry = self._knowledge[eid]
                if entry.失效:
                    continue
                if priority_order.index(entry.priority) < min_prio_idx:
                    continue
                if keyword:
                    content_str = json.dumps(entry.content, ensure_ascii=True)
                    if keyword.lower() not in content_str.lower():
                        continue
                results.append(entry)

            # 按优先级和时间排序
            results.sort(key=lambda e: (
                priority_order.index(e.priority),
                -e.source_timestamp
            ), reverse=True)

            return results[:limit]

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """获取知识条目"""
        with self._kb_lock:
            return self._knowledge.get(entry_id)

    def invalidate(self, entry_id: str, reason: str = ""):
        """标记知识为失效"""
        with self._kb_lock:
            entry = self._knowledge.get(entry_id)
            if entry:
                entry.失效 = True
                logger.info(f"📚 [KnowledgeSync] 知识已失效: {entry_id} reason={reason}")

    def increment_ref(self, entry_id: str):
        """增加引用计数"""
        with self._kb_lock:
            entry = self._knowledge.get(entry_id)
            if entry:
                entry.refs += 1

    # ===== 状态 =====

    def get_status(self) -> dict:
        """获取同步状态"""
        with self._kb_lock:
            return {
                "total_knowledge": len(self._knowledge),
                "by_type": {t: len(ids) for t, ids in self._by_type.items()},
                "synced_hashes": len(self._synced_hashes),
                "last_sync": self._last_sync,
                "running": self._running,
            }


# ===== 全局单例 =====

_knowledge_sync: Optional[KnowledgeSync] = None
_sync_lock = threading.Lock()


def get_knowledge_sync() -> Optional[KnowledgeSync]:
    """获取知识同步器单例"""
    return _knowledge_sync


def init_knowledge_sync(federation, **kwargs) -> KnowledgeSync:
    """初始化知识同步器"""
    global _knowledge_sync
    with _sync_lock:
        if _knowledge_sync is not None and _knowledge_sync._running:
            return _knowledge_sync
        _knowledge_sync = KnowledgeSync(federation, **kwargs)
        _knowledge_sync.start()
    return _knowledge_sync
