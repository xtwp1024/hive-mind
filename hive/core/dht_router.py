# -*- coding: utf-8 -*-
"""
DHTRouter - 去中心化哈希表路由器
设计灵感来源于 learning-at-home/hivemind 的 DHT (Kademlia) 实现。

Kademlia 核心思想：
- 每个节点拥有一个唯一 ID（160-bit 数字）
- 节点距离用 XOR 度量（非地理距离）
- 任何键值对都路由存储于距该 key XOR 最近的 K 个节点上
- 无中心节点，任意节点加入/退出不影响网络寻址
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Hive.DHTRouter")

# K-bucket 参数（对标 Kademlia 论文中 k=20）
K_BUCKET_SIZE = 8
KEY_BITS      = 64  # 简化版：64 位 ID 空间（hivemind 原版 160 位）


def _node_id(name: str) -> int:
    """将节点名字哈希为 64 位整数 ID"""
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2 ** KEY_BITS)


def _xor_distance(a: int, b: int) -> int:
    """Kademlia XOR 距离度量"""
    return a ^ b


@dataclass
class DHTEntry:
    """DHT 中存储的一条知识记录"""
    key:     str
    value:   Any
    owner:   str        # 发布者节点名
    ttl:     float      # 过期时间戳（unix）
    hops:    int = 0    # 经过转发的跳数


class DHTRouter:
    """
    单节点的 Kademlia DHT 路由器（in-process 版本）。
    每只工蜂或每个系统组件持有一个实例，共同构成去中心化知识网络。
    """

    # 全局 peer 注册表（模拟真实网络中的 bootstrap 连接）
    _global_peers: dict[str, "DHTRouter"] = {}
    _peers_lock = threading.Lock()

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.node_id   = _node_id(node_name)
        self._store: dict[str, DHTEntry] = {}     # 本地存储
        self._routing_table: list[str] = []        # 已知邻居节点名列表
        self._lock = threading.Lock()

        # 自注册到全局 peer 表
        with DHTRouter._peers_lock:
            DHTRouter._global_peers[node_name] = self
        logger.info(f"[DHT:{self.node_name}] 节点上线 ID={self.node_id:016x}")

    # ─────────────────────────────────────────── 网络发现 ──

    def bootstrap(self, known_peer: str) -> None:
        """加入网络（对标 DHT bootstrap），通过已知节点发现邻居"""
        with DHTRouter._peers_lock:
            all_names = list(DHTRouter._global_peers.keys())

        # 将所有已知节点按 XOR 距离排序，取最近的 K 个加入路由表
        distances = [
            (n, _xor_distance(self.node_id, _node_id(n)))
            for n in all_names if n != self.node_name
        ]
        distances.sort(key=lambda x: x[1])
        with self._lock:
            self._routing_table = [n for n, _ in distances[:K_BUCKET_SIZE]]
        logger.debug(f"[DHT:{self.node_name}] 路由表更新：{self._routing_table}")

    # ─────────────────────────────────────────── 存储/获取 ──

    def store(self, key: str, value: Any, ttl_seconds: float = 300.0) -> None:
        """
        存储知识碎片（对标 DHT store）。
        本地保存，同时广播给 XOR 最近的 K 个邻居节点。
        """
        entry = DHTEntry(
            key=key,
            value=value,
            owner=self.node_name,
            ttl=time.time() + ttl_seconds,
        )
        with self._lock:
            self._store[key] = entry
        logger.info(f"[DHT:{self.node_name}] store({key!r}) → 本地已写入")

        # 广播给最近邻 (最多 K 个)
        key_id = _node_id(key)
        with self._lock:
            neighbors = list(self._routing_table)
        neighbors_sorted = sorted(
            neighbors,
            key=lambda n: _xor_distance(key_id, _node_id(n))
        )[:K_BUCKET_SIZE]

        with DHTRouter._peers_lock:
            peers = dict(DHTRouter._global_peers)

        for peer_name in neighbors_sorted:
            if peer_name in peers and peer_name != self.node_name:
                peer = peers[peer_name]
                forwarded = DHTEntry(
                    key=entry.key,
                    value=entry.value,
                    owner=entry.owner,
                    ttl=entry.ttl,
                    hops=entry.hops + 1,
                )
                with peer._lock:
                    peer._store[key] = forwarded
                logger.debug(f"[DHT:{self.node_name}] 广播 {key!r} → {peer_name} (hop+1)")

    def get(self, key: str) -> Any | None:
        """
        查找知识碎片（对标 DHT get + traverse）。
        先查本地，未命中则遍历路由表询问最近邻节点。
        """
        # 1. 本地命中
        with self._lock:
            entry = self._store.get(key)
        if entry and entry.ttl > time.time():
            logger.info(f"[DHT:{self.node_name}] get({key!r}) → 本地命中")
            return entry.value

        # 2. 向最近邻询问（容错：节点不可达则跳过）
        key_id = _node_id(key)
        with self._lock:
            neighbors = list(self._routing_table)
        neighbors_sorted = sorted(
            neighbors,
            key=lambda n: _xor_distance(key_id, _node_id(n))
        )

        with DHTRouter._peers_lock:
            peers = dict(DHTRouter._global_peers)

        for peer_name in neighbors_sorted:
            if peer_name not in peers:
                continue
            try:
                peer = peers[peer_name]
                with peer._lock:
                    remote_entry = peer._store.get(key)
                if remote_entry and remote_entry.ttl > time.time():
                    logger.info(f"[DHT:{self.node_name}] get({key!r}) → 从 {peer_name} 命中")
                    return remote_entry.value
            except Exception:
                # 容错：节点不响应则跳过（对标 hivemind 容错反向传播）
                pass

        logger.warning(f"[DHT:{self.node_name}] get({key!r}) → 未找到")
        return None

    def leave(self) -> None:
        """节点离线，从全局注册表中注销"""
        with DHTRouter._peers_lock:
            DHTRouter._global_peers.pop(self.node_name, None)
        logger.info(f"[DHT:{self.node_name}] 节点下线")

    def status(self) -> dict:
        with self._lock:
            return {
                "node":           self.node_name,
                "node_id":        f"{self.node_id:016x}",
                "local_entries":  len(self._store),
                "known_peers":    len(self._routing_table),
            }
