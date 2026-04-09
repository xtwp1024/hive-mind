# -*- coding: utf-8 -*-
"""
Hive Federation (蜂巢联邦)

Phase 5 P1: 多实例协作
多个 Hive 实例互联互通，形成分布式蜂群网络。

核心职责:
1. 节点注册与发现 — 多 Hive 实例自动发现彼此
2. 知识广播 — 优秀基因/策略在节点间同步
3. 任务委托 — 将任务委托给有能力空闲的节点执行
4. 状态同步 — 节点状态 (active/idle/overloaded) 全网可见
5. 联邦负载均衡 — 自动将任务路由到负载最低的节点

架构:
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  Hive-A    │◄───►│  Hive-B    │◄───►│  Hive-C    │
  │ (节点A)    │     │ (节点B)    │     │ (节点C)    │
  └─────────────┘     └─────────────┘     └─────────────┘
         ▲                   ▲                   ▲
         │    ┌──────────────────────────────┘
         │    │  知识广播 / 任务委托 / 状态同步
         ▼    ▼
  ┌─────────────────────────────────────────┐
  │         Hive Federation (联邦中枢)       │
  │  - 节点注册表 (谁在线?)                   │
  │  - 知识库同步 (学到的知识共享)             │
  │  - 任务路由 (委托到最适合的节点)           │
  │  - 健康检查 (心跳检测节点存活性)            │
  └─────────────────────────────────────────┘

使用方式:
    federation = HiveFederation(node_id="hive-1", port=8765)
    federation.start()
    federation.broadcast_knowledge({"type": "strategy", "content": "..."})
    result = federation.delegate_task(task, target_capabilities=["vision"])
    federation.stop()
"""

import json
import logging
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import urllib.request
import urllib.error

logger = logging.getLogger("Hive.Federation")


# ===== 节点描述 =====

@dataclass
class HiveNode:
    """蜂巢节点描述"""
    node_id: str
    host: str
    port: int
    status: str = "active"  # active, idle, overloaded, offline
    capabilities: list[str] = field(default_factory=list)
    load_factor: float = 0.0  # 0.0=空闲, 1.0=满载
    last_heartbeat: float = field(default_factory=time.time)
    version: str = "1.0"
    metadata: dict = field(default_factory=dict)

    def is_alive(self, timeout: float = 30.0) -> bool:
        """检查节点是否存活"""
        return (time.time() - self.last_heartbeat) < timeout

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "capabilities": self.capabilities,
            "load_factor": self.load_factor,
            "last_heartbeat": self.last_heartbeat,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HiveNode":
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            status=data.get("status", "active"),
            capabilities=data.get("capabilities", []),
            load_factor=data.get("load_factor", 0.0),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    def get_url(self, path: str = "") -> str:
        return f"http://{self.host}:{self.port}/{path.lstrip('/')}"


# ===== 蜂巢联邦 =====

class HiveFederation:
    """
    蜂巢联邦核心

    Phase 5 P1: 多 Hive 实例联邦网络
    """

    # 联邦协议消息类型
    MSG_NODE_REGISTER = "node_register"
    MSG_NODE_HEARTBEAT = "node_heartbeat"
    MSG_NODE_DEREGISTER = "node_deregister"
    MSG_KNOWLEDGE_BROADCAST = "knowledge_broadcast"
    MSG_TASK_DELEGATE = "task_delegate"
    MSG_TASK_RESULT = "task_result"
    MSG_SYNC_REQUEST = "sync_request"
    MSG_SYNC_RESPONSE = "sync_response"

    def __init__(
        self,
        node_id: Optional[str] = None,
        host: str = "localhost",
        port: int = 8765,
        seed_nodes: Optional[list[str]] = None,
        discovery_mode: str = "broadcast",  # "broadcast" | "seed" | "manual"
        broadcast_group: str = "239.255.255.250",
        broadcast_port: int = 9999,
    ):
        """
        初始化蜂巢联邦

        Args:
            node_id: 本节点ID (默认自动生成)
            host: 本节点 host
            port: 本节点 HTTP API 端口
            seed_nodes: 种子节点列表 ["host:port", ...]
            discovery_mode: 发现模式
                - "broadcast": UDP 多播发现同网段节点
                - "seed": 通过种子节点发现其他节点
                - "manual": 仅手动添加节点
            broadcast_group: UDP 多播地址
            broadcast_port: UDP 多播端口
        """
        self.node_id = node_id or f"hive-{uuid.uuid4().hex[:8]}"
        self.host = host
        self.port = port
        self.seed_nodes = seed_nodes or []
        self.discovery_mode = discovery_mode
        self.broadcast_group = broadcast_group
        self.broadcast_port = broadcast_port

        # 节点注册表
        self._nodes: dict[str, HiveNode] = {}
        self._nodes_lock = threading.RLock()

        # 知识库 (联邦共享)
        self._knowledge_base: list[dict] = []
        self._kb_lock = threading.RLock()

        # 任务追踪
        self._pending_tasks: dict[str, dict] = {}
        self._tasks_lock = threading.RLock()

        # 运行状态
        self._running = False
        self._threads: list[threading.Thread] = []

        # HTTP 服务器 (简化版，用线程)
        self._http_thread: Optional[threading.Thread] = None
        self._stop_http = threading.Event()

        # 心跳间隔
        self.heartbeat_interval = 10.0  # 秒
        self.node_timeout = 60.0  # 秒

        logger.info(f"🔗 [Federation] 蜂巢联邦初始化: node_id={self.node_id}, mode={discovery_mode}")

    # ===== 生命周期 =====

    def start(self):
        """启动联邦"""
        if self._running:
            return
        self._running = True

        # 启动 HTTP API 服务器
        self._start_http_server()

        # 启动心跳
        t = threading.Thread(target=self._heartbeat_loop, daemon=True, name="federation-heartbeat")
        t.start()
        self._threads.append(t)

        # 启动节点发现
        t = threading.Thread(target=self._discovery_loop, daemon=True, name="federation-discovery")
        t.start()
        self._threads.append(t)

        # 注册自己到联邦
        self._register_self()

        logger.info(f"🔗 [Federation] 蜂巢联邦已启动")

    def stop(self):
        """停止联邦"""
        if not self._running:
            return
        self._running = False

        # 注销自己
        self._deregister_self()

        # 停止 HTTP 服务器
        self._stop_http.set()
        if self._http_thread:
            self._http_thread.join(timeout=3.0)

        # 等待其他线程
        for t in self._threads:
            t.join(timeout=3.0)

        logger.info(f"🔗 [Federation] 蜂巢联邦已停止")

    # ===== 节点管理 =====

    def register_node(self, node: HiveNode) -> bool:
        """注册节点到联邦"""
        with self._nodes_lock:
            existing = self._nodes.get(node.node_id)
            if existing and existing.host == node.host and existing.port == node.port:
                # 更新心跳
                existing.last_heartbeat = time.time()
                logger.debug(f"🔗 [Federation] 节点心跳更新: {node.node_id}")
                return True

            self._nodes[node.node_id] = node
            logger.info(f"🔗 [Federation] 新节点注册: {node.node_id} ({node.host}:{node.port}), cap={node.capabilities}")
            return True

    def deregister_node(self, node_id: str):
        """从联邦注销节点"""
        with self._nodes_lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.info(f"🔗 [Federation] 节点注销: {node_id}")

    def get_node(self, node_id: str) -> Optional[HiveNode]:
        """获取节点信息"""
        with self._nodes_lock:
            return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[HiveNode]:
        """获取所有活动节点"""
        with self._nodes_lock:
            return [n for n in self._nodes.values() if n.is_alive(self.node_timeout)]

    def get_nodes_by_capability(self, capability: str) -> list[HiveNode]:
        """获取具有特定能力的节点"""
        with self._nodes_lock:
            return [
                n for n in self._nodes.values()
                if n.is_alive(self.node_timeout) and capability in n.capabilities
            ]

    def get_best_node(self, capabilities: list[str]) -> Optional[HiveNode]:
        """获取最适合的节点 (负载最低且具备能力)"""
        candidates = self.get_all_nodes()
        if not candidates:
            return None

        # 过滤具备所需能力的节点
        for cap in capabilities:
            candidates = [n for n in candidates if cap in n.capabilities]
            if not candidates:
                break

        if not candidates:
            return None

        # 选择负载最低的
        candidates.sort(key=lambda n: n.load_factor)
        return candidates[0]

    # ===== 知识广播 =====

    def broadcast_knowledge(self, knowledge: dict, exclude_self: bool = True) -> int:
        """
        广播知识到所有节点

        Args:
            knowledge: 要广播的知识 dict
            exclude_self: 是否排除自己

        Returns:
            广播到的节点数量
        """
        with self._kb_lock:
            self._knowledge_base.append({
                "knowledge": knowledge,
                "timestamp": time.time(),
                "origin": self.node_id,
            })
            # 保留最近 1000 条知识
            if len(self._knowledge_base) > 1000:
                self._knowledge_base = self._knowledge_base[-1000:]

        count = 0
        for node in self.get_all_nodes():
            if exclude_self and node.node_id == self.node_id:
                continue
            if self._send_to_node(node, self.MSG_KNOWLEDGE_BROADCAST, {
                "knowledge": knowledge,
                "origin": self.node_id,
            }):
                count += 1

        logger.info(f"📡 [Federation] 知识广播完成: {count} 个节点")
        return count

    def get_recent_knowledge(self, limit: int = 50) -> list[dict]:
        """获取最近的知识"""
        with self._kb_lock:
            return self._knowledge_base[-limit:]

    # ===== 任务委托 =====

    def delegate_task(
        self,
        task: dict,
        target_capabilities: Optional[list[str]] = None,
        target_node_id: Optional[str] = None,
        timeout: float = 60.0
    ) -> Optional[dict]:
        """
        将任务委托给其他节点

        Args:
            task: 任务描述 dict
            target_capabilities: 所需能力列表 (如 ["vision", "audio"])
            target_node_id: 指定节点ID (优先于此参数)
            timeout: 超时时间 (秒)

        Returns:
            任务结果 dict，失败返回 None
        """
        task_id = task.get("task_id") or f"task-{uuid.uuid4().hex[:8]}"

        # 选择目标节点
        if target_node_id:
            node = self.get_node(target_node_id)
        else:
            node = self.get_best_node(target_capabilities or [])

        if not node:
            logger.warning(f"🔗 [Federation] 找不到合适的节点: cap={target_capabilities}")
            return None

        with self._tasks_lock:
            self._pending_tasks[task_id] = {"node": node.node_id, "task": task}

        # 发送任务
        success = self._send_to_node(node, self.MSG_TASK_DELEGATE, {
            "task_id": task_id,
            "task": task,
            "origin": self.node_id,
        })

        if not success:
            with self._tasks_lock:
                self._pending_tasks.pop(task_id, None)
            return None

        # 等待结果 (简化: 轮询)
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._tasks_lock:
                result = self._pending_tasks[task_id].get("result")
                if result is not None:
                    self._pending_tasks.pop(task_id, None)
                    return result
            time.sleep(0.5)

        logger.warning(f"🔗 [Federation] 任务超时: {task_id}")
        with self._tasks_lock:
            self._pending_tasks.pop(task_id, None)
        return None

    def on_task_result(self, task_id: str, result: dict):
        """接收任务结果 (由 HTTP 处理器调用)"""
        with self._tasks_lock:
            if task_id in self._pending_tasks:
                self._pending_tasks[task_id]["result"] = result

    # ===== 状态更新 =====

    def update_status(self, status: str, load_factor: float = 0.0, capabilities: Optional[list[str]] = None):
        """更新本节点状态"""
        self.status = status
        self.load_factor = load_factor
        if capabilities:
            self.capabilities = capabilities

        # 广播更新
        for node in self.get_all_nodes():
            if node.node_id != self.node_id:
                self._send_to_node(node, self.MSG_NODE_HEARTBEAT, {
                    "node_id": self.node_id,
                    "status": status,
                    "load_factor": load_factor,
                    "capabilities": self.capabilities or [],
                })

    # ===== HTTP API (简化版) =====

    def _start_http_server(self):
        """启动简单 HTTP 服务器"""
        import http.server
        import socketserver

        class Handler(http.server.BaseHTTPRequestHandler):
            federation = self

            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")

                try:
                    data = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    self.send_error(400, "Invalid JSON")
                    return

                msg_type = data.get("type", "")
                payload = data.get("payload", {})

                result = self.federation._handle_http_request(msg_type, payload)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result or {"status": "ok"}).encode())

            def log_message(self, format, *args):
                pass  # 抑制日志

        class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            allow_reuse_address = True

        self._stop_http.clear()
        self._http_server = ThreadedHTTPServer(("0.0.0.0", self.port), Handler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True,
            name="federation-http"
        )
        self._http_thread.start()
        logger.info(f"🔗 [Federation] HTTP API 启动: {self.host}:{self.port}")

    def _handle_http_request(self, msg_type: str, payload: dict) -> Optional[dict]:
        """处理 HTTP API 请求"""
        if msg_type == self.MSG_NODE_REGISTER:
            node = HiveNode.from_dict(payload)
            self.register_node(node)
            return {"status": "ok", "node_id": self.node_id}
        elif msg_type == self.MSG_NODE_HEARTBEAT:
            node_id = payload.get("node_id")
            with self._nodes_lock:
                if node_id in self._nodes:
                    self._nodes[node_id].last_heartbeat = time.time()
                    self._nodes[node_id].status = payload.get("status", "active")
                    self._nodes[node_id].load_factor = payload.get("load_factor", 0.0)
            return None
        elif msg_type == self.MSG_KNOWLEDGE_BROADCAST:
            with self._kb_lock:
                self._knowledge_base.append({
                    "knowledge": payload.get("knowledge"),
                    "timestamp": time.time(),
                    "origin": payload.get("origin"),
                })
            return None
        elif msg_type == self.MSG_TASK_DELEGATE:
            # 收到委托任务，执行并返回结果 (这里只是模拟)
            task = payload.get("task", {})
            task_id = payload.get("task_id", "")
            logger.info(f"🔗 [Federation] 收到委托任务: {task_id}")
            return {"status": "received", "task_id": task_id}
        elif msg_type == self.MSG_TASK_RESULT:
            task_id = payload.get("task_id", "")
            self.on_task_result(task_id, payload.get("result"))
            return None
        elif msg_type == self.MSG_SYNC_REQUEST:
            return {
                "nodes": [n.to_dict() for n in self.get_all_nodes()],
                "knowledge": self.get_recent_knowledge(100),
            }
        return None

    def _send_to_node(self, node: HiveNode, msg_type: str, payload: dict) -> bool:
        """发送消息到节点"""
        try:
            data = json.dumps({"type": msg_type, "payload": payload}).encode("utf-8")
            req = urllib.request.Request(
                node.get_url("api"),
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"🔗 [Federation] 发送消息到 {node.node_id} 失败: {e}")
            return False

    # ===== 心跳 & 发现 =====

    def _register_self(self):
        """注册自己到种子节点 (如果有)"""
        if not self.seed_nodes:
            # 自己是第一个节点
            logger.info(f"🔗 [Federation] 作为种子节点启动: {self.node_id}")
            return

        self_node = HiveNode(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            status="active",
            capabilities=[],  # 由调用者通过 update_status 更新
        )

        for seed in self.seed_nodes:
            try:
                host, port = seed.split(":")
                seed_node = HiveNode(node_id="seed", host=host, port=int(port))
                if self._send_to_node(seed_node, self.MSG_NODE_REGISTER, self_node.to_dict()):
                    logger.info(f"🔗 [Federation] 已注册到种子节点: {seed}")
                    break
            except Exception as e:
                logger.warning(f"🔗 [Federation] 注册到种子节点失败: {seed}: {e}")

    def _deregister_self(self):
        """注销自己"""
        for node in self.get_all_nodes():
            if node.node_id != self.node_id:
                self._send_to_node(node, self.MSG_NODE_DEREGISTER, {"node_id": self.node_id})

    def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            time.sleep(self.heartbeat_interval)

            # 清理超时的节点
            with self._nodes_lock:
                dead = [nid for nid, n in self._nodes.items() if not n.is_alive(self.node_timeout)]
                for nid in dead:
                    del self._nodes[nid]
                    logger.info(f"🔗 [Federation] 节点超时移除: {nid}")

            # 发送自己的心跳
            for node in self.get_all_nodes():
                if node.node_id != self.node_id:
                    self._send_to_node(node, self.MSG_NODE_HEARTBEAT, {
                        "node_id": self.node_id,
                        "status": getattr(self, "status", "active"),
                        "load_factor": getattr(self, "load_factor", 0.0),
                    })

    def _discovery_loop(self):
        """节点发现循环"""
        if self.discovery_mode == "broadcast":
            self._discovery_broadcast()
        elif self.discovery_mode == "seed":
            self._discovery_seed()

    def _discovery_broadcast(self):
        """UDP 多播发现"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        try:
            sock.bind(("", self.broadcast_port))
            mreq = socket.inet_aton(self.broadcast_group) + socket.inet_aton("0.0.0.0")
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.settimeout(1.0)

            while self._running:
                try:
                    data, addr = sock.recvfrom(4096)
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("type") == "hive_discovery":
                        node_info = msg.get("node")
                        if node_info:
                            node = HiveNode.from_dict(node_info)
                            if node.node_id != self.node_id:
                                self.register_node(node)
                            # 回复自己的信息
                            self_node = HiveNode(
                                node_id=self.node_id,
                                host=self.host,
                                port=self.port,
                            )
                            sock.sendto(
                                json.dumps({"type": "hive_response", "node": self_node.to_dict()}).encode(),
                                (addr[0], self.broadcast_port)
                            )
                except socket.timeout:
                    # 定期广播自己的存在
                    self_node = HiveNode(node_id=self.node_id, host=self.host, port=self.port)
                    sock.sendto(
                        json.dumps({"type": "hive_discovery", "node": self_node.to_dict()}).encode(),
                        (self.broadcast_group, self.broadcast_port)
                    )
                except Exception as e:
                    logger.debug(f"🔗 [Federation] 多播发现异常: {e}")
        finally:
            sock.close()

    def _discovery_seed(self):
        """通过种子节点发现"""
        while self._running:
            time.sleep(self.heartbeat_interval * 3)
            for seed in self.seed_nodes:
                try:
                    host, port = seed.split(":")
                    seed_node = HiveNode(node_id="seed", host=host, port=int(port))
                    response = self._send_and_receive(seed_node, self.MSG_SYNC_REQUEST, {})
                    if response:
                        nodes = response.get("nodes", [])
                        for node_dict in nodes:
                            node = HiveNode.from_dict(node_dict)
                            if node.node_id != self.node_id:
                                self.register_node(node)
                except Exception as e:
                    logger.debug(f"🔗 [Federation] 种子节点发现异常: {e}")

    def _send_and_receive(self, node: HiveNode, msg_type: str, payload: dict) -> Optional[dict]:
        """发送并等待响应"""
        try:
            data = json.dumps({"type": msg_type, "payload": payload}).encode("utf-8")
            req = urllib.request.Request(
                node.get_url("api"),
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    # ===== 状态 =====

    def get_status(self) -> dict:
        """获取联邦状态"""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "mode": self.discovery_mode,
            "nodes_online": len(self.get_all_nodes()),
            "knowledge_count": len(self._knowledge_base),
            "pending_tasks": len(self._pending_tasks),
        }


# ===== 全局单例 =====

_federation: Optional[HiveFederation] = None
_federation_lock = threading.Lock()


def get_federation() -> HiveFederation:
    """获取蜂巢联邦单例"""
    global _federation
    if _federation is None:
        with _federation_lock:
            if _federation is None:
                _federation = HiveFederation()
    return _federation


def init_federation(**kwargs) -> HiveFederation:
    """初始化蜂巢联邦"""
    global _federation
    with _federation_lock:
        if _federation is not None and _federation._running:
            return _federation
        _federation = HiveFederation(**kwargs)
        _federation.start()
    return _federation
