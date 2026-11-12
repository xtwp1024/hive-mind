# -*- coding: utf-8 -*-
"""
SwarmCommunication - 蜂群通信系统

实现蜂种之间的协作和集体智慧

增强功能 (Phase 1 增强):
1. 优先级队列 - 高优先级消息优先处理
2. 消息过期机制 - 防止过时消息堆积
3. 请求-响应模式 - send_and_wait 等待响应

功能:
- Bee-to-Bee通信
- 集体决策
- 任务协商
- 知识广播
"""

import heapq
import queue
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from hive.utils.logger import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """消息类型"""

    # ===== 基础类型 (原有) =====
    DISCOVERY = "discovery"  # 发现新知识
    TASK_REQUEST = "task_request"  # 任务请求
    TASK_BID = "task_bid"  # 任务投标
    VOTE = "vote"  # 投票
    KNOWLEDGE = "knowledge"  # 知识分享
    STATUS = "status"  # 状态更新
    ALERT = "alert"  # 警报
    RESPONSE = "response"  # 响应消息
    HEARTBEAT = "heartbeat"  # 心跳消息

    # ===== 请求类 (Phase 2 新增) =====
    REQUEST_ASSISTANCE = "request_assistance"  # 请求协助
    REQUEST_INFORMATION = "request_information"  # 请求信息
    REQUEST_APPROVAL = "request_approval"  # 请求审批

    # ===== 响应类 (Phase 2 新增) =====
    RESPONSE_SUCCESS = "response_success"  # 成功响应
    RESPONSE_FAILURE = "response_failure"  # 失败响应
    RESPONSE_PARTIAL = "response_partial"  # 部分响应

    # ===== 广播类 (Phase 2 新增) =====
    BROADCAST_DISCOVERY = "broadcast_discovery"  # 发现广播
    BROADCAST_STATUS = "broadcast_status"  # 状态广播
    BROADCAST_EMERGENCY = "broadcast_emergency"  # 紧急广播

    # ===== 协作类 (Phase 2 新增) =====
    COLLABORATION_PROPOSAL = "collab_proposal"  # 协作提议
    COLLABORATION_ACCEPT = "collab_accept"  # 接受协作
    COLLABORATION_REJECT = "collab_reject"  # 拒绝协作

    # ===== 知识类 (Phase 2 新增) =====
    KNOWLEDGE_SHARE = "knowledge_share"  # 知识分享
    KNOWLEDGE_QUERY = "knowledge_query"  # 知识查询


class MessagePriority(Enum):
    """消息优先级 (Phase 2 新增)"""

    LOW = 0  # 低优先级 - 日志、统计
    NORMAL = 1  # 普通优先级 - 常规消息
    HIGH = 2  # 高优先级 - 任务请求、知识查询
    CRITICAL = 3  # 关键优先级 - 警报、审批请求
    EMERGENCY = 4  # 紧急优先级 - 紧急广播

    @classmethod
    def to_int(cls, priority: "MessagePriority") -> int:
        """转换为整数优先级 (1-10 范围)"""
        mapping = {
            cls.LOW: 2,
            cls.NORMAL: 5,
            cls.HIGH: 7,
            cls.CRITICAL: 9,
            cls.EMERGENCY: 10,
        }
        return mapping.get(priority, 5)

    @classmethod
    def from_int(cls, priority_int: int) -> "MessagePriority":
        """从整数转换为优先级枚举"""
        if priority_int <= 2:
            return cls.LOW
        elif priority_int <= 5:
            return cls.NORMAL
        elif priority_int <= 7:
            return cls.HIGH
        elif priority_int <= 9:
            return cls.CRITICAL
        else:
            return cls.EMERGENCY


@dataclass(order=True)
class PriorityQueueItem:
    """优先级队列项"""

    priority: int  # 注意：值越小优先级越高（heapq 是最小堆）
    timestamp: float
    message: "SwarmMessage" = field(compare=False)

    def __post_init__(self):
        self.priority = 10 - self.message.priority


@dataclass
class SwarmMessage:
    """蜂群消息"""

    id: str
    type: MessageType
    from_bee: str
    to_bee: Optional[str] = None  # None表示广播
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5  # 1-10, 10最高

    # 增强字段
    expires_at: Optional[float] = None  # 过期时间戳（秒）
    correlation_id: Optional[str] = None  # 关联ID（用于请求-响应）
    hop_count: int = 0  # 经过的节点数
    trace_path: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def is_broadcast(self) -> bool:
        """是否为广播消息"""
        return self.to_bee is None

    def needs_response(self) -> bool:
        """是否需要响应"""
        return (
            self.type
            in [
                MessageType.TASK_REQUEST,
                MessageType.VOTE,
                MessageType.REQUEST_ASSISTANCE,
                MessageType.REQUEST_INFORMATION,
                MessageType.REQUEST_APPROVAL,
                MessageType.KNOWLEDGE_QUERY,
                MessageType.COLLABORATION_PROPOSAL,
            ]
            or self.correlation_id is not None
        )

    def is_request_type(self) -> bool:
        """是否为请求类型消息 (Phase 2 新增)"""
        return self.type in [
            MessageType.TASK_REQUEST,
            MessageType.REQUEST_ASSISTANCE,
            MessageType.REQUEST_INFORMATION,
            MessageType.REQUEST_APPROVAL,
            MessageType.KNOWLEDGE_QUERY,
            MessageType.COLLABORATION_PROPOSAL,
        ]

    def is_response_type(self) -> bool:
        """是否为响应类型消息 (Phase 2 新增)"""
        return self.type in [
            MessageType.RESPONSE,
            MessageType.RESPONSE_SUCCESS,
            MessageType.RESPONSE_FAILURE,
            MessageType.RESPONSE_PARTIAL,
        ]

    def is_broadcast_type(self) -> bool:
        """是否为广播类型消息 (Phase 2 新增)"""
        return self.type in [
            MessageType.BROADCAST_DISCOVERY,
            MessageType.BROADCAST_STATUS,
            MessageType.BROADCAST_EMERGENCY,
            MessageType.KNOWLEDGE_SHARE,
            MessageType.DISCOVERY,
            MessageType.KNOWLEDGE,
            MessageType.STATUS,
            MessageType.ALERT,
            MessageType.HEARTBEAT,
        ]

    def get_priority_enum(self) -> MessagePriority:
        """获取消息优先级枚举 (Phase 2 新增)"""
        return MessagePriority.from_int(self.priority)

    def set_priority_enum(self, priority: MessagePriority):
        """设置消息优先级 (Phase 2 新增)"""
        self.priority = MessagePriority.to_int(priority)


@dataclass
class MessageEnvelope:
    """消息信封 - 用于队列传输和追踪"""

    message: SwarmMessage
    received_at: float = field(default_factory=time.time)
    read_count: int = 0
    is_acknowledged: bool = False


@dataclass
class TaskBid:
    """任务投标"""

    bee_id: str
    bee_type: str
    capability_score: float
    estimated_time: int
    confidence: float
    reasoning: str


@dataclass
class Vote:
    """投票"""

    voter_id: str
    option: str
    weight: float  # 投票权重
    reasoning: str


class SwarmCommunication:
    """蜂群通信系统

    实现蜂种之间的通信和协作

    增强功能:
    1. 优先级队列 - 高优先级消息优先处理
    2. 消息过期机制 - 防止过时消息堆积
    3. 请求-响应模式 - send_and_wait 等待响应

    功能:
    1. 消息广播和点对点通信
    2. 集体决策（投票）
    3. 任务协商和分配
    4. 知识共享
    """

    def __init__(self):
        # 优先级消息队列（使用 heapq 实现最小堆）
        self._priority_queue_lock = threading.Lock()
        self._priority_queue: List[PriorityQueueItem] = []
        self._queue_not_empty = threading.Condition(self._priority_queue_lock)

        # 已注册的蜂
        self.registered_bees: Dict[str, Any] = {}

        # 消息订阅
        self.subscriptions: Dict[str, List[Callable]] = defaultdict(list)

        # 消息历史（限制大小防止内存泄漏）
        self.message_history: List[SwarmMessage] = []
        self._max_history_size = 10000

        # 活跃任务
        self.active_tasks: Dict[str, Dict] = {}

        # 投票记录
        self.votes: Dict[str, List[Vote]] = {}

        # 通信线程
        self.running = False
        self.comm_thread: Optional[threading.Thread] = None

        # 增强功能: 待处理响应
        self.pending_responses: Dict[str, Tuple[threading.Event, Optional[SwarmMessage]]] = {}
        self._pending_lock = threading.Lock()

        # 增强功能: 按蜜蜂ID的专用队列
        self._bee_queues: Dict[str, List[SwarmMessage]] = defaultdict(list)
        self._bee_queues_lock = threading.Lock()

        # 增强功能: 消息过期检查间隔（秒）
        self._expiry_check_interval = 5.0

        # 统计信息
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_expired": 0,
            "responses_received": 0,
            "timeouts_occurred": 0,
        }
        self._stats_lock = threading.Lock()

        logger.info("🐝 [SwarmComm] 蜂群通信系统初始化完成 (支持优先级队列、消息过期、请求-响应)")

    def start(self):
        """启动通信系统"""
        if self.running:
            logger.warning("⚠️ [SwarmComm] 通信系统已在运行")
            return

        self.running = True
        self.comm_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.comm_thread.start()

        logger.info("✅ [SwarmComm] 通信系统已启动")

    def stop(self):
        """停止通信系统"""
        self.running = False
        if self.comm_thread:
            self.comm_thread.join(timeout=5)

        logger.info("🛑 [SwarmComm] 通信系统已停止")

    def register_bee(self, bee_id: str, bee_instance: Any):
        """注册蜂到通信系统

        Args:
            bee_id: 蜂ID
            bee_instance: 蜂实例
        """
        self.registered_bees[bee_id] = bee_instance
        logger.info(f"📝 [SwarmComm] 蜂 {bee_id} 已注册")

    def unregister_bee(self, bee_id: str):
        """注销蜂"""
        if bee_id in self.registered_bees:
            del self.registered_bees[bee_id]
            logger.info(f"📝 [SwarmComm] 蜂 {bee_id} 已注销")

    def _put_priority_message(self, message: SwarmMessage):
        """将消息放入优先级队列"""
        with self._queue_not_empty:
            item = PriorityQueueItem(
                priority=message.priority, timestamp=time.time(), message=message
            )
            heapq.heappush(self._priority_queue, item)
            self._queue_not_empty.notify()

        with self._stats_lock:
            self._stats["messages_sent"] += 1

    def broadcast(self, message: SwarmMessage):
        """广播消息

        Args:
            message: 要广播的消息
        """
        message.to_bee = None  # 确保是广播
        self._put_priority_message(message)

        logger.info(
            f"📢 [SwarmComm] 广播消息: {message.type.value} from {message.from_bee}, priority={message.priority}"
        )

    def send_message(self, message: SwarmMessage):
        """发送点对点消息

        Args:
            message: 要发送的消息
        """
        self._put_priority_message(message)

        if message.to_bee:
            logger.info(
                f"📨 [SwarmComm] 发送消息: {message.type.value} from {message.from_bee} to {message.to_bee}, priority={message.priority}"
            )
        else:
            logger.info(
                f"📨 [SwarmComm] 发送消息: {message.type.value} from {message.from_bee}, priority={message.priority}"
            )

    def subscribe(self, message_type: MessageType, callback: Callable):
        """订阅消息类型

        Args:
            message_type: 消息类型
            callback: 回调函数
        """
        self.subscriptions[message_type.value].append(callback)
        logger.info(f"📬 [SwarmComm] 订阅消息类型: {message_type.value}")

    def broadcast_discovery(self, bee_id: str, discovery: Dict[str, Any]):
        """广播发现

        Args:
            bee_id: 发现者ID
            discovery: 发现内容
        """
        message = SwarmMessage(
            id=f"discovery_{datetime.now().timestamp()}",
            type=MessageType.DISCOVERY,
            from_bee=bee_id,
            content=discovery,
            priority=8,
        )

        self.broadcast(message)

        logger.info(f"🔍 [SwarmComm] {bee_id} 广播发现: {discovery.get('title', 'Untitled')}")

    def _broadcast_vote_request(self, vote_id: str, question: str, options: List[str]):
        message = SwarmMessage(
            id=vote_id,
            type=MessageType.VOTE,
            from_bee="system",
            content={"question": question, "options": options, "vote_id": vote_id},
            priority=9,
        )
        self.broadcast(message)

    def collective_decision(self, question: str, options: List[str], timeout: int = 60) -> str:
        logger.info(f"🗳️ [SwarmComm] 发起集体决策: {question}")

        vote_id = f"vote_{datetime.now().timestamp()}"
        self.votes[vote_id] = []
        self._broadcast_vote_request(vote_id, question, options)

        import time

        time.sleep(min(timeout, 5))

        decision = self._count_votes(vote_id, options)
        logger.info(f"✅ [SwarmComm] 决策结果: {decision}")
        return decision

    def task_negotiation(self, task: Dict[str, Any]) -> List[TaskBid]:
        """任务协商

        Args:
            task: 任务描述

        Returns:
            投标列表
        """
        logger.info(f"🤝 [SwarmComm] 发起任务协商: {task.get('type', 'unknown')}")

        task_id = f"task_{datetime.now().timestamp()}"
        self.active_tasks[task_id] = {"task": task, "bids": [], "status": "negotiating"}

        # 广播任务请求
        message = SwarmMessage(
            id=task_id,
            type=MessageType.TASK_REQUEST,
            from_bee="system",
            content={"task_id": task_id, "task": task},
            priority=7,
        )
        self.broadcast(message)

        # 等待投标（简化版）
        import time

        time.sleep(3)

        bids = self.active_tasks[task_id]["bids"]

        logger.info(f"📊 [SwarmComm] 收到 {len(bids)} 个投标")

        return bids

    def select_optimal_team(self, bids: List[TaskBid], task: Dict[str, Any]) -> List[str]:
        """选择最优团队

        Args:
            bids: 投标列表
            task: 任务描述

        Returns:
            选中的蜂ID列表
        """
        if not bids:
            logger.warning("⚠️ [SwarmComm] 无投标，无法组建团队")
            return []

        # 按能力分数排序
        sorted_bids = sorted(bids, key=lambda b: b.capability_score * b.confidence, reverse=True)

        # 选择前N个（根据任务复杂度）
        team_size = min(3, len(sorted_bids))  # 最多3个蜂
        selected = [bid.bee_id for bid in sorted_bids[:team_size]]

        logger.info(f"👥 [SwarmComm] 选中团队: {selected}")

        return selected

    def share_knowledge(self, bee_id: str, knowledge: Dict[str, Any]):
        """分享知识

        Args:
            bee_id: 分享者ID
            knowledge: 知识内容
        """
        message = SwarmMessage(
            id=f"knowledge_{datetime.now().timestamp()}",
            type=MessageType.KNOWLEDGE,
            from_bee=bee_id,
            content=knowledge,
            priority=6,
        )

        self.broadcast(message)

        logger.info(f"📚 [SwarmComm] {bee_id} 分享知识: {knowledge.get('topic', 'Unknown')}")

    # ===== 增强功能: 请求-响应模式 =====

    def send_and_wait(
        self, message: SwarmMessage, timeout: float = 30.0
    ) -> Tuple[bool, Optional[SwarmMessage]]:
        """
        发送消息并等待响应

        Args:
            message: 要发送的消息
            timeout: 超时时间（秒）

        Returns:
            (是否成功, 响应消息或 None)
        """
        if not message.correlation_id:
            message.correlation_id = f"corr_{uuid.uuid4().hex[:12]}"

        message.expires_at = time.time() + timeout

        event = threading.Event()
        with self._pending_lock:
            self.pending_responses[message.correlation_id] = (event, None)

        self.send_message(message)

        logger.info(
            f"📨 [SwarmComm] 发送并等待响应: {message.id}, correlation_id={message.correlation_id}, timeout={timeout}s"
        )

        success = event.wait(timeout=timeout)

        with self._pending_lock:
            if message.correlation_id in self.pending_responses:
                _, response = self.pending_responses.pop(message.correlation_id)
            else:
                response = None

        if success and response:
            with self._stats_lock:
                self._stats["responses_received"] += 1
            logger.info(f"✅ [SwarmComm] 收到响应: {message.correlation_id}")
            return True, response
        else:
            with self._stats_lock:
                self._stats["timeouts_occurred"] += 1
            logger.warning(f"⏰ [SwarmComm] 等待响应超时: {message.correlation_id}")
            return False, None

    def send_response(
        self, original_message: SwarmMessage, response_content: Dict[str, Any], success: bool = True
    ):
        """
        发送响应消息

        Args:
            original_message: 原始请求消息
            response_content: 响应内容
            success: 是否成功
        """
        if not original_message.correlation_id:
            logger.warning(f"⚠️ [SwarmComm] 原始消息没有 correlation_id，无法发送响应")
            return

        response = SwarmMessage(
            id=f"resp_{uuid.uuid4().hex[:12]}",
            type=MessageType.RESPONSE,
            from_bee=original_message.to_bee or "unknown",
            to_bee=original_message.from_bee,
            content={**response_content, "_success": success, "_original_id": original_message.id},
            priority=original_message.priority,
            correlation_id=original_message.correlation_id,
        )

        self.send_message(response)

        logger.info(f"📨 [SwarmComm] 发送响应: {response.id} → {original_message.from_bee}")

    def _handle_response(self, message: SwarmMessage):
        """处理响应消息"""
        if not message.correlation_id:
            logger.warning(f"⚠️ [SwarmComm] 响应消息没有 correlation_id: {message.id}")
            return

        with self._pending_lock:
            if message.correlation_id in self.pending_responses:
                event, _ = self.pending_responses[message.correlation_id]
                self.pending_responses[message.correlation_id] = (event, message)
                event.set()
                logger.info(f"✅ [SwarmComm] 响应已匹配: {message.correlation_id}")
            else:
                logger.warning(f"⚠️ [SwarmComm] 未找到等待的请求: {message.correlation_id}")

    # ===== 增强功能: 统计和状态 =====

    def get_stats(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        with self._stats_lock:
            return dict(self._stats)

    def get_swarm_status(self) -> Dict[str, Any]:
        """获取蜂群状态

        Returns:
            蜂群状态
        """
        with self._stats_lock:
            stats = dict(self._stats)

        return {
            "registered_bees": len(self.registered_bees),
            "active_bees": [bee_id for bee_id in self.registered_bees.keys()],
            "message_count": len(self.message_history),
            "active_tasks": len(self.active_tasks),
            "active_votes": len(self.votes),
            "pending_responses": len(self.pending_responses),
            "queue_size": len(self._priority_queue),
            "stats": stats,
            "running": self.running,
        }

    # ===== 私有方法 =====

    def _get_next_message(self, timeout: float = 1.0) -> Optional[SwarmMessage]:
        """从优先级队列获取下一条消息

        Args:
            timeout: 超时时间（秒）

        Returns:
            消息或 None
        """
        with self._queue_not_empty:
            if not self._priority_queue:
                self._queue_not_empty.wait(timeout=timeout)
                if not self._priority_queue:
                    return None

            item = heapq.heappop(self._priority_queue)
            message = item.message

            # 检查是否过期
            if message.is_expired():
                with self._stats_lock:
                    self._stats["messages_expired"] += 1
                logger.debug(f"⏰ [SwarmComm] 消息已过期: {message.id}")
                return None

            return message

    def _check_expired_messages(self):
        """检查并清理过期消息"""
        with self._queue_not_empty:
            expired_count = 0
            valid_items = []

            while self._priority_queue:
                item = heapq.heappop(self._priority_queue)
                if item.message.is_expired():
                    expired_count += 1
                else:
                    valid_items.append(item)

            # 重新放入有效消息
            for item in valid_items:
                heapq.heappush(self._priority_queue, item)

            if expired_count > 0:
                with self._stats_lock:
                    self._stats["messages_expired"] += expired_count
                logger.debug(f"⏰ [SwarmComm] 清理过期消息: {expired_count} 条")

    def _communication_loop(self):
        """通信循环（后台线程）"""
        logger.info("🔄 [SwarmComm] 通信循环启动 (优先级队列模式)")

        last_expiry_check = time.time()

        while self.running:
            try:
                # 定期检查过期消息
                now = time.time()
                if now - last_expiry_check > self._expiry_check_interval:
                    self._check_expired_messages()
                    last_expiry_check = now

                # 从优先级队列获取消息
                message = self._get_next_message(timeout=1.0)
                if message is None:
                    continue

                # 更新统计
                with self._stats_lock:
                    self._stats["messages_received"] += 1

                # 处理响应消息
                if message.type == MessageType.RESPONSE:
                    self._handle_response(message)

                # 处理消息
                self._process_message(message)

                # 记录历史（限制大小防止内存泄漏）
                self.message_history.append(message)
                if len(self.message_history) > self._max_history_size:
                    self.message_history = self.message_history[-self._max_history_size :]

            except Exception as e:
                logger.error(f"❌ [SwarmComm] 处理消息错误: {e}")

        logger.info("🛑 [SwarmComm] 通信循环停止")

    def _process_message(self, message: SwarmMessage):
        """处理消息

        Args:
            message: 消息
        """
        # 1. 调用订阅者
        for callback in self.subscriptions.get(message.type.value, []):
            try:
                callback(message)
            except Exception as e:
                logger.error(f"❌ [SwarmComm] 回调错误: {e}")

        # 2. 特殊处理
        if message.type == MessageType.TASK_BID:
            self._handle_task_bid(message)
        elif message.type == MessageType.VOTE:
            self._handle_vote(message)

    def _handle_task_bid(self, message: SwarmMessage):
        """处理任务投标"""
        task_id = message.content.get("task_id")
        if task_id in self.active_tasks:
            bid = TaskBid(**message.content.get("bid", {}))
            self.active_tasks[task_id]["bids"].append(bid)

    def _handle_vote(self, message: SwarmMessage):
        """处理投票"""
        vote_id = message.content.get("vote_id")
        if vote_id in self.votes:
            vote = Vote(**message.content.get("vote", {}))
            self.votes[vote_id].append(vote)

    def _count_votes(self, vote_id: str, options: List[str]) -> str:
        """统计投票

        Args:
            vote_id: 投票ID
            options: 选项列表

        Returns:
            获胜选项
        """
        votes = self.votes.get(vote_id, [])

        if not votes:
            return options[0] if options else "no_decision"

        # 加权统计
        vote_counts = defaultdict(float)
        for vote in votes:
            vote_counts[vote.option] += vote.weight

        # 返回得票最多的选项
        winner = max(vote_counts, key=vote_counts.get)

        logger.info(f"📊 [SwarmComm] 投票统计:")
        for option, count in vote_counts.items():
            logger.info(f"  - {option}: {count:.2f}")

        return winner


# 全局实例
_swarm_comm = None
_swarm_comm_lock = __import__("threading").Lock()


def get_swarm_communication() -> SwarmCommunication:
    """获取全局蜂群通信实例（线程安全）"""
    global _swarm_comm
    if _swarm_comm is None:
        with _swarm_comm_lock:
            if _swarm_comm is None:
                _swarm_comm = SwarmCommunication()
                _swarm_comm.start()
    return _swarm_comm


# 导出
__all__ = [
    "SwarmCommunication",
    "SwarmMessage",
    "MessageType",
    "TaskBid",
    "Vote",
    "get_swarm_communication",
]
