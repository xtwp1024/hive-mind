"""
CognitiveBus - 认知消息总线
===========================

各认知模块通过消息总线进行松耦合通信。

消息流向:
    Perceiver → [PERCEPTION] → GoalDecomposer, Planner, Memory
    GoalDecomposer → [GOAL] → Planner
    Planner → [PLAN] → Executor / QueenBee
    Reflector → [INSIGHT] → Memory, Learner
    SelfMonitor → [METRIC] → KingBee

Usage:
    bus = CognitiveBus()
    bus.subscribe(MessageType.PERCEPTION, my_handler)
    bus.publish(Message(type=MessageType.PERCEPTION, data={...}))
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional

logger = logging.getLogger("Hive.Brain.Bus")


class MessageType(str, Enum):
    """认知消息类型"""

    # 感知层
    PERCEPTION = "perception"  # 感知结果
    INTENT = "intent"  # 意图识别结果

    # 记忆层
    MEMORY_STORE = "memory_store"  # 存储记忆
    MEMORY_RECALL = "memory_recall"  # 记忆检索
    MEMORY_CONSOLIDATE = "memory_consolidate"  # 记忆整合

    # 目标层
    GOAL_DECOMPOSE = "goal_decompose"  # 目标分解请求
    GOAL = "goal"  # 分解后的子目标
    SUBGOAL_COMPLETE = "subgoal_complete"  # 子目标完成

    # 规划层
    PLAN_REQUEST = "plan_request"  # 规划请求
    PLAN = "plan"  # 生成的执行计划
    PLAN_REVISION = "plan_revision"  # 计划修订

    # 执行层
    EXECUTE = "execute"  # 执行指令
    EXECUTION_RESULT = "execution_result"  # 执行结果

    # 反思层
    REFLECT = "reflect"  # 反思请求
    INSIGHT = "insight"  # 反思洞察
    LEARN = "learn"  # 学习请求

    # 元认知
    METRIC = "metric"  # 认知指标
    ALERT = "alert"  # 异常警报
    STRATEGY_CHANGE = "strategy_change"  # 策略变更


@dataclass
class Message:
    """认知消息"""

    type: MessageType
    data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    source: str = "system"
    target: Optional[str] = None  # None = 广播
    correlation_id: Optional[str] = None  # 用于关联相关消息
    ttl: int = 10  # 消息最大跳数

    def __post_init__(self):
        if self.ttl <= 0:
            raise ValueError("TTL must be positive")

    def with_ttl(self, ttl: int) -> "Message":
        """创建递减TTL的副本"""
        return Message(
            type=self.type,
            data=self.data,
            id=self.id,
            timestamp=self.timestamp,
            source=self.source,
            target=self.target,
            correlation_id=self.correlation_id,
            ttl=ttl,
        )

    def is_broadcast(self) -> bool:
        return self.target is None


Handler = Callable[[Message], Awaitable[Any]]


class CognitiveBus:
    """
    认知消息总线

    提供异步publish/subscribe模式，连接各认知模块。
    支持:
    - 广播/单播消息
    - 消息相关性追踪(correlation_id)
    - TTL防止消息无限循环
    - 死信队列处理无法投递的消息
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        persistence_path: Optional[str] = None,
        max_dead_letter_size: int = 1000,
    ):
        """
        Args:
            max_queue_size: 消息队列最大长度
            persistence_path: 消息持久化文件路径（可选）
            max_dead_letter_size: 死信队列最大长度，防止内存耗尽
        """
        self._subscribers: Dict[MessageType, List[Handler]] = {}
        self._global_handlers: List[Handler] = []  # 处理所有消息
        self._queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._dead_letter_queue: List[Message] = []
        self._max_dead_letter_size = max_dead_letter_size
        self._lock = asyncio.Lock()
        self._circuit_open = False  # 断路器状态

        # 消息持久化配置
        self._persistence_path = persistence_path
        self._persistence_enabled = persistence_path is not None

        # 统计
        self._stats = {
            "messages_published": 0,
            "messages_delivered": 0,
            "messages_dead_lettered": 0,
            "messages_persisted": 0,
            "messages_loaded": 0,
        }

    async def start(self):
        """启动消息处理器"""
        if self._running:
            return
        self._running = True
        self._processor_task = asyncio.create_task(self._process_messages())
        logger.info("🧠 CognitiveBus started")

    async def stop(self):
        """停止消息处理器"""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("🧠 CognitiveBus stopped")

    def subscribe(self, message_type: MessageType, handler: Handler):
        """订阅特定类型的消息"""
        if message_type not in self._subscribers:
            self._subscribers[message_type] = []
        self._subscribers[message_type].append(handler)
        logger.debug(f"📝 Subscribed to {message_type.value}")

    def subscribe_global(self, handler: Handler):
        """订阅所有消息"""
        self._global_handlers.append(handler)

    def unsubscribe(self, message_type: MessageType, handler: Handler):
        """取消订阅"""
        if message_type in self._subscribers:
            try:
                self._subscribers[message_type].remove(handler)
            except ValueError:
                pass

    def _add_to_dead_letter(self, message: Message):
        """添加消息到死信队列，带大小限制防止内存耗尽"""
        self._dead_letter_queue.append(message)
        self._stats["messages_dead_lettered"] += 1
        # 超过限制时移除最旧的消息
        while len(self._dead_letter_queue) > self._max_dead_letter_size:
            removed = self._dead_letter_queue.pop(0)
            logger.debug(f"🗑️ Dead letter queue overflow, dropped oldest message {removed.id}")

    async def publish(self, message: Message, source: str = "unknown") -> None:
        """发布消息到总线"""
        message.source = source
        self._stats["messages_published"] += 1

        # 断路器模式：队列满时拒绝新消息，防止级联失败
        if self._circuit_open:
            raise asyncio.QueueFull(f"CognitiveBus circuit open, message {message.id} rejected")

        # 持久化消息（如果启用）
        if self._persistence_enabled:
            await self._persist_message(message)

        try:
            await asyncio.wait_for(self._queue.put(message), timeout=1.0)
        except asyncio.QueueFull:
            # 打开断路器，连续失败时快速失败
            self._circuit_open = True
            logger.warning(f"⚠️ CognitiveBus queue full, message {message.id} rejected")
            # 启动定时重置断路器
            asyncio.create_task(self._reset_circuit_breaker())
            raise  # 传播异常，让调用者知道消息被拒绝

    async def _reset_circuit_breaker(self):
        """定时重置断路器"""
        await asyncio.sleep(5.0)
        self._circuit_open = False
        logger.info("🔄 CognitiveBus circuit breaker reset")

    async def publish_reply(
        self,
        original: Message,
        reply_type: MessageType,
        data: Dict[str, Any],
    ) -> None:
        """发布回复消息，关联原消息"""
        reply = Message(
            type=reply_type,
            data=data,
            source="bus",
            target=original.source,
            correlation_id=original.id,
        )
        await self.publish(reply, source="bus")

    async def _process_messages(self):
        """异步处理消息队列"""
        while self._running:
            try:
                message = await asyncio.wait_for(self._queue.get(), timeout=0.1)

                # TTL检查
                message.ttl -= 1
                if message.ttl <= 0:
                    logger.debug(f"💀 Message {message.id} TTL expired")
                    self._add_to_dead_letter(message)
                    continue

                # 广播给特定类型订阅者
                handlers = self._subscribers.get(message.type, [])
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                        self._stats["messages_delivered"] += 1
                    except Exception as e:
                        logger.error(f"❌ Handler error for {message.type.value}: {e}")

                # 广播给全局订阅者
                for handler in self._global_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                        self._stats["messages_delivered"] += 1
                    except Exception as e:
                        logger.error(f"❌ Global handler error: {e}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ CognitiveBus processing error: {e}")

    def get_stats(self) -> Dict[str, int]:
        """获取总线统计"""
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "dead_letter_count": len(self._dead_letter_queue),
        }

    async def clear_dead_letters(self) -> int:
        """清空死信队列"""
        async with self._lock:
            count = len(self._dead_letter_queue)
            self._dead_letter_queue.clear()
            return count

    async def _persist_message(self, message: Message):
        """持久化单条消息到文件"""
        if not self._persistence_path:
            return

        try:
            msg_dict = {
                "type": message.type.value,
                "data": message.data,
                "id": message.id,
                "timestamp": message.timestamp,
                "source": message.source,
                "target": message.target,
                "correlation_id": message.correlation_id,
            }

            # 追加到文件（JSONL格式，每行一条消息）
            # 使用 asyncio.to_thread 避免阻塞事件循环
            def _write_message():
                with open(self._persistence_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")

            async with self._lock:
                await asyncio.to_thread(_write_message)
                self._stats["messages_persisted"] += 1

        except Exception as e:
            logger.warning(f"⚠️ Failed to persist message: {e}")

    async def persist_messages(self, messages: List[Message]):
        """
        批量持久化消息

        Args:
            messages: 消息列表
        """
        if not self._persistence_path:
            return

        try:
            # 准备消息数据
            msg_lines = []
            for msg in messages:
                msg_dict = {
                    "type": msg.type.value,
                    "data": msg.data,
                    "id": msg.id,
                    "timestamp": msg.timestamp,
                    "source": msg.source,
                    "target": msg.target,
                    "correlation_id": msg.correlation_id,
                }
                msg_lines.append(json.dumps(msg_dict, ensure_ascii=False) + "\n")

            # 使用 asyncio.to_thread 避免阻塞事件循环
            def _write_messages():
                with open(self._persistence_path, "a", encoding="utf-8") as f:
                    f.writelines(msg_lines)

            async with self._lock:
                await asyncio.to_thread(_write_messages)
                self._stats["messages_persisted"] += len(messages)

        except Exception as e:
            logger.warning(f"⚠️ Failed to persist messages: {e}")

    def load_persisted_messages(
        self,
        limit: int = 100,
    ) -> List[Message]:
        """
        加载持久化的消息

        Args:
            limit: 最多加载的消息数

        Returns:
            List[Message]: 消息列表
        """
        if not self._persistence_path or not os.path.exists(self._persistence_path):
            return []

        messages = []
        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(messages) >= limit:
                        break
                    try:
                        msg_dict = json.loads(line.strip())
                        msg = Message(
                            type=MessageType(msg_dict["type"]),
                            data=msg_dict["data"],
                            id=msg_dict["id"],
                            timestamp=msg_dict["timestamp"],
                            source=msg_dict.get("source", "unknown"),
                            target=msg_dict.get("target"),
                            correlation_id=msg_dict.get("correlation_id"),
                        )
                        messages.append(msg)
                        self._stats["messages_loaded"] += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to parse persisted message: {e}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load persisted messages: {e}")

        return messages

    def clear_persistence(self) -> bool:
        """
        清空持久化文件

        Returns:
            bool: 是否成功
        """
        if not self._persistence_path:
            return False

        try:
            if os.path.exists(self._persistence_path):
                os.remove(self._persistence_path)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to clear persistence: {e}")
            return False


# 全局单例
_global_bus: Optional[CognitiveBus] = None


def get_cognitive_bus() -> CognitiveBus:
    """获取全局CognitiveBus单例（同步版本，仅返回已初始化的实例）"""
    global _global_bus
    if _global_bus is None:
        raise RuntimeError(
            "CognitiveBus not initialized. Use 'await get_or_create_cognitive_bus()' instead."
        )
    return _global_bus


async def get_or_create_cognitive_bus() -> CognitiveBus:
    """获取或创建全局CognitiveBus单例（异步安全）"""
    global _global_bus
    if _global_bus is None:
        _global_bus = CognitiveBus()
        await _global_bus.start()
    return _global_bus
