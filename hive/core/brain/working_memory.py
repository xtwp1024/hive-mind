"""
WorkingMemory - 工作记忆
========================

工作记忆是当前思考的上下文空间，类似人类的前额叶工作记忆。
存储:
- 当前目标/子目标栈
- 近期重要事件 (Last-K)
- 注意力焦点
- 置信度

Capacity: 默认最多存储 50 个注意力项

Usage:
    wm = WorkingMemory()
    wm.push_goal("分析项目架构")
    wm.add_attention("正在查看 core/swarm.py")
    context = wm.get_context()
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("Hive.Brain.WorkingMemory")


@dataclass
class AttentionItem:
    """注意力项"""

    id: str
    content: Any  # 内容（任意类型）
    attention_level: float = 1.0  # 注意力水平 0-1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def access(self):
        """访问时更新last_accessed"""
        self.last_accessed = time.time()


@dataclass
class Goal:
    """目标"""

    id: str
    description: str
    status: str = "active"  # active, paused, completed, failed
    parent_id: Optional[str] = None  # 父目标ID
    children_ids: List[str] = field(default_factory=list)
    priority: int = 1  # 1-5, 5最高
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self):
        self.status = "completed"
        self.completed_at = time.time()


class WorkingMemory:
    """
    工作记忆

    特性:
    - 最大容量限制 (max_items)
    - 基于访问频率的衰减
    - 目标栈支持层次结构
    - 注意力焦点跟踪
    """

    DEFAULT_MAX_ITEMS = 50
    DEFAULT_MAX_GOALS = 10
    ATTENTION_DECAY_RATE = 0.95  # 每分钟衰减率
    LOW_ATTENTION_THRESHOLD = 0.1

    def __init__(
        self,
        max_items: int = DEFAULT_MAX_ITEMS,
        max_goals: int = DEFAULT_MAX_GOALS,
    ):
        self.max_items = max_items
        self.max_goals = max_goals

        self._items: Dict[str, AttentionItem] = {}
        self._item_order: Deque[str] = deque()  # FIFO顺序
        self._attention_focus: Optional[str] = None  # 当前注意力焦点ID

        self._goal_stack: List[Goal] = []  # 目标栈
        self._goals: Dict[str, Goal] = {}  # 所有目标

        self._lock = asyncio.Lock()
        self._last_decay = time.time()

        logger.debug(f"🧠 WorkingMemory initialized (max_items={max_items}, max_goals={max_goals})")

    @property
    def attention_focus(self) -> Optional[AttentionItem]:
        """获取当前注意力焦点"""
        if self._attention_focus and self._attention_focus in self._items:
            item = self._items[self._attention_focus]
            item.access()
            return item
        return None

    def get_context(self) -> Dict[str, Any]:
        """获取当前上下文快照"""
        now = time.time()
        return {
            "focus": (
                {
                    "id": self._attention_focus,
                    "content": item.content,
                    "attention_level": item.attention_level,
                }
                if (item := self._items.get(self._attention_focus))
                else None
            ),
            "recent_items": [
                {
                    "id": item.id,
                    "content": item.content,
                    "attention_level": item.attention_level,
                    "age_seconds": now - item.created_at,
                }
                for item in list(self._items.values())[-10:]
            ],
            "active_goals": [
                {
                    "id": g.id,
                    "description": g.description,
                    "priority": g.priority,
                    "progress": self._calc_goal_progress(g),
                }
                for g in self._goal_stack
            ],
            "total_items": len(self._items),
            "memory_pressure": len(self._items) / self.max_items,
        }

    async def add(
        self,
        content: Any,
        tags: Optional[List[str]] = None,
        attention_level: float = 1.0,
    ) -> str:
        """添加注意力项"""
        async with self._lock:
            self._decay_with_lock()

            item_id = f"wm_{len(self._items)}_{int(time.time() * 1000) % 10000}"
            item = AttentionItem(
                id=item_id,
                content=content,
                attention_level=attention_level,
                tags=tags or [],
            )

            self._items[item_id] = item
            self._item_order.append(item_id)
            self._attention_focus = item_id

            # 容量检查
            while len(self._items) > self.max_items:
                oldest_id = self._item_order.popleft()
                if oldest_id in self._items:
                    del self._items[oldest_id]
                    if self._attention_focus == oldest_id:
                        self._attention_focus = (
                            list(self._items.keys())[-1] if self._items else None
                        )

            logger.debug(f"🧠 WorkingMemory added item {item_id}, total={len(self._items)}")
            return item_id

    async def set_focus(self, item_id: str) -> bool:
        """设置注意力焦点"""
        async with self._lock:
            if item_id in self._items:
                self._attention_focus = item_id
                self._items[item_id].access()
                self._items[item_id].attention_level = min(
                    1.0, self._items[item_id].attention_level + 0.2
                )
                return True
            return False

    async def update_attention(self, item_id: str, delta: float) -> bool:
        """更新注意力水平"""
        async with self._lock:
            if item_id in self._items:
                self._items[item_id].attention_level = max(
                    0.0, min(1.0, self._items[item_id].attention_level + delta)
                )
                return True
            return False

    async def push_goal(
        self,
        description: str,
        priority: int = 1,
        parent_id: Optional[str] = None,
    ) -> Goal:
        """压入新目标到栈顶"""
        async with self._lock:
            goal_id = f"goal_{len(self._goals)}_{int(time.time() * 1000) % 10000}"
            goal = Goal(
                id=goal_id,
                description=description,
                priority=priority,
                parent_id=parent_id,
            )

            self._goals[goal_id] = goal
            self._goal_stack.append(goal)

            # 更新父目标的children
            if parent_id and parent_id in self._goals:
                self._goals[parent_id].children_ids.append(goal_id)

            # 容量检查
            while len(self._goal_stack) > self.max_goals:
                removed = self._goal_stack.pop(0)
                removed.status = "paused"

            logger.debug(f"🧠 WorkingMemory pushed goal: {description[:50]}")
            return goal

    async def pop_goal(self) -> Optional[Goal]:
        """弹出栈顶目标"""
        async with self._lock:
            if self._goal_stack:
                goal = self._goal_stack.pop()
                return goal
            return None

    async def complete_goal(self, goal_id: str) -> bool:
        """标记目标完成"""
        async with self._lock:
            if goal_id in self._goals:
                goal = self._goals[goal_id]
                goal.complete()
                if goal in self._goal_stack:
                    self._goal_stack.remove(goal)
                logger.debug(f"🧠 WorkingMemory completed goal: {goal.description[:50]}")
                return True
            return False

    async def get_active_goals(self) -> List[Goal]:
        """获取当前活跃目标"""
        async with self._lock:
            return list(self._goal_stack)

    async def get_goal_by_id(self, goal_id: str) -> Optional[Goal]:
        return self._goals.get(goal_id)

    def get_active_goal_count(self) -> int:
        """获取活跃目标数量（线程安全）"""
        return len(self._goal_stack)

    def _calc_goal_progress(self, goal: Goal) -> float:
        """计算目标完成进度"""
        if not goal.children_ids:
            return 1.0 if goal.status == "completed" else 0.0
        completed = sum(
            1
            for cid in goal.children_ids
            if self._goals.get(cid, Goal(id="", description="", status="failed")).status
            == "completed"
        )
        return completed / len(goal.children_ids) if goal.children_ids else 0.0

    def _decay_attention(self):
        """对注意力水平进行衰减"""
        now = time.time()
        elapsed = now - self._last_decay
        if elapsed < 60:  # 至少60秒才衰减
            return

        decay_factor = self.ATTENTION_DECAY_RATE ** (elapsed / 60)
        for item in self._items.values():
            item.attention_level *= decay_factor

        self._last_decay = now

    async def _decay_attention_async(self):
        """异步安全的衰减操作（内部调用，不在已持锁的情况下使用）"""
        async with self._lock:
            self._decay_attention()

    async def _decay_with_lock(self):
        """在已持锁的情况下调用的衰减操作"""
        self._decay_attention()

    def get_memory_pressure(self) -> float:
        """获取记忆压力 (0-1)"""
        return len(self._items) / self.max_items

    def get_attention_load(self) -> float:
        """获取注意力负载 (0-1)"""
        active_items = sum(
            1
            for item in self._items.values()
            if item.attention_level > self.LOW_ATTENTION_THRESHOLD
        )
        return active_items / self.max_items

    async def clear(self):
        """清空工作记忆"""
        async with self._lock:
            self._items.clear()
            self._item_order.clear()
            self._goal_stack.clear()
            self._goals.clear()
            self._attention_focus = None
            logger.info("🧠 WorkingMemory cleared")

    def snapshot(self) -> Dict[str, Any]:
        """获取快照用于调试"""
        return {
            "item_count": len(self._items),
            "goal_count": len(self._goal_stack),
            "attention_focus": self._attention_focus,
            "memory_pressure": self.get_memory_pressure(),
            "attention_load": self.get_attention_load(),
            "decay_needed": time.time() - self._last_decay > 60,
        }
