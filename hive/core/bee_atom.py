# -*- coding: utf-8 -*-
"""
BeeAtom - 蜂群最小原子

最小单位：DNA + 知识 + 状态 + 连接
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hive.core.dna import BeeDNA


class AtomState(Enum):
    """原子状态"""

    ACTIVE = "active"  # 活跃 - 参与任务
    DORMANT = "dormant"  # 休眠 - 储备力量
    RETIRED = "retired"  # 退休 - 知识传承者
    DEAD = "dead"  # 死亡 - 被回收


@dataclass
class KnowledgeAtom:
    """知识原子"""

    vectors: list[float] = field(default_factory=list)
    summary: str = ""
    source: str = ""
    importance: float = 1.0
    age: int = 0
    born_at: int = 0  # 出生世代


@dataclass
class BeeAtom:
    """
    蜂群原子 - 蜂群的最小单位

    包含：
    - id: 唯一标识
    - dna: 基因信息
    - knowledge: 知识片段
    - state: 当前状态
    - connections: 与其他原子的连接
    - born_at: 出生世代
    - fitness: 适应度 (0-1)
    - age: 年龄
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dna: BeeDNA | None = None
    knowledge: KnowledgeAtom = field(default_factory=KnowledgeAtom)
    state: AtomState = AtomState.ACTIVE
    connections: set[str] = field(default_factory=set)
    born_at: int = 0
    fitness: float = 0.0
    age: int = 0

    # 统计信息
    tasks_completed: int = 0
    offspring_count: int = 0

    def __post_init__(self):
        if self.dna is None:
            self.dna = BeeDNA(species="bee", role="worker", id=self.id, target_name="default")

    @property
    def is_alive(self) -> bool:
        """是否存活"""
        return self.state != AtomState.DEAD

    @property
    def is_active(self) -> bool:
        """是否活跃"""
        return self.state == AtomState.ACTIVE

    def activate(self) -> None:
        """激活"""
        if self.state != AtomState.DEAD:
            self.state = AtomState.ACTIVE

    def deactivate(self) -> None:
        """休眠"""
        if self.state == AtomState.ACTIVE:
            self.state = AtomState.DORMANT

    def retire(self) -> None:
        """退休"""
        if self.state != AtomState.DEAD:
            self.state = AtomState.RETIRED

    def die(self) -> None:
        """死亡"""
        self.state = AtomState.DEAD
        self.connections.clear()

    def add_connection(self, other_id: str) -> None:
        """添加连接"""
        if other_id != self.id:
            self.connections.add(other_id)

    def remove_connection(self, other_id: str) -> None:
        """移除连接"""
        self.connections.discard(other_id)

    def aging(self) -> None:
        """老化"""
        self.age += 1
        if self.knowledge:
            self.knowledge.age = self.age

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "dna": self.dna.__dict__ if self.dna else None,
            "knowledge": (
                {
                    "summary": self.knowledge.summary,
                    "importance": self.knowledge.importance,
                    "age": self.knowledge.age,
                }
                if self.knowledge
                else None
            ),
            "state": self.state.value,
            "connections": list(self.connections),
            "born_at": self.born_at,
            "fitness": self.fitness,
            "age": self.age,
            "tasks_completed": self.tasks_completed,
            "offspring_count": self.offspring_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BeeAtom":
        """从字典反序列化"""
        state_str = data.get("state", "active")
        try:
            state = AtomState(state_str)
        except ValueError:
            state = AtomState.ACTIVE
        atom = cls(
            id=data.get("id", str(uuid.uuid4())),
            state=state,
            born_at=data.get("born_at", 0),
            fitness=data.get("fitness", 0.0),
            age=data.get("age", 0),
            tasks_completed=data.get("tasks_completed", 0),
            offspring_count=data.get("offspring_count", 0),
        )

        if data.get("dna"):
            atom.dna = BeeDNA(**data["dna"])

        if data.get("knowledge"):
            atom.knowledge = KnowledgeAtom(
                summary=data["knowledge"].get("summary", ""),
                importance=data["knowledge"].get("importance", 1.0),
                age=data["knowledge"].get("age", 0),
            )

        atom.connections = set(data.get("connections", []))

        return atom

    def __repr__(self) -> str:
        return f"BeeAtom(id={self.id[:8]}, state={self.state.value}, fitness={self.fitness:.2f}, age={self.age})"
