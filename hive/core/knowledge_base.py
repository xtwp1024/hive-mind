# -*- coding: utf-8 -*-
"""
KnowledgeBase - 知识库

存储和管理蜂群积累的知识，支持代际传承。
"""

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hive.core.bee_atom import KnowledgeAtom

if TYPE_CHECKING:
    from hive.core.bee_atom import BeeAtom


@dataclass
class KnowledgeEntry:
    """知识条目"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    vectors: list[float] = field(default_factory=list)
    importance: float = 1.0
    born_at: int = 0  # 出生世代
    last_accessed: int = 0
    access_count: int = 0

    # 元数据
    source_atom_id: str = ""
    source_type: str = "atom"  # atom, cluster, external

    def update_access(self, generation: int) -> None:
        """更新访问信息"""
        self.last_accessed = generation
        self.access_count += 1

    def age(self, current_generation: int) -> int:
        """知识年龄"""
        return current_generation - self.born_at

    def effective_importance(self, current_generation: int, decay_rate: float = 0.95) -> float:
        """有效重要性（考虑年龄衰减）"""
        return self.importance * (decay_rate ** self.age(current_generation))


class KnowledgeBase:
    """
    知识库 - 蜂群的知识存储

    支持：
    - 添加知识
    - 查询知识
    - 重要性衰减
    - 知识遗忘
    """

    # 配置
    MAX_KNOWLEDGE_SIZE = 10000  # 最大知识条目
    MIN_IMPORTANCE = 0.01  # 最小重要性阈值
    DECAY_RATE = 0.95  # 每代衰减率

    def __init__(self):
        self.knowledge: dict[str, KnowledgeEntry] = {}
        self.generation = 0

    def set_generation(self, generation: int) -> None:
        """设置当前世代"""
        self.generation = generation

    def add_knowledge(
        self,
        content: str,
        importance: float = 1.0,
        source_atom_id: str = "",
        vectors: list[float] = None,
    ) -> str:
        """
        添加知识

        Returns:
            知识 ID
        """
        # 检查是否需要遗忘
        if len(self.knowledge) >= self.MAX_KNOWLEDGE_SIZE:
            self._forget_least_important()

        entry = KnowledgeEntry(
            content=content,
            importance=importance,
            born_at=self.generation,
            source_atom_id=source_atom_id,
            vectors=vectors or [],
        )

        self.knowledge[entry.id] = entry
        return entry.id

    def add_from_atom(self, atom: "BeeAtom") -> str:
        """从原子添加知识"""
        if not atom.knowledge:
            return ""

        return self.add_knowledge(
            content=atom.knowledge.summary,
            importance=atom.knowledge.importance,
            source_atom_id=atom.id,
            vectors=atom.knowledge.vectors,
        )

    def query(
        self, query_vector: list[float] = None, keywords: str = "", limit: int = 10
    ) -> list[KnowledgeEntry]:
        """
        查询知识

        Args:
            query_vector: 查询向量
            keywords: 关键词
            limit: 返回数量

        Returns:
            匹配的知识列表
        """
        if not query_vector and not keywords:
            # 返回最重要的知识
            return self._get_most_important(limit)

        results = []

        for entry in self.knowledge.values():
            score = self._calculate_relevance(entry, query_vector, keywords)

            if score > 0:
                # 考虑有效重要性
                effective = entry.effective_importance(self.generation, self.DECAY_RATE)
                results.append((entry, score * effective))

        # 排序并返回
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]

    def _calculate_relevance(
        self, entry: KnowledgeEntry, query_vector: list[float], keywords: str
    ) -> float:
        """计算相关性"""
        score = 0.0

        # 关键词匹配
        if keywords:
            keywords_lower = keywords.lower()
            if keywords_lower in entry.content.lower():
                score += 0.5

        # 向量相似度
        if query_vector and entry.vectors:
            score += self._cosine_similarity(query_vector, entry.vectors)

        return score

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """余弦相似度"""
        if not a or not b:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _get_most_important(self, limit: int) -> list[KnowledgeEntry]:
        """获取最重要的知识"""
        entries = [
            (e, e.effective_importance(self.generation, self.DECAY_RATE))
            for e in self.knowledge.values()
        ]
        entries.sort(key=lambda x: x[1], reverse=True)
        return [e[0] for e in entries[:limit]]

    def _forget_least_important(self) -> None:
        """遗忘最不重要的知识"""
        if not self.knowledge:
            return

        # 按有效重要性排序
        entries = [
            (eid, e.effective_importance(self.generation, self.DECAY_RATE))
            for eid, e in self.knowledge.items()
        ]
        entries.sort(key=lambda x: x[1])

        # 删除最不重要的 10%
        count = max(1, len(entries) // 10)
        for eid, _ in entries[:count]:
            del self.knowledge[eid]

    def get_knowledge_by_id(self, knowledge_id: str) -> KnowledgeEntry | None:
        """获取指定知识"""
        entry = self.knowledge.get(knowledge_id)
        if entry:
            entry.update_access(self.generation)
        return entry

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        total_importance = sum(
            e.effective_importance(self.generation, self.DECAY_RATE)
            for e in self.knowledge.values()
        )

        return {
            "total_knowledge": len(self.knowledge),
            "total_importance": total_importance,
            "average_importance": total_importance / max(1, len(self.knowledge)),
            "generation": self.generation,
        }

    def to_dict(self) -> dict[str, Any]:
        """序列化"""
        return {
            "generation": self.generation,
            "knowledge": {
                k: {
                    "id": v.id,
                    "content": v.content[:100],  # 截断
                    "importance": v.importance,
                    "born_at": v.born_at,
                    "access_count": v.access_count,
                }
                for k, v in self.knowledge.items()
            },
        }

    def __len__(self) -> int:
        return len(self.knowledge)

    def __repr__(self) -> str:
        return f"KnowledgeBase(generation={self.generation}, size={len(self)})"
