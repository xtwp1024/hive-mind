# -*- coding: utf-8 -*-
"""
Attraction - 自由结合规则

定义原子之间的吸引规则，实现自由结合。
"""

import math
import random
from dataclasses import dataclass

from hive.core.bee_atom import BeeAtom


@dataclass
class AttractionResult:
    """吸引结果"""

    target_id: str
    attraction_score: float  # 0-1
    reason: str


class AttractionRule:
    """
    吸引规则集合

    规则：
    - 基因相似吸引
    - 知识互补吸引
    - 任务协同吸引
    - 适应度吸引（精英吸引追随者）
    - 随机探索
    """

    # 权重配置
    GENE_WEIGHT = 0.3  # 基因相似权重
    KNOWLEDGE_WEIGHT = 0.2  # 知识互补权重
    TASK_WEIGHT = 0.3  # 任务协同权重
    FITNESS_WEIGHT = 0.1  # 适应度权重
    RANDOM_WEIGHT = 0.1  # 随机探索权重

    # 阈值
    GENE_SIMILARITY_THRESHOLD = 0.7  # 基因相似度阈值
    MIN_ATTRACTION = 0.1  # 最小吸引度

    @classmethod
    def calculate_attraction(cls, atom: BeeAtom, others: list[BeeAtom]) -> list[AttractionResult]:
        """
        计算对其他所有原子的吸引度

        Args:
            atom: 源原子
            others: 其他原子列表

        Returns:
            按吸引度排序的结果列表
        """
        if not others:
            return []

        results = []

        for other in others:
            if other.id == atom.id or not other.is_alive:
                continue

            score = cls._calculate_score(atom, other)

            if score > cls.MIN_ATTRACTION:
                reason = cls._get_attraction_reason(atom, other)
                results.append(
                    AttractionResult(target_id=other.id, attraction_score=score, reason=reason)
                )

        # 按吸引度排序
        results.sort(key=lambda x: x.attraction_score, reverse=True)

        return results

    @classmethod
    def _calculate_score(cls, atom: BeeAtom, other: BeeAtom) -> float:
        """计算综合吸引度"""

        gene_score = cls._gene_similarity(atom, other)
        knowledge_score = cls._knowledge_distance(atom, other)
        task_score = cls._task_synergy(atom, other)
        fitness_score = cls._fitness_gravity(atom, other)
        random_score = cls._random_exploration()

        total = (
            gene_score * cls.GENE_WEIGHT
            + knowledge_score * cls.KNOWLEDGE_WEIGHT
            + task_score * cls.TASK_WEIGHT
            + fitness_score * cls.FITNESS_WEIGHT
            + random_score * cls.RANDOM_WEIGHT
        )

        return min(1.0, max(0.0, total))

    @classmethod
    def _gene_similarity(cls, a: BeeAtom, b: BeeAtom) -> float:
        """基因相似度 (0-1)"""
        if not a.dna or not b.dna:
            return 0.5

        # 基于 role 相似度
        if a.dna.role == b.dna.role:
            role_sim = 1.0
        else:
            role_sim = 0.0

        # 基于 species 相似度
        if a.dna.species == b.dna.species:
            species_sim = 1.0
        else:
            species_sim = 0.5

        # 基于能力权重相似度
        cap_keys = [
            "mining_capability",
            "gathering_capability",
            "architectural_capability",
            "librarian_capability",
            "sentinel_capability",
        ]

        cap_sim = 0.0
        for key in cap_keys:
            a_val = getattr(a.dna, key, 0.5)
            b_val = getattr(b.dna, key, 0.5)
            cap_sim += 1 - abs(a_val - b_val)

        cap_sim /= len(cap_keys)

        return (role_sim + species_sim + cap_sim) / 3

    @classmethod
    def _knowledge_distance(cls, a: BeeAtom, b: BeeAtom) -> float:
        """知识距离/互补性 (0-1, 距离越远越互补)"""
        if not a.knowledge.vectors or not b.knowledge.vectors:
            return 0.5

        # 余弦相似度
        dot = sum(a * b for a, b in zip(a.knowledge.vectors, b.knowledge.vectors))
        norm_a = math.sqrt(sum(x * x for x in a.knowledge.vectors))
        norm_b = math.sqrt(sum(x * x for x in b.knowledge.vectors))

        if norm_a == 0 or norm_b == 0:
            return 0.5

        similarity = dot / (norm_a * norm_b)

        # 转换为距离（相似度高 = 互补性低）
        return 1.0 - similarity

    @classmethod
    def _task_synergy(cls, a: BeeAtom, b: BeeAtom) -> float:
        """任务协同 (0-1)"""
        if not a.dna or not b.dna:
            return 0.5

        # 处理同一目标 = 强协同
        if (
            a.dna.target_name
            and b.dna.target_name
            and a.dna.target_name == b.dna.target_name
            and a.dna.target_name != "default"
        ):
            return 1.0

        return 0.3

    @classmethod
    def _fitness_gravity(cls, a: BeeAtom, b: BeeAtom) -> float:
        """适应度引力 - 精英吸引追随者"""
        if a.fitness == 0:
            return 0.5

        # 高适应度吸引低适应度
        if a.fitness > b.fitness:
            return (a.fitness - b.fitness) / a.fitness

        return 0.1

    @classmethod
    def _random_exploration(cls) -> float:
        """随机探索 (0-1)"""
        return random.random()

    @classmethod
    def _get_attraction_reason(cls, a: BeeAtom, b: BeeAtom) -> str:
        """获取吸引原因"""
        gene_sim = cls._gene_similarity(a, b)
        task_score = cls._task_synergy(a, b)

        if gene_sim > cls.GENE_SIMILARITY_THRESHOLD:
            return "基因相似"
        elif task_score > 0.8:
            return "任务协同"
        elif a.fitness > b.fitness + 0.3:
            return "精英吸引"
        else:
            return "探索连接"


class ClusterFormation:
    """簇形成算法"""

    MAX_CLUSTER_SIZE = 50  # 最大簇大小
    ATTRACTION_THRESHOLD = 0.3  # 形成连接的最低吸引度

    @classmethod
    def form_clusters(cls, atoms: list[BeeAtom]) -> list[list[str]]:
        """
        根据吸引规则形成簇

        Returns:
            簇列表，每个簇是原子 ID 列表
        """
        if not atoms:
            return []

        # 只处理活跃原子
        active = [a for a in atoms if a.is_active]
        if not active:
            return []

        # 构建吸引图
        connections = cls._build_attraction_graph(active)

        # 使用 Union-Find 形成连通分量
        clusters = cls._union_find(active, connections)

        return clusters

    @classmethod
    def _build_attraction_graph(cls, atoms: list[BeeAtom]) -> dict[str, list[str]]:
        """构建吸引图"""
        connections: dict[str, list[str]] = {a.id: [] for a in atoms}

        for atom in atoms:
            attractions = AttractionRule.calculate_attraction(atom, atoms)

            for attr in attractions[:10]:  # 最多保留 top 10
                if attr.attraction_score > cls.ATTRACTION_THRESHOLD:
                    connections[atom.id].append(attr.target_id)

        return connections

    @classmethod
    def _union_find(
        cls, atoms: list[BeeAtom], connections: dict[str, list[str]]
    ) -> list[list[str]]:
        """Union-Find 算法形成簇"""
        parent = {a.id: a.id for a in atoms}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 合并连接的原子
        for atom_id, targets in connections.items():
            for target_id in targets:
                union(atom_id, target_id)

        # 收集所有簇
        clusters_dict: dict[str, list[str]] = {}
        for atom in atoms:
            root = find(atom.id)
            if root not in clusters_dict:
                clusters_dict[root] = []
            clusters_dict[root].append(atom.id)

        # 过滤太小的簇
        return [c for c in clusters_dict.values() if len(c) >= 2]
