# 阶段 E: 万年蜂群系统 - 详细实现计划

> 基于 `docs/eternal_swarm_design.md`
> 创建日期: 2026-04-22
> 版本: v1.0

---

## 📋 任务分解

| 子任务 | 优先级 | 估算时间 | 状态 |
|--------|--------|----------|------|
| E1: BeeAtom 数据类实现 | 高 | 30-45 分钟 | ⏳ 待开始 |
| E2: KnowledgeAtom 数据类 | 高 | 15-20 分钟 | ⏳ 待开始 |
| E3: AtomState 枚举 | 高 | 5-10 分钟 | ⏳ 待开始 |
| E4: 吸引规则实现 | 高 | 30-45 分钟 | ⏳ 待开始 |
| E5: Cluster 涌现结构 | 中 | 20-30 分钟 | ⏳ 待开始 |
| E6: Swarm 主类 | 高 | 30-45 分钟 | ⏳ 待开始 |
| E7: 知识衰减机制 | 中 | 15-20 分钟 | ⏳ 待开始 |
| E8: 检查点持久化 | 中 | 20-30 分钟 | ⏳ 待开始 |
| E9: 进化循环算法 | 高 | 30-45 分钟 | ⏳ 待开始 |
| E10: 单元测试 | 中 | 20-30 分钟 | ⏳ 待开始 |

---

## 📝 详细任务设计

### E1: BeeAtom 数据类实现

**文件位置:** `hive/core/bee_atom.py`

**数据结构:**
```python
from dataclasses import dataclass, field, asdict
from typing import Set, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
from enum import Enum


class AtomState(Enum):
    """原子状态"""
    ACTIVE = "active"      # 活跃 - 参与任务
    DORMANT = "dormant"     # 休眠 - 储备力量
    RETIRED = "retired"     # 退休 - 知识传承者
    DEAD = "dead"           # 死亡 - 被回收


@dataclass
class KnowledgeAtom:
    """知识原子"""
    vectors: List[float] = field(default_factory=list)
    summary: str = ""
    source: str = ""
    importance: float = 1.0
    age: int = 0
    source_gen: int = 0
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeAtom':
        return cls(**data)
    
    def get_current_importance(self, current_gen: int, decay_rate: float = 0.95) -> float:
        """计算当前重要性（衰减后）"""
        if self.source_gen <= 0:
            return self.importance
        age = current_gen - self.source_gen
        return self.importance * (decay_rate ** age)


@dataclass
class BeeAtom:
    """蜂群原子 - 最小智能单元"""
    id: str = field(default_factory=lambda: str(uuid4()))
    
    # 基因信息 (与现有 SpeciesBlueprint 集成)
    blueprint_id: Optional[str] = None
    dna_hash: str = ""
    capabilities: Dict[str, float] = field(default_factory=dict)
    
    # 知识
    knowledge: Optional[KnowledgeAtom] = None
    memory_ids: Set[str] = field(default_factory=set)
    
    # 状态
    state: AtomState = AtomState.ACTIVE
    connections: Set[str] = field(default_factory=set)
    
    # 生命周期
    born_at: int = 0
    generation: int = 0
    fitness: float = 0.0
    fitness_history: List[float] = field(default_factory=list)
    
    # 统计
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "blueprint_id": self.blueprint_id,
            "dna_hash": self.dna_hash,
            "capabilities": self.capabilities,
            "knowledge": self.knowledge.to_dict() if self.knowledge else None,
            "memory_ids": list(self.memory_ids),
            "state": self.state.value,
            "connections": list(self.connections),
            "born_at": self.born_at,
            "generation": self.generation,
            "fitness": self.fitness,
            "fitness_history": self.fitness_history,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "last_heartbeat": self.last_heartbeat,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeeAtom':
        knowledge = None
        if data.get("knowledge"):
            knowledge = KnowledgeAtom.from_dict(data["knowledge"])
        
        return cls(
            id=data.get("id", str(uuid4())),
            blueprint_id=data.get("blueprint_id"),
            dna_hash=data.get("dna_hash", ""),
            capabilities=data.get("capabilities", {}),
            knowledge=knowledge,
            memory_ids=set(data.get("memory_ids", [])),
            state=AtomState(data.get("state", "active")),
            connections=set(data.get("connections", [])),
            born_at=data.get("born_at", 0),
            generation=data.get("generation", 0),
            fitness=data.get("fitness", 0.0),
            fitness_history=data.get("fitness_history", []),
            tasks_completed=data.get("tasks_completed", 0),
            tasks_failed=data.get("tasks_failed", 0),
            last_heartbeat=data.get("last_heartbeat", time.time()),
        )
    
    def add_connection(self, other_id: str) -> bool:
        """添加连接"""
        if other_id == self.id:
            return False
        self.connections.add(other_id)
        return True
    
    def remove_connection(self, other_id: str) -> bool:
        """移除连接"""
        if other_id in self.connections:
            self.connections.remove(other_id)
            return True
        return False
    
    def update_fitness(self, new_fitness: float, max_history: int = 100) -> None:
        """更新适应度"""
        self.fitness = new_fitness
        self.fitness_history.append(new_fitness)
        if len(self.fitness_history) > max_history:
            self.fitness_history = self.fitness_history[-max_history:]
    
    def get_avg_fitness(self, last_n: int = 10) -> float:
        """获取平均适应度"""
        if not self.fitness_history:
            return self.fitness
        history = self.fitness_history[-last_n:]
        return sum(history) / len(history)
    
    def should_be_eliminated(self, threshold: float = 0.1, max_age: int = 100) -> bool:
        """检查是否应该被淘汰"""
        # 适应度过低
        if self.fitness < threshold:
            return True
        
        # 年龄过大
        if self.generation - self.born_at > max_age:
            # 概率淘汰，随年龄增长
            age = self.generation - self.born_at
            elimination_chance = min(0.01 * (age - max_age), 0.5)
            import random
            return random.random() < elimination_chance
        
        return False
```

---

### E4: 吸引规则实现

**文件位置:** `hive/core/attraction.py`

```python
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from enum import Enum
import math
import random

from hive.core.bee_atom import BeeAtom, KnowledgeAtom


class AttractionStrength(Enum):
    """吸引强度"""
    STRONG = 1.0
    MEDIUM = 0.5
    WEAK = 0.2
    NONE = 0.0


@dataclass
class AttractionRule:
    """吸引规则"""
    name: str
    weight: float
    calculator: Callable[[BeeAtom, BeeAtom, Optional[Dict]], float]
    
    def calculate(self, a: BeeAtom, b: BeeAtom, context: Optional[Dict] = None) -> float:
        return self.calculator(a, b, context) * self.weight


class AttractionEngine:
    """吸引规则引擎"""
    
    def __init__(self):
        self.rules: Dict[str, AttractionRule] = {}
        self._register_default_rules()
    
    def _register_default_rules(self):
        """注册默认吸引规则"""
        
        # 1. 基因相似
        self.rules["gene_similarity"] = AttractionRule(
            name="基因相似",
            weight=0.4,
            calculator=self._calculate_gene_similarity,
        )
        
        # 2. 知识互补
        self.rules["knowledge_complementarity"] = AttractionRule(
            name="知识互补",
            weight=0.3,
            calculator=self._calculate_knowledge_distance,
        )
        
        # 3. 适应度吸引
        self.rules["fitness_gravity"] = AttractionRule(
            name="适应度吸引",
            weight=0.2,
            calculator=self._calculate_fitness_gravity,
        )
        
        # 4. 随机探索
        self.rules["random_exploration"] = AttractionRule(
            name="随机探索",
            weight=0.1,
            calculator=self._calculate_random,
        )
    
    def _calculate_gene_similarity(self, a: BeeAtom, b: BeeAtom, 
                                    context: Optional[Dict] = None) -> float:
        """计算基因相似度"""
        # 相同 blueprint 直接强吸引
        if a.blueprint_id and a.blueprint_id == b.blueprint_id:
            return 1.0
        
        # 相同 DNA hash
        if a.dna_hash and a.dna_hash == b.dna_hash:
            return 1.0
        
        # 能力相似度
        if not a.capabilities or not b.capabilities:
            return 0.0
        
        all_caps = set(a.capabilities.keys()).union(set(b.capabilities.keys()))
        if not all_caps:
            return 0.0
        
        similarity = 0.0
        for cap in all_caps:
            val_a = a.capabilities.get(cap, 0.0)
            val_b = b.capabilities.get(cap, 0.0)
            # 计算能力值的相似度
            diff = abs(val_a - val_b)
            similarity += (1.0 - diff)
        
        similarity /= len(all_caps)
        
        # 70% 以上为强吸引
        return similarity if similarity > 0.7 else 0.0
    
    def _calculate_knowledge_distance(self, a: BeeAtom, b: BeeAtom,
                                        context: Optional[Dict] = None) -> float:
        """计算知识距离（互补性）"""
        if not a.knowledge or not b.knowledge:
            return 0.5  # 默认中等吸引
        
        vec_a = a.knowledge.vectors
        vec_b = b.knowledge.vectors
        
        if not vec_a or not vec_b:
            return 0.5
        
        # 计算欧氏距离
        if len(vec_a) != len(vec_b):
            return 0.5
        
        distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(vec_a, vec_b)))
        
        # 距离越大，互补性越强
        max_distance = math.sqrt(len(vec_a))
        complementarity = distance / max_distance
        
        return complementarity
    
    def _calculate_fitness_gravity(self, a: BeeAtom, b: BeeAtom,
                                     context: Optional[Dict] = None) -> float:
        """计算适应度引力（高适应度吸引低适应度）"""
        if a.fitness <= 0 or b.fitness <= 0:
            return 0.0
        
        # 高适应度吸引低适应度
        # 差距越大，引力越强（最高 0.5）
        fitness_diff = abs(a.fitness - b.fitness)
        max_fitness = max(a.fitness, b.fitness)
        
        if max_fitness <= 0:
            return 0.0
        
        # 归一化到 0-0.5
        gravity = min(fitness_diff / max_fitness, 0.5)
        
        return gravity
    
    def _calculate_random(self, a: BeeAtom, b: BeeAtom,
                          context: Optional[Dict] = None) -> float:
        """随机探索"""
        # 10% 的概率产生随机吸引
        return random.random() if random.random() < 0.1 else 0.0
    
    def calculate_total_attraction(self, a: BeeAtom, b: BeeAtom,
                                    context: Optional[Dict] = None) -> float:
        """计算两个原子之间的总吸引度"""
        if a.id == b.id:
            return 0.0
        
        total = 0.0
        for rule in self.rules.values():
            total += rule.calculate(a, b, context)
        
        # 归一化到 0-1
        total_weight = sum(r.weight for r in self.rules.values())
        if total_weight > 0:
            total /= total_weight
        
        return min(total, 1.0)
    
    def should_connect(self, a: BeeAtom, b: BeeAtom,
                       threshold: float = 0.3,
                       context: Optional[Dict] = None) -> bool:
        """判断两个原子是否应该建立连接"""
        attraction = self.calculate_total_attraction(a, b, context)
        
        # 基础阈值
        if attraction < threshold:
            return False
        
        # 概率连接，吸引度越高概率越大
        return random.random() < attraction


# 全局单例
_global_attraction_engine: Optional[AttractionEngine] = None


def get_attraction_engine() -> AttractionEngine:
    """获取全局吸引引擎"""
    global _global_attraction_engine
    if _global_attraction_engine is None:
        _global_attraction_engine = AttractionEngine()
    return _global_attraction_engine
```

---

### E6: Swarm 主类

**文件位置:** `hive/core/swarm_eternal.py`

```python
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Any, Tuple
from uuid import uuid4
from datetime import datetime
import json
import time
import random
import math
from pathlib import Path

from hive.core.bee_atom import BeeAtom, KnowledgeAtom, AtomState
from hive.core.attraction import AttractionEngine, get_attraction_engine
from hive.core.memory_box import get_memory_box
from hive.core.memory_store import MemoryStore


@dataclass
class Cluster:
    """簇 - 原子的临时/永久组合"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    cluster_type: str = "temporary"  # task-oriented, gene-oriented, temporary
    members: Set[str] = field(default_factory=set)
    leader_id: Optional[str] = None
    goal: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "cluster_type": self.cluster_type,
            "members": list(self.members),
            "leader_id": self.leader_id,
            "goal": self.goal,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
        }


@dataclass
class SwarmMetrics:
    """蜂群指标"""
    generation: int = 0
    total_atoms: int = 0
    active_atoms: int = 0
    dormant_atoms: int = 0
    retired_atoms: int = 0
    dead_atoms: int = 0
    avg_fitness: float = 0.0
    max_fitness: float = 0.0
    min_fitness: float = 0.0
    total_connections: int = 0
    clusters_count: int = 0
    tasks_completed_since_last_check: int = 0
    tasks_failed_since_last_check: int = 0


@dataclass
class SwarmSnapshot:
    """蜂群检查点快照"""
    generation: int
    timestamp: float
    atoms: List[Dict[str, Any]]
    clusters: List[Dict[str, Any]]
    metrics: SwarmMetrics
    checkpoint_id: str = field(default_factory=lambda: str(uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "timestamp": self.timestamp,
            "atoms": self.atoms,
            "clusters": self.clusters,
            "metrics": asdict(self.metrics),
            "checkpoint_id": self.checkpoint_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwarmSnapshot':
        metrics_dict = data.get("metrics", {})
        metrics = SwarmMetrics(**metrics_dict) if isinstance(metrics_dict, dict) else metrics_dict
        return cls(
            generation=data["generation"],
            timestamp=data["timestamp"],
            atoms=data.get("atoms", []),
            clusters=data.get("clusters", []),
            metrics=metrics,
            checkpoint_id=data.get("checkpoint_id", str(uuid4())),
        )


class EternalSwarm:
    """万年蜂群 - 主类"""
    
    def __init__(
        self,
        swarm_id: Optional[str] = None,
        checkpoint_interval: int = 1000,
        initial_population: int = 100,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.swarm_id = swarm_id or str(uuid4())
        
        # 配置
        self.checkpoint_interval = checkpoint_interval
        self.initial_population = initial_population
        
        # 存储
        self._memory_store = memory_store or MemoryStore(
            store_name=f"eternal_swarm_{self.swarm_id[:8]}",
            auto_save=True,
        )
        
        # 吸引引擎
        self._attraction_engine = get_attraction_engine()
        
        # 状态
        self.generation: int = 0
        self.atoms: Dict[str, BeeAtom] = {}
        self.clusters: Dict[str, Cluster] = {}
        
        # 统计
        self.metrics = SwarmMetrics()
        self._metrics_history: List[SwarmMetrics] = []
        
        # 配置参数
        self.params = {
            "elite_count": 10,
            "mutation_probability": 0.1,
            "knowledge_decay_rate": 0.95,
            "elimination_threshold": 0.1,
            "max_age": 100,
            "active_threshold": 1000,
            "connection_threshold": 0.3,
        }
    
    def add_atom(self, atom: BeeAtom) -> bool:
        """添加原子"""
        if atom.id in self.atoms:
            return False
        self.atoms[atom.id] = atom
        return True
    
    def remove_atom(self, atom_id: str) -> Optional[BeeAtom]:
        """移除原子"""
        if atom_id not in self.atoms:
            return None
        
        atom = self.atoms.pop(atom_id)
        
        # 从所有连接中移除
        for other in self.atoms.values():
            other.remove_connection(atom_id)
        
        # 从所有簇中移除
        for cluster in self.clusters.values():
            if atom_id in cluster.members:
                cluster.members.remove(atom_id)
        
        return atom
    
    def get_active_atoms(self) -> List[BeeAtom]:
        """获取活跃原子"""
        return [a for a in self.atoms.values() if a.state == AtomState.ACTIVE]
    
    def get_dormant_atoms(self) -> List[BeeAtom]:
        """获取休眠原子"""
        return [a for a in self.atoms.values() if a.state == AtomState.DORMANT]
    
    def get_computing_set(self) -> List[BeeAtom]:
        """获取计算集合（采样活跃原子）"""
        active = self.get_active_atoms()
        threshold = self.params["active_threshold"]
        
        if len(active) <= threshold:
            return active
        
        # 采样
        return random.sample(active, threshold)
    
    def form_clusters(self) -> List[Cluster]:
        """自由结合形成簇"""
        computing_set = self.get_computing_set()
        
        if len(computing_set) < 2:
            return []
        
        # 计算所有原子对的吸引度
        new_clusters: List[Cluster] = []
        used_atoms: Set[str] = set()
        
        # 简单聚类：按吸引度分组
        for i, atom in enumerate(computing_set):
            if atom.id in used_atoms:
                continue
            
            # 创建新簇
            cluster = Cluster(
                name=f"Cluster_{self.generation}_{len(new_clusters)}",
                cluster_type="temporary",
            )
            cluster.members.add(atom.id)
            cluster.leader_id = atom.id
            used_atoms.add(atom.id)
            
            # 寻找强连接的原子
            for other in computing_set[i+1:]:
                if other.id in used_atoms:
                    continue
                
                attraction = self._attraction_engine.calculate_total_attraction(atom, other)
                
                if self._attraction_engine.should_connect(
                    atom, other, 
                    threshold=self.params["connection_threshold"]
                ):
                    # 建立双向连接
                    atom.add_connection(other.id)
                    other.add_connection(atom.id)
                    
                    # 加入簇
                    cluster.members.add(other.id)
                    used_atoms.add(other.id)
            
            # 只有多个成员的簇才有意义
            if len(cluster.members) >= 2:
                new_clusters.append(cluster)
                self.clusters[cluster.id] = cluster
        
        return new_clusters
    
    def select_elites(self, top_k: int = 10) -> List[BeeAtom]:
        """选择精英"""
        active = self.get_active_atoms()
        
        if not active:
            return []
        
        # 按适应度排序
        sorted_atoms = sorted(active, key=lambda a: a.get_avg_fitness(), reverse=True)
        
        return sorted_atoms[:top_k]
    
    def eliminate_weak(self) -> int:
        """淘汰弱势个体"""
        eliminated = 0
        
        for atom_id in list(self.atoms.keys()):
            atom = self.atoms[atom_id]
            
            if atom.should_be_eliminated(
                threshold=self.params["elimination_threshold"],
                max_age=self.params["max_age"],
            ):
                atom.state = AtomState.DEAD
                self.remove_atom(atom_id)
                eliminated += 1
        
        return eliminated
    
    def reproduce(self, elites: List[BeeAtom], count: int = 10) -> List[BeeAtom]:
        """繁殖新个体"""
        new_atoms: List[BeeAtom] = []
        
        if not elites:
            return new_atoms
        
        for _ in range(count):
            # 选择两个精英作为父母
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            
            # 创建新原子
            child = BeeAtom(
                generation=self.generation,
                born_at=self.generation,
                state=AtomState.ACTIVE,
            )
            
            # 继承能力（混合父母）
            all_caps = set(parent1.capabilities.keys()).union(set(parent2.capabilities.keys()))
            for cap in all_caps:
                val1 = parent1.capabilities.get(cap, 0.0)
                val2 = parent2.capabilities.get(cap, 0.0)
                
                # 混合 + 可能的变异
                child_val = (val1 + val2) / 2
                
                # 变异
                if random.random() < self.params["mutation_probability"]:
                    child_val += random.uniform(-0.2, 0.2)
                    child_val = max(0.0, min(1.0, child_val))
                
                child.capabilities[cap] = child_val
            
            # 继承基因信息
            child.blueprint_id = parent1.blueprint_id if random.random() < 0.5 else parent2.blueprint_id
            
            # 继承知识衰减
            if parent1.knowledge or parent2.knowledge:
                source = parent1.knowledge if random.random() < 0.5 else parent2.knowledge
                if source:
                    child.knowledge = KnowledgeAtom(
                        vectors=source.vectors.copy(),
                        summary=f"Inherited from generation {self.generation}",
                        source="inheritance",
                        importance=source.importance * 0.9,  # 稍微衰减
                        source_gen=self.generation,
                    )
            
            new_atoms.append(child)
            self.add_atom(child)
        
        return new_atoms
    
    def calculate_metrics(self) -> SwarmMetrics:
        """计算指标"""
        active = self.get_active_atoms()
        dormant = self.get_dormant_atoms()
        retired = [a for a in self.atoms.values() if a.state == AtomState.RETIRED]
        dead = [a for a in self.atoms.values() if a.state == AtomState.DEAD]
        
        fitness_values = [a.fitness for a in active if a.fitness > 0]
        
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0.0
        max_fitness = max(fitness_values) if fitness_values else 0.0
        min_fitness = min(fitness_values) if fitness_values else 0.0
        
        total_connections = sum(len(a.connections) for a in self.atoms.values())
        
        # 统计任务
        total_tasks_completed = sum(a.tasks_completed for a in self.atoms.values())
        total_tasks_failed = sum(a.tasks_failed for a in self.atoms.values())
        
        metrics = SwarmMetrics(
            generation=self.generation,
            total_atoms=len(self.atoms),
            active_atoms=len(active),
            dormant_atoms=len(dormant),
            retired_atoms=len(retired),
            dead_atoms=len(dead),
            avg_fitness=avg_fitness,
            max_fitness=max_fitness,
            min_fitness=min_fitness,
            total_connections=total_connections,
            clusters_count=len(self.clusters),
        )
        
        self.metrics = metrics
        self._metrics_history.append(metrics)
        
        # 限制历史大小
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-1000:]
        
        return metrics
    
    def create_checkpoint(self) -> SwarmSnapshot:
        """创建检查点"""
        metrics = self.calculate_metrics()
        
        snapshot = SwarmSnapshot(
            generation=self.generation,
            timestamp=time.time(),
            atoms=[a.to_dict() for a in self.atoms.values()],
            clusters=[c.to_dict() for c in self.clusters.values()],
            metrics=metrics,
        )
        
        # 保存到 MemoryBox
        try:
            mb = get_memory_box()
            key = f"eternal_swarm_checkpoint_{self.generation:08d}"
            mb[key] = snapshot.to_dict()
            
            # 同时保存到最近检查点
            mb[f"eternal_swarm_checkpoint_latest"] = snapshot.to_dict()
            
            # 保存到 MemoryStore
            if self._memory_store:
                self._memory_store.memorize(
                    content=f"Checkpoint at generation {self.generation}",
                    agent_id="eternal_swarm",
                    source="checkpoint",
                    category="system",
                    importance=1.0,
                )
            
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
        
        return snapshot
    
    def load_latest_checkpoint(self) -> Optional[SwarmSnapshot]:
        """加载最新检查点"""
        try:
            mb = get_memory_box()
            key = "eternal_swarm_checkpoint_latest"
            
            if key not in mb:
                return None
            
            data = mb[key]
            return SwarmSnapshot.from_dict(data)
            
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None
    
    def evolve_once(self) -> SwarmMetrics:
        """进化一代"""
        self.generation += 1
        
        # 1. 自由结合形成簇
        self.form_clusters()
        
        # 2. 选择精英
        elites = self.select_elites(top_k=self.params["elite_count"])
        
        # 3. 繁殖新个体
        if elites:
            # 基于精英数量决定繁殖数量
            reproduce_count = min(len(elites) * 2, 20)
            self.reproduce(elites, count=reproduce_count)
        
        # 4. 淘汰弱势
        eliminated = self.eliminate_weak()
        
        # 5. 计算指标
        metrics = self.calculate_metrics()
        
        # 6. 检查点保存
        if self.generation % self.checkpoint_interval == 0:
            self.create_checkpoint()
        
        return metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        metrics = self.calculate_metrics()
        
        return {
            "swarm_id": self.swarm_id,
            "generation": self.generation,
            "metrics": asdict(metrics),
            "params": self.params,
            "checkpoint_interval": self.checkpoint_interval,
        }


# 全局单例
_global_eternal_swarm: Optional[EternalSwarm] = None


def get_eternal_swarm() -> EternalSwarm:
    """获取全局万年蜂群实例"""
    global _global_eternal_swarm
    if _global_eternal_swarm is None:
        _global_eternal_swarm = EternalSwarm()
    return _global_eternal_swarm
```

---

## 📁 目标文件结构

```
hive/
├── core/
│   ├── bee_atom.py           # E1, E2, E3: 数据类
│   ├── attraction.py         # E4: 吸引规则
│   ├── swarm_eternal.py      # E5, E6, E7, E8, E9: Swarm 主类
│   └── ...
├── ministers/
│   └── eternal.py            # 万年繁衍控制器 (已有)
└── tests/
    └── test_eternal.py       # E10: 单元测试 (待创建)
```

---

## 🔗 与现有系统的集成点

| 现有模块 | 集成方式 |
|----------|----------|
| `SpeciesBlueprint` | BeeAtom 使用 blueprint_id 引用 |
| `MemoryStore` | 用于知识存储和检查点 |
| `MemoryBox` | 用于持久化检查点 |
| `CapabilityGapAnalyzer` | 用于能力评估和适应度计算 |
| `Cluster` (现有) | 与新的 Cluster 概念整合 |

---

## ✅ 验收标准

- [ ] `BeeAtom` 可序列化/反序列化
- [ ] 吸引规则计算正确
- [ ] 进化循环正常运行
- [ ] 检查点可保存/加载
- [ ] 单元测试通过
- [ ] 简单模拟运行 1000 代无崩溃
