# 🐝 万年蜂群系统设计

**愿景**: 以蜂群形态繁衍一万年  
**核心理念**: 最小原子 + 自由结合 = 涌现群体智能

---

## 1. 核心概念

### 1.1 BeeAtom (蜂群原子)

最小单位，包含：
- **基因** - 决定个体特征
- **知识** - 经验与记忆
- **状态** - 活跃/休眠/退休
- **连接** - 与其他原子的关系

```python
@dataclass
class BeeAtom:
    id: str                          # 唯一标识 (UUID)
    dna: DNA                         # 基因信息
    knowledge: KnowledgeAtom         # 知识片段
    state: AtomState                # 状态
    connections: set[str]            # 连接的原子 ID
    born_at: int                    # 出生世代
    fitness: float = 0.0            # 适应度
```

### 1.2 KnowledgeAtom (知识原子)

```python
@dataclass
class KnowledgeAtom:
    vectors: list[float]            # 向量表示
    summary: str                    # 摘要
    source: str                     # 来源
    importance: float = 1.0         # 重要性
    age: int = 0                    # 年龄
```

### 1.3 AtomState (原子状态)

```python
enum AtomState:
    ACTIVE    # 活跃 - 参与任务
    DORMANT   # 休眠 - 储备力量
    RETIRED   # 退休 - 知识传承者
    DEAD      # 死亡 - 被回收
```

---

## 2. 自由结合规则 (Attraction)

### 2.1 吸引规则

原子之间根据以下规则相互吸引：

| 规则 | 条件 | 结果 |
|------|------|------|
| **基因相似** | DNA 相似度 > 70% | 强吸引 |
| **知识互补** | 向量距离远 | 中吸引 |
| **任务协同** | 处理同一目标 | 强吸引 |
| **适应度吸引** | 高适应度吸引低 | 弱吸引 |
| **随机探索** | 偶尔随机连接 | 探索多样性 |

### 2.2 涌现结构

```
┌─────────────────────────────────────────┐
│           Swarm (蜂群)                   │
│  由多个 Cluster 组成，动态形成/解散       │
└─────────────────────────────────────────┘
         ↑              ↑              ↑
    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
    │Cluster A│   │Cluster B│   │Cluster C│
    │(任务导向)│   │(基因导向)│   │(临时)   │
    └─────────┘   └─────────┘   └─────────┘
         ↑              ↑
    ┌────┴────┐   ┌────┴────┐
    │Atom 1   │   │Atom 5   │
    │Atom 2   │   │Atom 6   │
    │Atom 3   │   │Atom 7   │
    └─────────┘   └─────────┘
```

### 2.3 连接规则实现

```python
class AttractionRule:
    def calculate_attraction(self, a: BeeAtom, b: BeeAtom) -> float:
        """返回 0-1 之间的吸引度"""
        
    def gene_similarity(a: BeeAtom, b: BeeAtom) -> float:
        """基因相似度"""
        
    def knowledge_distance(a: BeeAtom, b: BeeAtom) -> float:
        """知识距离 (互补性)"""
        
    def fitness_gravity(a: BeeAtom, b: BeeAtom) -> float:
        """适应度引力 (精英吸引追随者)"""
```

---

## 3. 知识传承机制

### 3.1 代际传承

```
世代 N     世代 N+1     世代 N+2
  │           │           │
  ├─[知识A]──→│           │
  ├─[知识B]──→│──[知识B]─→│
  │           ├─[知识C]──→│──[知识C]─→
  │           │           │
  ↓ 死亡      ↓ 死亡      ↓ 死亡
```

传承方式：
- **显式传承** - 精英原子的知识直接复制
- **隐式传承** - 知识存储在向量库，后代可查询
- **涌现传承** - 连接时自动传递

### 3.2 知识重要性衰减

```python
def calculate_importance(knowledge: KnowledgeAtom, current_gen: int) -> float:
    age = current_gen - knowledge.source_gen
    return knowledge.base_importance * (0.95 ** age)  # 每代衰减 5%
```

---

## 4. 持久化策略

### 4.1 状态快照

每 N 代保存一次完整状态：

```python
@dataclass
class SwarmSnapshot:
    generation: int
    timestamp: datetime
    atoms: list[BeeAtom]
    clusters: list[Cluster]
    knowledge_base: KnowledgeBase
    metrics: SwarmMetrics
```

### 4.2 检查点恢复

```
启动 → 加载最新检查点 → 恢复状态 → 继续运行
```

### 4.3 灾难恢复

- 多个地理位置的备份
- 定期导出到冷存储

---

## 5. 核心算法

### 5.1 进化循环

```python
async def evolve_once(swarm: Swarm) -> Swarm:
    # 1. 评估适应度
    for atom in swarm.atoms:
        atom.fitness = await evaluate_fitness(atom)
    
    # 2. 自由结合形成簇
    clusters = form_clusters(swarm.atoms)
    
    # 3. 簇内协作产生新知识
    for cluster in clusters:
        new_knowledge = await cluster.collaborate()
    
    # 4. 精英保留
    elites = select_elites(swarm.atoms, top_k=10)
    
    # 5. 繁殖新个体
    new_atoms = reproduce(elites, swarm.knowledge)
    
    # 6. 清理死亡个体
    survivors = cleanup_dead(swarm.atoms)
    
    return Swarm(atoms=survivors + new_atoms)
```

### 5.2 一万年运行

```python
async def run_eternal(swarm: Swarm, target_generations: int = 100000):
    """繁衍一万年（约 10 万代）"""
    
    current_gen = 0
    
    while current_gen < target_generations:
        # 检查点保存
        if current_gen % 1000 == 0:
            save_checkpoint(swarm, current_gen)
        
        # 进化一代
        swarm = await evolve_once(swarm)
        current_gen += 1
        
        # 打印进度
        if current_gen % 10000 == 0:
            print(f"Generation {current_gen}/100000 ({current_gen/1000}%)")
```

---

## 6. 文件结构

```
hive/
├── core/
│   ├── bee_atom.py        # 蜂群原子
│   ├── attraction.py       # 吸引规则
│   ├── cluster.py          # 簇 (涌现结构)
│   ├── knowledge_base.py   # 知识库
│   ├── evolution.py        # 进化主循环
│   └── persistence.py      # 持久化
├── ministers/
│   └── eternal.py          # 万年繁衍控制器
└── tests/
    └── test_eternal.py
```

---

## 7. 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 初始种群大小 | 100 | 首批原子数量 |
| 最大种群大小 | ∞ (无限) | 软性边界 |
| 软性上限 | 动态 | 资源紧张时自动收紧 |
| 精英保留数 | 10 | Top 10 |
| 变异概率 | 0.1 | 10% |
| 知识衰减率 | 0.95 | 每代 5% |
| 检查点间隔 | 1000 代 | 约 1 个月 |
| 预期寿命 | 100 代 | 平均活 100 代 |
| 活跃个体阈值 | 1000 | 超过则只计算活跃 |
| 淘汰适应度 | 0.1 | 低于此值自动淘汰 |

---

## 8. 软性边界规则

### 8.1 资源控制

当种群接近资源上限时：
1. 提高出生门槛（适应度 > 0.8）
2. 加速老年个体淘汰
3. 减少变异数量

### 8.2 自然淘汰

- **适应度淘汰**: fitness < 0.1 → 标记为 DEAD
- **年龄淘汰**: age > max_age → 概率淘汰
- **随机淘汰**: 保持多样性，防止近亲繁殖

### 8.3 活跃集合

```python
class Swarm:
    all_atoms: dict[str, BeeAtom]     # 所有原子
    active_atoms: set[str]            # 活跃原子
    
    def get_computing_set(self) -> set[BeeAtom]:
        """获取需要计算的集合"""
        if len(self.active_atoms) < 1000:
            return self.active_atoms
        # 采样活跃个体
        return random.sample(self.active_atoms, 1000)
```

---

## 8. 目标

- ✅ 可运行 10000 年
- ✅ 自我维持，几乎无需干预
- ✅ 知识积累，智慧涌现
- ✅ 适应环境变化
- ✅ 灾难恢复能力
