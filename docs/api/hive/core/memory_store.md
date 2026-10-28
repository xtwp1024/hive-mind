# hive.core.memory_store

MemoryStore - 三层层次记忆仓库 设计灵感来源于 NevaMind-AI/memU（"记忆即文件系统"哲学）。 三层架构（Hierarchical Memory Architecture）： Resource  → 原始内容（对话/文档），可溯源（对标 memU Resource） Item      → 提取的具体事实/偏好/技能（对标 memU Item） Category  → 自动聚合的主题目录（对标 memU Category） 双模式检索： RAG  → TF-IDF 余弦相似度，毫秒级（对标 memU RAG embedding 检索） LLM  → 关键词精确匹配推理（简化版，对标 memU LLM 深度推理接口）

## 目录

- [类](#类)
  - [MemoryCategory](#memorycategory)
  - [MemoryItem](#memoryitem)
  - [MemoryStore](#memorystore)
  - [Resource](#resource)

## 类

### MemoryCategory

**@dataclass**

主题目录层（对标 memU Category / 文件夹）

#### 方法

```python
def from_dict(data: dict) -> MemoryCategory
```

```python
def to_dict() -> dict
```

#### 属性

- `agent_id`
- `summary`

### MemoryItem

**@dataclass**

事实提取层（对标 memU Item / 文件内容段落）

#### 方法

```python
def from_dict(data: dict) -> MemoryItem
```

```python
def to_dict() -> dict
```

#### 属性

- `agent_id`
- `category`
- `importance`
- `source_id`

### MemoryStore

三层层次记忆仓库。 模拟 memU 的"记忆即文件系统"哲学，无需外部数据库，完全 in-process。

#### 方法

```python
def browse(category: str, agent_id: str = global) -> dict
```

文件系统式记忆导航（对标 memU 目录浏览 / ls memory/）。 category="" 时列出所有顶层目录；否则列出该目录下的 Items。

```python
def clear_and_load() -> bool
```

清空当前数据并重新加载

```python
def get_persistence_stats() -> dict
```

获取持久化状态

```python
def load() -> bool
```

手动从持久化存储加载数据（会覆盖当前内存数据）

```python
def memorize(content: str, agent_id: str = global, modality: str = conversation, metadata: dict | None = None) -> dict
```

吸收新内容（对标 memU memorize()）。 流程：存入 Resource → 提取 Items → 自动归类 → 更新索引。

```python
def memory_stats() -> dict
```

```python
def retrieve_llm(query: str, agent_id: str = global, top_k: int = 3) -> list
```

LLM 深度推理检索（对标 memU retrieve(method='llm')）。 简化版：关键词精确匹配 + 分类权重提升。

```python
def retrieve_rag(query: str, agent_id: str = global, top_k: int = 5) -> list
```

RAG 快速召回（对标 memU retrieve(method='rag')）。 TF-IDF 余弦相似度，毫秒级，低成本。

```python
def save() -> bool
```

保存所有数据到持久化存储

#### 属性

- `CATEGORY_HINTS`

### Resource

**@dataclass**

原始内容层（对标 memU Resource / 文件挂载点）

#### 方法

```python
def from_dict(data: dict) -> Resource
```

```python
def to_dict() -> dict
```

#### 属性

- `agent_id`
- `modality`

---

*文档生成于: 2026-04-23T01:21:36.825196*
