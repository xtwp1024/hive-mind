# hive.core.knowledge_sync

Knowledge Sync (知识同步) Phase 5 P2: 全球知识共享 任意蜂群学到的知识可被其他蜂群使用。 核心职责: 1. 增量同步 — 只同步新增知识，避免重复传输 2. 知识去重 — 基于内容hash去重，避免存储冗余 3. 优先级同步 — 重要的知识优先同步 4. 知识版本管理 — 知识的创建/更新/失效 5. 联邦知识路由 — 根据知识类型路由到合适的节点 与 HiveFederation 集成: - 监听知识广播，自动接收同步 - 主动从其他节点拉取缺失知识 - 推送本节点新知识到联邦 使用方式: sync = KnowledgeSync(federation) sync.start() sync.push_knowledge({"type": "pattern", "content": "...", "priority": "high"}) result = sync.pull_knowledge(filter={"type": "pattern"})

## 目录

- [类](#类)
  - [KnowledgeEntry](#knowledgeentry)
  - [KnowledgeSync](#knowledgesync)
- [函数](#函数)
  - [get_knowledge_sync](#get_knowledge_sync)
  - [init_knowledge_sync](#init_knowledge_sync)

## 类

### KnowledgeEntry

**@dataclass**

知识条目

#### 方法

```python
def compute_hash(content: Any) -> str
```

计算内容hash

```python
def from_dict(data: dict) -> KnowledgeEntry
```

```python
def to_dict() -> dict
```

#### 属性

- `local_timestamp`
- `priority`
- `refs`
- `source_node`
- `source_timestamp`
- `version`
- `失效`

### KnowledgeSync

知识同步器 Phase 5 P2: 全球知识共享

#### 方法

```python
def get(entry_id: str) -> Optional
```

获取知识条目

```python
def get_status() -> dict
```

获取同步状态

```python
def increment_ref(entry_id: str)
```

增加引用计数

```python
def invalidate(entry_id: str, reason: str)
```

标记知识为失效

```python
def push_knowledge(content: Any, knowledge_type: str = insight, priority: str = normal, tags: Optional = None, allow_duplicate: bool = False) -> Optional
```

推送知识到联邦

**参数:**

- `content`: 知识内容
- `knowledge_type`: 知识类型 (pattern/strategy/code/insight/lesson)
- `priority`: 优先级 (critical/high/normal/low)
- `tags`: 标签列表
- `allow_duplicate`: 是否允许重复

**返回:** 知识ID，失败返回 None

```python
def query(knowledge_type: Optional = None, tags: Optional = None, min_priority: Optional = None, keyword: Optional = None, limit: int = 50) -> list
```

查询知识

**参数:**

- `knowledge_type`: 按类型过滤
- `tags`: 按标签过滤 (AND)
- `min_priority`: 最低优先级
- `keyword`: 关键词 (模糊匹配 content)
- `limit`: 返回条数限制

**返回:** 匹配的知识条目列表

```python
def start()
```

启动知识同步

```python
def stop()
```

停止知识同步

#### 属性

- `PRIORITY_CRITICAL`
- `PRIORITY_HIGH`
- `PRIORITY_LOW`
- `PRIORITY_NORMAL`
- `TYPE_CODE`
- `TYPE_INSIGHT`
- `TYPE_LESSON`
- `TYPE_PATTERN`
- `TYPE_STRATEGY`

## 函数

### get_knowledge_sync

```python
def get_knowledge_sync() -> Optional
```

获取知识同步器单例

### init_knowledge_sync

```python
def init_knowledge_sync(federation, kwargs) -> KnowledgeSync
```

初始化知识同步器

---

*文档生成于: 2026-04-23T01:21:36.649212*
