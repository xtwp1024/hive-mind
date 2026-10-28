# hive.core.conflict_resolution

Conflict Resolution (冲突解决策略) Phase 5: 分布式网络 - 知识同步增强 核心职责: 1. 知识版本冲突检测 - 检测同时更新导致的冲突 2. 自动冲突解决 - 基于规则的自动合并 3. 手动冲突标记 - 需要人工干预的冲突 4. 冲突日志记录 - 完整的冲突历史 5. 回滚机制 - 恢复到冲突前状态 冲突类型: ┌─────────────────────────────────────────────────────────────┐ │  冲突类型              │  严重程度  │  解决策略               │ ├─────────────────────────────────────────────────────────────┤ │  版本冲突 (Version)   │  中       │  时间戳优先/版本号优先  │ │  内容冲突 (Content)   │  高       │  自动合并/标记待审核    │ │  来源冲突 (Source)    │  中       │  来源优先级              │ │  依赖冲突 (Dependency)│  高       │  递归解析                │ └─────────────────────────────────────────────────────────────┘ 解决策略: ┌─────────────────────────────────────────────────────────────┐ │  策略名称              │  描述                               │ ├─────────────────────────────────────────────────────────────┤ │  LAST_WRITE_WINS       │  最后写入优先 (时间戳)             │ │  HIGHER_VERSION        │  高版本优先                         │ │  MERGE_AUTO            │  尝试自动合并                       │ │  SOURCE_PRIORITY       │  来源优先级 (可信节点优先)          │ │  VOTE_RESOLUTION       │  投票决议 (多节点投票)             │ │  MANUAL_REVIEW         │  标记待人工审核                     │ └─────────────────────────────────────────────────────────────┘

## 目录

- [类](#类)
  - [ConflictDetector](#conflictdetector)
  - [ConflictLog](#conflictlog)
  - [ConflictRecord](#conflictrecord)
  - [ConflictResolutionEngine](#conflictresolutionengine)
  - [ConflictResolver](#conflictresolver)
  - [ConflictSeverity](#conflictseverity)
  - [ConflictType](#conflicttype)
  - [KnowledgeVersion](#knowledgeversion)
  - [ResolutionStatus](#resolutionstatus)
  - [ResolutionStrategy](#resolutionstrategy)
- [函数](#函数)
  - [get_conflict_engine](#get_conflict_engine)
  - [init_conflict_engine](#init_conflict_engine)

## 类

### ConflictDetector

冲突检测器 检测知识更新时的各种冲突。

#### 方法

```python
def detect_conflict(local_version: KnowledgeVersion, remote_version: KnowledgeVersion) -> Optional
```

检测两个版本之间的冲突

**参数:**

- `local_version`: 本地版本
- `remote_version`: 远程版本

**返回:** (ConflictType, ConflictSeverity, details) 或 None

### ConflictLog

冲突日志 记录所有冲突历史，支持查询和统计。

#### 方法

```python
def get_by_entry(entry_id: str) -> List
```

获取某个条目的所有冲突

```python
def get_by_status(status: ResolutionStatus) -> List
```

获取特定状态的冲突

```python
def get_conflict(conflict_id: str) -> Optional
```

获取冲突记录

```python
def get_pending_review() -> List
```

获取待审核的冲突

```python
def get_stats() -> Dict
```

获取统计信息

```python
def log_conflict(conflict: ConflictRecord) -> None
```

记录冲突

```python
def update_conflict(conflict_id: str, updates) -> bool
```

更新冲突记录

### ConflictRecord

**@dataclass**

冲突记录

#### 方法

```python
def from_dict(data: Dict) -> ConflictRecord
```

```python
def to_dict() -> Dict
```

#### 属性

- `local_version`
- `remote_version`
- `resolution_status`
- `resolution_strategy`
- `resolved_at`
- `resolved_by`
- `resolved_version`

### ConflictResolutionEngine

冲突解决引擎 整合检测器、解决器、日志的统一接口。

#### 方法

```python
def detect_and_resolve(local_version: KnowledgeVersion, remote_version: KnowledgeVersion, strategy: Optional = None) -> Tuple
```

检测并解决冲突

**参数:**

- `local_version`: 本地版本
- `remote_version`: 远程版本
- `strategy`: 解决策略

**返回:** (ConflictRecord|None, resolved_version|None) - 如果没有冲突: (None, remote_version) - 如果有冲突: (ConflictRecord, resolved_version|None)

```python
def get_pending_conflicts() -> List
```

获取待审核的冲突

```python
def get_stats() -> Dict
```

获取统计信息

```python
def resolve_manually(conflict_id: str, chosen_version: KnowledgeVersion, resolved_by: str = manual) -> bool
```

手动解决冲突

**参数:**

- `conflict_id`: 冲突ID
- `chosen_version`: 选择的版本
- `resolved_by`: 解决者标识

**返回:** bool - 是否成功

```python
def set_source_priority(source: str, priority: int) -> None
```

设置来源优先级

### ConflictResolver

冲突解决器 基于策略自动解决冲突。

#### 方法

```python
def register_merge_callback(content_type: str, callback: Callable) -> None
```

注册自定义合并回调

```python
def resolve(conflict: ConflictRecord, strategy: Optional = None) -> ConflictRecord
```

解决冲突

**参数:**

- `conflict`: 冲突记录
- `strategy`: 解决策略 (默认使用 default_strategy)

**返回:** ConflictRecord - 更新后的冲突记录

```python
def set_source_priority(source: str, priority: int) -> None
```

设置来源优先级

### ConflictSeverity

**Enum**

**继承:** `str, Enum`

冲突严重程度

#### 属性

- `CRITICAL`
- `HIGH`
- `LOW`
- `MEDIUM`
- `name`
- `value`

### ConflictType

**Enum**

**继承:** `str, Enum`

冲突类型

#### 属性

- `CONTENT`
- `DEPENDENCY`
- `METADATA`
- `SOURCE`
- `UNKNOWN`
- `VERSION`
- `name`
- `value`

### KnowledgeVersion

**@dataclass**

知识版本

#### 方法

```python
def from_dict(data: Dict) -> KnowledgeVersion
```

```python
def to_dict() -> Dict
```

#### 属性

- `content`

### ResolutionStatus

**Enum**

**继承:** `str, Enum`

解决状态

#### 属性

- `AUTO_RESOLVED`
- `DETECTED`
- `PENDING_REVIEW`
- `RESOLVED`
- `REVERTED`
- `name`
- `value`

### ResolutionStrategy

**Enum**

**继承:** `str, Enum`

解决策略

#### 属性

- `HIGHER_VERSION`
- `LAST_WRITE_WINS`
- `MANUAL_REVIEW`
- `MERGE_AUTO`
- `SOURCE_PRIORITY`
- `VOTE_RESOLUTION`
- `name`
- `value`

## 函数

### get_conflict_engine

```python
def get_conflict_engine() -> ConflictResolutionEngine
```

获取冲突解决引擎单例

### init_conflict_engine

```python
def init_conflict_engine(kwargs) -> ConflictResolutionEngine
```

初始化冲突解决引擎

---

*文档生成于: 2026-04-23T01:21:36.709187*
