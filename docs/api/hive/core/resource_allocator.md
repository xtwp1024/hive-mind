# hive.core.resource_allocator

Dynamic Resource Allocation - 动态资源分配系统 Phase 3: 自主进化能力 基于适应度和优先级动态分配系统资源。 核心功能: 1. 资源配额管理 - 为每个蜂种分配资源配额 2. 动态调整 - 根据适应度实时调整配额 3. 优先级调度 - 高优先级任务优先获得资源 4. 资源监控 - 实时监控资源使用情况 5. 过载保护 - 防止系统资源耗尽 资源类型: - CPU 时间片 - 内存配额 - 并发任务数 - API 调用限额 - 网络带宽

## 目录

- [类](#类)
  - [AllocationDecision](#allocationdecision)
  - [AllocationPriority](#allocationpriority)
  - [AllocationRecord](#allocationrecord)
  - [ConsumerProfile](#consumerprofile)
  - [DynamicResourceAllocator](#dynamicresourceallocator)
  - [ResourceQuota](#resourcequota)
  - [ResourceType](#resourcetype)
- [函数](#函数)
  - [get_resource_allocator](#get_resource_allocator)

## 类

### AllocationDecision

**@dataclass**

分配决策

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `estimated_wait_seconds`
- `granted_amount`
- `reason`
- `wait_required`

### AllocationPriority

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `BACKGROUND`
- `CRITICAL`
- `HIGH`
- `LOW`
- `NORMAL`
- `name`
- `value`

### AllocationRecord

**@dataclass**

分配记录

#### 属性

- `duration_seconds`
- `granted_at`
- `preemption_reason`
- `priority`
- `released_at`
- `was_preempted`

### ConsumerProfile

**@dataclass**

消费者配置文件

#### 方法

```python
def current_priority() -> AllocationPriority
```

根据适应度计算当前优先级

```python
def success_rate() -> float
```

```python
def to_dict() -> Dict
```

```python
def update_fitness(new_score: float)
```

更新适应度分数

#### 属性

- `base_priority`
- `error_count`
- `fitness_score`
- `success_count`
- `task_count`

### DynamicResourceAllocator

动态资源分配器 Phase 3 核心组件。基于适应度和优先级动态分配系统资源。 主要功能: 1. 消费者管理 - 注册和管理资源消费者 2. 配额管理 - 为每个消费者分配资源配额 3. 动态调整 - 根据适应度实时调整配额 4. 分配决策 - 处理资源请求并做出分配决策 5. 资源监控 - 实时监控资源使用情况 6. 过载保护 - 防止系统资源耗尽 分配策略: - 高适应度消费者获得更多配额 - 高优先级请求优先处理 - 支持资源抢占（低优先级让给高优先级） - 支持突发资源使用（burst mode） 使用方式: allocator = DynamicResourceAllocator() allocator.register_consumer("bee_001", "miner_bee") decision = allocator.request_resource("bee_001", ResourceType.CPU, 1.0)

#### 方法

```python
def get_consumer(consumer_id: str) -> Optional
```

获取消费者配置文件

```python
def get_consumer_stats(consumer_id: str) -> Optional
```

获取消费者统计

```python
def get_system_status() -> Dict
```

获取系统状态

```python
def get_top_consumers(resource_type: ResourceType, count: int = 10) -> List
```

获取资源使用最多的消费者

```python
def register_consumer(consumer_id: str, consumer_type: str, base_priority: AllocationPriority = AllocationPriority.NORMAL, initial_quotas: Optional = None) -> ConsumerProfile
```

注册资源消费者

**参数:**

- `consumer_id`: 消费者唯一标识
- `consumer_type`: 消费者类型（如 bee 类型）
- `base_priority`: 基础优先级
- `initial_quotas`: 初始配额配置

**返回:** ConsumerProfile - 消费者配置文件

```python
def release_allocation(record_id: str) -> bool
```

释放资源分配

```python
def request_resource(consumer_id: str, resource_type: ResourceType, amount: float, priority: Optional = None, duration_seconds: Optional = None, metadata: Optional = None) -> AllocationDecision
```

请求资源

**参数:**

- `consumer_id`: 消费者 ID
- `resource_type`: 资源类型
- `amount`: 请求数量
- `priority`: 请求优先级（覆盖消费者优先级）
- `duration_seconds`: 预计使用时长
- `metadata`: 附加元数据

**返回:** AllocationDecision - 分配决策

```python
def start_monitoring()
```

启动监控线程

```python
def stop_monitoring()
```

停止监控线程

```python
def unregister_consumer(consumer_id: str) -> bool
```

注销消费者

```python
def update_consumer_fitness(consumer_id: str, fitness_score: float) -> bool
```

更新消费者适应度

```python
def update_consumer_task_result(consumer_id: str, success: bool, fitness_delta: Optional = None) -> bool
```

更新消费者任务结果

#### 属性

- `FITNESS_ADJUSTMENT_FACTOR`
- `PRIORITY_WEIGHTS`

### ResourceQuota

**@dataclass**

资源配额

#### 方法

```python
def available() -> float
```

```python
def effective_limit() -> float
```

```python
def is_over_hard_limit() -> bool
```

```python
def is_over_soft_limit() -> bool
```

```python
def to_dict() -> Dict
```

#### 属性

- `burst_allowed`
- `burst_limit`
- `current_usage`
- `guaranteed`
- `reserved`

### ResourceType

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `API_CALLS`
- `CONCURRENCY`
- `CPU`
- `MEMORY`
- `NETWORK`
- `STORAGE`
- `name`
- `value`

## 函数

### get_resource_allocator

```python
def get_resource_allocator() -> DynamicResourceAllocator
```

获取资源分配器单例

---

*文档生成于: 2026-04-23T01:21:36.427591*
