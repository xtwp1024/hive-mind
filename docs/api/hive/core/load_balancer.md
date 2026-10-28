# hive.core.load_balancer

Load Balancer & Failover (负载均衡与故障转移) Phase 5: 分布式网络 - 高可用增强 核心职责: 1. 负载均衡策略 - 多种算法选择最优节点 2. 健康检查 - 持续监控节点健康状态 3. 故障检测 - 快速识别故障节点 4. 故障转移 - 自动迁移任务到健康节点 5. 重试机制 - 失败任务自动重试 6. 熔断机制 - 防止故障扩散 负载均衡策略: ┌─────────────────────────────────────────────────────────────┐ │  策略名称              │  描述                               │ ├─────────────────────────────────────────────────────────────┤ │  ROUND_ROBIN           │  轮询 (简单公平)                    │ │  WEIGHTED_ROUND_ROBIN  │  加权轮询 (根据能力权重)            │ │  LEAST_LOAD            │  最少负载 (选择负载最低的节点)       │ │  LEAST_CONNECTIONS     │  最少连接 (选择连接数最少的)        │ │  CONSISTENT_HASHING    │  一致性哈希 (相同任务到相同节点)    │ │  RANDOM                │  随机选择                           │ │  CAPABILITY_AWARE      │  能力感知 (根据任务选择合适节点)    │ └─────────────────────────────────────────────────────────────┘ 健康状态: ┌─────────────────────────────────────────────────────────────┐ │  状态        │  描述                    │  可分配任务       │ ├─────────────────────────────────────────────────────────────┤ │  HEALTHY     │  健康，正常工作          │  是              │ │  DEGRADED    │  性能下降，但可用        │  是 (低优先级)   │ │  UNHEALTHY   │  不健康                  │  否              │ │  FAILED      │  已失败                  │  否              │ │  DRAINING    │  正在排空 (准备下线)     │  否 (仅现有任务)  │ └─────────────────────────────────────────────────────────────┘

## 目录

- [类](#类)
  - [ConsistentHashRing](#consistenthashring)
  - [FailoverAction](#failoveraction)
  - [FailoverManager](#failovermanager)
  - [HealthCheckResult](#healthcheckresult)
  - [HealthChecker](#healthchecker)
  - [LoadBalanceStrategy](#loadbalancestrategy)
  - [LoadBalancer](#loadbalancer)
  - [LoadBalancerEngine](#loadbalancerengine)
  - [NodeHealthStatus](#nodehealthstatus)
  - [NodeMetrics](#nodemetrics)
  - [TaskAssignment](#taskassignment)
- [函数](#函数)
  - [get_load_balancer_engine](#get_load_balancer_engine)
  - [init_load_balancer_engine](#init_load_balancer_engine)

## 类

### ConsistentHashRing

一致性哈希环 用于实现一致性哈希负载均衡策略。

#### 方法

```python
def add_node(node_id: str) -> None
```

添加节点到哈希环

```python
def get_node(key: str) -> Optional
```

获取 key 对应的节点

```python
def get_nodes() -> List
```

获取所有节点

```python
def remove_node(node_id: str) -> None
```

从哈希环移除节点

### FailoverAction

**Enum**

**继承:** `str, Enum`

故障转移动作

#### 属性

- `CIRCUIT_BREAK`
- `FALLBACK`
- `REDIRECT`
- `RETRY`
- `name`
- `value`

### FailoverManager

故障转移管理器 处理节点故障和任务重试。

#### 方法

```python
def assign_task(task_id: str, required_capabilities: Optional = None, preferred_node: Optional = None, max_retries: Optional = None) -> Optional
```

分配任务到节点

**参数:**

- `task_id`: 任务ID
- `required_capabilities`: 要求的能力
- `preferred_node`: 首选节点
- `max_retries`: 最大重试次数 (覆盖默认)

**返回:** TaskAssignment|None - 任务分配

```python
def complete_task(assignment_id: str, result: Any = None) -> bool
```

标记任务完成

```python
def fail_task(assignment_id: str, error_message: str, can_retry: bool = True) -> Tuple
```

标记任务失败并尝试重试

**参数:**

- `assignment_id`: 分配ID
- `error_message`: 错误信息
- `can_retry`: 是否允许重试

**返回:** (success, new_assignment|None)

```python
def get_assignment(assignment_id: str) -> Optional
```

获取任务分配

```python
def get_assignment_by_task(task_id: str) -> List
```

获取任务的所有分配

```python
def get_stats() -> Dict
```

获取故障转移统计

```python
def is_circuit_open(node_id: str) -> bool
```

检查节点是否熔断

```python
def start_task(assignment_id: str) -> bool
```

标记任务开始

### HealthCheckResult

**@dataclass**

健康检查结果

#### 属性

- `response_time_ms`
- `status`

### HealthChecker

健康检查器 持续监控节点健康状态。

#### 方法

```python
def check_all_nodes() -> Dict
```

检查所有节点

```python
def check_node(node_id: str) -> HealthCheckResult
```

检查单个节点健康状态

**参数:**

- `node_id`: 节点ID

**返回:** HealthCheckResult - 检查结果

```python
def drain_node(node_id: str) -> bool
```

排空节点 (准备下线)

```python
def get_node_status(node_id: str) -> Optional
```

获取节点状态

```python
def get_stats() -> Dict
```

获取健康检查统计

```python
def register_custom_checker(node_id: str, checker: Callable) -> None
```

注册自定义检查器

```python
def register_node(node_id: str, initial_status: NodeHealthStatus = NodeHealthStatus.HEALTHY) -> None
```

注册节点进行健康检查

```python
def set_node_status(node_id: str, status: NodeHealthStatus) -> bool
```

手动设置节点状态

```python
def unregister_node(node_id: str) -> None
```

注销节点

### LoadBalanceStrategy

**Enum**

**继承:** `str, Enum`

负载均衡策略

#### 属性

- `CAPABILITY_AWARE`
- `CONSISTENT_HASHING`
- `LEAST_CONNECTIONS`
- `LEAST_LOAD`
- `RANDOM`
- `ROUND_ROBIN`
- `WEIGHTED_ROUND_ROBIN`
- `name`
- `value`

### LoadBalancer

负载均衡器 根据策略选择最优节点。

#### 方法

```python
def add_node(node_id: str, weight: int = 1, capabilities: Optional = None) -> bool
```

添加节点

**参数:**

- `node_id`: 节点ID
- `weight`: 权重 (用于加权轮询)
- `capabilities`: 能力列表

**返回:** bool - 是否成功

```python
def get_all_metrics() -> Dict
```

获取所有节点指标

```python
def get_available_nodes(required_capabilities: Optional = None) -> List
```

获取可用节点ID列表

```python
def get_node_metrics(node_id: str) -> Optional
```

获取节点指标

```python
def get_stats() -> Dict
```

获取负载均衡器统计

```python
def record_failure(node_id: str) -> None
```

记录失败

```python
def record_success(node_id: str, response_time_ms: float) -> None
```

记录成功

```python
def remove_node(node_id: str) -> bool
```

移除节点

```python
def select_node(task_id: Optional = None, required_capabilities: Optional = None, strategy: Optional = None) -> Optional
```

选择节点

**参数:**

- `task_id`: 任务ID (用于一致性哈希)
- `required_capabilities`: 要求的能力列表
- `strategy`: 覆盖默认策略

**返回:** str|None - 选中的节点ID

```python
def update_node_metrics(node_id: str, metrics) -> bool
```

更新节点指标

### LoadBalancerEngine

负载均衡引擎 整合负载均衡、健康检查、故障转移的统一接口。

#### 方法

```python
def add_node(node_id: str, weight: int = 1, capabilities: Optional = None, initial_status: NodeHealthStatus = NodeHealthStatus.HEALTHY) -> bool
```

添加节点

```python
def drain_node(node_id: str) -> bool
```

排空节点

```python
def execute_with_failover(task_id: str, executor: Callable, required_capabilities: Optional = None) -> Tuple
```

执行任务并自动处理故障转移

**参数:**

- `task_id`: 任务ID
- `executor`: 执行函数 (node_id) -> (success, result, error_message)
- `required_capabilities`: 要求的能力

**返回:** (success, result, last_assignment)

```python
def get_stats() -> Dict
```

获取完整统计

```python
def remove_node(node_id: str) -> bool
```

移除节点

```python
def select_and_assign(task_id: str, required_capabilities: Optional = None) -> Optional
```

选择节点并分配任务

```python
def start_health_monitoring() -> None
```

启动健康监控

```python
def stop_health_monitoring() -> None
```

停止健康监控

### NodeHealthStatus

**Enum**

**继承:** `str, Enum`

节点健康状态

#### 属性

- `DEGRADED`
- `DRAINING`
- `FAILED`
- `HEALTHY`
- `UNHEALTHY`
- `name`
- `value`

### NodeMetrics

**@dataclass**

节点指标

#### 方法

```python
def can_accept_task() -> bool
```

检查节点是否能接受新任务

```python
def from_dict(data: Dict) -> NodeMetrics
```

```python
def is_available() -> bool
```

检查节点是否可用

```python
def to_dict() -> Dict
```

#### 属性

- `connection_count`
- `cpu_usage`
- `failure_count`
- `health_status`
- `last_health_check`
- `load_factor`
- `memory_usage`
- `pending_tasks`
- `response_time_ms`
- `success_count`
- `success_rate`
- `weight`

### TaskAssignment

**@dataclass**

任务分配记录

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `completed_at`
- `error_message`
- `failed_at`
- `max_retries`
- `result`
- `retry_count`
- `started_at`
- `status`

## 函数

### get_load_balancer_engine

```python
def get_load_balancer_engine() -> LoadBalancerEngine
```

获取负载均衡引擎单例

### init_load_balancer_engine

```python
def init_load_balancer_engine(kwargs) -> LoadBalancerEngine
```

初始化负载均衡引擎

---

*文档生成于: 2026-04-23T01:21:36.755552*
