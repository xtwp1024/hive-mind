# hive.core.dispatcher

Hive V2.0 Neural Orchestration Engine (Dispatcher) -------------------------------------------------- Implements a DAG-based Task Scheduler. Orchestrates `HiveActors` to execute complex workflows with dependencies. V2.5 Enhancements: - SSE (Server-Sent Events) Message Bus for real-time UI updates - Background worker process for non-blocking execution - Task queue for decoupled execution Concepts: - TaskNode: A single step in the workflow (e.g., "Summarize File"). - TaskGraph: A Directed Acyclic Graph of TaskNodes. - Dispatcher: The engine that traverses the graph and assigns nodes to compatible Actors. - MessageBus: Pub/Sub system for real-time event streaming. - TaskQueue: Background job queue.

## 目录

- [类](#类)
  - [HiveDispatcher](#hivedispatcher)
  - [MessageBus](#messagebus)
  - [SSEEventType](#sseeventtype)
  - [SSEMessage](#ssemessage)
  - [TaskGraph](#taskgraph)
  - [TaskNode](#tasknode)
  - [TaskQueue](#taskqueue)
  - [TaskStatus](#taskstatus)
- [函数](#函数)
  - [get_message_bus](#get_message_bus)
  - [get_task_queue](#get_task_queue)

## 类

### HiveDispatcher

The Central Nervous System. Manages a pool of Actors and executes TaskGraphs. V2.5 Enhancements: - Integrates with MessageBus for real-time SSE events - Publishes task/workflow events to UI

#### 方法

```python
async def execute_task(node: TaskNode)
```

Execute a single node

```python
async def run_workflow(graph: TaskGraph)
```

Main execution loop (事件驱动，无轮询）

```python
def set_message_bus(message_bus: MessageBus) -> None
```

设置消息总线

### MessageBus

SSE 消息总线 - 发布订阅系统 用于 UI 实时订阅任务流、集群状态等事件。

#### 方法

```python
def get_history(event_type: Optional = None, limit: int = 100) -> List
```

获取历史消息

```python
def get_stats() -> Dict
```

获取消息总线统计

```python
def publish(message: SSEMessage) -> None
```

发布消息到所有订阅者

```python
def publish_cluster_event(event_type: SSEEventType, cluster_id: str, cluster_info: Dict, extra: Optional = None) -> None
```

发布集群相关事件

```python
def publish_system_metrics(cpu_percent: float, memory_percent: float, active_bees: int, extra: Optional = None) -> None
```

发布系统指标

```python
def publish_task_event(event_type: SSEEventType, task_id: str, task_node: TaskNode, workflow_name: str, extra: Optional = None) -> None
```

发布任务相关事件

```python
def publish_workflow_event(event_type: SSEEventType, workflow_name: str, summary: Dict, extra: Optional = None) -> None
```

发布工作流相关事件

```python
def subscribe() -> Queue
```

订阅消息总线

**返回:** 消息队列，用于接收 SSE 流

```python
def unsubscribe(queue: Queue) -> None
```

取消订阅

### SSEEventType

**Enum**

**继承:** `Enum`

SSE 消息事件类型

#### 属性

- `BEE_STATUS`
- `CLUSTER_DISBANDED`
- `CLUSTER_TIMEOUT`
- `CLUSTER_UPDATED`
- `CUSTOM`
- `HEALTH_CHECK`
- `SYSTEM_METRICS`
- `TASK_COMPLETED`
- `TASK_FAILED`
- `TASK_SKIPPED`
- `TASK_STARTED`
- `WORKFLOW_COMPLETED`
- `WORKFLOW_FAILED`
- `WORKFLOW_STARTED`
- `name`
- `value`

### SSEMessage

**@dataclass**

SSE 消息数据结构

#### 方法

```python
def to_dict() -> Dict
```

Convert to dictionary

```python
def to_sse_format() -> str
```

Convert to SSE format string

### TaskGraph

Represents a workflow

#### 方法

```python
def add_dependency(child_id: str, parent_id: str)
```

```python
def add_node(node: TaskNode)
```

```python
def get_ready_nodes() -> List
```

Return nodes that are PENDING and have all dependencies COMPLETED. Nodes with any FAILED or SKIPPED parent are marked SKIPPED immediately.

```python
def is_complete() -> bool
```

### TaskNode

**@dataclass**

A single node in the execution graph

#### 方法

```python
def to_dict() -> Dict
```

Convert to dictionary for serialization

#### 属性

- `assigned_actor`
- `completed_at`
- `error`
- `result`
- `started_at`
- `status`

### TaskQueue

后台任务队列 - 任务解耦执行 使用后台线程执行任务，不阻塞主线程。

#### 方法

```python
def get_result(task_id: str) -> Optional
```

获取任务结果

```python
def get_stats() -> Dict
```

获取队列统计

```python
def start() -> None
```

启动后台工作线程

```python
def stop() -> None
```

停止后台工作线程

```python
def submit(task_id: str, coro, callback: Optional = None) -> bool
```

提交任务到队列

**参数:**

- `task_id`: 任务ID
- `coro`: 协程或可调用对象
- `callback`: 完成回调函数 signature: (task_id, result, error)

**返回:** 是否成功提交

### TaskStatus

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `COMPLETED`
- `FAILED`
- `PENDING`
- `RUNNING`
- `SKIPPED`
- `name`
- `value`

## 函数

### get_message_bus

```python
def get_message_bus() -> MessageBus
```

获取全局消息总线实例

### get_task_queue

```python
def get_task_queue() -> TaskQueue
```

获取全局任务队列实例

---

*文档生成于: 2026-04-23T01:21:36.780516*
