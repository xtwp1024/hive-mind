# hive.core.swarm_communication

SwarmCommunication - 蜂群通信系统 实现蜂种之间的协作和集体智慧 增强功能 (Phase 1 增强): 1. 优先级队列 - 高优先级消息优先处理 2. 消息过期机制 - 防止过时消息堆积 3. 请求-响应模式 - send_and_wait 等待响应 功能: - Bee-to-Bee通信 - 集体决策 - 任务协商 - 知识广播

## 目录

- [类](#类)
  - [MessageEnvelope](#messageenvelope)
  - [MessagePriority](#messagepriority)
  - [MessageType](#messagetype)
  - [PriorityQueueItem](#priorityqueueitem)
  - [SwarmCommunication](#swarmcommunication)
  - [SwarmMessage](#swarmmessage)
  - [TaskBid](#taskbid)
  - [Vote](#vote)
- [函数](#函数)
  - [get_swarm_communication](#get_swarm_communication)

## 类

### MessageEnvelope

**@dataclass**

消息信封 - 用于队列传输和追踪

#### 属性

- `is_acknowledged`
- `read_count`

### MessagePriority

**Enum**

**继承:** `Enum`

消息优先级 (Phase 2 新增)

#### 属性

- `CRITICAL`
- `EMERGENCY`
- `HIGH`
- `LOW`
- `NORMAL`
- `name`
- `value`

### MessageType

**Enum**

**继承:** `Enum`

消息类型

#### 属性

- `ALERT`
- `BROADCAST_DISCOVERY`
- `BROADCAST_EMERGENCY`
- `BROADCAST_STATUS`
- `COLLABORATION_ACCEPT`
- `COLLABORATION_PROPOSAL`
- `COLLABORATION_REJECT`
- `DISCOVERY`
- `HEARTBEAT`
- `KNOWLEDGE`
- `KNOWLEDGE_QUERY`
- `KNOWLEDGE_SHARE`
- `REQUEST_APPROVAL`
- `REQUEST_ASSISTANCE`
- `REQUEST_INFORMATION`
- `RESPONSE`
- `RESPONSE_FAILURE`
- `RESPONSE_PARTIAL`
- `RESPONSE_SUCCESS`
- `STATUS`
- `TASK_BID`
- `TASK_REQUEST`
- `VOTE`
- `name`
- `value`

### PriorityQueueItem

**@dataclass**

优先级队列项

### SwarmCommunication

蜂群通信系统 实现蜂种之间的通信和协作 增强功能: 1. 优先级队列 - 高优先级消息优先处理 2. 消息过期机制 - 防止过时消息堆积 3. 请求-响应模式 - send_and_wait 等待响应 功能: 1. 消息广播和点对点通信 2. 集体决策（投票） 3. 任务协商和分配 4. 知识共享

#### 方法

```python
def broadcast(message: SwarmMessage)
```

广播消息

**参数:**

- `message`: 要广播的消息

```python
def broadcast_discovery(bee_id: str, discovery: Dict)
```

广播发现

**参数:**

- `bee_id`: 发现者ID
- `discovery`: 发现内容

```python
def collective_decision(question: str, options: List, timeout: int = 60) -> str
```

```python
def get_stats() -> Dict
```

获取通信统计信息

```python
def get_swarm_status() -> Dict
```

获取蜂群状态

**返回:** 蜂群状态

```python
def register_bee(bee_id: str, bee_instance: Any)
```

注册蜂到通信系统

**参数:**

- `bee_id`: 蜂ID
- `bee_instance`: 蜂实例

```python
def select_optimal_team(bids: List, task: Dict) -> List
```

选择最优团队

**参数:**

- `bids`: 投标列表
- `task`: 任务描述

**返回:** 选中的蜂ID列表

```python
def send_and_wait(message: SwarmMessage, timeout: float = 30.0) -> Tuple
```

发送消息并等待响应

**参数:**

- `message`: 要发送的消息
- `timeout`: 超时时间（秒）

**返回:** (是否成功, 响应消息或 None)

```python
def send_message(message: SwarmMessage)
```

发送点对点消息

**参数:**

- `message`: 要发送的消息

```python
def send_response(original_message: SwarmMessage, response_content: Dict, success: bool = True)
```

发送响应消息

**参数:**

- `original_message`: 原始请求消息
- `response_content`: 响应内容
- `success`: 是否成功

```python
def share_knowledge(bee_id: str, knowledge: Dict)
```

分享知识

**参数:**

- `bee_id`: 分享者ID
- `knowledge`: 知识内容

```python
def start()
```

启动通信系统

```python
def stop()
```

停止通信系统

```python
def subscribe(message_type: MessageType, callback: Callable)
```

订阅消息类型

**参数:**

- `message_type`: 消息类型
- `callback`: 回调函数

```python
def task_negotiation(task: Dict) -> List
```

任务协商

**参数:**

- `task`: 任务描述

**返回:** 投标列表

```python
def unregister_bee(bee_id: str)
```

注销蜂

### SwarmMessage

**@dataclass**

蜂群消息

#### 方法

```python
def get_priority_enum() -> MessagePriority
```

获取消息优先级枚举 (Phase 2 新增)

```python
def is_broadcast() -> bool
```

是否为广播消息

```python
def is_broadcast_type() -> bool
```

是否为广播类型消息 (Phase 2 新增)

```python
def is_expired() -> bool
```

检查消息是否过期

```python
def is_request_type() -> bool
```

是否为请求类型消息 (Phase 2 新增)

```python
def is_response_type() -> bool
```

是否为响应类型消息 (Phase 2 新增)

```python
def needs_response() -> bool
```

是否需要响应

```python
def set_priority_enum(priority: MessagePriority)
```

设置消息优先级 (Phase 2 新增)

#### 属性

- `correlation_id`
- `expires_at`
- `hop_count`
- `priority`
- `to_bee`

### TaskBid

**@dataclass**

任务投标

### Vote

**@dataclass**

投票

## 函数

### get_swarm_communication

```python
def get_swarm_communication() -> SwarmCommunication
```

获取全局蜂群通信实例（线程安全）

## 导出

```python
__all__ = ['SwarmCommunication', 'SwarmMessage', 'MessageType', 'TaskBid', 'Vote', 'get_swarm_communication']
```

---

*文档生成于: 2026-04-23T01:21:36.239495*
