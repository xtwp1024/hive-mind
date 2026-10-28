# hive.core.hive_federation

Hive Federation (蜂巢联邦) Phase 5 P1: 多实例协作 多个 Hive 实例互联互通，形成分布式蜂群网络。 核心职责: 1. 节点注册与发现 — 多 Hive 实例自动发现彼此 2. 知识广播 — 优秀基因/策略在节点间同步 3. 任务委托 — 将任务委托给有能力空闲的节点执行 4. 状态同步 — 节点状态 (active/idle/overloaded) 全网可见 5. 联邦负载均衡 — 自动将任务路由到负载最低的节点 架构: ┌─────────────┐     ┌─────────────┐     ┌─────────────┐ │  Hive-A    │◄───►│  Hive-B    │◄───►│  Hive-C    │ │ (节点A)    │     │ (节点B)    │     │ (节点C)    │ └─────────────┘     └─────────────┘     └─────────────┘ ▲                   ▲                   ▲ │    ┌──────────────────────────────┘ │    │  知识广播 / 任务委托 / 状态同步 ▼    ▼ ┌─────────────────────────────────────────┐ │         Hive Federation (联邦中枢)       │ │  - 节点注册表 (谁在线?)                   │ │  - 知识库同步 (学到的知识共享)             │ │  - 任务路由 (委托到最适合的节点)           │ │  - 健康检查 (心跳检测节点存活性)            │ └─────────────────────────────────────────┘ 使用方式: federation = HiveFederation(node_id="hive-1", port=8765) federation.start() federation.broadcast_knowledge({"type": "strategy", "content": "..."}) result = federation.delegate_task(task, target_capabilities=["vision"]) federation.stop()

## 目录

- [类](#类)
  - [HiveFederation](#hivefederation)
  - [HiveNode](#hivenode)
- [函数](#函数)
  - [get_federation](#get_federation)
  - [init_federation](#init_federation)

## 类

### HiveFederation

蜂巢联邦核心 Phase 5 P1: 多 Hive 实例联邦网络

#### 方法

```python
def broadcast_knowledge(knowledge: dict, exclude_self: bool = True) -> int
```

广播知识到所有节点

**参数:**

- `knowledge`: 要广播的知识 dict
- `exclude_self`: 是否排除自己

**返回:** 广播到的节点数量

```python
def delegate_task(task: dict, target_capabilities: Optional = None, target_node_id: Optional = None, timeout: float = 60.0) -> Optional
```

将任务委托给其他节点

**参数:**

- `task`: 任务描述 dict
- `target_capabilities`: 所需能力列表 (如 ["vision", "audio"])
- `target_node_id`: 指定节点ID (优先于此参数)
- `timeout`: 超时时间 (秒)

**返回:** 任务结果 dict，失败返回 None

```python
def deregister_node(node_id: str)
```

从联邦注销节点

```python
def get_all_nodes() -> list
```

获取所有活动节点

```python
def get_best_node(capabilities: list) -> Optional
```

获取最适合的节点 (负载最低且具备能力)

```python
def get_knowledge_base() -> list
```

获取本地知识库

```python
def get_node(node_id: str) -> Optional
```

获取节点信息

```python
def get_nodes_by_capability(capability: str) -> list
```

获取具有特定能力的节点

```python
def get_recent_knowledge(limit: int = 50) -> list
```

获取最近的知识

```python
def get_status() -> dict
```

获取联邦状态

```python
def on_task_result(task_id: str, result: dict)
```

接收任务结果 (由 HTTP 处理器调用)

```python
def register_node(node: HiveNode) -> bool
```

注册节点到联邦

```python
def set_knowledge_callback(callback: Optional)
```

设置知识同步回调 (Phase 5 P2) callback(msg_type, data) 会被调用: - msg_type="broadcast": data 是 {"knowledge": ..., "origin": ...} - msg_type="push": data 是 KnowledgeEntry dict

```python
def start()
```

启动联邦

```python
def stop()
```

停止联邦

```python
def update_status(status: str, load_factor: float = 0.0, capabilities: Optional = None)
```

更新本节点状态

#### 属性

- `MSG_KNOWLEDGE_BROADCAST`
- `MSG_NODE_DEREGISTER`
- `MSG_NODE_HEARTBEAT`
- `MSG_NODE_REGISTER`
- `MSG_SYNC_REQUEST`
- `MSG_SYNC_RESPONSE`
- `MSG_TASK_DELEGATE`
- `MSG_TASK_RESULT`

### HiveNode

**@dataclass**

蜂巢节点描述

#### 方法

```python
def from_dict(data: dict) -> HiveNode
```

```python
def get_url(path: str) -> str
```

```python
def is_alive(timeout: float = 30.0) -> bool
```

检查节点是否存活

```python
def to_dict() -> dict
```

#### 属性

- `load_factor`
- `status`
- `version`

## 函数

### get_federation

```python
def get_federation() -> HiveFederation
```

获取蜂巢联邦单例

### init_federation

```python
def init_federation(kwargs) -> HiveFederation
```

初始化蜂巢联邦

---

*文档生成于: 2026-04-23T01:21:36.626940*
