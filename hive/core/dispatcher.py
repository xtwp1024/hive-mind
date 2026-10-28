
"""
Hive V2.0 Neural Orchestration Engine (Dispatcher)
--------------------------------------------------
Implements a DAG-based Task Scheduler.
Orchestrates `HiveActors` to execute complex workflows with dependencies.

V2.5 Enhancements:
- SSE (Server-Sent Events) Message Bus for real-time UI updates
- Background worker process for non-blocking execution
- Task queue for decoupled execution

Concepts:
- TaskNode: A single step in the workflow (e.g., "Summarize File").
- TaskGraph: A Directed Acyclic Graph of TaskNodes.
- Dispatcher: The engine that traverses the graph and assigns nodes to compatible Actors.
- MessageBus: Pub/Sub system for real-time event streaming.
- TaskQueue: Background job queue.
"""

import asyncio
import structlog
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import time
import threading
from queue import Queue, Empty
from datetime import datetime
import json

from hive.core.component import HiveActor

logger = structlog.get_logger()

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TaskNode:
    """A single node in the execution graph"""
    id: str
    action: str  # The action verb (e.g., "read_file", "run_shell")
    params: Dict[str, Any]
    dependencies: Set[str] = field(default_factory=set) # IDs of parent tasks
    required_capabilities: List[str] = field(default_factory=list)
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    assigned_actor: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "action": self.action,
            "params": {k: str(v) for k, v in self.params.items()},
            "dependencies": list(self.dependencies),
            "required_capabilities": self.required_capabilities,
            "status": self.status.value,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "assigned_actor": self.assigned_actor,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.completed_at - self.started_at if (self.completed_at and self.started_at) else None,
        }


class SSEEventType(Enum):
    """SSE 消息事件类型"""
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_SKIPPED = "task_skipped"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    CLUSTER_TIMEOUT = "cluster_timeout"
    CLUSTER_DISBANDED = "cluster_disbanded"
    CLUSTER_UPDATED = "cluster_updated"
    BEE_STATUS = "bee_status"
    SYSTEM_METRICS = "system_metrics"
    HEALTH_CHECK = "health_check"
    CUSTOM = "custom"


@dataclass
class SSEMessage:
    """SSE 消息数据结构"""
    event_type: SSEEventType
    data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    def to_sse_format(self) -> str:
        """Convert to SSE format string"""
        event_str = f"event: {self.event_type.value}\n"
        id_str = f"id: {self.id}\n"
        data_str = f"data: {json.dumps(self.data, default=str)}\n\n"
        return event_str + id_str + data_str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
        }


class MessageBus:
    """
    SSE 消息总线 - 发布订阅系统
    
    用于 UI 实时订阅任务流、集群状态等事件。
    """
    
    def __init__(self, max_history: int = 1000):
        self._subscribers: List[Queue] = []
        self._history: List[SSEMessage] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self.logger = logger.bind(component="MessageBus")
        
    def subscribe(self) -> Queue:
        """
        订阅消息总线
        
        Returns:
            消息队列，用于接收 SSE 流
        """
        queue: Queue = Queue()
        with self._lock:
            self._subscribers.append(queue)
            # 发送历史消息给新订阅者
            for msg in self._history[-100:]:  # 最多发送最近100条
                queue.put(msg)
        self.logger.info("New subscriber connected", total_subscribers=len(self._subscribers))
        return queue
    
    def unsubscribe(self, queue: Queue) -> None:
        """取消订阅"""
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)
        self.logger.info("Subscriber disconnected", remaining_subscribers=len(self._subscribers))
    
    def publish(self, message: SSEMessage) -> None:
        """发布消息到所有订阅者"""
        with self._lock:
            # 添加到历史
            self._history.append(message)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            
            # 发布到所有订阅者
            for queue in self._subscribers:
                try:
                    queue.put_nowait(message)
                except Exception as e:
                    self.logger.warning("Failed to publish to subscriber", error=str(e))
        
        self.logger.debug("Message published", event_type=message.event_type.value)
    
    def publish_task_event(
        self,
        event_type: SSEEventType,
        task_id: str,
        task_node: TaskNode,
        workflow_name: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """发布任务相关事件"""
        data = {
            "task_id": task_id,
            "task": task_node.to_dict(),
            "workflow_name": workflow_name,
            "timestamp": time.time(),
        }
        if extra:
            data.update(extra)
        
        message = SSEMessage(event_type=event_type, data=data)
        self.publish(message)
    
    def publish_workflow_event(
        self,
        event_type: SSEEventType,
        workflow_name: str,
        summary: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """发布工作流相关事件"""
        data = {
            "workflow_name": workflow_name,
            "summary": summary,
            "timestamp": time.time(),
        }
        if extra:
            data.update(extra)
        
        message = SSEMessage(event_type=event_type, data=data)
        self.publish(message)
    
    def publish_cluster_event(
        self,
        event_type: SSEEventType,
        cluster_id: str,
        cluster_info: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """发布集群相关事件"""
        data = {
            "cluster_id": cluster_id,
            "cluster_info": cluster_info,
            "timestamp": time.time(),
        }
        if extra:
            data.update(extra)
        
        message = SSEMessage(event_type=event_type, data=data)
        self.publish(message)
    
    def publish_system_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        active_bees: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """发布系统指标"""
        data = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "active_bees": active_bees,
            "timestamp": time.time(),
        }
        if extra:
            data.update(extra)
        
        message = SSEMessage(event_type=SSEEventType.SYSTEM_METRICS, data=data)
        self.publish(message)
    
    def get_history(
        self,
        event_type: Optional[SSEEventType] = None,
        limit: int = 100,
    ) -> List[SSEMessage]:
        """获取历史消息"""
        with self._lock:
            if event_type:
                filtered = [m for m in self._history if m.event_type == event_type]
                return filtered[-limit:]
            return self._history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取消息总线统计"""
        with self._lock:
            return {
                "subscribers": len(self._subscribers),
                "history_size": len(self._history),
                "max_history": self._max_history,
            }


class TaskQueue:
    """
    后台任务队列 - 任务解耦执行
    
    使用后台线程执行任务，不阻塞主线程。
    """
    
    def __init__(self, max_size: int = 1000):
        self._queue: Queue = Queue(maxsize=max_size)
        self._results: Dict[str, Any] = {}
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self.logger = logger.bind(component="TaskQueue")
        
    def start(self) -> None:
        """启动后台工作线程"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="TaskQueue-Worker",
        )
        self._worker_thread.start()
        self.logger.info("TaskQueue worker started")
    
    def stop(self) -> None:
        """停止后台工作线程"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self.logger.info("TaskQueue worker stopped")
    
    def _worker_loop(self) -> None:
        """后台工作循环"""
        while self._running:
            try:
                # 从队列获取任务，超时1秒
                task_item = self._queue.get(timeout=1.0)
                
                task_id = task_item.get("task_id")
                coro = task_item.get("coro")
                callback = task_item.get("callback")
                
                self.logger.info("Executing queued task", task_id=task_id)
                
                try:
                    # 执行任务
                    if asyncio.iscoroutine(coro):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(coro)
                        finally:
                            loop.close()
                    else:
                        result = coro()
                    
                    # 存储结果
                    with self._lock:
                        self._results[task_id] = {
                            "status": "completed",
                            "result": result,
                            "error": None,
                            "completed_at": time.time(),
                        }
                    
                    # 执行回调
                    if callback:
                        try:
                            callback(task_id, result, None)
                        except Exception as e:
                            self.logger.error("Callback failed", task_id=task_id, error=str(e))
                            
                except Exception as e:
                    self.logger.error("Queued task failed", task_id=task_id, error=str(e))
                    with self._lock:
                        self._results[task_id] = {
                            "status": "failed",
                            "result": None,
                            "error": str(e),
                            "completed_at": time.time(),
                        }
                    if callback:
                        try:
                            callback(task_id, None, str(e))
                        except Exception as cb_err:
                            self.logger.error("Error callback failed", task_id=task_id, error=str(cb_err))
                
            except Empty:
                # 超时，继续循环
                continue
            except Exception as e:
                self.logger.error("Worker loop error", error=str(e))
    
    def submit(
        self,
        task_id: str,
        coro,
        callback: Optional[Callable] = None,
    ) -> bool:
        """
        提交任务到队列
        
        Args:
            task_id: 任务ID
            coro: 协程或可调用对象
            callback: 完成回调函数 signature: (task_id, result, error)
        
        Returns:
            是否成功提交
        """
        try:
            self._queue.put_nowait({
                "task_id": task_id,
                "coro": coro,
                "callback": callback,
                "submitted_at": time.time(),
            })
            with self._lock:
                self._results[task_id] = {
                    "status": "pending",
                    "result": None,
                    "error": None,
                    "submitted_at": time.time(),
                }
            self.logger.info("Task submitted", task_id=task_id, queue_size=self._queue.qsize())
            return True
        except Exception as e:
            self.logger.error("Failed to submit task", task_id=task_id, error=str(e))
            return False
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        with self._lock:
            return self._results.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计"""
        return {
            "queue_size": self._queue.qsize(),
            "total_results": len(self._results),
            "running": self._running,
        }


# 全局单例
_global_message_bus: Optional[MessageBus] = None
_global_task_queue: Optional[TaskQueue] = None


def get_message_bus() -> MessageBus:
    """获取全局消息总线实例"""
    global _global_message_bus
    if _global_message_bus is None:
        _global_message_bus = MessageBus()
    return _global_message_bus


def get_task_queue() -> TaskQueue:
    """获取全局任务队列实例"""
    global _global_task_queue
    if _global_task_queue is None:
        _global_task_queue = TaskQueue()
        _global_task_queue.start()
    return _global_task_queue

class TaskGraph:
    """Represents a workflow"""
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, TaskNode] = {}
        
    def add_node(self, node: TaskNode):
        self.nodes[node.id] = node
        
    def add_dependency(self, child_id: str, parent_id: str):
        if child_id in self.nodes and parent_id in self.nodes:
            self.nodes[child_id].dependencies.add(parent_id)
            if self._has_cycle():
                self.nodes[child_id].dependencies.discard(parent_id)
                raise ValueError(f"Adding dependency {parent_id} -> {child_id} would create a cycle")
        else:
            raise ValueError("Node ID not found")

    def _has_cycle(self) -> bool:
        """DFS 检测 DAG 中是否存在环"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self.nodes}

        def dfs(nid: str) -> bool:
            color[nid] = GRAY
            for dep_id in self.nodes[nid].dependencies:
                if dep_id not in color:
                    continue
                if color[dep_id] == GRAY:
                    return True
                if color[dep_id] == WHITE and dfs(dep_id):
                    return True
            color[nid] = BLACK
            return False

        return any(dfs(nid) for nid in self.nodes if color[nid] == WHITE)

    def get_ready_nodes(self) -> List[TaskNode]:
        """Return nodes that are PENDING and have all dependencies COMPLETED.
        Nodes with any FAILED or SKIPPED parent are marked SKIPPED immediately."""
        ready = []
        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            failed_parent = False
            skipped_parent = False
            parents_ok = True
            for parent_id in node.dependencies:
                parent = self.nodes.get(parent_id)
                if parent is None:
                    continue
                if parent.status == TaskStatus.FAILED:
                    failed_parent = True
                    break
                if parent.status == TaskStatus.SKIPPED:
                    skipped_parent = True
                    break
                if parent.status != TaskStatus.COMPLETED:
                    parents_ok = False
            if failed_parent or skipped_parent:
                node.status = TaskStatus.SKIPPED
            elif parents_ok:
                ready.append(node)
        return ready

    def is_complete(self) -> bool:
        return all(n.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED] for n in self.nodes.values())

class HiveDispatcher:
    """
    The Central Nervous System.
    Manages a pool of Actors and executes TaskGraphs.
    
    V2.5 Enhancements:
    - Integrates with MessageBus for real-time SSE events
    - Publishes task/workflow events to UI
    """
    
    def __init__(self, actors: List[HiveActor], message_bus: Optional[MessageBus] = None):
        self.actors = actors
        self._actors_lock = __import__('threading').Lock()
        self.logger = logger.bind(component="Dispatcher")
        self._task_completed = asyncio.Event()
        self._running_tasks = 0
        
        # V2.5: MessageBus integration
        self._message_bus = message_bus or get_message_bus()
        self._workflow_name: str = ""
    
    def set_message_bus(self, message_bus: MessageBus) -> None:
        """设置消息总线"""
        self._message_bus = message_bus
    
    def _find_capable_actor(self, capabilities: List[str]) -> Optional[HiveActor]:
        """Matchmaking logic: Find an actor that has ALL required capabilities"""
        with self._actors_lock:
            for actor in self.actors:
                match = True
                for cap in capabilities:
                    if not actor.has_capability(cap):
                        match = False
                        break
                if match:
                    return actor
        return None

    async def execute_task(self, node: TaskNode):
        """Execute a single node"""
        actor = self._find_capable_actor(node.required_capabilities)

        if not actor:
            self.logger.error("No actor found for task", task_id=node.id, required=node.required_capabilities)
            node.status = TaskStatus.FAILED
            node.error = "No Capable Actor Available"
            
            # V2.5: 发布任务失败事件
            if self._message_bus:
                self._message_bus.publish_task_event(
                    event_type=SSEEventType.TASK_FAILED,
                    task_id=node.id,
                    task_node=node,
                    workflow_name=self._workflow_name,
                    extra={"reason": "No actor available"},
                )
            
            self._task_completed.set()
            return

        node.status = TaskStatus.RUNNING
        node.assigned_actor = actor.name
        node.started_at = time.time()
        self._running_tasks += 1
        self.logger.info("Dispatching task", task_id=node.id, actor=actor.name, action=node.action)
        
        # V2.5: 发布任务开始事件
        if self._message_bus:
            self._message_bus.publish_task_event(
                event_type=SSEEventType.TASK_STARTED,
                task_id=node.id,
                task_node=node,
                workflow_name=self._workflow_name,
            )

        try:
            # Resolve params (V2.2: Expand to support output from previous tasks via templating)
            # For now, static params

            result = await actor.perform(node.action, **node.params)
            node.result = result
            node.status = TaskStatus.COMPLETED
            node.completed_at = time.time()
            self.logger.info("Task completed", task_id=node.id)
            
            # V2.5: 发布任务完成事件
            if self._message_bus:
                self._message_bus.publish_task_event(
                    event_type=SSEEventType.TASK_COMPLETED,
                    task_id=node.id,
                    task_node=node,
                    workflow_name=self._workflow_name,
                )

        except Exception as e:
            self.logger.error("Task failed", task_id=node.id, error=str(e))
            node.error = str(e)
            node.status = TaskStatus.FAILED
            node.completed_at = time.time()
            
            # V2.5: 发布任务失败事件
            if self._message_bus:
                self._message_bus.publish_task_event(
                    event_type=SSEEventType.TASK_FAILED,
                    task_id=node.id,
                    task_node=node,
                    workflow_name=self._workflow_name,
                    extra={"error": str(e)},
                )
        finally:
            self._running_tasks -= 1
            self._task_completed.set()

    async def run_workflow(self, graph: TaskGraph):
        """Main execution loop (事件驱动，无轮询）"""
        self.logger.info("Starting workflow", workflow=graph.name)
        self._workflow_name = graph.name
        self._task_completed.clear()
        self._running_tasks = 0
        
        # V2.5: 发布工作流开始事件
        if self._message_bus:
            self._message_bus.publish_workflow_event(
                event_type=SSEEventType.WORKFLOW_STARTED,
                workflow_name=graph.name,
                summary={
                    "total_tasks": len(graph.nodes),
                    "status": "started",
                },
            )
        
        try:
            while not graph.is_complete():
                # 1. Get ready tasks
                ready_nodes = graph.get_ready_nodes()

                if not ready_nodes and not graph.is_complete():
                    # Deadlock check
                    if self._running_tasks == 0:
                        self.logger.error("Deadlock detected in workflow. Aborting.")
                        break

                    # 等待任务完成事件（无轮询）
                    await self._task_completed.wait()
                    self._task_completed.clear()
                    continue

                # 2. Launch tasks concurrently
                tasks = []
                for node in ready_nodes:
                    tasks.append(self.execute_task(node))

                if tasks:
                    # 并发执行，gather 会等待所有任务完成
                    await asyncio.gather(*tasks)

            self.logger.info("Workflow finished", workflow=graph.name)

            # Report
            summary = {n.id: n.status.value for n in graph.nodes.values()}
            
            # V2.5: 发布工作流完成事件
            if self._message_bus:
                # 统计
                completed = sum(1 for n in graph.nodes.values() if n.status == TaskStatus.COMPLETED)
                failed = sum(1 for n in graph.nodes.values() if n.status == TaskStatus.FAILED)
                skipped = sum(1 for n in graph.nodes.values() if n.status == TaskStatus.SKIPPED)
                
                is_success = failed == 0
                self._message_bus.publish_workflow_event(
                    event_type=SSEEventType.WORKFLOW_COMPLETED if is_success else SSEEventType.WORKFLOW_FAILED,
                    workflow_name=graph.name,
                    summary={
                        "total": len(graph.nodes),
                        "completed": completed,
                        "failed": failed,
                        "skipped": skipped,
                        "status": "completed" if is_success else "failed",
                        "task_details": summary,
                    },
                )
            
            return summary
            
        except Exception as e:
            self.logger.error("Workflow execution failed", workflow=graph.name, error=str(e))
            
            # V2.5: 发布工作流失败事件
            if self._message_bus:
                self._message_bus.publish_workflow_event(
                    event_type=SSEEventType.WORKFLOW_FAILED,
                    workflow_name=graph.name,
                    summary={
                        "status": "failed",
                        "error": str(e),
                    },
                )
            raise
