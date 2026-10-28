# -*- coding: utf-8 -*-
"""
Load Balancer & Failover (负载均衡与故障转移)

Phase 5: 分布式网络 - 高可用增强

核心职责:
1. 负载均衡策略 - 多种算法选择最优节点
2. 健康检查 - 持续监控节点健康状态
3. 故障检测 - 快速识别故障节点
4. 故障转移 - 自动迁移任务到健康节点
5. 重试机制 - 失败任务自动重试
6. 熔断机制 - 防止故障扩散

负载均衡策略:
┌─────────────────────────────────────────────────────────────┐
│  策略名称              │  描述                               │
├─────────────────────────────────────────────────────────────┤
│  ROUND_ROBIN           │  轮询 (简单公平)                    │
│  WEIGHTED_ROUND_ROBIN  │  加权轮询 (根据能力权重)            │
│  LEAST_LOAD            │  最少负载 (选择负载最低的节点)       │
│  LEAST_CONNECTIONS     │  最少连接 (选择连接数最少的)        │
│  CONSISTENT_HASHING    │  一致性哈希 (相同任务到相同节点)    │
│  RANDOM                │  随机选择                           │
│  CAPABILITY_AWARE      │  能力感知 (根据任务选择合适节点)    │
└─────────────────────────────────────────────────────────────┘

健康状态:
┌─────────────────────────────────────────────────────────────┐
│  状态        │  描述                    │  可分配任务       │
├─────────────────────────────────────────────────────────────┤
│  HEALTHY     │  健康，正常工作          │  是              │
│  DEGRADED    │  性能下降，但可用        │  是 (低优先级)   │
│  UNHEALTHY   │  不健康                  │  否              │
│  FAILED      │  已失败                  │  否              │
│  DRAINING    │  正在排空 (准备下线)     │  否 (仅现有任务)  │
└─────────────────────────────────────────────────────────────┘
"""

import heapq
import logging
import random
import threading
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict, deque
from copy import deepcopy
from statistics import mean

logger = logging.getLogger("Hive.LoadBalancer")


class LoadBalanceStrategy(str, Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_LOAD = "least_load"
    LEAST_CONNECTIONS = "least_connections"
    CONSISTENT_HASHING = "consistent_hashing"
    RANDOM = "random"
    CAPABILITY_AWARE = "capability_aware"


class NodeHealthStatus(str, Enum):
    """节点健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    DRAINING = "draining"


class FailoverAction(str, Enum):
    """故障转移动作"""
    REDIRECT = "redirect"
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"


@dataclass
class NodeMetrics:
    """节点指标"""

    node_id: str

    load_factor: float = 0.0
    connection_count: int = 0
    pending_tasks: int = 0

    response_time_ms: float = 0.0
    success_rate: float = 1.0

    cpu_usage: float = 0.0
    memory_usage: float = 0.0

    last_health_check: float = 0.0
    health_status: NodeHealthStatus = NodeHealthStatus.HEALTHY

    failure_count: int = 0
    success_count: int = 0

    capabilities: List[str] = field(default_factory=list)
    weight: int = 1

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """检查节点是否可用"""
        return self.health_status in [
            NodeHealthStatus.HEALTHY,
            NodeHealthStatus.DEGRADED,
        ]

    def can_accept_task(self) -> bool:
        """检查节点是否能接受新任务"""
        if self.health_status in [NodeHealthStatus.FAILED, NodeHealthStatus.DRAINING]:
            return False
        if self.health_status == NodeHealthStatus.DEGRADED and self.load_factor > 0.7:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "load_factor": self.load_factor,
            "connection_count": self.connection_count,
            "pending_tasks": self.pending_tasks,
            "response_time_ms": self.response_time_ms,
            "success_rate": self.success_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "last_health_check": self.last_health_check,
            "health_status": self.health_status.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "capabilities": self.capabilities,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeMetrics":
        status_str = data.get("health_status", "healthy")
        try:
            health_status = NodeHealthStatus(status_str)
        except ValueError:
            health_status = NodeHealthStatus.HEALTHY

        return cls(
            node_id=data["node_id"],
            load_factor=data.get("load_factor", 0.0),
            connection_count=data.get("connection_count", 0),
            pending_tasks=data.get("pending_tasks", 0),
            response_time_ms=data.get("response_time_ms", 0.0),
            success_rate=data.get("success_rate", 1.0),
            cpu_usage=data.get("cpu_usage", 0.0),
            memory_usage=data.get("memory_usage", 0.0),
            last_health_check=data.get("last_health_check", 0.0),
            health_status=health_status,
            failure_count=data.get("failure_count", 0),
            success_count=data.get("success_count", 0),
            capabilities=data.get("capabilities", []),
            weight=data.get("weight", 1),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TaskAssignment:
    """任务分配记录"""

    assignment_id: str
    task_id: str
    node_id: str

    assigned_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    failed_at: Optional[float] = None

    status: str = "assigned"
    retry_count: int = 0
    max_retries: int = 3

    error_message: Optional[str] = None
    result: Optional[Any] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "task_id": self.task_id,
            "node_id": self.node_id,
            "assigned_at": self.assigned_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "failed_at": self.failed_at,
            "status": self.status,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class HealthCheckResult:
    """健康检查结果"""

    node_id: str
    is_healthy: bool

    response_time_ms: float = 0.0
    status: NodeHealthStatus = NodeHealthStatus.HEALTHY

    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: float = field(default_factory=time.time)


class ConsistentHashRing:
    """
    一致性哈希环

    用于实现一致性哈希负载均衡策略。
    """

    def __init__(self, virtual_nodes: int = 100, hash_function: Optional[Callable] = None):
        self.virtual_nodes = virtual_nodes
        self.hash_function = hash_function or self._default_hash

        self._ring: Dict[int, str] = {}
        self._sorted_keys: List[int] = []
        self._node_map: Dict[str, List[int]] = defaultdict(list)
        self._lock = threading.RLock()

    def _default_hash(self, key: str) -> int:
        """默认哈希函数"""
        h = hashlib.md5(key.encode("utf-8"))
        return int(h.hexdigest(), 16)

    def add_node(self, node_id: str) -> None:
        """添加节点到哈希环"""
        with self._lock:
            for i in range(self.virtual_nodes):
                key = self.hash_function(f"{node_id}:virtual:{i}")
                self._ring[key] = node_id
                self._node_map[node_id].append(key)

            self._sorted_keys = sorted(self._ring.keys())

    def remove_node(self, node_id: str) -> None:
        """从哈希环移除节点"""
        with self._lock:
            if node_id not in self._node_map:
                return

            for key in self._node_map[node_id]:
                del self._ring[key]

            del self._node_map[node_id]
            self._sorted_keys = sorted(self._ring.keys())

    def get_node(self, key: str) -> Optional[str]:
        """获取 key 对应的节点"""
        with self._lock:
            if not self._ring:
                return None

            hash_key = self.hash_function(key)

            idx = self._bisect_right(self._sorted_keys, hash_key)
            if idx == len(self._sorted_keys):
                idx = 0

            return self._ring[self._sorted_keys[idx]]

    def _bisect_right(self, sorted_list: List[int], value: int) -> int:
        """二分查找"""
        low = 0
        high = len(sorted_list)
        while low < high:
            mid = (low + high) // 2
            if sorted_list[mid] > value:
                high = mid
            else:
                low = mid + 1
        return low

    def get_nodes(self) -> List[str]:
        """获取所有节点"""
        with self._lock:
            return list(self._node_map.keys())


class HealthChecker:
    """
    健康检查器

    持续监控节点健康状态。
    """

    def __init__(
        self,
        check_interval: float = 10.0,
        timeout_seconds: float = 5.0,
        unhealthy_threshold: int = 3,
        healthy_threshold: int = 2,
    ):
        self.check_interval = check_interval
        self.timeout_seconds = timeout_seconds
        self.unhealthy_threshold = unhealthy_threshold
        self.healthy_threshold = healthy_threshold

        self._node_checks: Dict[str, Dict[str, Any]] = {}
        self._custom_checkers: Dict[str, Callable] = {}
        self._lock = threading.RLock()

        logger.info(f"🏥 [HealthChecker] 初始化，检查间隔: {check_interval}s")

    def register_node(
        self,
        node_id: str,
        initial_status: NodeHealthStatus = NodeHealthStatus.HEALTHY,
    ) -> None:
        """注册节点进行健康检查"""
        with self._lock:
            self._node_checks[node_id] = {
                "status": initial_status,
                "consecutive_failures": 0,
                "consecutive_successes": 0,
                "last_check": 0.0,
                "last_response_time_ms": 0.0,
                "history": deque(maxlen=20),
            }

        logger.info(f"🏥 [HealthChecker] 注册节点: {node_id}")

    def unregister_node(self, node_id: str) -> None:
        """注销节点"""
        with self._lock:
            self._node_checks.pop(node_id, None)

        logger.info(f"🏥 [HealthChecker] 注销节点: {node_id}")

    def register_custom_checker(
        self,
        node_id: str,
        checker: Callable[[str], HealthCheckResult],
    ) -> None:
        """注册自定义检查器"""
        with self._lock:
            self._custom_checkers[node_id] = checker

    def check_node(self, node_id: str) -> HealthCheckResult:
        """
        检查单个节点健康状态

        Args:
            node_id: 节点ID

        Returns:
            HealthCheckResult - 检查结果
        """
        start_time = time.time()

        with self._lock:
            if node_id not in self._node_checks:
                return HealthCheckResult(
                    node_id=node_id,
                    is_healthy=False,
                    status=NodeHealthStatus.UNHEALTHY,
                    details={"error": "Node not registered"},
                )

            check_data = self._node_checks[node_id]

        if node_id in self._custom_checkers:
            try:
                result = self._custom_checkers[node_id](node_id)
                response_time = (time.time() - start_time) * 1000
                result.response_time_ms = response_time
            except Exception as e:
                result = HealthCheckResult(
                    node_id=node_id,
                    is_healthy=False,
                    response_time_ms=(time.time() - start_time) * 1000,
                    status=NodeHealthStatus.UNHEALTHY,
                    details={"error": str(e)},
                )
        else:
            result = HealthCheckResult(
                node_id=node_id,
                is_healthy=True,
                response_time_ms=(time.time() - start_time) * 1000,
                status=NodeHealthStatus.HEALTHY,
                details={"info": "Default health check passed"},
            )

        with self._lock:
            check_data = self._node_checks[node_id]
            check_data["last_check"] = time.time()
            check_data["last_response_time_ms"] = result.response_time_ms
            check_data["history"].append(result.is_healthy)

            if result.is_healthy:
                check_data["consecutive_successes"] += 1
                check_data["consecutive_failures"] = 0

                if check_data["consecutive_successes"] >= self.healthy_threshold:
                    check_data["status"] = NodeHealthStatus.HEALTHY
            else:
                check_data["consecutive_failures"] += 1
                check_data["consecutive_successes"] = 0

                if check_data["consecutive_failures"] >= self.unhealthy_threshold:
                    check_data["status"] = NodeHealthStatus.UNHEALTHY

            result.status = check_data["status"]

        return result

    def check_all_nodes(self) -> Dict[str, HealthCheckResult]:
        """检查所有节点"""
        results = {}
        with self._lock:
            nodes = list(self._node_checks.keys())

        for node_id in nodes:
            results[node_id] = self.check_node(node_id)

        return results

    def get_node_status(self, node_id: str) -> Optional[NodeHealthStatus]:
        """获取节点状态"""
        with self._lock:
            if node_id not in self._node_checks:
                return None
            return self._node_checks[node_id]["status"]

    def set_node_status(self, node_id: str, status: NodeHealthStatus) -> bool:
        """手动设置节点状态"""
        with self._lock:
            if node_id not in self._node_checks:
                return False

            self._node_checks[node_id]["status"] = status

            if status == NodeHealthStatus.HEALTHY:
                self._node_checks[node_id]["consecutive_successes"] = self.healthy_threshold
                self._node_checks[node_id]["consecutive_failures"] = 0
            elif status in [NodeHealthStatus.UNHEALTHY, NodeHealthStatus.FAILED]:
                self._node_checks[node_id]["consecutive_failures"] = self.unhealthy_threshold
                self._node_checks[node_id]["consecutive_successes"] = 0

            logger.info(f"🏥 [HealthChecker] 设置节点状态: {node_id} -> {status.value}")
            return True

    def drain_node(self, node_id: str) -> bool:
        """排空节点 (准备下线)"""
        return self.set_node_status(node_id, NodeHealthStatus.DRAINING)

    def get_stats(self) -> Dict[str, Any]:
        """获取健康检查统计"""
        with self._lock:
            by_status = defaultdict(int)
            for node_id, data in self._node_checks.items():
                by_status[data["status"].value] += 1

            return {
                "total_nodes": len(self._node_checks),
                "by_status": dict(by_status),
                "check_interval": self.check_interval,
                "unhealthy_threshold": self.unhealthy_threshold,
                "healthy_threshold": self.healthy_threshold,
            }


class LoadBalancer:
    """
    负载均衡器

    根据策略选择最优节点。
    """

    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_LOAD,
        health_checker: Optional[HealthChecker] = None,
    ):
        self.strategy = strategy
        self.health_checker = health_checker or HealthChecker()

        self._nodes: Dict[str, NodeMetrics] = {}
        self._rr_counter = 0
        self._hash_ring = ConsistentHashRing()
        self._lock = threading.RLock()

        logger.info(f"⚖️ [LoadBalancer] 初始化，策略: {strategy.value}")

    def add_node(
        self,
        node_id: str,
        weight: int = 1,
        capabilities: Optional[List[str]] = None,
    ) -> bool:
        """
        添加节点

        Args:
            node_id: 节点ID
            weight: 权重 (用于加权轮询)
            capabilities: 能力列表

        Returns:
            bool - 是否成功
        """
        with self._lock:
            if node_id in self._nodes:
                return False

            self._nodes[node_id] = NodeMetrics(
                node_id=node_id,
                weight=weight,
                capabilities=capabilities or [],
            )

            self.health_checker.register_node(node_id)
            self._hash_ring.add_node(node_id)

        logger.info(f"⚖️ [LoadBalancer] 添加节点: {node_id}, 权重: {weight}")
        return True

    def remove_node(self, node_id: str) -> bool:
        """移除节点"""
        with self._lock:
            if node_id not in self._nodes:
                return False

            del self._nodes[node_id]
            self.health_checker.unregister_node(node_id)
            self._hash_ring.remove_node(node_id)

        logger.info(f"⚖️ [LoadBalancer] 移除节点: {node_id}")
        return True

    def update_node_metrics(self, node_id: str, **metrics) -> bool:
        """更新节点指标"""
        with self._lock:
            if node_id not in self._nodes:
                return False

            node = self._nodes[node_id]
            for key, value in metrics.items():
                if hasattr(node, key):
                    setattr(node, key, value)

            return True

    def select_node(
        self,
        task_id: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        strategy: Optional[LoadBalanceStrategy] = None,
    ) -> Optional[str]:
        """
        选择节点

        Args:
            task_id: 任务ID (用于一致性哈希)
            required_capabilities: 要求的能力列表
            strategy: 覆盖默认策略

        Returns:
            str|None - 选中的节点ID
        """
        use_strategy = strategy or self.strategy

        with self._lock:
            candidates = self._get_available_nodes(required_capabilities)

            if not candidates:
                logger.warning(f"⚖️ [LoadBalancer] 没有可用节点 (需要能力: {required_capabilities})")
                return None

            if use_strategy == LoadBalanceStrategy.ROUND_ROBIN:
                return self._select_round_robin(candidates)
            elif use_strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
                return self._select_weighted_round_robin(candidates)
            elif use_strategy == LoadBalanceStrategy.LEAST_LOAD:
                return self._select_least_load(candidates)
            elif use_strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
                return self._select_least_connections(candidates)
            elif use_strategy == LoadBalanceStrategy.CONSISTENT_HASHING:
                return self._select_consistent_hash(task_id, candidates)
            elif use_strategy == LoadBalanceStrategy.RANDOM:
                return self._select_random(candidates)
            elif use_strategy == LoadBalanceStrategy.CAPABILITY_AWARE:
                return self._select_capability_aware(candidates, required_capabilities)
            else:
                return self._select_least_load(candidates)

    def _get_available_nodes(self, required_capabilities: Optional[List[str]]) -> List[NodeMetrics]:
        """获取可用节点列表"""
        candidates = []
        for node in self._nodes.values():
            if not node.can_accept_task():
                continue

            if required_capabilities:
                has_all = True
                for cap in required_capabilities:
                    if cap not in node.capabilities:
                        has_all = False
                        break
                if not has_all:
                    continue

            candidates.append(node)
        return candidates

    def _select_round_robin(self, candidates: List[NodeMetrics]) -> str:
        """轮询选择"""
        self._rr_counter = (self._rr_counter + 1) % len(candidates)
        return candidates[self._rr_counter].node_id

    def _select_weighted_round_robin(self, candidates: List[NodeMetrics]) -> str:
        """加权轮询"""
        total_weight = sum(n.weight for n in candidates)
        if total_weight == 0:
            return candidates[0].node_id

        r = random.randint(1, total_weight)
        cumulative = 0
        for node in candidates:
            cumulative += node.weight
            if cumulative >= r:
                return node.node_id

        return candidates[0].node_id

    def _select_least_load(self, candidates: List[NodeMetrics]) -> str:
        """最少负载"""
        candidates.sort(key=lambda n: n.load_factor)
        return candidates[0].node_id

    def _select_least_connections(self, candidates: List[NodeMetrics]) -> str:
        """最少连接"""
        candidates.sort(key=lambda n: (n.connection_count, n.pending_tasks))
        return candidates[0].node_id

    def _select_consistent_hash(self, task_id: Optional[str], candidates: List[NodeMetrics]) -> str:
        """一致性哈希"""
        if task_id is None:
            return self._select_least_load(candidates)

        node_id = self._hash_ring.get_node(task_id)

        if node_id and any(n.node_id == node_id for n in candidates):
            return node_id

        return self._select_least_load(candidates)

    def _select_random(self, candidates: List[NodeMetrics]) -> str:
        """随机选择"""
        return random.choice(candidates).node_id

    def _select_capability_aware(
        self,
        candidates: List[NodeMetrics],
        required_capabilities: Optional[List[str]],
    ) -> str:
        """能力感知选择"""
        if not required_capabilities:
            return self._select_least_load(candidates)

        scored = []
        for node in candidates:
            score = 0.0

            for cap in required_capabilities:
                if cap in node.capabilities:
                    score += 1.0

            score -= node.load_factor
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0].node_id

    def get_available_nodes(self, required_capabilities: Optional[List[str]] = None) -> List[str]:
        """获取可用节点ID列表"""
        with self._lock:
            candidates = self._get_available_nodes(required_capabilities)
            return [n.node_id for n in candidates]

    def get_node_metrics(self, node_id: str) -> Optional[NodeMetrics]:
        """获取节点指标"""
        with self._lock:
            return self._nodes.get(node_id)

    def get_all_metrics(self) -> Dict[str, NodeMetrics]:
        """获取所有节点指标"""
        with self._lock:
            return dict(self._nodes)

    def record_success(self, node_id: str, response_time_ms: float) -> None:
        """记录成功"""
        with self._lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                node.success_count += 1
                node.response_time_ms = (node.response_time_ms * 0.9 + response_time_ms * 0.1)
                node.connection_count = max(0, node.connection_count - 1)

                total = node.success_count + node.failure_count
                if total > 0:
                    node.success_rate = node.success_count / total

    def record_failure(self, node_id: str) -> None:
        """记录失败"""
        with self._lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                node.failure_count += 1
                node.connection_count = max(0, node.connection_count - 1)

                total = node.success_count + node.failure_count
                if total > 0:
                    node.success_rate = node.success_count / total

    def get_stats(self) -> Dict[str, Any]:
        """获取负载均衡器统计"""
        with self._lock:
            by_status = defaultdict(int)
            total_load = 0.0
            total_connections = 0

            for node in self._nodes.values():
                by_status[node.health_status.value] += 1
                total_load += node.load_factor
                total_connections += node.connection_count

            available = len(self._get_available_nodes(None))

            return {
                "strategy": self.strategy.value,
                "total_nodes": len(self._nodes),
                "available_nodes": available,
                "by_status": dict(by_status),
                "avg_load": total_load / len(self._nodes) if self._nodes else 0.0,
                "total_connections": total_connections,
                "health_checker": self.health_checker.get_stats(),
            }


class FailoverManager:
    """
    故障转移管理器

    处理节点故障和任务重试。
    """

    def __init__(
        self,
        load_balancer: LoadBalancer,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        circuit_break_threshold: int = 5,
        circuit_break_duration: float = 60.0,
    ):
        self.load_balancer = load_balancer
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.circuit_break_threshold = circuit_break_threshold
        self.circuit_break_duration = circuit_break_duration

        self._assignments: Dict[str, TaskAssignment] = {}
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        logger.info(
            f"🔄 [FailoverManager] 初始化，最大重试: {max_retries}, "
            f"熔断阈值: {circuit_break_threshold}"
        )

    def assign_task(
        self,
        task_id: str,
        required_capabilities: Optional[List[str]] = None,
        preferred_node: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[TaskAssignment]:
        """
        分配任务到节点

        Args:
            task_id: 任务ID
            required_capabilities: 要求的能力
            preferred_node: 首选节点
            max_retries: 最大重试次数 (覆盖默认)

        Returns:
            TaskAssignment|None - 任务分配
        """
        if preferred_node:
            with self._lock:
                metrics = self.load_balancer.get_node_metrics(preferred_node)
                if metrics and metrics.can_accept_task():
                    node_id = preferred_node
                else:
                    node_id = self.load_balancer.select_node(
                        task_id, required_capabilities
                    )
        else:
            node_id = self.load_balancer.select_node(task_id, required_capabilities)

        if not node_id:
            logger.warning(f"🔄 [FailoverManager] 无法分配任务: {task_id} - 没有可用节点")
            return None

        assignment = TaskAssignment(
            assignment_id=f"assign_{uuid.uuid4().hex[:12]}",
            task_id=task_id,
            node_id=node_id,
            max_retries=max_retries or self.max_retries,
        )

        with self._lock:
            self._assignments[assignment.assignment_id] = assignment

            metrics = self.load_balancer.get_node_metrics(node_id)
            if metrics:
                metrics.connection_count += 1
                metrics.pending_tasks += 1

        logger.info(f"🔄 [FailoverManager] 分配任务: {task_id} -> {node_id}")
        return assignment

    def start_task(self, assignment_id: str) -> bool:
        """标记任务开始"""
        with self._lock:
            if assignment_id not in self._assignments:
                return False

            assignment = self._assignments[assignment_id]
            assignment.status = "running"
            assignment.started_at = time.time()

            return True

    def complete_task(self, assignment_id: str, result: Any = None) -> bool:
        """标记任务完成"""
        with self._lock:
            if assignment_id not in self._assignments:
                return False

            assignment = self._assignments[assignment_id]
            assignment.status = "completed"
            assignment.completed_at = time.time()
            assignment.result = result

            self.load_balancer.record_success(
                assignment.node_id,
                (time.time() - (assignment.started_at or time.time())) * 1000,
            )

            metrics = self.load_balancer.get_node_metrics(assignment.node_id)
            if metrics:
                metrics.pending_tasks = max(0, metrics.pending_tasks - 1)

        logger.info(f"🔄 [FailoverManager] 任务完成: {assignment.task_id}")
        return True

    def fail_task(
        self,
        assignment_id: str,
        error_message: str,
        can_retry: bool = True,
    ) -> Tuple[bool, Optional[TaskAssignment]]:
        """
        标记任务失败并尝试重试

        Args:
            assignment_id: 分配ID
            error_message: 错误信息
            can_retry: 是否允许重试

        Returns:
            (success, new_assignment|None)
        """
        with self._lock:
            if assignment_id not in self._assignments:
                return False, None

            assignment = self._assignments[assignment_id]
            assignment.status = "failed"
            assignment.failed_at = time.time()
            assignment.error_message = error_message

            self.load_balancer.record_failure(assignment.node_id)

            metrics = self.load_balancer.get_node_metrics(assignment.node_id)
            if metrics:
                metrics.pending_tasks = max(0, metrics.pending_tasks - 1)

            self._check_circuit_breaker(assignment.node_id)

            if not can_retry:
                logger.warning(
                    f"🔄 [FailoverManager] 任务失败: {assignment.task_id}, "
                    f"错误: {error_message} (不重试)"
                )
                return True, None

            if assignment.retry_count >= assignment.max_retries:
                logger.warning(
                    f"🔄 [FailoverManager] 任务失败: {assignment.task_id}, "
                    f"已达到最大重试次数 ({assignment.max_retries})"
                )
                return True, None

        new_assignment = self._retry_task(assignment)
        return True, new_assignment

    def _retry_task(self, assignment: TaskAssignment) -> Optional[TaskAssignment]:
        """重试任务"""
        assignment.retry_count += 1

        logger.info(
            f"🔄 [FailoverManager] 重试任务: {assignment.task_id}, "
            f"第 {assignment.retry_count} 次"
        )

        if self.retry_delay_seconds > 0:
            delay = self.retry_delay_seconds * (2 ** (assignment.retry_count - 1))
            time.sleep(min(delay, 10.0))

        required_caps = assignment.metadata.get("required_capabilities")
        exclude_node = assignment.node_id

        candidates = self.load_balancer.get_available_nodes(required_caps)
        if len(candidates) > 1:
            candidates = [c for c in candidates if c != exclude_node]

        if not candidates:
            return self.assign_task(
                task_id=assignment.task_id,
                required_capabilities=required_caps,
                max_retries=assignment.max_retries,
            )

        return self.assign_task(
            task_id=assignment.task_id,
            required_capabilities=required_caps,
            preferred_node=candidates[0] if candidates else None,
            max_retries=assignment.max_retries,
        )

    def _check_circuit_breaker(self, node_id: str) -> None:
        """检查熔断器"""
        with self._lock:
            if node_id not in self._circuit_breakers:
                self._circuit_breakers[node_id] = {
                    "failure_count": 0,
                    "circuit_open": False,
                    "circuit_open_since": None,
                    "last_failure_at": None,
                }

            cb = self._circuit_breakers[node_id]
            cb["failure_count"] += 1
            cb["last_failure_at"] = time.time()

            if not cb["circuit_open"] and cb["failure_count"] >= self.circuit_break_threshold:
                cb["circuit_open"] = True
                cb["circuit_open_since"] = time.time()

                self.load_balancer.health_checker.set_node_status(
                    node_id, NodeHealthStatus.UNHEALTHY
                )

                logger.warning(
                    f"🔄 [FailoverManager] 熔断器打开: {node_id}, "
                    f"连续失败 {cb['failure_count']} 次"
                )

            if cb["circuit_open"] and cb["circuit_open_since"]:
                elapsed = time.time() - cb["circuit_open_since"]
                if elapsed >= self.circuit_break_duration:
                    cb["circuit_open"] = False
                    cb["failure_count"] = 0

                    self.load_balancer.health_checker.set_node_status(
                        node_id, NodeHealthStatus.HEALTHY
                    )

                    logger.info(
                        f"🔄 [FailoverManager] 熔断器关闭: {node_id}, "
                        f"冷却时间已过 ({elapsed:.1f}s)"
                    )

    def get_assignment(self, assignment_id: str) -> Optional[TaskAssignment]:
        """获取任务分配"""
        with self._lock:
            return self._assignments.get(assignment_id)

    def get_assignment_by_task(self, task_id: str) -> List[TaskAssignment]:
        """获取任务的所有分配"""
        with self._lock:
            return [
                a for a in self._assignments.values()
                if a.task_id == task_id
            ]

    def is_circuit_open(self, node_id: str) -> bool:
        """检查节点是否熔断"""
        with self._lock:
            if node_id not in self._circuit_breakers:
                return False
            return self._circuit_breakers[node_id].get("circuit_open", False)

    def get_stats(self) -> Dict[str, Any]:
        """获取故障转移统计"""
        with self._lock:
            by_status = defaultdict(int)
            for assignment in self._assignments.values():
                by_status[assignment.status] += 1

            circuit_open_count = sum(
                1 for cb in self._circuit_breakers.values()
                if cb.get("circuit_open")
            )

            return {
                "total_assignments": len(self._assignments),
                "by_status": dict(by_status),
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay_seconds,
                "circuit_break_threshold": self.circuit_break_threshold,
                "circuit_break_duration": self.circuit_break_duration,
                "circuits_open": circuit_open_count,
                "total_circuits": len(self._circuit_breakers),
            }


class LoadBalancerEngine:
    """
    负载均衡引擎

    整合负载均衡、健康检查、故障转移的统一接口。
    """

    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_LOAD,
        health_check_interval: float = 10.0,
        max_retries: int = 3,
    ):
        self.health_checker = HealthChecker(check_interval=health_check_interval)
        self.load_balancer = LoadBalancer(
            strategy=strategy,
            health_checker=self.health_checker,
        )
        self.failover_manager = FailoverManager(
            load_balancer=self.load_balancer,
            max_retries=max_retries,
        )

        self._running = False
        self._health_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        logger.info(f"🚀 [LoadBalancerEngine] 初始化，策略: {strategy.value}")

    def add_node(
        self,
        node_id: str,
        weight: int = 1,
        capabilities: Optional[List[str]] = None,
        initial_status: NodeHealthStatus = NodeHealthStatus.HEALTHY,
    ) -> bool:
        """添加节点"""
        with self._lock:
            if not self.load_balancer.add_node(node_id, weight, capabilities):
                return False

            return True

    def remove_node(self, node_id: str) -> bool:
        """移除节点"""
        with self._lock:
            return self.load_balancer.remove_node(node_id)

    def drain_node(self, node_id: str) -> bool:
        """排空节点"""
        with self._lock:
            return self.health_checker.drain_node(node_id)

    def select_and_assign(
        self,
        task_id: str,
        required_capabilities: Optional[List[str]] = None,
    ) -> Optional[TaskAssignment]:
        """选择节点并分配任务"""
        assignment = self.failover_manager.assign_task(
            task_id=task_id,
            required_capabilities=required_capabilities,
        )

        if assignment:
            self.failover_manager.start_task(assignment.assignment_id)

        return assignment

    def execute_with_failover(
        self,
        task_id: str,
        executor: Callable[[str], Tuple[bool, Any, str]],
        required_capabilities: Optional[List[str]] = None,
    ) -> Tuple[bool, Any, Optional[TaskAssignment]]:
        """
        执行任务并自动处理故障转移

        Args:
            task_id: 任务ID
            executor: 执行函数 (node_id) -> (success, result, error_message)
            required_capabilities: 要求的能力

        Returns:
            (success, result, last_assignment)
        """
        assignment = self.select_and_assign(task_id, required_capabilities)
        if not assignment:
            return False, "No available nodes", None

        while assignment:
            try:
                success, result, error_msg = executor(assignment.node_id)

                if success:
                    self.failover_manager.complete_task(assignment.assignment_id, result)
                    return True, result, assignment
                else:
                    can_retry, new_assignment = self.failover_manager.fail_task(
                        assignment.assignment_id,
                        error_msg,
                    )
                    assignment = new_assignment

            except Exception as e:
                can_retry, new_assignment = self.failover_manager.fail_task(
                    assignment.assignment_id,
                    str(e),
                )
                assignment = new_assignment

        return False, "All retries failed", None

    def start_health_monitoring(self) -> None:
        """启动健康监控"""
        if self._running:
            return

        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="load-balancer-health",
        )
        self._health_thread.start()

        logger.info("🚀 [LoadBalancerEngine] 健康监控已启动")

    def stop_health_monitoring(self) -> None:
        """停止健康监控"""
        self._running = False
        logger.info("🚀 [LoadBalancerEngine] 健康监控已停止")

    def _health_monitor_loop(self) -> None:
        """健康监控循环"""
        while self._running:
            time.sleep(self.health_checker.check_interval)

            try:
                self.health_checker.check_all_nodes()
            except Exception as e:
                logger.error(f"🚀 [LoadBalancerEngine] 健康检查异常: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取完整统计"""
        return {
            "load_balancer": self.load_balancer.get_stats(),
            "health_checker": self.health_checker.get_stats(),
            "failover_manager": self.failover_manager.get_stats(),
            "running": self._running,
        }


_lb_engine: Optional[LoadBalancerEngine] = None
_lb_lock = threading.Lock()


def get_load_balancer_engine() -> LoadBalancerEngine:
    """获取负载均衡引擎单例"""
    global _lb_engine
    if _lb_engine is None:
        with _lb_lock:
            if _lb_engine is None:
                _lb_engine = LoadBalancerEngine()
    return _lb_engine


def init_load_balancer_engine(**kwargs) -> LoadBalancerEngine:
    """初始化负载均衡引擎"""
    global _lb_engine
    with _lb_lock:
        if _lb_engine is not None:
            return _lb_engine
        _lb_engine = LoadBalancerEngine(**kwargs)
    return _lb_engine
