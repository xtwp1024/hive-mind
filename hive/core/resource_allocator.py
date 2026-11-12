# -*- coding: utf-8 -*-
"""
Dynamic Resource Allocation - 动态资源分配系统

Phase 3: 自主进化能力
基于适应度和优先级动态分配系统资源。

核心功能:
1. 资源配额管理 - 为每个蜂种分配资源配额
2. 动态调整 - 根据适应度实时调整配额
3. 优先级调度 - 高优先级任务优先获得资源
4. 资源监控 - 实时监控资源使用情况
5. 过载保护 - 防止系统资源耗尽

资源类型:
- CPU 时间片
- 内存配额
- 并发任务数
- API 调用限额
- 网络带宽
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("Hive.ResourceAllocator")


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    CONCURRENCY = "concurrency"
    API_CALLS = "api_calls"
    NETWORK = "network"
    STORAGE = "storage"


class AllocationPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class ResourceQuota:
    """资源配额"""

    resource_type: ResourceType

    hard_limit: float
    soft_limit: float
    current_usage: float = 0.0

    reserved: float = 0.0
    guaranteed: float = 0.0

    burst_allowed: bool = True
    burst_limit: Optional[float] = None

    def is_over_soft_limit(self) -> bool:
        return self.current_usage > self.soft_limit

    def is_over_hard_limit(self) -> bool:
        return self.current_usage > self.hard_limit

    def available(self) -> float:
        return self.hard_limit - self.current_usage

    def effective_limit(self) -> float:
        if self.burst_allowed and self.burst_limit:
            return self.burst_limit
        return self.hard_limit

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type.value,
            "hard_limit": self.hard_limit,
            "soft_limit": self.soft_limit,
            "current_usage": self.current_usage,
            "reserved": self.reserved,
            "guaranteed": self.guaranteed,
            "burst_allowed": self.burst_allowed,
            "burst_limit": self.burst_limit,
        }


@dataclass
class AllocationRecord:
    """分配记录"""

    record_id: str
    consumer_id: str
    resource_type: ResourceType
    amount: float

    priority: AllocationPriority = AllocationPriority.NORMAL

    requested_at: float = field(default_factory=time.time)
    granted_at: Optional[float] = None
    released_at: Optional[float] = None

    duration_seconds: Optional[float] = None

    was_preempted: bool = False
    preemption_reason: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsumerProfile:
    """消费者配置文件"""

    consumer_id: str
    consumer_type: str

    base_priority: AllocationPriority = AllocationPriority.NORMAL

    fitness_score: float = 0.5
    historical_fitness: List[float] = field(default_factory=list)

    resource_quotas: Dict[ResourceType, ResourceQuota] = field(default_factory=dict)

    task_count: int = 0
    success_count: int = 0
    error_count: int = 0

    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    tags: List[str] = field(default_factory=list)

    def current_priority(self) -> AllocationPriority:
        """根据适应度计算当前优先级"""
        if self.fitness_score >= 0.8:
            if self.base_priority.value in ["critical", "high"]:
                return self.base_priority
            return AllocationPriority.HIGH
        elif self.fitness_score >= 0.5:
            return self.base_priority
        elif self.fitness_score >= 0.3:
            return AllocationPriority.LOW
        else:
            return AllocationPriority.BACKGROUND

    def success_rate(self) -> float:
        return self.success_count / max(self.task_count, 1)

    def update_fitness(self, new_score: float):
        """更新适应度分数"""
        self.historical_fitness.append(new_score)
        if len(self.historical_fitness) > 100:
            self.historical_fitness = self.historical_fitness[-100:]

        alpha = 0.1
        self.fitness_score = alpha * new_score + (1 - alpha) * self.fitness_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consumer_id": self.consumer_id,
            "consumer_type": self.consumer_type,
            "base_priority": self.base_priority.value,
            "fitness_score": self.fitness_score,
            "current_priority": self.current_priority().value,
            "task_count": self.task_count,
            "success_rate": self.success_rate(),
            "resource_quotas": {rt.value: q.to_dict() for rt, q in self.resource_quotas.items()},
            "created_at": self.created_at,
            "last_active": self.last_active,
            "tags": self.tags,
        }


@dataclass
class AllocationDecision:
    """分配决策"""

    decision_id: str
    consumer_id: str
    resource_type: ResourceType
    requested_amount: float

    granted: bool
    granted_amount: float = 0.0

    reason: str = ""

    wait_required: bool = False
    estimated_wait_seconds: float = 0.0

    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "consumer_id": self.consumer_id,
            "resource_type": self.resource_type.value,
            "requested_amount": self.requested_amount,
            "granted": self.granted,
            "granted_amount": self.granted_amount,
            "reason": self.reason,
            "wait_required": self.wait_required,
            "estimated_wait_seconds": self.estimated_wait_seconds,
            "alternatives": self.alternatives,
        }


class DynamicResourceAllocator:
    """
    动态资源分配器

    Phase 3 核心组件。基于适应度和优先级动态分配系统资源。

    主要功能:
    1. 消费者管理 - 注册和管理资源消费者
    2. 配额管理 - 为每个消费者分配资源配额
    3. 动态调整 - 根据适应度实时调整配额
    4. 分配决策 - 处理资源请求并做出分配决策
    5. 资源监控 - 实时监控资源使用情况
    6. 过载保护 - 防止系统资源耗尽

    分配策略:
    - 高适应度消费者获得更多配额
    - 高优先级请求优先处理
    - 支持资源抢占（低优先级让给高优先级）
    - 支持突发资源使用（burst mode）

    使用方式:
        allocator = DynamicResourceAllocator()
        allocator.register_consumer("bee_001", "miner_bee")
        decision = allocator.request_resource("bee_001", ResourceType.CPU, 1.0)
    """

    PRIORITY_WEIGHTS = {
        AllocationPriority.CRITICAL: 10.0,
        AllocationPriority.HIGH: 5.0,
        AllocationPriority.NORMAL: 1.0,
        AllocationPriority.LOW: 0.5,
        AllocationPriority.BACKGROUND: 0.1,
    }

    FITNESS_ADJUSTMENT_FACTOR = 0.3

    def __init__(self, system_resources: Optional[Dict[ResourceType, float]] = None):
        self._consumers: Dict[str, ConsumerProfile] = {}
        self._allocations: Dict[str, AllocationRecord] = {}
        self._lock = threading.Lock()

        self._system_resources = system_resources or {
            ResourceType.CPU: 8.0,
            ResourceType.MEMORY: 16.0,
            ResourceType.CONCURRENCY: 100.0,
            ResourceType.API_CALLS: 1000.0,
            ResourceType.NETWORK: 100.0,
            ResourceType.STORAGE: 100.0,
        }

        self._system_usage: Dict[ResourceType, float] = defaultdict(float)

        self._allocation_history: deque = deque(maxlen=10000)
        self._adjustment_count = 0

        self._monitoring_enabled = True
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

        self._persistence_path = Path("hive/data/resource_allocator")
        self._persistence_path.mkdir(parents=True, exist_ok=True)

        logger.info("⚡ [DynamicResourceAllocator] 动态资源分配器已初始化")
        logger.info(f"   系统资源: { {rt.value: v for rt, v in self._system_resources.items()} }")

    def start_monitoring(self):
        """启动监控线程"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="resource_monitor"
        )
        self._monitor_thread.start()

        logger.info("⚡ [DynamicResourceAllocator] 监控线程已启动")

    def stop_monitoring(self):
        """停止监控线程"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        logger.info("⚡ [DynamicResourceAllocator] 监控线程已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                time.sleep(5.0)
                self._periodic_adjustment()
            except Exception as e:
                logger.error(f"⚡ [DynamicResourceAllocator] 监控循环错误: {e}")

    def register_consumer(
        self,
        consumer_id: str,
        consumer_type: str,
        base_priority: AllocationPriority = AllocationPriority.NORMAL,
        initial_quotas: Optional[Dict[ResourceType, Dict[str, Any]]] = None,
    ) -> ConsumerProfile:
        """
        注册资源消费者

        Args:
            consumer_id: 消费者唯一标识
            consumer_type: 消费者类型（如 bee 类型）
            base_priority: 基础优先级
            initial_quotas: 初始配额配置

        Returns:
            ConsumerProfile - 消费者配置文件
        """
        with self._lock:
            if consumer_id in self._consumers:
                logger.warning(f"⚡ [DynamicResourceAllocator] 消费者 {consumer_id} 已注册，跳过")
                return self._consumers[consumer_id]

            profile = ConsumerProfile(
                consumer_id=consumer_id,
                consumer_type=consumer_type,
                base_priority=base_priority,
            )

            if initial_quotas:
                for rt, quota_config in initial_quotas.items():
                    profile.resource_quotas[rt] = ResourceQuota(
                        resource_type=rt,
                        hard_limit=quota_config.get("hard_limit", 1.0),
                        soft_limit=quota_config.get("soft_limit", 0.8),
                        reserved=quota_config.get("reserved", 0.0),
                        guaranteed=quota_config.get("guaranteed", 0.0),
                        burst_allowed=quota_config.get("burst_allowed", True),
                        burst_limit=quota_config.get("burst_limit"),
                    )
            else:
                for rt in ResourceType:
                    profile.resource_quotas[rt] = ResourceQuota(
                        resource_type=rt,
                        hard_limit=1.0,
                        soft_limit=0.8,
                        burst_allowed=True,
                    )

            self._consumers[consumer_id] = profile

            logger.info(
                f"⚡ [DynamicResourceAllocator] 注册消费者: {consumer_id} ({consumer_type}), "
                f"优先级: {base_priority.value}"
            )

            return profile

    def unregister_consumer(self, consumer_id: str) -> bool:
        """注销消费者"""
        with self._lock:
            if consumer_id not in self._consumers:
                return False

            for record_id, record in list(self._allocations.items()):
                if record.consumer_id == consumer_id and record.released_at is None:
                    self._release_allocation_internal(record_id, "consumer_unregistered")

            del self._consumers[consumer_id]

            logger.info(f"⚡ [DynamicResourceAllocator] 注销消费者: {consumer_id}")
            return True

    def get_consumer(self, consumer_id: str) -> Optional[ConsumerProfile]:
        """获取消费者配置文件"""
        with self._lock:
            return self._consumers.get(consumer_id)

    def request_resource(
        self,
        consumer_id: str,
        resource_type: ResourceType,
        amount: float,
        priority: Optional[AllocationPriority] = None,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AllocationDecision:
        """
        请求资源

        Args:
            consumer_id: 消费者 ID
            resource_type: 资源类型
            amount: 请求数量
            priority: 请求优先级（覆盖消费者优先级）
            duration_seconds: 预计使用时长
            metadata: 附加元数据

        Returns:
            AllocationDecision - 分配决策
        """
        import uuid

        decision_id = f"dec_{uuid.uuid4().hex[:12]}"

        with self._lock:
            consumer = self._consumers.get(consumer_id)

            if consumer is None:
                return AllocationDecision(
                    decision_id=decision_id,
                    consumer_id=consumer_id,
                    resource_type=resource_type,
                    requested_amount=amount,
                    granted=False,
                    reason="consumer_not_registered",
                )

            effective_priority = priority or consumer.current_priority()

            quota = consumer.resource_quotas.get(resource_type)
            if quota is None:
                return AllocationDecision(
                    decision_id=decision_id,
                    consumer_id=consumer_id,
                    resource_type=resource_type,
                    requested_amount=amount,
                    granted=False,
                    reason="no_quota_defined",
                )

            available_in_quota = quota.effective_limit() - quota.current_usage

            if available_in_quota < amount:
                if quota.burst_allowed and quota.burst_limit:
                    burst_available = quota.burst_limit - quota.current_usage
                    if burst_available >= amount:
                        pass
                    else:
                        return AllocationDecision(
                            decision_id=decision_id,
                            consumer_id=consumer_id,
                            resource_type=resource_type,
                            requested_amount=amount,
                            granted=False,
                            reason="quota_exceeded",
                            alternatives=[
                                {
                                    "suggestion": "reduce_request",
                                    "max_available": min(available_in_quota, 0),
                                }
                            ],
                        )

            system_available = self._system_resources.get(
                resource_type, float("inf")
            ) - self._system_usage.get(resource_type, 0.0)

            if system_available < amount:
                if effective_priority in [AllocationPriority.CRITICAL, AllocationPriority.HIGH]:
                    preempted = self._try_preempt(resource_type, amount - system_available)
                    if preempted:
                        system_available = self._system_resources.get(
                            resource_type, float("inf")
                        ) - self._system_usage.get(resource_type, 0.0)

                if system_available < amount:
                    return AllocationDecision(
                        decision_id=decision_id,
                        consumer_id=consumer_id,
                        resource_type=resource_type,
                        requested_amount=amount,
                        granted=False,
                        reason="system_resource_exhausted",
                        wait_required=True,
                        estimated_wait_seconds=self._estimate_wait_time(resource_type, amount),
                        alternatives=[
                            {
                                "suggestion": "wait_or_reduce",
                                "system_available": system_available,
                            }
                        ],
                    )

            granted_amount = self._calculate_granted_amount(
                consumer, resource_type, amount, effective_priority
            )

            record = AllocationRecord(
                record_id=f"alloc_{uuid.uuid4().hex[:12]}",
                consumer_id=consumer_id,
                resource_type=resource_type,
                amount=granted_amount,
                priority=effective_priority,
                duration_seconds=duration_seconds,
                metadata=metadata or {},
            )
            record.granted_at = time.time()

            self._allocations[record.record_id] = record
            self._allocation_history.append(record)

            quota.current_usage += granted_amount
            self._system_usage[resource_type] += granted_amount

            consumer.last_active = time.time()

            logger.debug(
                f"⚡ [DynamicResourceAllocator] 分配资源: {consumer_id} -> "
                f"{resource_type.value} x {granted_amount:.2f}, 优先级: {effective_priority.value}"
            )

            return AllocationDecision(
                decision_id=decision_id,
                consumer_id=consumer_id,
                resource_type=resource_type,
                requested_amount=amount,
                granted=True,
                granted_amount=granted_amount,
                reason="granted",
            )

    def _calculate_granted_amount(
        self,
        consumer: ConsumerProfile,
        resource_type: ResourceType,
        requested: float,
        priority: AllocationPriority,
    ) -> float:
        """计算实际分配数量"""
        quota = consumer.resource_quotas.get(resource_type)
        if not quota:
            return 0.0

        priority_bonus = self.PRIORITY_WEIGHTS.get(priority, 1.0) / 2.0
        fitness_bonus = consumer.fitness_score * self.FITNESS_ADJUSTMENT_FACTOR

        max_available = quota.effective_limit() - quota.current_usage
        adjusted_max = max_available * (1.0 + priority_bonus + fitness_bonus)

        return min(requested, adjusted_max, quota.effective_limit())

    def _try_preempt(self, resource_type: ResourceType, amount_needed: float) -> bool:
        """尝试抢占低优先级资源"""
        candidates = []
        for record_id, record in self._allocations.items():
            if record.released_at is not None:
                continue
            if record.resource_type != resource_type:
                continue

            priority_weight = self.PRIORITY_WEIGHTS.get(record.priority, 1.0)
            if priority_weight < self.PRIORITY_WEIGHTS[AllocationPriority.CRITICAL]:
                candidates.append((record_id, record, priority_weight))

        if not candidates:
            return False

        candidates.sort(key=lambda x: x[2])

        total_preempted = 0.0
        for record_id, record, _ in candidates:
            if total_preempted >= amount_needed:
                break

            self._release_allocation_internal(record_id, "preempted_by_higher_priority")
            total_preempted += record.amount

            record.was_preempted = True
            record.preemption_reason = "preempted_by_higher_priority"

            logger.warning(
                f"⚡ [DynamicResourceAllocator] 抢占资源: {record.consumer_id} -> "
                f"{resource_type.value} x {record.amount:.2f}"
            )

        return total_preempted >= amount_needed

    def _estimate_wait_time(self, resource_type: ResourceType, amount: float) -> float:
        """估计等待时间"""
        recent = [
            r
            for r in self._allocation_history
            if r.resource_type == resource_type and r.released_at
        ]

        if not recent:
            return 30.0

        avg_duration = statistics.mean(
            (r.released_at - r.granted_at) for r in recent[-20:] if r.released_at and r.granted_at
        )

        return avg_duration * 2

    def release_allocation(self, record_id: str) -> bool:
        """释放资源分配"""
        with self._lock:
            return self._release_allocation_internal(record_id, "voluntary_release")

    def _release_allocation_internal(self, record_id: str, reason: str) -> bool:
        """内部释放方法"""
        record = self._allocations.get(record_id)
        if record is None or record.released_at is not None:
            return False

        consumer = self._consumers.get(record.consumer_id)
        if consumer:
            quota = consumer.resource_quotas.get(record.resource_type)
            if quota:
                quota.current_usage = max(0.0, quota.current_usage - record.amount)

        self._system_usage[record.resource_type] = max(
            0.0, self._system_usage.get(record.resource_type, 0.0) - record.amount
        )

        record.released_at = time.time()

        logger.debug(
            f"⚡ [DynamicResourceAllocator] 释放资源: {record.consumer_id} -> "
            f"{record.resource_type.value} x {record.amount:.2f} ({reason})"
        )

        return True

    def update_consumer_fitness(self, consumer_id: str, fitness_score: float) -> bool:
        """更新消费者适应度"""
        with self._lock:
            consumer = self._consumers.get(consumer_id)
            if not consumer:
                return False

            consumer.update_fitness(fitness_score)

            logger.debug(
                f"⚡ [DynamicResourceAllocator] 更新适应度: {consumer_id} -> {fitness_score:.2%}"
            )

            return True

    def update_consumer_task_result(
        self,
        consumer_id: str,
        success: bool,
        fitness_delta: Optional[float] = None,
    ) -> bool:
        """更新消费者任务结果"""
        with self._lock:
            consumer = self._consumers.get(consumer_id)
            if not consumer:
                return False

            consumer.task_count += 1
            if success:
                consumer.success_count += 1
            else:
                consumer.error_count += 1

            if fitness_delta is not None:
                new_fitness = max(0.0, min(1.0, consumer.fitness_score + fitness_delta))
                consumer.update_fitness(new_fitness)

            consumer.last_active = time.time()

            return True

    def _periodic_adjustment(self):
        """周期性调整配额"""
        with self._lock:
            if not self._consumers:
                return

            total_fitness = sum(c.fitness_score for c in self._consumers.values())
            if total_fitness <= 0:
                return

            self._adjustment_count += 1

            for consumer_id, consumer in self._consumers.items():
                fitness_ratio = consumer.fitness_score / total_fitness

                for rt, quota in consumer.resource_quotas.items():
                    system_total = self._system_resources.get(rt, 1.0)
                    target_quota = system_total * fitness_ratio * 0.8

                    alpha = 0.1
                    quota.hard_limit = quota.hard_limit * (1 - alpha) + target_quota * alpha
                    quota.soft_limit = quota.hard_limit * 0.8

            logger.debug(
                f"⚡ [DynamicResourceAllocator] 周期性调整完成 (#{self._adjustment_count})"
            )

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self._lock:
            status = {
                "timestamp": time.time(),
                "total_consumers": len(self._consumers),
                "active_allocations": sum(
                    1 for r in self._allocations.values() if r.released_at is None
                ),
                "adjustment_count": self._adjustment_count,
                "system_resources": {
                    rt.value: {
                        "total": total,
                        "used": self._system_usage.get(rt, 0.0),
                        "available": total - self._system_usage.get(rt, 0.0),
                        "utilization": self._system_usage.get(rt, 0.0) / max(total, 0.01),
                    }
                    for rt, total in self._system_resources.items()
                },
                "consumers_by_priority": self._count_by_priority(),
            }

            return status

    def _count_by_priority(self) -> Dict[str, int]:
        """按优先级统计消费者"""
        counts = defaultdict(int)
        for consumer in self._consumers.values():
            priority = consumer.current_priority().value
            counts[priority] += 1
        return dict(counts)

    def get_consumer_stats(self, consumer_id: str) -> Optional[Dict[str, Any]]:
        """获取消费者统计"""
        with self._lock:
            consumer = self._consumers.get(consumer_id)
            if not consumer:
                return None

            consumer_allocations = [
                r for r in self._allocations.values() if r.consumer_id == consumer_id
            ]

            return {
                "profile": consumer.to_dict(),
                "allocation_stats": {
                    "total_requests": len(consumer_allocations),
                    "active_allocations": sum(
                        1 for r in consumer_allocations if r.released_at is None
                    ),
                },
            }

    def get_top_consumers(
        self,
        resource_type: ResourceType,
        count: int = 10,
    ) -> List[Tuple[str, float]]:
        """获取资源使用最多的消费者"""
        with self._lock:
            usage = []
            for consumer_id, consumer in self._consumers.items():
                quota = consumer.resource_quotas.get(resource_type)
                if quota:
                    usage.append((consumer_id, quota.current_usage))

            usage.sort(key=lambda x: x[1], reverse=True)
            return usage[:count]


_allocator: Optional[DynamicResourceAllocator] = None
_allocator_lock = threading.Lock()


def get_resource_allocator() -> DynamicResourceAllocator:
    """获取资源分配器单例"""
    global _allocator
    if _allocator is None:
        with _allocator_lock:
            if _allocator is None:
                _allocator = DynamicResourceAllocator()
    return _allocator
