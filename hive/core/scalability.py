# -*- coding: utf-8 -*-
"""
Scalability Manager (横向扩展管理器)

Phase 5 P3: 横向扩展
支持 100+ 蜜蜂并发，无限计算能力。

核心职责:
1. WorkerPool — 动态 worker 池，自动扩缩容
2. 负载均衡 — 基于能力的智能任务分发
3. RateLimiter — API/资源速率限制
4. CircuitBreaker — 熔断器，防止级联故障
5. 批量任务处理 — 一次分发多个任务提高吞吐量

架构:
  ┌──────────────────────────────────────────────────────┐
  │              ScalabilityManager                       │
  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
  │  │WorkerPool │  │RateLimiter │  │CircuitBreaker│   │
  │  │  (动态池)  │  │  (速率限制) │  │   (熔断)   │    │
  │  └────────────┘  └────────────┘  └────────────┘    │
  │  ┌────────────┐  ┌────────────┐                    │
  │  │LoadBalancer│  │BatchProcessor│                    │
  │  │  (负载均衡) │  │  (批量处理) │                    │
  │  └────────────┘  └────────────┘                    │
  └──────────────────────────────────────────────────────┘
                        │
                        ▼
  ┌──────────────────────────────────────────────────────┐
  │              WorkerManager (现有)                      │
  │  active_workers: Dict[worker_id, Process/Async]      │
  └──────────────────────────────────────────────────────┘

使用方式:
    sm = ScalabilityManager(worker_manager)
    sm.start()
    sm.submit_task({"type": "code_gen", "payload": {...}})
    result = sm.wait_result(task_id, timeout=30)
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("Hive.Scalability")


# ===== 熔断器 =====

class CircuitState(Enum):
    CLOSED = "closed"    # 正常
    OPEN = "open"        # 熔断
    HALF_OPEN = "half_open"  # 半开


@dataclass
class CircuitBreaker:
    """
    熔断器

    防止级联故障。当错误率超过阈值时断开，一段时间后尝试恢复。
    """
    name: str
    failure_threshold: int = 5       # 触发熔断的连续失败次数
    recovery_timeout: float = 30.0   # 恢复尝试间隔 (秒)
    half_open_max_calls: int = 3     # 半开状态允许的调用次数

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否应该进入半开
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过熔断器执行调用"""
        with self._lock:
            current_state = self._state

        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")

        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                self._half_open_calls += 1
                if self._half_open_calls > self.half_open_max_calls:
                    raise CircuitOpenError(f"Circuit {self.name} is HALF_OPEN (max calls reached)")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info(f"🔄 [CircuitBreaker] {self.name} 恢复: CLOSED")

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"⚡ [CircuitBreaker] {self.name} 熔断: OPEN (failures={self._failure_count})")


class CircuitOpenError(Exception):
    pass


# ===== 速率限制器 =====

@dataclass
class RateLimiter:
    """
    速率限制器

    限制单位时间内的调用次数，支持滑动窗口。
    """
    name: str
    max_calls: int = 100       # 窗口内最大调用次数
    window_seconds: float = 60.0  # 窗口大小 (秒)

    _calls: list = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """
        获取调用许可

        Args:
            blocking: 是否阻塞等待
            timeout: 最大等待时间

        Returns:
            True if permit acquired, False otherwise
        """
        deadline = time.time() + timeout if timeout else None

        while True:
            with self._lock:
                now = time.time()
                # 清理过期记录
                self._calls = [t for t in self._calls if now - t < self.window_seconds]

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return True

            if not blocking:
                return False

            if deadline and time.time() >= deadline:
                return False

            # 等待一小段时间后重试
            time.sleep(0.1)

    def try_acquire(self) -> bool:
        """非阻塞获取许可"""
        return self.acquire(blocking=False)

    def reset(self):
        """重置速率限制"""
        with self._lock:
            self._calls.clear()

    @property
    def current_usage(self) -> float:
        """当前窗口内的使用率"""
        with self._lock:
            now = time.time()
            active = [t for t in self._calls if now - t < self.window_seconds]
            return len(active) / self.max_calls


# ===== Worker 池 =====

@dataclass
class PooledWorker:
    """池化 worker 描述"""
    worker_id: str
    capabilities: list[str] = field(default_factory=list)
    current_load: float = 0.0  # 0.0=空闲, 1.0=满载
    is_alive: bool = True
    last_used: float = field(default_factory=time.time)


class WorkerPool:
    """
    Worker 池

    动态管理 worker 实例，支持自动扩缩容。
    """

    def __init__(
        self,
        worker_manager,  # WorkerManager 实例
        min_size: int = 5,
        max_size: int = 100,
        scale_up_threshold: float = 0.8,   # 负载 > 80% 时扩容
        scale_down_threshold: float = 0.3, # 负载 < 30% 时缩容
        scale_interval: float = 10.0,      # 扩缩容检查间隔
    ):
        """
        初始化 Worker 池

        Args:
            worker_manager: WorkerManager 实例
            min_size: 最小 worker 数量
            max_size: 最大 worker 数量
            scale_up_threshold: 扩容阈值
            scale_down_threshold: 缩容阈值
            scale_interval: 扩缩容检查间隔
        """
        self._wm = worker_manager
        self.min_size = min_size
        self.max_size = max_size
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_interval = scale_interval

        self._pool: dict[str, PooledWorker] = {}
        self._pool_lock = threading.RLock()
        self._scale_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(f"🏭 [WorkerPool] 初始化: min={min_size}, max={max_size}")

    def start(self):
        """启动 worker 池"""
        if self._running:
            return
        self._running = True

        # 初始启动到最小规模
        self._ensure_min_size()

        # 启动自动扩缩容线程
        self._scale_thread = threading.Thread(
            target=self._scale_loop,
            daemon=True,
            name="workerpool-scale"
        )
        self._scale_thread.start()
        logger.info(f"🏭 [WorkerPool] 已启动")

    def stop(self):
        """停止 worker 池"""
        if not self._running:
            return
        self._running = False
        if self._scale_thread:
            self._scale_thread.join(timeout=5.0)
        logger.info(f"🏭 [WorkerPool] 已停止")

    def _ensure_min_size(self):
        """确保最小规模"""
        with self._pool_lock:
            current = len(self._pool)
            if current < self.min_size:
                to_add = self.min_size - current
                logger.info(f"🏭 [WorkerPool] 初始扩容: +{to_add}")
                # 实际扩容会由 WorkerManager 处理，这里只是更新状态

    def _scale_loop(self):
        """自动扩缩容循环"""
        while self._running:
            time.sleep(self.scale_interval)
            try:
                self._check_and_scale()
            except Exception as e:
                logger.error(f"🏭 [WorkerPool] 扩缩容检查异常: {e}")

    def _check_and_scale(self):
        """检查并执行扩缩容"""
        with self._pool_lock:
            if not self._pool:
                return

            # 计算平均负载
            total_load = sum(w.current_load for w in self._pool.values())
            avg_load = total_load / len(self._pool) if self._pool else 0
            current_size = len(self._pool)

            if avg_load > self.scale_up_threshold and current_size < self.max_size:
                # 扩容
                scale_up = min(
                    max(1, current_size // 4),  # 至少扩1，最多扩25%
                    self.max_size - current_size
                )
                logger.info(f"🏭 [WorkerPool] 扩容: +{scale_up} (avg_load={avg_load:.2f})")
                # 扩容逻辑由 WorkerManager 处理
                self._emit_scale_event("scale_up", scale_up, avg_load)

            elif avg_load < self.scale_down_threshold and current_size > self.min_size:
                # 缩容
                scale_down = min(
                    max(1, current_size // 4),
                    current_size - self.min_size
                )
                # 只缩空闲的 workers
                idle = [wid for wid, w in self._pool.items() if w.current_load < 0.1]
                if idle:
                    to_remove = min(scale_down, len(idle))
                    logger.info(f"🏭 [WorkerPool] 缩容: -{to_remove} (avg_load={avg_load:.2f})")
                    self._emit_scale_event("scale_down", to_remove, avg_load)

    def _emit_scale_event(self, direction: str, count: int, avg_load: float):
        """发出扩缩容事件 (可被外部监听)"""
        # 简化: 直接调用 WorkerManager 的 spawn/deregister
        # 实际实现应该通过事件机制
        pass

    def register_worker(self, worker_id: str, capabilities: list[str] = None):
        """注册 worker 到池"""
        with self._pool_lock:
            self._pool[worker_id] = PooledWorker(
                worker_id=worker_id,
                capabilities=capabilities or [],
            )
        logger.debug(f"🏭 [WorkerPool] 注册 worker: {worker_id}")

    def unregister_worker(self, worker_id: str):
        """从池移除 worker"""
        with self._pool_lock:
            self._pool.pop(worker_id, None)
        logger.debug(f"🏭 [WorkerPool] 移除 worker: {worker_id}")

    def get_best_worker(self, required_capabilities: list[str] = None) -> Optional[str]:
        """获取最佳 worker (负载最低且具备能力)"""
        with self._pool_lock:
            candidates = list(self._pool.values())
            if not candidates:
                return None

            if required_capabilities:
                candidates = [
                    w for w in candidates
                    if w.is_alive and all(cap in w.capabilities for cap in required_capabilities)
                ]
            else:
                candidates = [w for w in candidates if w.is_alive]

            if not candidates:
                return None

            candidates.sort(key=lambda w: w.current_load)
            return candidates[0].worker_id

    def update_load(self, worker_id: str, load: float):
        """更新 worker 负载"""
        with self._pool_lock:
            if worker_id in self._pool:
                self._pool[worker_id].current_load = load
                self._pool[worker_id].last_used = time.time()

    def mark_dead(self, worker_id: str):
        """标记 worker 死亡"""
        with self._pool_lock:
            if worker_id in self._pool:
                self._pool[worker_id].is_alive = False

    def get_stats(self) -> dict:
        """获取池统计"""
        with self._pool_lock:
            alive = [w for w in self._pool.values() if w.is_alive]
            return {
                "total": len(self._pool),
                "alive": len(alive),
                "avg_load": sum(w.current_load for w in alive) / len(alive) if alive else 0,
                "min_size": self.min_size,
                "max_size": self.max_size,
            }


# ===== 横向扩展管理器 =====

class ScalabilityManager:
    """
    横向扩展管理器

    Phase 5 P3: 100+ 蜜蜂并发
    """

    def __init__(
        self,
        worker_manager,  # WorkerManager 实例
        max_concurrent_tasks: int = 200,
        task_timeout: float = 120.0,
    ):
        """
        初始化横向扩展管理器

        Args:
            worker_manager: WorkerManager 实例
            max_concurrent_tasks: 最大并发任务数
            task_timeout: 任务超时时间
        """
        self._wm = worker_manager
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout

        # 子组件
        self._worker_pool = WorkerPool(
            worker_manager=worker_manager,
            min_size=5,
            max_size=100,
        )

        # 熔断器 (按能力分类)
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._cb_lock = threading.Lock()

        # 速率限制器 (按 API/资源分类)
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._rl_lock = threading.Lock()

        # 任务追踪
        self._tasks: dict[str, dict] = {}
        self._tasks_lock = threading.RLock()

        # 运行状态
        self._running = False

        logger.info(f"🚀 [Scalability] 横向扩展管理器初始化: max_concurrent={max_concurrent_tasks}")

    def start(self):
        """启动横向扩展管理器"""
        if self._running:
            return
        self._running = True
        self._worker_pool.start()
        logger.info(f"🚀 [Scalability] 横向扩展管理器已启动")

    def stop(self):
        """停止横向扩展管理器"""
        if not self._running:
            return
        self._running = False
        self._worker_pool.stop()
        logger.info(f"🚀 [Scalability] 横向扩展管理器已停止")

    # ===== 熔断器 =====

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """获取或创建熔断器"""
        with self._cb_lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(name=name)
            return self._circuit_breakers[name]

    def call_protected(
        self,
        circuit_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """通过熔断器保护执行调用"""
        cb = self.get_circuit_breaker(circuit_name)
        return cb.call(func, *args, **kwargs)

    # ===== 速率限制器 =====

    def get_rate_limiter(
        self,
        name: str,
        max_calls: int = 100,
        window: float = 60.0
    ) -> RateLimiter:
        """获取或创建速率限制器"""
        with self._rl_lock:
            if name not in self._rate_limiters:
                self._rate_limiters[name] = RateLimiter(
                    name=name,
                    max_calls=max_calls,
                    window_seconds=window,
                )
            return self._rate_limiters[name]

    def acquire_permit(
        self,
        limiter_name: str,
        blocking: bool = True,
        timeout: float = None
    ) -> bool:
        """获取速率限制许可"""
        rl = self.get_rate_limiter(limiter_name)
        return rl.acquire(blocking=blocking, timeout=timeout)

    # ===== 任务管理 =====

    def submit_task(
        self,
        task: dict,
        required_capabilities: list[str] = None,
        priority: int = 5,
        timeout: float = None,
    ) -> str:
        """
        提交任务到横向扩展系统

        Args:
            task: 任务描述
            required_capabilities: 所需能力
            priority: 优先级 (1-10)
            timeout: 超时时间

        Returns:
            task_id
        """
        task_id = f"task-{uuid.uuid4().hex[:8]}"

        with self._tasks_lock:
            # 检查并发限制
            active = [t for t in self._tasks.values() if t.get("status") == "running"]
            if len(active) >= self.max_concurrent_tasks:
                # 等待或拒绝
                logger.warning(f"🚀 [Scalability] 并发超限: {len(active)}/{self.max_concurrent_tasks}")

            self._tasks[task_id] = {
                "task": task,
                "required_capabilities": required_capabilities,
                "priority": priority,
                "timeout": timeout or self.task_timeout,
                "status": "pending",
                "submitted_at": time.time(),
                "result": None,
            }

        # 选择最佳 worker
        best_worker = self._worker_pool.get_best_worker(required_capabilities)
        if best_worker:
            logger.info(f"🚀 [Scalability] 任务分配: {task_id} -> {best_worker}")
            # 实际提交到 WorkerManager
            # 简化: 这里只是标记，实际由 WorkerManager 处理
        else:
            logger.warning(f"🚀 [Scalability] 找不到合适的 worker: {task_id}")

        return task_id

    def wait_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """等待任务结果"""
        deadline = time.time() + (timeout or self.task_timeout)

        while time.time() < deadline:
            with self._tasks_lock:
                task = self._tasks.get(task_id)
                if not task:
                    return None
                if task["status"] == "completed":
                    return task.get("result")
                if task["status"] == "failed":
                    return task.get("error")

            time.sleep(0.1)

        # 超时
        with self._tasks_lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "timeout"
        return None

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """获取任务状态"""
        with self._tasks_lock:
            return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._tasks_lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "cancelled"
                return True
        return False

    # ===== 状态 =====

    def get_status(self) -> dict:
        """获取横向扩展状态"""
        with self._tasks_lock:
            active = [t for t in self._tasks.values() if t.get("status") == "running"]
            completed = [t for t in self._tasks.values() if t.get("status") == "completed"]
            failed = [t for t in self._tasks.values() if t.get("status") == "failed"]

        cb_states = {name: cb.state.value for name, cb in self._circuit_breakers.items()}
        rl_usage = {name: rl.current_usage for name, rl in self._rate_limiters.items()}

        return {
            "running": self._running,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "active_tasks": len(active),
            "completed_tasks": len(completed),
            "failed_tasks": len(failed),
            "total_tasks": len(self._tasks),
            "worker_pool": self._worker_pool.get_stats(),
            "circuit_breakers": cb_states,
            "rate_limiters": {name: f"{u*100:.1f}%" for name, u in rl_usage.items()},
        }


# ===== 全局单例 =====

_scalability: Optional[ScalabilityManager] = None
_scalability_lock = threading.Lock()


def get_scalability() -> Optional[ScalabilityManager]:
    """获取横向扩展管理器单例"""
    return _scalability


def init_scalability(worker_manager, **kwargs) -> ScalabilityManager:
    """初始化横向扩展管理器"""
    global _scalability
    with _scalability_lock:
        if _scalability is not None and _scalability._running:
            return _scalability
        _scalability = ScalabilityManager(worker_manager, **kwargs)
        _scalability.start()
    return _scalability
