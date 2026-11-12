# -*- coding: utf-8 -*-
"""
DNA Adaptive Modification - DNA 自适应修改系统

Phase 3: 自主进化能力
为 DNA 系统添加参数自适应调整能力。

核心功能:
1. 性能反馈驱动 - 基于任务结果自动调整 DNA 参数
2. 学习率自适应 - 根据性能变化动态调整学习率
3. 变异率自适应 - 根据适应度动态调整变异率
4. 能力权重优化 - 基于任务表现优化能力权重
5. 进化方向引导 - 根据缺口分析引导进化方向

工作原理:
- 收集性能反馈 → 分析趋势 → 调整参数 → 验证效果 → 持续迭代
"""

import logging
import math
import statistics
import threading
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("Hive.DNAAdaptive")


class AdaptationType(Enum):
    LEARNING_RATE = "learning_rate"
    MUTATION_RATE = "mutation_rate"
    CAPABILITY_WEIGHT = "capability_weight"
    IMMUNE_RESPONSE = "immune_response"
    ALL_PARAMS = "all_params"


class PerformanceTrend(Enum):
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class PerformanceSample:
    """性能样本"""

    timestamp: float

    success_rate: float
    avg_duration: float
    error_rate: float

    fitness_score: float

    capabilities_used: Dict[str, float] = field(default_factory=dict)

    task_type: Optional[str] = None
    task_complexity: float = 0.5

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "success_rate": self.success_rate,
            "avg_duration": self.avg_duration,
            "error_rate": self.error_rate,
            "fitness_score": self.fitness_score,
            "capabilities_used": self.capabilities_used,
            "task_type": self.task_type,
            "task_complexity": self.task_complexity,
            "metadata": self.metadata,
        }


@dataclass
class AdaptationAction:
    """适应动作"""

    action_id: str
    adaptation_type: AdaptationType

    target_parameter: str
    current_value: float
    target_value: float
    delta: float

    reason: str
    confidence: float

    triggered_at: float = field(default_factory=time.time)
    applied: bool = False
    applied_at: Optional[float] = None

    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "adaptation_type": self.adaptation_type.value,
            "target_parameter": self.target_parameter,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "delta": self.delta,
            "reason": self.reason,
            "confidence": self.confidence,
            "triggered_at": self.triggered_at,
            "applied": self.applied,
            "applied_at": self.applied_at,
            "result": self.result,
        }


@dataclass
class ParameterState:
    """参数状态"""

    name: str
    current_value: float

    min_value: float = 0.0
    max_value: float = 1.0

    historical_values: deque = field(default_factory=lambda: deque(maxlen=100))
    historical_gradients: deque = field(default_factory=lambda: deque(maxlen=100))

    adaptation_count: int = 0
    successful_adaptations: int = 0

    last_adaptation_at: Optional[float] = None

    def record_value(self, value: float, timestamp: Optional[float] = None):
        """记录新值"""
        if self.historical_values:
            gradient = value - self.historical_values[-1]
            self.historical_gradients.append(gradient)

        self.historical_values.append(value)
        self.current_value = value

    def get_trend(self) -> Tuple[PerformanceTrend, float]:
        """获取趋势"""
        if len(self.historical_values) < 5:
            return PerformanceTrend.STABLE, 0.0

        values = list(self.historical_values)[-20:]

        if len(values) < 2:
            return PerformanceTrend.STABLE, 0.0

        n = len(values)
        x = list(range(n))
        y = values

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return PerformanceTrend.STABLE, 0.0

        slope = numerator / denominator

        variance = statistics.variance(values) if len(values) > 1 else 0

        if variance > 0.1:
            return PerformanceTrend.VOLATILE, slope

        if abs(slope) < 0.01:
            return PerformanceTrend.STABLE, slope
        elif slope > 0:
            return PerformanceTrend.IMPROVING, slope
        else:
            return PerformanceTrend.DECLINING, slope

    def get_average(self, window: int = 10) -> float:
        """获取平均值"""
        if not self.historical_values:
            return self.current_value

        values = list(self.historical_values)[-window:]
        return sum(values) / len(values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "historical_count": len(self.historical_values),
            "adaptation_count": self.adaptation_count,
            "successful_adaptations": self.successful_adaptations,
            "success_rate": self.successful_adaptations / max(self.adaptation_count, 1),
            "last_adaptation_at": self.last_adaptation_at,
        }


class DNAAdaptiveEngine:
    """
    DNA 自适应引擎

    Phase 3 核心组件。为 DNA 系统添加参数自适应调整能力。

    主要功能:
    1. 性能监控 - 持续收集 DNA 性能数据
    2. 趋势分析 - 分析性能变化趋势
    3. 参数调整 - 根据趋势自动调整 DNA 参数
    4. 效果验证 - 验证调整是否有效
    5. 反馈闭环 - 根据验证结果优化调整策略

    自适应策略:
    - 当性能下降时：增加学习率，增加变异率
    - 当性能提升时：降低学习率，稳定变异率
    - 当性能波动时：降低学习率，增加探索
    - 当性能稳定时：微调参数，优化细节

    使用方式:
        engine = DNAAdaptiveEngine()
        engine.monitor_dna(bee_dna, "bee_001")
        engine.record_performance("bee_001", sample)
        actions = engine.suggest_adaptations("bee_001")
    """

    LEARNING_RATE_MIN = 0.01
    LEARNING_RATE_MAX = 0.9
    LEARNING_RATE_INITIAL = 0.1

    MUTATION_RATE_MIN = 0.001
    MUTATION_RATE_MAX = 0.5
    MUTATION_RATE_INITIAL = 0.1

    ADAPTATION_COOLDOWN = 60.0

    def __init__(self):
        self._monitored_dnas: Dict[str, Dict[str, Any]] = {}
        self._parameter_states: Dict[str, Dict[str, ParameterState]] = defaultdict(dict)

        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        self._adaptation_history: Dict[str, List[AdaptationAction]] = defaultdict(list)

        self._lock = threading.Lock()

        self._auto_adapt_enabled = False
        self._adapt_thread: Optional[threading.Thread] = None
        self._running = False

        self._adaptation_callbacks: List[Callable] = []

        self._persistence_path = Path("hive/data/dna_adaptive")
        self._persistence_path.mkdir(parents=True, exist_ok=True)

        logger.info("🧬 [DNAAdaptiveEngine] DNA 自适应引擎已初始化")

    def monitor_dna(
        self,
        dna,
        dna_id: str,
        dna_type: str = "bee",
    ):
        """
        开始监控 DNA

        Args:
            dna: DNA 实例（BeeDNA 或 DNA）
            dna_id: DNA 唯一标识
            dna_type: DNA 类型
        """
        with self._lock:
            if dna_id in self._monitored_dnas:
                logger.warning(f"🧬 [DNAAdaptiveEngine] DNA {dna_id} 已在监控中")
                return

            self._monitored_dnas[dna_id] = {
                "dna": dna,
                "dna_type": dna_type,
                "monitored_at": time.time(),
                "last_adaptation_at": None,
            }

            self._init_parameter_states(dna_id, dna)

            logger.info(f"🧬 [DNAAdaptiveEngine] 开始监控 DNA: {dna_id} ({dna_type})")

    def _init_parameter_states(self, dna_id: str, dna):
        """初始化参数状态"""
        params = [
            ("learning_rate", dna.learning_rate, 0.0, 1.0),
            ("mutation_rate", dna.mutation_rate, 0.0, 1.0),
            ("immune_response", getattr(dna, "immune_response", 0.5), 0.0, 1.0),
        ]

        capability_params = [
            "mining_capability",
            "gathering_capability",
            "architectural_capability",
            "librarian_capability",
            "sentinel_capability",
        ]

        for param_name in capability_params:
            if hasattr(dna, param_name):
                value = getattr(dna, param_name)
                params.append((param_name, value, 0.0, 1.0))

        for name, initial_value, min_val, max_val in params:
            state = ParameterState(
                name=name,
                current_value=initial_value,
                min_value=min_val,
                max_value=max_val,
            )
            state.record_value(initial_value)
            self._parameter_states[dna_id][name] = state

    def stop_monitoring(self, dna_id: str) -> bool:
        """停止监控 DNA"""
        with self._lock:
            if dna_id not in self._monitored_dnas:
                return False

            del self._monitored_dnas[dna_id]

            if dna_id in self._parameter_states:
                del self._parameter_states[dna_id]

            logger.info(f"🧬 [DNAAdaptiveEngine] 停止监控 DNA: {dna_id}")
            return True

    def record_performance(
        self,
        dna_id: str,
        success: bool,
        duration: float,
        error: bool = False,
        task_type: Optional[str] = None,
        capabilities_used: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        记录性能数据

        Args:
            dna_id: DNA 标识
            success: 是否成功
            duration: 耗时
            error: 是否错误
            task_type: 任务类型
            capabilities_used: 使用的能力

        Returns:
            bool - 是否成功记录
        """
        with self._lock:
            if dna_id not in self._monitored_dnas:
                return False

            history = self._performance_history[dna_id]

            if history:
                last_sample = history[-1]
                total_tasks = len(history)
                success_count = sum(1 for s in history if s.success_rate > 0.5)

                new_success_rate = (success_count + (1 if success else 0)) / (total_tasks + 1)
                new_avg_duration = (last_sample.avg_duration * total_tasks + duration) / (
                    total_tasks + 1
                )
                new_error_rate = (last_sample.error_rate * total_tasks + (1 if error else 0)) / (
                    total_tasks + 1
                )
            else:
                new_success_rate = 1.0 if success else 0.0
                new_avg_duration = duration
                new_error_rate = 1.0 if error else 0.0

            monitored = self._monitored_dnas.get(dna_id, {})
            dna = monitored.get("dna")
            fitness_score = getattr(dna, "fitness", 0.5) if dna else 0.5

            sample = PerformanceSample(
                timestamp=time.time(),
                success_rate=new_success_rate,
                avg_duration=new_avg_duration,
                error_rate=new_error_rate,
                fitness_score=fitness_score,
                capabilities_used=capabilities_used or {},
                task_type=task_type,
            )

            history.append(sample)

            if dna_id in self._parameter_states:
                for name, state in self._parameter_states[dna_id].items():
                    if dna and hasattr(dna, name):
                        current_val = getattr(dna, name)
                        state.record_value(current_val)

            logger.debug(
                f"🧬 [DNAAdaptiveEngine] 记录性能: {dna_id} - "
                f"success={success}, duration={duration:.2f}s"
            )

            return True

    def suggest_adaptations(
        self,
        dna_id: str,
        adaptation_types: Optional[List[AdaptationType]] = None,
    ) -> List[AdaptationAction]:
        """
        建议适应动作

        Args:
            dna_id: DNA 标识
            adaptation_types: 要考虑的适应类型

        Returns:
            List[AdaptationAction] - 建议的适应动作列表
        """
        with self._lock:
            if dna_id not in self._monitored_dnas:
                return []

            if dna_id not in self._parameter_states:
                return []

            history = self._performance_history.get(dna_id, deque())
            if len(history) < 10:
                logger.debug(f"🧬 [DNAAdaptiveEngine] DNA {dna_id} 历史数据不足，跳过适应建议")
                return []

            monitored = self._monitored_dnas.get(dna_id, {})
            last_adapt = monitored.get("last_adaptation_at")
            if last_adapt and time.time() - last_adapt < self.ADAPTATION_COOLDOWN:
                logger.debug(f"🧬 [DNAAdaptiveEngine] DNA {dna_id} 在冷却期内，跳过适应建议")
                return []

            actions = []

            trend, slope = self._analyze_performance_trend(history)

            adaptation_types = adaptation_types or [
                AdaptationType.LEARNING_RATE,
                AdaptationType.MUTATION_RATE,
                AdaptationType.CAPABILITY_WEIGHT,
            ]

            for adapt_type in adaptation_types:
                type_actions = self._generate_adaptations_for_type(
                    dna_id, adapt_type, trend, slope, history
                )
                actions.extend(type_actions)

            actions.sort(key=lambda a: a.confidence, reverse=True)

            logger.info(f"🧬 [DNAAdaptiveEngine] 为 DNA {dna_id} 建议 {len(actions)} 个适应动作")

            return actions

    def _analyze_performance_trend(self, history: deque) -> Tuple[PerformanceTrend, float]:
        """分析性能趋势"""
        recent = list(history)[-30:]

        if len(recent) < 5:
            return PerformanceTrend.STABLE, 0.0

        fitness_values = [s.fitness_score for s in recent]
        success_values = [s.success_rate for s in recent]

        combined = [(f * 0.7 + s * 0.3) for f, s in zip(fitness_values, success_values)]

        n = len(combined)
        x = list(range(n))
        y = combined

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return PerformanceTrend.STABLE, 0.0

        slope = numerator / denominator

        variance = statistics.variance(combined) if len(combined) > 1 else 0

        if variance > 0.15:
            return PerformanceTrend.VOLATILE, slope

        if abs(slope) < 0.005:
            return PerformanceTrend.STABLE, slope
        elif slope > 0:
            return PerformanceTrend.IMPROVING, slope
        else:
            return PerformanceTrend.DECLINING, slope

    def _generate_adaptations_for_type(
        self,
        dna_id: str,
        adapt_type: AdaptationType,
        trend: PerformanceTrend,
        slope: float,
        history: deque,
    ) -> List[AdaptationAction]:
        """为特定类型生成适应动作"""
        import uuid

        actions = []
        states = self._parameter_states.get(dna_id, {})

        if adapt_type == AdaptationType.LEARNING_RATE:
            state = states.get("learning_rate")
            if state:
                action = self._adapt_learning_rate(state, trend, slope, dna_id)
                if action:
                    actions.append(action)

        elif adapt_type == AdaptationType.MUTATION_RATE:
            state = states.get("mutation_rate")
            if state:
                action = self._adapt_mutation_rate(state, trend, slope, dna_id)
                if action:
                    actions.append(action)

        elif adapt_type == AdaptationType.CAPABILITY_WEIGHT:
            recent = list(history)[-20:]
            capability_usage = defaultdict(list)

            for sample in recent:
                for cap, usage in sample.capabilities_used.items():
                    capability_usage[cap].append(usage)

            for cap_name, usages in capability_usage.items():
                if not usages:
                    continue

                avg_usage = sum(usages) / len(usages)
                success_correlation = self._calculate_capability_correlation(cap_name, recent)

                param_name = f"{cap_name}_capability"
                state = states.get(param_name)

                if state and abs(success_correlation) > 0.1:
                    action = self._adapt_capability_weight(
                        state, cap_name, avg_usage, success_correlation, dna_id
                    )
                    if action:
                        actions.append(action)

        return actions

    def _calculate_capability_correlation(
        self,
        cap_name: str,
        samples: List[PerformanceSample],
    ) -> float:
        """计算能力使用与成功率的相关性"""
        if len(samples) < 5:
            return 0.0

        usage_values = []
        success_values = []

        for sample in samples:
            usage = sample.capabilities_used.get(cap_name, 0.5)
            usage_values.append(usage)
            success_values.append(sample.success_rate)

        if len(set(usage_values)) <= 1:
            return 0.0

        try:
            import statistics

            x = usage_values
            y = success_values

            n = len(x)
            x_mean = sum(x) / n
            y_mean = sum(y) / n

            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            std_x = statistics.stdev(x) if n > 1 else 0
            std_y = statistics.stdev(y) if n > 1 else 0

            if std_x == 0 or std_y == 0:
                return 0.0

            correlation = numerator / ((n - 1) * std_x * std_y)
            return max(-1.0, min(1.0, correlation))

        except Exception:
            return 0.0

    def _adapt_learning_rate(
        self,
        state: ParameterState,
        trend: PerformanceTrend,
        slope: float,
        dna_id: str,
    ) -> Optional[AdaptationAction]:
        """适应学习率"""
        import uuid

        current = state.current_value
        confidence = 0.5

        if trend == PerformanceTrend.DECLINING:
            target = min(self.LEARNING_RATE_MAX, current * 1.5)
            reason = f"性能下降 (slope={slope:.4f})，增加学习率以加速适应"
            confidence = 0.8

        elif trend == PerformanceTrend.IMPROVING:
            target = max(self.LEARNING_RATE_MIN, current * 0.8)
            reason = f"性能提升 (slope={slope:.4f})，降低学习率以稳定收敛"
            confidence = 0.7

        elif trend == PerformanceTrend.VOLATILE:
            target = max(self.LEARNING_RATE_MIN, current * 0.6)
            reason = "性能波动，降低学习率以减少振荡"
            confidence = 0.6

        else:
            avg = state.get_average(window=20)
            if abs(current - avg) > 0.1:
                target = avg
                reason = f"恢复到历史平均值 (当前={current:.2f}, 平均={avg:.2f})"
                confidence = 0.5
            else:
                return None

        target = max(self.LEARNING_RATE_MIN, min(self.LEARNING_RATE_MAX, target))

        if abs(target - current) < 0.001:
            return None

        return AdaptationAction(
            action_id=f"adapt_lr_{uuid.uuid4().hex[:8]}",
            adaptation_type=AdaptationType.LEARNING_RATE,
            target_parameter="learning_rate",
            current_value=current,
            target_value=target,
            delta=target - current,
            reason=reason,
            confidence=confidence,
        )

    def _adapt_mutation_rate(
        self,
        state: ParameterState,
        trend: PerformanceTrend,
        slope: float,
        dna_id: str,
    ) -> Optional[AdaptationAction]:
        """适应变异率"""
        import uuid

        current = state.current_value
        confidence = 0.5

        if trend == PerformanceTrend.DECLINING:
            target = min(self.MUTATION_RATE_MAX, current * 2.0)
            reason = f"性能下降，增加变异率以探索更多可能性"
            confidence = 0.8

        elif trend == PerformanceTrend.IMPROVING:
            target = max(self.MUTATION_RATE_MIN, current * 0.7)
            reason = f"性能提升，降低变异率以利用当前方向"
            confidence = 0.7

        elif trend == PerformanceTrend.VOLATILE:
            target = max(self.MUTATION_RATE_MIN, current * 0.5)
            reason = "性能波动，降低变异率以稳定系统"
            confidence = 0.6

        else:
            if current < 0.05:
                target = min(self.MUTATION_RATE_INITIAL, current * 1.5)
                reason = "变异率过低，适度增加以保持探索能力"
                confidence = 0.5
            else:
                return None

        target = max(self.MUTATION_RATE_MIN, min(self.MUTATION_RATE_MAX, target))

        if abs(target - current) < 0.0001:
            return None

        return AdaptationAction(
            action_id=f"adapt_mr_{uuid.uuid4().hex[:8]}",
            adaptation_type=AdaptationType.MUTATION_RATE,
            target_parameter="mutation_rate",
            current_value=current,
            target_value=target,
            delta=target - current,
            reason=reason,
            confidence=confidence,
        )

    def _adapt_capability_weight(
        self,
        state: ParameterState,
        cap_name: str,
        avg_usage: float,
        correlation: float,
        dna_id: str,
    ) -> Optional[AdaptationAction]:
        """适应能力权重"""
        import uuid

        current = state.current_value

        if correlation > 0.1:
            target = min(1.0, current + correlation * 0.1)
            reason = f"{cap_name} 能力与成功率正相关 (r={correlation:.2f})，增加权重"
            confidence = 0.6 + abs(correlation) * 0.3

        elif correlation < -0.1:
            target = max(0.0, current + correlation * 0.1)
            reason = f"{cap_name} 能力与成功率负相关 (r={correlation:.2f})，降低权重"
            confidence = 0.6 + abs(correlation) * 0.3

        else:
            return None

        target = max(state.min_value, min(state.max_value, target))

        if abs(target - current) < 0.01:
            return None

        return AdaptationAction(
            action_id=f"adapt_cap_{uuid.uuid4().hex[:8]}",
            adaptation_type=AdaptationType.CAPABILITY_WEIGHT,
            target_parameter=state.name,
            current_value=current,
            target_value=target,
            delta=target - current,
            reason=reason,
            confidence=confidence,
        )

    def apply_adaptation(
        self,
        dna_id: str,
        action: AdaptationAction,
    ) -> bool:
        """
        应用适应动作

        Args:
            dna_id: DNA 标识
            action: 适应动作

        Returns:
            bool - 是否成功应用
        """
        with self._lock:
            if dna_id not in self._monitored_dnas:
                return False

            monitored = self._monitored_dnas[dna_id]
            dna = monitored.get("dna")

            if dna is None:
                return False

            param_name = action.target_parameter

            if not hasattr(dna, param_name):
                logger.warning(f"🧬 [DNAAdaptiveEngine] DNA {dna_id} 没有参数 {param_name}")
                return False

            current_value = getattr(dna, param_name)
            target_value = action.target_value

            setattr(dna, param_name, target_value)

            action.applied = True
            action.applied_at = time.time()
            action.result = {
                "old_value": current_value,
                "new_value": target_value,
            }

            if dna_id in self._parameter_states:
                state = self._parameter_states[dna_id].get(param_name)
                if state:
                    state.record_value(target_value, time.time())
                    state.adaptation_count += 1

            monitored["last_adaptation_at"] = time.time()

            if dna_id not in self._adaptation_history:
                self._adaptation_history[dna_id] = []
            self._adaptation_history[dna_id].append(action)

            for callback in self._adaptation_callbacks:
                try:
                    callback(dna_id, action)
                except Exception as e:
                    logger.warning(f"🧬 [DNAAdaptiveEngine] 回调执行失败: {e}")

            logger.info(
                f"🧬 [DNAAdaptiveEngine] 应用适应: {dna_id} -> "
                f"{param_name}: {current_value:.4f} -> {target_value:.4f}"
            )

            return True

    def apply_all_suggested(
        self,
        dna_id: str,
        min_confidence: float = 0.6,
        max_actions: int = 5,
    ) -> int:
        """
        应用所有建议的适应动作

        Args:
            dna_id: DNA 标识
            min_confidence: 最小置信度阈值
            max_actions: 最大动作数量

        Returns:
            int - 成功应用的动作数量
        """
        actions = self.suggest_adaptations(dna_id)

        filtered = [a for a in actions if a.confidence >= min_confidence][:max_actions]

        applied_count = 0
        for action in filtered:
            if self.apply_adaptation(dna_id, action):
                applied_count += 1

        return applied_count

    def add_adaptation_callback(
        self,
        callback: Callable[[str, AdaptationAction], None],
    ):
        """添加适应动作回调"""
        self._adaptation_callbacks.append(callback)

    def get_status(self, dna_id: Optional[str] = None) -> Dict[str, Any]:
        """获取状态"""
        with self._lock:
            if dna_id:
                if dna_id not in self._monitored_dnas:
                    return {}

                history = self._performance_history.get(dna_id, deque())
                states = self._parameter_states.get(dna_id, {})
                adaptations = self._adaptation_history.get(dna_id, [])

                return {
                    "dna_id": dna_id,
                    "monitored": True,
                    "performance_samples": len(history),
                    "parameter_states": {name: state.to_dict() for name, state in states.items()},
                    "adaptation_count": len(adaptations),
                    "recent_adaptations": [a.to_dict() for a in adaptations[-5:]],
                }

            return {
                "total_monitored": len(self._monitored_dnas),
                "monitored_dnas": list(self._monitored_dnas.keys()),
                "auto_adapt_enabled": self._auto_adapt_enabled,
            }

    def start_auto_adapt(self, interval_seconds: float = 300.0):
        """启动自动适应"""
        if self._running:
            return

        self._auto_adapt_enabled = True
        self._running = True
        self._adapt_thread = threading.Thread(
            target=self._auto_adapt_loop,
            args=(interval_seconds,),
            daemon=True,
            name="dna_auto_adapt",
        )
        self._adapt_thread.start()

        logger.info(f"🧬 [DNAAdaptiveEngine] 自动适应已启动，间隔: {interval_seconds} 秒")

    def stop_auto_adapt(self):
        """停止自动适应"""
        self._auto_adapt_enabled = False
        self._running = False

        if self._adapt_thread:
            self._adapt_thread.join(timeout=5.0)

        logger.info("🧬 [DNAAdaptiveEngine] 自动适应已停止")

    def _auto_adapt_loop(self, interval_seconds: float):
        """自动适应循环"""
        while self._running:
            try:
                time.sleep(interval_seconds)

                if not self._auto_adapt_enabled:
                    continue

                with self._lock:
                    dna_ids = list(self._monitored_dnas.keys())

                for dna_id in dna_ids:
                    try:
                        applied = self.apply_all_suggested(
                            dna_id, min_confidence=0.7, max_actions=3
                        )
                        if applied > 0:
                            logger.info(
                                f"🧬 [DNAAdaptiveEngine] 自动适应: {dna_id} 应用了 {applied} 个动作"
                            )
                    except Exception as e:
                        logger.error(f"🧬 [DNAAdaptiveEngine] 自动适应错误 {dna_id}: {e}")

            except Exception as e:
                logger.error(f"🧬 [DNAAdaptiveEngine] 自动适应循环错误: {e}")


_engine: Optional[DNAAdaptiveEngine] = None
_engine_lock = threading.Lock()


def get_dna_adaptive_engine() -> DNAAdaptiveEngine:
    """获取 DNA 自适应引擎单例"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = DNAAdaptiveEngine()
    return _engine
