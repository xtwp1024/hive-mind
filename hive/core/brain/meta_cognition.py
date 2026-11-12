"""
MetaCognitionLayer - 元认知层
============================

高层认知能力，包括:
- 注意力分配优化
- 认知资源调度
- 自我建模
- 认知策略优化

Usage:
    meta = MetaCognitionLayer(hive_brain)
    allocation = await meta.optimize_attentionallocation()
"""

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Hive.Brain.MetaCognition")

# 可自动调整的参数配置
AUTOTUNABLE_PARAMS = {
    "ATTENTION_DECAY_RATE": {"range": (0.8, 0.99), "type": "float"},
    "reflection_threshold": {"range": (0.1, 0.5), "type": "float"},
    "max_think_depth": {"range": (5, 20), "type": "int"},
    "working_memory_capacity": {"range": (30, 100), "type": "int"},
}


@dataclass
class PerformanceMetrics:
    """性能指标"""

    task_type: str
    success_rate: float
    avg_duration: float
    confidence_delta: float  # 预期 vs 实际置信度差异
    reflection_accuracy: float  # 反思洞察被验证的比例


@dataclass
class Adjustment:
    """调整记录"""

    parameter: str
    current_value: Any
    suggested_value: Any
    reason: str
    confidence: float
    applied_at: float = field(default_factory=time.time)
    verified: bool = False


@dataclass
class AttentionAllocation:
    """注意力分配"""

    focus_areas: List[str] = field(default_factory=list)  # 重点关注领域
    deprioritized_areas: List[str] = field(default_factory=list)  # 降低优先级领域
    suggested_strategies: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.5


@dataclass
class CognitiveModel:
    """
    认知模型

    大脑对自身的建模
    """

    strengths: List[str] = field(default_factory=list)  # 优势
    weaknesses: List[str] = field(default_factory=list)  # 劣势
    preferred_modes: List[str] = field(default_factory=list)  # 偏好模式
    learned_patterns: Dict[str, int] = field(default_factory=dict)  # 学习到的模式
    optimal_contexts: Dict[str, float] = field(default_factory=dict)  # 最优场景

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "preferred_modes": self.preferred_modes,
            "learned_patterns": dict(
                sorted(self.learned_patterns.items(), key=lambda x: -x[1])[:10]
            ),
            "optimal_contexts": self.optimal_contexts,
        }


@dataclass
class OptimizationResult:
    """优化结果"""

    success: bool
    allocation: Optional[AttentionAllocation] = None
    model_updates: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


_UNSET = object()


class SelfImprovementEngine:
    """
    自我改进引擎

    分析表现历史，识别模式和弱点，提出并应用参数调整
    """

    def __init__(self, hive_brain):
        self.brain = hive_brain
        self.performance_history: List[PerformanceMetrics] = []
        self.parameter_adjustments: List[Adjustment] = []
        self._last_analysis_time = 0.0
        self._analysis_interval = 300.0  # 5分钟分析一次
        self._pending_verifications: List[Adjustment] = []

    async def analyze_and_adjust(self) -> List[Adjustment]:
        """
        分析表现并调整参数

        Returns:
            List[Adjustment]: 应用的调整列表
        """
        now = time.time()
        if now - self._last_analysis_time < self._analysis_interval:
            return []

        self._last_analysis_time = now

        recent = (
            self.performance_history[-50:]
            if len(self.performance_history) >= 50
            else self.performance_history
        )
        if len(recent) < 5:
            return []

        applied = []
        patterns = self._analyze_patterns(recent)

        # 分析每个可调参数
        for param, config in AUTOTUNABLE_PARAMS.items():
            current = self._get_param_value(param)
            if current is None:
                continue

            adjustment = self._suggest_adjustment(param, current, patterns)
            if adjustment:
                await self._apply_adjustment(adjustment)
                await self._schedule_verification(adjustment)
                applied.append(adjustment)

        return applied

    def _analyze_patterns(self, recent: List[PerformanceMetrics]) -> Dict[str, Any]:
        """
        识别模式和弱点

        Args:
            recent: 最近的性能指标

        Returns:
            Dict: 分析结果
        """
        analysis: Dict[str, Any] = {
            "weak_task_types": [],
            "declining_success_rates": {},
            "duration_trends": {},
            "low_reflection_accuracy": [],
        }

        if not recent:
            return analysis

        # 按任务类型分组
        by_type: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        for m in recent:
            by_type[m.task_type].append(m)

        # 分析每种任务类型
        for task_type, metrics in by_type.items():
            if not metrics:
                continue
            success_rates = [m.success_rate for m in metrics]
            avg_success = sum(success_rates) / len(success_rates)
            avg_duration = sum(m.avg_duration for m in metrics) / len(metrics)
            avg_reflection = sum(m.reflection_accuracy for m in metrics) / len(metrics)

            # 低成功率
            if avg_success < 0.6:
                analysis["weak_task_types"].append(task_type)

            # 低反思准确率
            if avg_reflection < 0.4:
                analysis["low_reflection_accuracy"].append(task_type)

            # 记录趋势（简化版：比较前后半段）
            mid = len(metrics) // 2
            if mid >= 2:
                first_half = sum(m.success_rate for m in metrics[:mid]) / mid
                second_half = sum(m.success_rate for m in metrics[mid:]) / (len(metrics) - mid)
                if second_half < first_half - 0.1:
                    analysis["declining_success_rates"][task_type] = second_half - first_half

        return analysis

    def _suggest_adjustment(
        self, param: str, current: Any, analysis: Dict[str, Any]
    ) -> Optional[Adjustment]:
        """
        建议参数调整

        Args:
            param: 参数名
            current: 当前值
            analysis: 分析结果

        Returns:
            Optional[Adjustment]: 调整建议，无调整则返回None
        """
        config = AUTOTUNABLE_PARAMS.get(param)
        if not config:
            return None

        low, high = config["range"]
        param_type = config["type"]

        reason = ""
        suggested_value = current
        confidence = 0.5

        # 基于分析结果调整
        if param == "ATTENTION_DECAY_RATE":
            if analysis["weak_task_types"]:
                # 成功率低，提高衰减阈值（更少走神）
                suggested_value = min(high, current + 0.02)
                reason = f"弱任务类型 {analysis['weak_task_types'][0]} 成功率低，提高注意力持续性"
                confidence = 0.65

        elif param == "reflection_threshold":
            if analysis["low_reflection_accuracy"]:
                # 反思准确率低，提高阈值（更严格的反思）
                suggested_value = min(high, current + 0.05)
                reason = f"反思准确率低，更严格反思标准"
                confidence = 0.6
            elif analysis["declining_success_rates"]:
                # 成功率下降，降低阈值（更多反思）
                suggested_value = max(low, current - 0.05)
                reason = "成功率下降，增加反思频率"
                confidence = 0.55

        elif param == "max_think_depth":
            weak_tasks = analysis.get("weak_task_types", [])
            if weak_tasks:
                # 复杂任务失败多，增加思考深度
                suggested_value = min(high, current + 2)
                reason = f"复杂任务 {weak_tasks[0]} 表现不佳，增加思考深度"
                confidence = 0.6

        elif param == "working_memory_capacity":
            if analysis["low_reflection_accuracy"]:
                # 反思准确率低可能是记忆不足
                suggested_value = min(high, current + 10)
                reason = "反思准确率低，增加工作记忆容量"
                confidence = 0.55

        # 检查是否有实际变化
        if suggested_value == current:
            return None

        # 验证范围
        if param_type == "float":
            suggested_value = max(low, min(high, float(suggested_value)))
        elif param_type == "int":
            suggested_value = max(low, min(high, int(suggested_value)))

        return Adjustment(
            parameter=param,
            current_value=current,
            suggested_value=suggested_value,
            reason=reason,
            confidence=confidence,
        )

    async def _apply_adjustment(self, adjustment: Adjustment):
        """
        应用参数调整（带安全检查）

        Args:
            adjustment: 调整记录
        """
        # 安全检查：置信度太低不应用
        if adjustment.confidence < 0.5:
            logger.info(f"调整 {adjustment.parameter} 置信度 {adjustment.confidence} 过低，跳过")
            return

        # 检查是否已有相同参数的待验证调整
        for pending in self._pending_verifications:
            if pending.parameter == adjustment.parameter:
                logger.info(f"参数 {adjustment.parameter} 已有待验证调整，跳过")
                return

        # 应用调整
        old_value = self._set_param_value(adjustment.parameter, adjustment.suggested_value)
        adjustment.current_value = old_value
        self.parameter_adjustments.append(adjustment)
        self._pending_verifications.append(adjustment)

        logger.info(
            f"应用参数调整: {adjustment.parameter} "
            f"{old_value} -> {adjustment.suggested_value} "
            f"(置信度: {adjustment.confidence:.2f})"
        )

    async def _schedule_verification(self, adjustment: Adjustment):
        """计划验证"""
        # 简化为在下次分析时验证
        # 实际实现可以放到队列中延迟执行
        pass

    def _get_param_value(self, param: str) -> Any:
        """获取参数当前值"""
        # 尝试从 hive_brain 获取
        value = getattr(self.brain, param, _UNSET)
        if value is not _UNSET:
            return value
        _config = getattr(self.brain, "_config", None)
        if _config is not None and param in _config:
            return _config.get(param)
        config = getattr(self.brain, "config", None)
        if config is not None:
            return getattr(config, param, None)
        return None

    def _set_param_value(self, param: str, value: Any) -> Any:
        """设置参数值，返回旧值"""
        old_value = self._get_param_value(param)

        if getattr(self.brain, param, _UNSET) is not _UNSET:
            setattr(self.brain, param, value)
        elif getattr(self.brain, "_config", None) is not None:
            self.brain._config[param] = value
        elif getattr(self.brain, "config", None) is not None:
            if getattr(self.brain.config, param, _UNSET) is not _UNSET:
                setattr(self.brain.config, param, value)

        return old_value

    def record_performance(self, metrics: PerformanceMetrics):
        """记录性能指标"""
        self.performance_history.append(metrics)
        # 保持历史在合理范围
        if len(self.performance_history) > 500:
            self.performance_history = self.performance_history[-400:]

    def get_adjustment_history(self) -> List[Adjustment]:
        """获取调整历史"""
        return self.parameter_adjustments

    def verify_adjustment(self, param: str, success: bool):
        """
        验证调整效果

        Args:
            param: 参数名
            success: 是否成功
        """
        for adj in reversed(self._pending_verifications):
            if adj.parameter == param:
                adj.verified = success
                if not success:
                    # 回滚
                    self._set_param_value(param, adj.current_value)
                    logger.info(f"调整 {param} 效果不佳，回滚到 {adj.current_value}")
                self._pending_verifications.remove(adj)
                break


class MetaCognitionLayer:
    """
    元认知层

    提供高层认知能力:
    1. 注意力分配优化
    2. 认知资源调度
    3. 自我建模
    4. 认知策略优化
    """

    # 认知模式偏好
    MODE_PREFERENCES = {
        "analysis": {"system2": 0.8, "system1": 0.2},
        "creation": {"system2": 0.7, "system1": 0.3},
        "review": {"system2": 0.6, "system1": 0.4},
        "optimization": {"system2": 0.9, "system1": 0.1},
        "debug": {"system2": 0.5, "system1": 0.5},
        "general": {"system2": 0.5, "system1": 0.5},
    }

    def __init__(self, hive_brain):
        self.hive_brain = hive_brain

        # 认知模型
        self.model = CognitiveModel(
            strengths=["pattern_recognition", "goal_decomposition"],
            weaknesses=["long_term_planning", "abstract_reasoning"],
            preferred_modes=["system2_deliberative"],
        )

        # 学习历史（使用 deque 自动管理长度，O(1) append/pop）
        self._max_history = 200
        self._learning_history = deque(maxlen=self._max_history)

        # 统计
        self._stats = {
            "optimizations": 0,
            "patterns_learned": 0,
            "adaptations": 0,
        }

        # 自我改进引擎
        self.self_improvement = SelfImprovementEngine(hive_brain)

        logger.info("MetaCognitionLayer initialized")

    async def optimize_attention_allocation(
        self,
        current_tasks: List[str],
        cognitive_state: Dict[str, Any],
    ) -> AttentionAllocation:
        """
        优化注意力分配

        分析当前状态，推荐应该重点关注什么

        Args:
            current_tasks: 当前任务列表
            cognitive_state: 当前认知状态

        Returns:
            AttentionAllocation: 注意力分配建议
        """
        allocation = AttentionAllocation()

        # 分析当前负载
        attention_load = cognitive_state.get("attention_load", 0.5)
        memory_pressure = cognitive_state.get("memory_pressure", 0.5)

        # 高负载时，聚焦核心任务
        if attention_load > 0.7:
            allocation.focus_areas = [current_tasks[0]] if current_tasks else []
            allocation.deprioritized_areas = current_tasks[3:] if len(current_tasks) > 3 else []
            allocation.reasoning = "高注意力负载，聚焦核心任务"
            allocation.confidence = 0.8

        # 高记忆压力时，减少新信息摄入
        elif memory_pressure > 0.7:
            allocation.focus_areas = ["consolidation"]
            allocation.deprioritized_areas = ["new_learning", "exploration"]
            allocation.reasoning = "高记忆压力，优先整合现有知识"
            allocation.confidence = 0.75

        # 正常状态，多元探索
        else:
            allocation.focus_areas = current_tasks[:3] if len(current_tasks) >= 3 else current_tasks
            allocation.deprioritized_areas = []
            allocation.reasoning = "正常认知状态，保持平衡"
            allocation.confidence = 0.6

        # 基于任务类型调整
        if current_tasks:
            first_task = current_tasks[0].lower()
            if "分析" in first_task:
                allocation.suggested_strategies = ["深度分析", "分步推理"]
            elif "创建" in first_task or "实现" in first_task:
                allocation.suggested_strategies = ["模块化", "增量开发"]
            elif "优化" in first_task:
                allocation.suggested_strategies = ["性能分析", "瓶颈定位"]

        self._stats["optimizations"] += 1
        return allocation

    async def learn_from_interaction(
        self,
        input_text: str,
        intent: str,
        success: bool,
        duration: float,
    ) -> OptimizationResult:
        """
        从交互中学习

        分析成功和失败，更新认知模型

        Args:
            input_text: 输入文本
            intent: 识别的意图
            success: 是否成功
            duration: 执行时长

        Returns:
            OptimizationResult: 优化结果
        """
        result = OptimizationResult(success=True)
        model_updates = []

        # 学习成功模式
        if success and duration < 60:
            key = f"{intent}_fast_success"
            self.model.learned_patterns[key] = self.model.learned_patterns.get(key, 0) + 1
            model_updates.append(f"学会: {intent}类任务快速完成")

            self._stats["patterns_learned"] += 1

        # 学习失败模式
        if not success:
            key = f"{intent}_failure"
            self.model.learned_patterns[key] = self.model.learned_patterns.get(key, 0) + 1
            model_updates.append(f"注意: {intent}类任务容易失败")

            # 记录弱点
            if intent not in self.model.weaknesses:
                self.model.weaknesses.append(intent)
                model_updates.append(f"识别弱点: {intent}")

        # 学习最优场景
        if success:
            key = f"{intent}_optimal"
            self.model.optimal_contexts[key] = self.model.optimal_contexts.get(key, 0) + 0.1
            if self.model.optimal_contexts[key] > 1.0:
                self.model.optimal_contexts[key] = 1.0

        # 保存学习历史（deque 自动管理长度，O(1) 复杂度）
        self._learning_history.append(
            {
                "timestamp": time.time(),
                "input": input_text[:50],
                "intent": intent,
                "success": success,
                "duration": duration,
            }
        )

        result.model_updates = model_updates
        result.recommendations = self._generate_recommendations(intent, success)

        return result

    def _generate_recommendations(self, intent: str, success: bool) -> List[str]:
        """生成建议"""
        recommendations = []

        if success:
            recommendations.append("继续保持当前策略")
        else:
            # 针对特定意图的建议
            suggestions = {
                "analysis": "分析任务建议增加推理时间",
                "creation": "创建任务建议先设计后实现",
                "review": "审查任务建议使用System2深度分析",
                "debug": "调试任务建议先复现问题",
                "optimization": "优化任务建议先定位瓶颈",
            }
            recommendations.append(suggestions.get(intent, "建议使用更谨慎的策略"))

        return recommendations

    async def adapt_strategy(
        self,
        current_strategy: Dict[str, Any],
        feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        适应策略

        根据反馈调整认知策略

        Args:
            current_strategy: 当前策略
            feedback: 反馈

        Returns:
            Dict: 调整后的策略
        """
        adapted = current_strategy.copy()
        self._stats["adaptations"] += 1

        success = feedback.get("success", False)
        duration = feedback.get("duration", 0)
        expected_duration = feedback.get("expected_duration", 60)

        # 如果超时，增加时间预算
        if duration > expected_duration * 0.9:
            new_max_duration = current_strategy.get("max_duration", 60) * 1.5
            adapted["max_duration"] = new_max_duration
            adapted["adaptation_reason"] = "延长执行时间"
            self._stats["adaptations"] += 1

        # 如果失败，切换策略
        if not success:
            current_mode = current_strategy.get("mode", "system2")
            if current_mode == "system1":
                adapted["mode"] = "system2"
                adapted["execution_mode"] = "deliberative"
                adapted["adaptation_reason"] = "System1失败，切换System2"
            else:
                # 增加注意力投入
                attention = current_strategy.get("attention_level", 0.5)
                adapted["attention_level"] = min(1.0, attention + 0.2)
                adapted["adaptation_reason"] = "增加注意力投入"

            self._stats["adaptations"] += 1

        return adapted

    async def predict_difficulty(self, task: str) -> float:
        """
        预测任务难度

        Args:
            task: 任务描述

        Returns:
            float: 难度评分 (0-1)
        """
        difficulty = 0.5  # 基础难度

        # 长度指标
        if len(task) > 200:
            difficulty += 0.15
        elif len(task) > 100:
            difficulty += 0.1

        # 复杂度关键词
        complex_keywords = [
            "分析",
            "设计",
            "实现",
            "重构",
            "优化",
            "系统",
            "架构",
            "分布式",
            "并发",
        ]
        complex_count = sum(1 for kw in complex_keywords if kw in task)
        difficulty += complex_count * 0.05

        # 多步骤指示
        multi_step_keywords = ["然后", "接着", "再", "并且"]
        if any(kw in task for kw in multi_step_keywords):
            difficulty += 0.15

        # 历史失败
        for entry in self._learning_history[-20:]:
            if task[:30] in entry.get("input", "") and not entry.get("success"):
                difficulty += 0.1
                break

        return min(1.0, difficulty)

    def get_cognitive_model(self) -> CognitiveModel:
        """获取认知模型"""
        return self.model

    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        total = len(self._learning_history)
        successes = sum(1 for e in self._learning_history if e.get("success"))

        return {
            **self._stats,
            "total_interactions": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "recent_learning_size": len(self._learning_history[-20:]),
        }

    async def run_self_check(self) -> Dict[str, Any]:
        """
        运行自我检查

        周期性检查认知状态，并触发自我改进
        """
        check_result = {
            "timestamp": time.time(),
            "model_health": "healthy",
            "recommendations": [],
            "adjustments_applied": [],
        }

        # 检查弱点数量
        if len(self.model.weaknesses) > 10:
            check_result["recommendations"].append("弱点过多，考虑简化任务")

        # 检查学习是否停滞
        recent = self._learning_history[-50:]
        if len(recent) >= 50:
            recent_success_rate = sum(1 for e in recent if e.get("success")) / 50
            if recent_success_rate < 0.5:
                check_result["recommendations"].append("成功率较低，建议调整策略")

        # 检查模式是否过度集中
        if self.model.learned_patterns:
            max_pattern = max(self.model.learned_patterns.values())
            total = sum(self.model.learned_patterns.values())
            if max_pattern / total > 0.8:
                check_result["recommendations"].append("模式过度集中，考虑扩展能力")

        # 触发自我改进分析
        try:
            adjustments = await self.self_improvement.analyze_and_adjust()
            if adjustments:
                check_result["adjustments_applied"] = [
                    {
                        "parameter": a.parameter,
                        "from": a.current_value,
                        "to": a.suggested_value,
                        "reason": a.reason,
                    }
                    for a in adjustments
                ]
                check_result["model_health"] = "improving"
        except Exception as e:
            logger.warning(f"自我改进分析失败: {e}")

        return check_result
