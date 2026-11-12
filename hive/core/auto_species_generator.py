# -*- coding: utf-8 -*-
"""
Auto Species Generator - 自动蜂种生成器

Phase 3: 自主进化能力
基于能力缺口分析自动生成新的蜂种蓝图。

核心功能:
1. 缺口分析驱动 - 从 CapabilityGapAnalyzer 获取进化需求
2. 蓝图模板系统 - 基于现有成功蓝图创建新蓝图
3. 参数优化 - 遗传算法优化蓝图参数
4. 蓝图验证 - 自动验证生成的蓝图
5. 孵化调度 - 将验证通过的蓝图提交给 Incubator

工作流程:
1. 收集缺口分析 → 2. 确定进化目标 → 3. 生成候选蓝图
   → 4. 优化筛选 → 5. 验证 → 6. 注册/孵化
"""

import logging
import random
import statistics
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    # 以下类只用于类型注解；运行时仍由各函数按需延迟导入，避免拉长启动链
    from hive.core.capability_gap_analyzer import CapabilityGapAnalyzer, GapAnalysisReport
    from hive.core.species_blueprint import (
        BlueprintRegistry,
        BlueprintValidationResult,
        SpeciesBlueprint,
    )

logger = logging.getLogger("Hive.AutoSpeciesGenerator")


class GenerationStrategy(Enum):
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    HYBRID = "hybrid"
    NOVEL = "novel"


class OptimizationObjective(Enum):
    SUCCESS_RATE = "success_rate"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    BALANCED = "balanced"


@dataclass
class GenerationResult:
    """生成结果"""

    generation_id: str
    strategy: GenerationStrategy

    candidate_blueprints: List[Tuple["SpeciesBlueprint", float]] = field(default_factory=list)
    selected_blueprints: List["SpeciesBlueprint"] = field(default_factory=list)

    validation_results: Dict[str, "BlueprintValidationResult"] = field(default_factory=dict)

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "strategy": self.strategy.value,
            "candidate_count": len(self.candidate_blueprints),
            "selected_count": len(self.selected_blueprints),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time else None,
            "metadata": self.metadata,
        }


@dataclass
class EvolutionTarget:
    """进化目标"""

    target_id: str

    capability_name: str
    task_category: str

    current_success_rate: float
    target_success_rate: float

    current_efficiency: Optional[float] = None
    target_efficiency: Optional[float] = None

    priority: float = 0.5
    urgency: str = "medium"

    source_gap_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "capability_name": self.capability_name,
            "task_category": self.task_category,
            "current_success_rate": self.current_success_rate,
            "target_success_rate": self.target_success_rate,
            "current_efficiency": self.current_efficiency,
            "target_efficiency": self.target_efficiency,
            "priority": self.priority,
            "urgency": self.urgency,
            "source_gap_id": self.source_gap_id,
        }


class BlueprintOptimizer:
    """
    蓝图优化器

    使用遗传算法优化蓝图参数。
    """

    def __init__(self, population_size: int = 20, generations: int = 10):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.2
        self.crossover_rate = 0.7

        self._evaluation_cache: Dict[str, float] = {}

    def optimize(
        self,
        base_blueprint: "SpeciesBlueprint",
        objectives: List[OptimizationObjective],
        evaluator,
    ) -> "SpeciesBlueprint":
        """
        优化蓝图

        Args:
            base_blueprint: 基础蓝图
            objectives: 优化目标列表
            evaluator: 评估函数，接收蓝图返回适应度分数

        Returns:
            优化后的蓝图
        """
        logger.info(f"🧬 [BlueprintOptimizer] 开始优化蓝图: {base_blueprint.name}")

        population = self._initialize_population(base_blueprint)

        best_fitness = 0.0
        best_individual = deepcopy(base_blueprint)

        for gen in range(self.generations):
            fitness_scores = [self._evaluate_individual(ind, evaluator) for ind in population]

            current_best_idx = fitness_scores.index(max(fitness_scores))
            current_best_fitness = fitness_scores[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = deepcopy(population[current_best_idx])
                logger.debug(
                    f"🧬 [BlueprintOptimizer] 第 {gen+1} 代找到更好的个体, 适应度: {best_fitness:.4f}"
                )

            population = self._evolve_population(population, fitness_scores)

            if best_fitness >= 0.95:
                logger.info(f"🧬 [BlueprintOptimizer] 适应度已达到阈值，提前结束优化")
                break

        logger.info(f"🧬 [BlueprintOptimizer] 优化完成, 最佳适应度: {best_fitness:.4f}")

        return best_individual

    def _initialize_population(self, base: "SpeciesBlueprint") -> List["SpeciesBlueprint"]:
        """初始化种群"""
        population = [deepcopy(base)]

        while len(population) < self.population_size:
            mutated = base.mutate(self.mutation_rate * 1.5)
            population.append(mutated)

        return population

    def _evaluate_individual(self, individual: "SpeciesBlueprint", evaluator) -> float:
        """评估个体适应度"""
        cache_key = f"{individual.blueprint_id}_{hash(str(individual.to_dict()))}"

        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]

        fitness = evaluator(individual)
        self._evaluation_cache[cache_key] = fitness

        return fitness

    def _evolve_population(
        self, population: List["SpeciesBlueprint"], fitness_scores: List[float]
    ) -> List["SpeciesBlueprint"]:
        """进化种群"""
        new_population = []

        sorted_indices = sorted(
            range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
        )

        elite_count = max(2, int(self.population_size * 0.1))
        for i in range(elite_count):
            new_population.append(deepcopy(population[sorted_indices[i]]))

        while len(new_population) < self.population_size:
            parent1 = self._select_parent(population, fitness_scores)
            parent2 = self._select_parent(population, fitness_scores)

            if random.random() < self.crossover_rate and parent1 != parent2:
                child1, child2 = parent1.crossover(parent2)
                if random.random() < self.mutation_rate:
                    child1 = child1.mutate(self.mutation_rate)
                if random.random() < self.mutation_rate:
                    child2 = child2.mutate(self.mutation_rate)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            else:
                child = parent1.mutate(self.mutation_rate)
                new_population.append(child)

        return new_population

    def _select_parent(
        self, population: List["SpeciesBlueprint"], fitness_scores: List[float]
    ) -> "SpeciesBlueprint":
        """选择父代（锦标赛选择）"""
        tournament_size = 3
        indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]


class AutoSpeciesGenerator:
    """
    自动蜂种生成器

    Phase 3 核心组件。基于能力缺口分析自动生成新的蜂种蓝图。

    主要功能:
    1. 从 CapabilityGapAnalyzer 获取进化需求
    2. 基于现有蓝图生成新蓝图（变异/交叉/混合）
    3. 使用遗传算法优化蓝图参数
    4. 自动验证生成的蓝图
    5. 将验证通过的蓝图注册到 BlueprintRegistry

    使用方式:
        generator = AutoSpeciesGenerator()
        result = generator.generate_from_gap_analysis(
            gap_analyzer,
            max_blueprints=3
        )
    """

    def __init__(
        self,
        registry: Optional["BlueprintRegistry"] = None,
        optimizer: Optional[BlueprintOptimizer] = None,
    ):
        self._registry = registry or self._get_registry()
        self._optimizer = optimizer or BlueprintOptimizer()

        self._generation_history: List[GenerationResult] = []
        self._lock = threading.Lock()

        self._default_objectives = [
            OptimizationObjective.SUCCESS_RATE,
            OptimizationObjective.ROBUSTNESS,
        ]

        logger.info("🧬 [AutoSpeciesGenerator] 自动蜂种生成器已初始化")

    def _get_registry(self) -> "BlueprintRegistry":
        """获取蓝图注册表"""
        try:
            from hive.core.species_blueprint import get_blueprint_registry

            return get_blueprint_registry()
        except ImportError:
            from hive.core.species_blueprint import BlueprintRegistry

            return BlueprintRegistry()

    def generate_from_gap_analysis(
        self,
        gap_analyzer: "CapabilityGapAnalyzer",
        max_blueprints: int = 3,
        strategy: GenerationStrategy = GenerationStrategy.HYBRID,
        analysis_window_hours: float = 24.0,
    ) -> GenerationResult:
        """
        基于能力缺口分析生成新蜂种

        Args:
            gap_analyzer: 能力缺口分析器
            max_blueprints: 最大生成蓝图数量
            strategy: 生成策略
            analysis_window_hours: 分析时间窗口

        Returns:
            GenerationResult - 生成结果
        """
        logger.info(f"🧬 [AutoSpeciesGenerator] 基于缺口分析生成新蜂种, 策略: {strategy.value}")

        result = GenerationResult(
            generation_id=f"gen_{int(time.time())}_{uuid.uuid4().hex[:6]}",
            strategy=strategy,
        )

        report = gap_analyzer.analyze(analysis_window_hours=analysis_window_hours)

        if report.total_gaps_detected == 0:
            logger.info("🧬 [AutoSpeciesGenerator] 未检测到能力缺口，无需生成新蜂种")
            result.end_time = time.time()
            result.metadata["reason"] = "no_gaps_detected"
            return result

        evolution_targets = self._extract_evolution_targets(report)

        if not evolution_targets:
            logger.info("🧬 [AutoSpeciesGenerator] 没有可操作的进化目标")
            result.end_time = time.time()
            result.metadata["reason"] = "no_actionable_targets"
            return result

        result.metadata["evolution_targets"] = [t.to_dict() for t in evolution_targets]

        candidate_blueprints = self._generate_candidates(
            evolution_targets, strategy, max_blueprints * 2
        )

        result.candidate_blueprints = [
            (bp, self._estimate_blueprint_fitness(bp, evolution_targets))
            for bp in candidate_blueprints
        ]

        validated_blueprints = []
        for blueprint, _ in result.candidate_blueprints:
            validation = blueprint.validate()
            result.validation_results[blueprint.blueprint_id] = validation

            if validation.is_valid:
                validated_blueprints.append((blueprint, validation.overall_score))

        validated_blueprints.sort(key=lambda x: x[1], reverse=True)

        selected = validated_blueprints[:max_blueprints]
        result.selected_blueprints = [bp for bp, _ in selected]

        for blueprint in result.selected_blueprints:
            self._registry.register(blueprint)
            self._registry.update_status(blueprint.blueprint_id, "validated")

            logger.info(
                f"🧬 [AutoSpeciesGenerator] 注册新蓝图: {blueprint.name} ({blueprint.blueprint_id})"
            )

        result.end_time = time.time()
        self._generation_history.append(result)

        logger.info(
            f"🧬 [AutoSpeciesGenerator] 生成完成: {len(result.selected_blueprints)} 个蓝图已注册"
        )

        return result

    def _extract_evolution_targets(self, report: "GapAnalysisReport") -> List[EvolutionTarget]:
        """从缺口报告提取进化目标"""
        targets = []

        all_gaps = report.critical_gaps + report.high_gaps

        for gap in all_gaps:
            target_success_rate = min(0.95, gap.success_rate + gap.estimated_improvement)

            target = EvolutionTarget(
                target_id=f"target_{uuid.uuid4().hex[:8]}",
                capability_name=gap.capability_name,
                task_category=gap.task_category,
                current_success_rate=gap.success_rate,
                target_success_rate=target_success_rate,
                current_efficiency=gap.efficiency_ratio,
                target_efficiency=1.0 if gap.efficiency_ratio else None,
                priority=gap.remediation_priority,
                urgency=gap.severity.value,
                source_gap_id=gap.gap_id,
            )
            targets.append(target)

        targets.sort(key=lambda t: t.priority, reverse=True)

        logger.info(f"🧬 [AutoSpeciesGenerator] 提取到 {len(targets)} 个进化目标")

        return targets

    def _generate_candidates(
        self, targets: List[EvolutionTarget], strategy: GenerationStrategy, count: int
    ) -> List["SpeciesBlueprint"]:
        """生成候选蓝图"""
        candidates = []

        existing_blueprints = self._registry.get_all()

        for target in targets[: max(1, count // 2)]:
            if strategy in [GenerationStrategy.MUTATION, GenerationStrategy.HYBRID]:
                mutated = self._generate_by_mutation(target, existing_blueprints)
                if mutated:
                    candidates.append(mutated)

            if strategy in [GenerationStrategy.CROSSOVER, GenerationStrategy.HYBRID]:
                crossed = self._generate_by_crossover(target, existing_blueprints)
                if crossed:
                    candidates.extend(crossed)

            if strategy in [GenerationStrategy.NOVEL, GenerationStrategy.HYBRID]:
                novel = self._generate_novel(target)
                candidates.append(novel)

        while len(candidates) < count:
            base = self._create_blueprint_from_target(targets[0] if targets else None)
            mutated = base.mutate(0.3)
            candidates.append(mutated)

        return candidates[:count]

    def _generate_by_mutation(
        self, target: EvolutionTarget, existing_blueprints: List["SpeciesBlueprint"]
    ) -> Optional["SpeciesBlueprint"]:
        """通过变异生成"""
        similar = [bp for bp in existing_blueprints if target.capability_name in bp.capabilities]

        if not similar:
            return None

        similar.sort(
            key=lambda bp: (
                bp.capabilities.get(target.capability_name, target).target_value
                if target.capability_name in bp.capabilities
                else 0
            ),
            reverse=True,
        )

        base = similar[0]
        mutated = base.mutate(0.2)

        mutated.blueprint_id = f"mut_{uuid.uuid4().hex[:8]}"
        mutated.name = f"{base.name} (Mutated for {target.task_category})"
        mutated.description = f"变异自 {base.name}，优化 {target.capability_name} 能力"
        mutated.origin = "mutated"
        mutated.parent_blueprints = [base.blueprint_id]

        if target.capability_name in mutated.capabilities:
            mutated.capabilities[target.capability_name].target_value = target.target_success_rate
            mutated.capabilities[target.capability_name].priority = "primary"

        mutated.add_task_affinity(
            task_category=target.task_category, affinity_score=0.8, is_exclusive=False
        )

        return mutated

    def _generate_by_crossover(
        self, target: EvolutionTarget, existing_blueprints: List["SpeciesBlueprint"]
    ) -> List["SpeciesBlueprint"]:
        """通过交叉生成"""
        candidates = []

        with_capability = [
            bp for bp in existing_blueprints if target.capability_name in bp.capabilities
        ]

        for_task = [
            bp
            for bp in existing_blueprints
            if any(a.task_category == target.task_category for a in bp.task_affinities)
        ]

        if len(with_capability) >= 2:
            parent1, parent2 = random.sample(with_capability, 2)
            child1, child2 = parent1.crossover(parent2)

            for child in [child1, child2]:
                child.name = f"Cross: {parent1.name} x {parent2.name}"
                child.description = f"交叉后代，优化 {target.capability_name} 能力"

                if target.capability_name in child.capabilities:
                    child.capabilities[target.capability_name].target_value = (
                        target.target_success_rate
                    )

                candidates.append(child)

        elif with_capability and for_task:
            parent1 = with_capability[0]
            parent2 = for_task[0]
            child1, child2 = parent1.crossover(parent2)

            for child in [child1, child2]:
                child.name = f"Hybrid: {parent1.name} x {parent2.name}"
                child.description = f"能力与任务亲和度的混合"
                candidates.append(child)

        return candidates

    def _generate_novel(self, target: EvolutionTarget) -> "SpeciesBlueprint":
        """生成全新蓝图"""
        from hive.core.species_blueprint import CapabilityPriority, SpeciesBlueprint

        blueprint = SpeciesBlueprint(
            blueprint_id=f"novel_{uuid.uuid4().hex[:8]}",
            name=f"Specialized {target.task_category} Bee",
            description=(
                f"专门处理 {target.task_category} 类型任务的新蜂种。"
                f"主要优化 {target.capability_name} 能力，目标成功率 {target.target_success_rate:.0%}"
            ),
            origin="auto_generated",
            generation=1,
        )

        blueprint.add_capability(
            name=target.capability_name,
            target_value=target.target_success_rate,
            priority=CapabilityPriority.PRIMARY,
            weight=2.0,
        )

        complementary = {
            "mining": ["gathering", "sentinel"],
            "gathering": ["mining", "librarian"],
            "architectural": ["mining", "cognitive"],
            "librarian": ["gathering", "architectural"],
            "sentinel": ["mining", "immune_response"],
            "cognitive": ["architectural", "librarian"],
        }

        for comp in complementary.get(target.capability_name, []):
            blueprint.add_capability(
                name=comp,
                target_value=0.6,
                priority=CapabilityPriority.SECONDARY,
                weight=1.0,
            )

        blueprint.add_task_affinity(
            task_category=target.task_category,
            affinity_score=0.9,
            is_exclusive=True,
        )

        blueprint.mutation_rate = 0.15
        blueprint.learning_rate = 0.6

        blueprint.success_criteria = {
            "min_success_rate": target.target_success_rate * 0.9,
            "min_task_count": 5,
            "max_avg_duration_seconds": 120.0,
        }

        blueprint.tags = [
            f"auto_generated",
            f"target_{target.task_category}",
            f"cap_{target.capability_name}",
        ]

        return blueprint

    def _create_blueprint_from_target(
        self, target: Optional[EvolutionTarget]
    ) -> "SpeciesBlueprint":
        """从目标创建蓝图"""
        if target:
            return self._generate_novel(target)

        from hive.core.species_blueprint import SpeciesBlueprint

        return SpeciesBlueprint(
            blueprint_id=f"blueprint_{uuid.uuid4().hex[:8]}",
            name="Generic Worker Blueprint",
            description="通用工作蜂蓝图",
            origin="manual",
        )

    def _estimate_blueprint_fitness(
        self, blueprint: "SpeciesBlueprint", targets: List[EvolutionTarget]
    ) -> float:
        """估计蓝图适应度"""
        fitness = 0.5

        for target in targets:
            if target.capability_name in blueprint.capabilities:
                spec = blueprint.capabilities[target.capability_name]

                target_achievement = min(
                    1.0, spec.target_value / max(target.target_success_rate, 0.01)
                )

                priority_bonus = 1.0
                if spec.priority == "primary":
                    priority_bonus = 1.2
                elif spec.priority == "secondary":
                    priority_bonus = 1.1

                cap_score = target_achievement * priority_bonus * target.priority
                fitness = max(fitness, 0.5 + cap_score * 0.3)

        task_match_score = 0.0
        for target in targets:
            for affinity in blueprint.task_affinities:
                if affinity.task_category == target.task_category:
                    task_match_score = max(
                        task_match_score, affinity.affinity_score * target.priority
                    )

        fitness += task_match_score * 0.2

        validation, _ = blueprint.validate()
        if validation:
            fitness += 0.1

        return min(1.0, max(0.0, fitness))

    def optimize_blueprint(
        self,
        blueprint: "SpeciesBlueprint",
        objectives: Optional[List[OptimizationObjective]] = None,
    ) -> "SpeciesBlueprint":
        """优化现有蓝图"""
        objectives = objectives or self._default_objectives

        def evaluator(bp):
            return self._evaluate_for_objectives(bp, objectives)

        return self._optimizer.optimize(blueprint, objectives, evaluator)

    def _evaluate_for_objectives(
        self, blueprint: "SpeciesBlueprint", objectives: List[OptimizationObjective]
    ) -> float:
        """按目标评估蓝图"""
        scores = []

        for obj in objectives:
            if obj == OptimizationObjective.SUCCESS_RATE:
                primary_caps = blueprint.get_primary_capabilities()
                if primary_caps:
                    score = sum(spec.target_value for _, spec in primary_caps) / len(primary_caps)
                else:
                    score = 0.5
                scores.append(score)

            elif obj == OptimizationObjective.EFFICIENCY:
                for aff in blueprint.task_affinities:
                    if aff.affinity_score > 0.7:
                        scores.append(0.8)
                        break
                else:
                    scores.append(0.5)

            elif obj == OptimizationObjective.ROBUSTNESS:
                cap_count = len(blueprint.capabilities)
                score = min(1.0, cap_count * 0.2)
                scores.append(score)

            elif obj == OptimizationObjective.BALANCED:
                if blueprint.capabilities:
                    values = [spec.target_value for spec in blueprint.capabilities.values()]
                    variance = statistics.variance(values) if len(values) > 1 else 0
                    balance = 1.0 - min(1.0, variance * 5)
                    scores.append(balance)
                else:
                    scores.append(0.5)

        return sum(scores) / len(scores) if scores else 0.5

    def get_generation_history(self, limit: int = 10) -> List[GenerationResult]:
        """获取生成历史"""
        with self._lock:
            return self._generation_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "total_generations": len(self._generation_history),
                "total_blueprints_generated": sum(
                    len(r.selected_blueprints) for r in self._generation_history
                ),
                "registry_stats": self._registry.get_stats(),
            }


_generator: Optional[AutoSpeciesGenerator] = None
_generator_lock = threading.Lock()


def get_auto_species_generator() -> AutoSpeciesGenerator:
    """获取自动蜂种生成器单例"""
    global _generator
    if _generator is None:
        with _generator_lock:
            if _generator is None:
                _generator = AutoSpeciesGenerator()
    return _generator
