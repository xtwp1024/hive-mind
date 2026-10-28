# -*- coding: utf-8 -*-
"""
Species Blueprint - 蜂种蓝图系统

Phase 3: 自主进化能力
定义新蜂种的完整规格，作为自动蜂种生成的模板。

核心概念:
1. SpeciesBlueprint - 蜂种蓝图，定义新蜂种的完整规格
2. CapabilitySpecification - 能力规格（目标值、权重、约束）
3. TaskAffinity - 任务亲和度（偏好的任务类型）
4. BlueprintOrigin - 蓝图来源（手动创建/自动生成/变异而来）
5. BlueprintRegistry - 蓝图注册表，管理所有蓝图

蓝图生命周期:
1. 设计 → 2. 验证 → 3. 孵化 → 4. 评估 → 5. 晋升/淘汰
"""

import logging
import uuid
import time
import json
import threading
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from copy import deepcopy

logger = logging.getLogger("Hive.SpeciesBlueprint")


class BlueprintStatus(Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    INCUBATING = "incubating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class BlueprintOrigin(Enum):
    MANUAL = "manual"
    AUTO_GENERATED = "auto_generated"
    MUTATED = "mutated"
    CROSSED = "crossed"
    EVOLVED = "evolved"


class CapabilityPriority(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    OPTIONAL = "optional"


@dataclass
class CapabilitySpecification:
    """
    能力规格

    定义单个能力的目标值、权重和约束条件。
    用于 AutoSpeciesGenerator 生成优化的 DNA。
    """

    capability_name: str

    target_value: float = 0.8
    min_value: float = 0.3
    max_value: float = 1.0

    priority: CapabilityPriority = CapabilityPriority.SECONDARY
    weight: float = 1.0

    is_mutable: bool = True
    mutation_rate_override: Optional[float] = None

    constraints: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """验证能力规格"""
        errors = []

        if not 0.0 <= self.min_value <= 1.0:
            errors.append(f"min_value {self.min_value} 超出范围 [0, 1]")

        if not 0.0 <= self.max_value <= 1.0:
            errors.append(f"max_value {self.max_value} 超出范围 [0, 1]")

        if not 0.0 <= self.target_value <= 1.0:
            errors.append(f"target_value {self.target_value} 超出范围 [0, 1]")

        if self.min_value > self.max_value:
            errors.append(f"min_value {self.min_value} 大于 max_value {self.max_value}")

        if not (self.min_value <= self.target_value <= self.max_value):
            errors.append(
                f"target_value {self.target_value} 不在 [{self.min_value}, {self.max_value}] 范围内"
            )

        if not 0.0 <= self.weight <= 10.0:
            errors.append(f"weight {self.weight} 超出范围 [0, 10]")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability_name": self.capability_name,
            "target_value": self.target_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "priority": self.priority.value,
            "weight": self.weight,
            "is_mutable": self.is_mutable,
            "mutation_rate_override": self.mutation_rate_override,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilitySpecification":
        priority = CapabilityPriority(data.get("priority", "secondary"))
        return cls(
            capability_name=data["capability_name"],
            target_value=data.get("target_value", 0.8),
            min_value=data.get("min_value", 0.3),
            max_value=data.get("max_value", 1.0),
            priority=priority,
            weight=data.get("weight", 1.0),
            is_mutable=data.get("is_mutable", True),
            mutation_rate_override=data.get("mutation_rate_override"),
            constraints=data.get("constraints", {}),
        )


@dataclass
class TaskAffinity:
    """
    任务亲和度

    定义蜂种偏好的任务类型，用于任务分配优化。
    """

    task_category: str
    affinity_score: float = 0.5

    min_task_complexity: float = 0.0
    max_task_complexity: float = 1.0

    preferred_task_size: str = "medium"

    is_exclusive: bool = False

    def validate(self) -> Tuple[bool, List[str]]:
        """验证任务亲和度"""
        errors = []

        if not 0.0 <= self.affinity_score <= 1.0:
            errors.append(f"affinity_score {self.affinity_score} 超出范围 [0, 1]")

        if self.min_task_complexity > self.max_task_complexity:
            errors.append("min_task_complexity 大于 max_task_complexity")

        valid_sizes = {"small", "medium", "large", "any"}
        if self.preferred_task_size not in valid_sizes:
            errors.append(
                f"preferred_task_size {self.preferred_task_size} 无效，有效值: {valid_sizes}"
            )

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_category": self.task_category,
            "affinity_score": self.affinity_score,
            "min_task_complexity": self.min_task_complexity,
            "max_task_complexity": self.max_task_complexity,
            "preferred_task_size": self.preferred_task_size,
            "is_exclusive": self.is_exclusive,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskAffinity":
        return cls(
            task_category=data["task_category"],
            affinity_score=data.get("affinity_score", 0.5),
            min_task_complexity=data.get("min_task_complexity", 0.0),
            max_task_complexity=data.get("max_task_complexity", 1.0),
            preferred_task_size=data.get("preferred_task_size", "medium"),
            is_exclusive=data.get("is_exclusive", False),
        )


@dataclass
class BlueprintValidationResult:
    """蓝图验证结果"""

    blueprint_id: str
    is_valid: bool
    validation_time: float = field(default_factory=time.time)

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    capability_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0

    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "is_valid": self.is_valid,
            "validation_time": self.validation_time,
            "errors": self.errors,
            "warnings": self.warnings,
            "capability_scores": self.capability_scores,
            "overall_score": self.overall_score,
            "recommendations": self.recommendations,
        }


@dataclass
class SpeciesBlueprint:
    """
    蜂种蓝图

    定义新蜂种的完整规格，包括：
    - 基础信息（名称、描述、版本）
    - 能力规格（目标值、权重、约束）
    - 任务亲和度（偏好的任务类型）
    - 遗传参数（变异率、学习率等）
    - 评估指标（用于判断孵化是否成功）

    蓝图是 AutoSpeciesGenerator 的输入，
    也是 Incubator 的孵化模板。
    """

    blueprint_id: str
    name: str
    description: str

    version: str = "1.0.0"
    status: BlueprintStatus = BlueprintStatus.DRAFT
    origin: BlueprintOrigin = BlueprintOrigin.MANUAL

    parent_blueprints: List[str] = field(default_factory=list)
    generation: int = 1

    capabilities: Dict[str, CapabilitySpecification] = field(default_factory=dict)
    task_affinities: List[TaskAffinity] = field(default_factory=list)

    base_species: str = "worker"
    role: str = "worker"

    mutation_rate: float = 0.1
    learning_rate: float = 0.5
    immune_response: float = 0.5

    success_criteria: Dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    validated_at: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.success_criteria:
            self.success_criteria = {
                "min_success_rate": 0.7,
                "min_task_count": 5,
                "max_avg_duration_seconds": 120.0,
            }

    def validate(self) -> BlueprintValidationResult:
        """
        验证蓝图是否有效

        检查:
        1. 必填字段是否存在
        2. 能力规格是否有效
        3. 任务亲和度是否有效
        4. 遗传参数是否在合理范围
        5. 成功标准是否合理

        Returns:
            BlueprintValidationResult - 验证结果
        """
        errors = []
        warnings = []
        recommendations = []
        capability_scores = {}

        if not self.name or not self.name.strip():
            errors.append("蓝图名称不能为空")

        if not self.description or len(self.description) < 10:
            warnings.append("蓝图描述过短，建议提供更详细的说明")

        if not self.capabilities:
            errors.append("蓝图必须至少定义一个能力规格")
        else:
            for cap_name, spec in self.capabilities.items():
                is_valid, cap_errors = spec.validate()
                if not is_valid:
                    errors.extend([f"[{cap_name}] {e}" for e in cap_errors])
                else:
                    score = self._calculate_capability_score(spec)
                    capability_scores[cap_name] = score

            primary_count = sum(
                1 for s in self.capabilities.values()
                if s.priority == CapabilityPriority.PRIMARY
            )
            if primary_count == 0:
                warnings.append("没有定义主要能力 (PRIMARY priority)")
                recommendations.append("建议指定 1-2 个主要能力")
            elif primary_count > 3:
                warnings.append(f"定义了 {primary_count} 个主要能力，建议不超过 3 个")

        for affinity in self.task_affinities:
            is_valid, aff_errors = affinity.validate()
            if not is_valid:
                errors.extend([f"[task_affinity] {e}" for e in aff_errors])

        if not 0.0 <= self.mutation_rate <= 1.0:
            errors.append(f"mutation_rate {self.mutation_rate} 超出范围 [0, 1]")

        if not 0.0 <= self.learning_rate <= 1.0:
            errors.append(f"learning_rate {self.learning_rate} 超出范围 [0, 1]")

        if not 0.0 <= self.immune_response <= 1.0:
            errors.append(f"immune_response {self.immune_response} 超出范围 [0, 1]")

        criteria = self.success_criteria
        if criteria:
            min_rate = criteria.get("min_success_rate", 0)
            if min_rate > 0.95:
                warnings.append(f"成功标准 min_success_rate={min_rate} 可能过高")
            elif min_rate < 0.3:
                warnings.append(f"成功标准 min_success_rate={min_rate} 可能过低")

        overall_score = 0.0
        if capability_scores:
            total_weight = sum(
                spec.weight for spec in self.capabilities.values()
            )
            if total_weight > 0:
                overall_score = sum(
                    capability_scores[name] * self.capabilities[name].weight
                    for name in capability_scores
                ) / total_weight

        result = BlueprintValidationResult(
            blueprint_id=self.blueprint_id,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            capability_scores=capability_scores,
            overall_score=overall_score,
            recommendations=recommendations,
        )

        if result.is_valid:
            self.validated_at = time.time()
            self.status = BlueprintStatus.VALIDATED
            logger.info(f"📋 [SpeciesBlueprint] 蓝图 {self.name} 验证通过")
        else:
            logger.warning(f"📋 [SpeciesBlueprint] 蓝图 {self.name} 验证失败: {len(errors)} 个错误")

        return result

    def _calculate_capability_score(self, spec: CapabilitySpecification) -> float:
        """计算能力规格的质量分数"""
        score = 1.0

        if spec.target_value < 0.5:
            score *= 0.8

        if spec.min_value > 0.5:
            score *= 0.9

        if not spec.is_mutable:
            score *= 0.7

        if spec.priority == CapabilityPriority.PRIMARY:
            score *= 1.1
        elif spec.priority == CapabilityPriority.OPTIONAL:
            score *= 0.9

        return min(1.0, max(0.0, score))

    def add_capability(
        self,
        name: str,
        target_value: float = 0.8,
        priority: CapabilityPriority = CapabilityPriority.SECONDARY,
        weight: float = 1.0,
        min_value: float = 0.3,
        max_value: float = 1.0,
        is_mutable: bool = True,
    ) -> "SpeciesBlueprint":
        """添加能力规格"""
        spec = CapabilitySpecification(
            capability_name=name,
            target_value=target_value,
            min_value=min_value,
            max_value=max_value,
            priority=priority,
            weight=weight,
            is_mutable=is_mutable,
        )
        self.capabilities[name] = spec
        self.updated_at = time.time()
        return self

    def add_task_affinity(
        self,
        task_category: str,
        affinity_score: float = 0.5,
        is_exclusive: bool = False,
    ) -> "SpeciesBlueprint":
        """添加任务亲和度"""
        affinity = TaskAffinity(
            task_category=task_category,
            affinity_score=affinity_score,
            is_exclusive=is_exclusive,
        )
        self.task_affinities.append(affinity)
        self.updated_at = time.time()
        return self

    def get_primary_capabilities(self) -> List[Tuple[str, CapabilitySpecification]]:
        """获取主要能力"""
        return [
            (name, spec)
            for name, spec in self.capabilities.items()
            if spec.priority == CapabilityPriority.PRIMARY
        ]

    def get_preferred_task_categories(self) -> List[str]:
        """获取偏好的任务类别"""
        sorted_affinities = sorted(
            self.task_affinities,
            key=lambda a: a.affinity_score,
            reverse=True
        )
        return [a.task_category for a in sorted_affinities]

    def mutate(self, mutation_rate: float = 0.2) -> "SpeciesBlueprint":
        """
        创建变异蓝图

        基于当前蓝图创建一个变异版本，
        用于进化探索。
        """
        import random

        mutated = deepcopy(self)
        mutated.blueprint_id = f"{self.blueprint_id}_mut_{uuid.uuid4().hex[:8]}"
        mutated.origin = BlueprintOrigin.MUTATED
        mutated.parent_blueprints = [self.blueprint_id]
        mutated.generation = self.generation + 1
        mutated.status = BlueprintStatus.DRAFT
        mutated.created_at = time.time()
        mutated.updated_at = time.time()
        mutated.validated_at = None

        for name, spec in mutated.capabilities.items():
            if spec.is_mutable and random.random() < mutation_rate:
                delta = random.gauss(0, 0.15)
                new_target = spec.target_value + delta
                new_target = max(spec.min_value, min(spec.max_value, new_target))
                spec.target_value = new_target

                if random.random() < 0.3:
                    spec.weight = max(0.1, spec.weight + random.gauss(0, 0.2))

        if random.random() < mutation_rate * 0.5:
            mutated.mutation_rate = max(0.01, min(0.5, mutated.mutation_rate + random.gauss(0, 0.05)))

        if random.random() < mutation_rate * 0.5:
            mutated.learning_rate = max(0.1, min(0.9, mutated.learning_rate + random.gauss(0, 0.1)))

        logger.info(f"🧬 [SpeciesBlueprint] 创建变异蓝图: {mutated.blueprint_id}")

        return mutated

    def crossover(self, other: "SpeciesBlueprint") -> Tuple["SpeciesBlueprint", "SpeciesBlueprint"]:
        """
        与另一个蓝图交叉

        使用遗传算法的交叉操作，
        创建两个子代蓝图。
        """
        child1_id = f"cross_{uuid.uuid4().hex[:8]}_1"
        child2_id = f"cross_{uuid.uuid4().hex[:8]}_2"

        child1 = SpeciesBlueprint(
            blueprint_id=child1_id,
            name=f"{self.name} x {other.name} (1)",
            description=f"交叉后代: {self.blueprint_id} x {other.blueprint_id}",
            origin=BlueprintOrigin.CROSSED,
            parent_blueprints=[self.blueprint_id, other.blueprint_id],
            generation=max(self.generation, other.generation) + 1,
        )

        child2 = SpeciesBlueprint(
            blueprint_id=child2_id,
            name=f"{self.name} x {other.name} (2)",
            description=f"交叉后代: {other.blueprint_id} x {self.blueprint_id}",
            origin=BlueprintOrigin.CROSSED,
            parent_blueprints=[self.blueprint_id, other.blueprint_id],
            generation=max(self.generation, other.generation) + 1,
        )

        all_capabilities = set(self.capabilities.keys()) | set(other.capabilities.keys())

        for cap_name in all_capabilities:
            spec1 = self.capabilities.get(cap_name)
            spec2 = other.capabilities.get(cap_name)

            if spec1 and spec2:
                import random

                if random.random() < 0.5:
                    child1.capabilities[cap_name] = deepcopy(spec1)
                    child2.capabilities[cap_name] = deepcopy(spec2)
                else:
                    child1.capabilities[cap_name] = deepcopy(spec2)
                    child2.capabilities[cap_name] = deepcopy(spec1)

                avg_target = (spec1.target_value + spec2.target_value) / 2
                if random.random() < 0.3:
                    child1.capabilities[cap_name].target_value = avg_target
                if random.random() < 0.3:
                    child2.capabilities[cap_name].target_value = avg_target

            elif spec1:
                child1.capabilities[cap_name] = deepcopy(spec1)
                if random.random() < 0.5:
                    child2.capabilities[cap_name] = deepcopy(spec1)
            elif spec2:
                child2.capabilities[cap_name] = deepcopy(spec2)
                if random.random() < 0.5:
                    child1.capabilities[cap_name] = deepcopy(spec2)

        child1.mutation_rate = (self.mutation_rate + other.mutation_rate) / 2
        child2.mutation_rate = (self.mutation_rate + other.mutation_rate) / 2

        child1.learning_rate = (self.learning_rate + other.learning_rate) / 2
        child2.learning_rate = (self.learning_rate + other.learning_rate) / 2

        all_affinities = {}
        for aff in self.task_affinities + other.task_affinities:
            key = aff.task_category
            if key not in all_affinities:
                all_affinities[key] = []
            all_affinities[key].append(aff)

        import random
        for key, affs in all_affinities.items():
            if len(affs) >= 2:
                avg_score = (affs[0].affinity_score + affs[1].affinity_score) / 2
                child1.task_affinities.append(TaskAffinity(
                    task_category=key,
                    affinity_score=avg_score if random.random() < 0.5 else affs[0].affinity_score
                ))
                child2.task_affinities.append(TaskAffinity(
                    task_category=key,
                    affinity_score=avg_score if random.random() < 0.5 else affs[1].affinity_score
                ))
            else:
                child1.task_affinities.append(deepcopy(affs[0]))
                if random.random() < 0.5:
                    child2.task_affinities.append(deepcopy(affs[0]))

        logger.info(f"🧬 [SpeciesBlueprint] 创建交叉蓝图: {child1_id}, {child2_id}")

        return child1, child2

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "blueprint_id": self.blueprint_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "origin": self.origin.value,
            "parent_blueprints": self.parent_blueprints,
            "generation": self.generation,
            "capabilities": {
                name: spec.to_dict()
                for name, spec in self.capabilities.items()
            },
            "task_affinities": [aff.to_dict() for aff in self.task_affinities],
            "base_species": self.base_species,
            "role": self.role,
            "mutation_rate": self.mutation_rate,
            "learning_rate": self.learning_rate,
            "immune_response": self.immune_response,
            "success_criteria": self.success_criteria,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "validated_at": self.validated_at,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeciesBlueprint":
        """从字典创建蓝图"""
        capabilities = {}
        if "capabilities" in data:
            for name, spec_data in data["capabilities"].items():
                capabilities[name] = CapabilitySpecification.from_dict(spec_data)

        task_affinities = []
        if "task_affinities" in data:
            for aff_data in data["task_affinities"]:
                task_affinities.append(TaskAffinity.from_dict(aff_data))

        return cls(
            blueprint_id=data["blueprint_id"],
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            status=BlueprintStatus(data.get("status", "draft")),
            origin=BlueprintOrigin(data.get("origin", "manual")),
            parent_blueprints=data.get("parent_blueprints", []),
            generation=data.get("generation", 1),
            capabilities=capabilities,
            task_affinities=task_affinities,
            base_species=data.get("base_species", "worker"),
            role=data.get("role", "worker"),
            mutation_rate=data.get("mutation_rate", 0.1),
            learning_rate=data.get("learning_rate", 0.5),
            immune_response=data.get("immune_response", 0.5),
            success_criteria=data.get("success_criteria", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            validated_at=data.get("validated_at"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )

    def to_dna_params(self) -> Dict[str, Any]:
        """
        转换为 DNA 参数

        从蓝图生成 BeeDNA 的初始化参数。
        """
        dna_params = {
            "species": self.base_species,
            "role": self.role,
            "mutation_rate": self.mutation_rate,
            "learning_rate": self.learning_rate,
            "immune_response": self.immune_response,
        }

        capability_mapping = {
            "mining": "mining_capability",
            "gathering": "gathering_capability",
            "architectural": "architectural_capability",
            "librarian": "librarian_capability",
            "sentinel": "sentinel_capability",
        }

        for cap_name, spec in self.capabilities.items():
            dna_key = capability_mapping.get(cap_name)
            if dna_key:
                dna_params[dna_key] = spec.target_value

        return dna_params


class BlueprintRegistry:
    """
    蓝图注册表

    管理所有蜂种蓝图，支持:
    - 注册/注销蓝图
    - 按状态/标签/能力查询
    - 版本管理
    - 持久化存储
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self._blueprints: Dict[str, SpeciesBlueprint] = {}
        self._lock = threading.Lock()

        if persistence_path is None:
            persistence_path = Path("hive/data/blueprints")
        self._persistence_path = persistence_path
        self._persistence_path.mkdir(parents=True, exist_ok=True)

        self._load_from_disk()

        logger.info(f"📋 [BlueprintRegistry] 初始化完成，已加载 {len(self._blueprints)} 个蓝图")

    def register(self, blueprint: SpeciesBlueprint) -> bool:
        """注册蓝图"""
        with self._lock:
            if blueprint.blueprint_id in self._blueprints:
                logger.warning(f"📋 [BlueprintRegistry] 蓝图 {blueprint.blueprint_id} 已存在，跳过注册")
                return False

            self._blueprints[blueprint.blueprint_id] = blueprint
            self._save_to_disk(blueprint)

            logger.info(f"📋 [BlueprintRegistry] 注册蓝图: {blueprint.name} ({blueprint.blueprint_id})")
            return True

    def unregister(self, blueprint_id: str) -> bool:
        """注销蓝图"""
        with self._lock:
            if blueprint_id not in self._blueprints:
                return False

            del self._blueprints[blueprint_id]

            file_path = self._persistence_path / f"{blueprint_id}.json"
            if file_path.exists():
                file_path.unlink()

            logger.info(f"📋 [BlueprintRegistry] 注销蓝图: {blueprint_id}")
            return True

    def get(self, blueprint_id: str) -> Optional[SpeciesBlueprint]:
        """获取蓝图"""
        with self._lock:
            return self._blueprints.get(blueprint_id)

    def get_all(self) -> List[SpeciesBlueprint]:
        """获取所有蓝图"""
        with self._lock:
            return list(self._blueprints.values())

    def get_by_status(self, status: BlueprintStatus) -> List[SpeciesBlueprint]:
        """按状态获取蓝图"""
        with self._lock:
            return [
                bp for bp in self._blueprints.values()
                if bp.status == status
            ]

    def get_by_tag(self, tag: str) -> List[SpeciesBlueprint]:
        """按标签获取蓝图"""
        with self._lock:
            return [
                bp for bp in self._blueprints.values()
                if tag in bp.tags
            ]

    def get_by_capability(self, capability_name: str) -> List[SpeciesBlueprint]:
        """按能力获取蓝图"""
        with self._lock:
            return [
                bp for bp in self._blueprints.values()
                if capability_name in bp.capabilities
            ]

    def get_for_task(self, task_category: str) -> List[SpeciesBlueprint]:
        """获取适合特定任务的蓝图"""
        candidates = []
        with self._lock:
            for bp in self._blueprints.values():
                if bp.status != BlueprintStatus.ACTIVE:
                    continue

                for affinity in bp.task_affinities:
                    if affinity.task_category == task_category and affinity.affinity_score >= 0.3:
                        candidates.append((bp, affinity.affinity_score))
                        break

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [bp for bp, _ in candidates]

    def update_status(self, blueprint_id: str, new_status: BlueprintStatus) -> bool:
        """更新蓝图状态"""
        with self._lock:
            bp = self._blueprints.get(blueprint_id)
            if not bp:
                return False

            old_status = bp.status
            bp.status = new_status
            bp.updated_at = time.time()

            self._save_to_disk(bp)

            logger.info(
                f"📋 [BlueprintRegistry] 蓝图 {blueprint_id} 状态变更: "
                f"{old_status.value} -> {new_status.value}"
            )
            return True

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            by_status = {}
            for bp in self._blueprints.values():
                status_val = bp.status.value
                by_status[status_val] = by_status.get(status_val, 0) + 1

            by_origin = {}
            for bp in self._blueprints.values():
                origin_val = bp.origin.value
                by_origin[origin_val] = by_origin.get(origin_val, 0) + 1

            return {
                "total_blueprints": len(self._blueprints),
                "by_status": by_status,
                "by_origin": by_origin,
            }

    def _save_to_disk(self, blueprint: SpeciesBlueprint) -> None:
        """保存蓝图到磁盘"""
        try:
            file_path = self._persistence_path / f"{blueprint.blueprint_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(blueprint.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"📋 [BlueprintRegistry] 保存蓝图失败: {e}")

    def _load_from_disk(self) -> None:
        """从磁盘加载蓝图"""
        if not self._persistence_path.exists():
            return

        for file_path in self._persistence_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                blueprint = SpeciesBlueprint.from_dict(data)
                self._blueprints[blueprint.blueprint_id] = blueprint
            except Exception as e:
                logger.warning(f"📋 [BlueprintRegistry] 加载蓝图失败 {file_path}: {e}")


_registry: Optional[BlueprintRegistry] = None
_registry_lock = threading.Lock()


def get_blueprint_registry() -> BlueprintRegistry:
    """获取蓝图注册表单例"""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = BlueprintRegistry()
    return _registry
