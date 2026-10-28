# -*- coding: utf-8 -*-
"""DNA数据结构

定义蜜蜂和蜂巢的核心DNA数据结构。
支持两种进化模式：
1. 传统模式：浮点向量 capability
2. GEP 模式：树结构基因表达式（genome 字段）
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional
import uuid
import random

from hive.utils.logger import get_logger

logger = get_logger(__name__)

# GEP imports - lazy loading to avoid circular dependency
try:
    from hive.core.gep_gene import (
        GEPGene, GEPChromosome, GEPGeneFactory,
        GEPFitnessEvaluator, GEPOperators, FunctionType
    )
    GEP_AVAILABLE = True
except ImportError:
    GEP_AVAILABLE = False
    GEPGene = None
    GEPChromosome = None
    GEPGeneFactory = None
    GEPFitnessEvaluator = None
    GEPOperators = None
    FunctionType = None


@dataclass
class DNA:
    """基础DNA类

    定义蜂巢系统的基础属性和能力。
    """

    # 基本属性
    species: str
    generation: int = 1
    version: str = "2.0.0"
    fitness: float = 0.0

    # 能力权重（0.0-1.0）
    mining_capability: float = 0.5
    gathering_capability: float = 0.5
    architectural_capability: float = 0.5
    librarian_capability: float = 0.5
    sentinel_capability: float = 0.5

    # 特殊能力
    immune_response: float = 0.5
    learning_rate: float = 0.5
    mutation_rate: float = 0.1

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """验证DNA是否有效

        Returns:
            是否有效
        """
        # 检查能力权重在0.0-1.0范围内
        capabilities = [
            self.mining_capability,
            self.gathering_capability,
            self.architectural_capability,
            self.librarian_capability,
            self.sentinel_capability,
            self.immune_response,
            self.learning_rate,
        ]

        for cap in capabilities:
            if not 0.0 <= cap <= 1.0:
                logger.error(
                    "Invalid DNA capability",
                    species=self.species,
                    capability_value=cap,
                )
                return False

        return True

    def _mutate_capability(self, value: float, rate: float) -> float:
        import random
        if random.random() < rate:
            delta = random.gauss(0, 0.1)
            return max(0.0, min(1.0, value + delta))
        return value

    def mutate(self, mutation_rate: Optional[float] = None) -> "DNA":
        rate = mutation_rate if mutation_rate is not None else self.mutation_rate

        mutated = DNA(
            species=self.species,
            generation=self.generation + 1,
            version=self.version,
            fitness=self.fitness,
            mining_capability=self._mutate_capability(self.mining_capability, rate),
            gathering_capability=self._mutate_capability(self.gathering_capability, rate),
            architectural_capability=self._mutate_capability(self.architectural_capability, rate),
            librarian_capability=self._mutate_capability(self.librarian_capability, rate),
            sentinel_capability=self._mutate_capability(self.sentinel_capability, rate),
            immune_response=self._mutate_capability(self.immune_response, rate),
            learning_rate=self._mutate_capability(self.learning_rate, rate),
            mutation_rate=self.mutation_rate,
            metadata=self.metadata.copy(),
        )

        logger.debug(
            "DNA mutated",
            species=self.species,
            generation=self.generation,
            new_generation=mutated.generation,
        )

        return mutated

    def _average_values(self, other: "DNA") -> dict:
        return {
            "fitness": (self.fitness + other.fitness) / 2,
            "mining_capability": (self.mining_capability + other.mining_capability) / 2,
            "gathering_capability": (self.gathering_capability + other.gathering_capability) / 2,
            "architectural_capability": (self.architectural_capability + other.architectural_capability) / 2,
            "librarian_capability": (self.librarian_capability + other.librarian_capability) / 2,
            "sentinel_capability": (self.sentinel_capability + other.sentinel_capability) / 2,
            "immune_response": (self.immune_response + other.immune_response) / 2,
            "learning_rate": (self.learning_rate + other.learning_rate) / 2,
            "mutation_rate": (self.mutation_rate + other.mutation_rate) / 2,
        }

    def crossover(self, other: "DNA") -> "DNA":
        if self.species != other.species:
            logger.warning(
                "DNA crossover between different species",
                species1=self.species,
                species2=other.species,
            )

        avg = self._average_values(other)
        child = DNA(
            species=self.species,
            generation=max(self.generation, other.generation) + 1,
            version=self.version,
            metadata={**self.metadata, **other.metadata},
            **avg,
        )

        logger.debug(
            "DNA crossover",
            species=self.species,
            generation1=self.generation,
            generation2=other.generation,
            child_generation=child.generation,
        )

        return child


@dataclass
class BeeDNA(DNA):
    """蜜蜂DNA

    特定于个体蜜蜂的DNA，包含个体特征。
    支持两种进化模式：
    - 无 genome: 传统浮点向量进化（calculate_fitness 使用加权求和）
    - 有 genome: GEP 基因进化（calculate_fitness 使用 GEPFitnessEvaluator）
    """

    # 个体特征
    id: str = ""
    role: str = "worker"  # worker, soldier, scout, etc.
    target_name: str = ""
    target_url: str = ""
    mutation_gene: str = "standard"
    can_merge_back: bool = False # 是否允许该个体导回修改

    # 性能指标
    success_rate: float = 0.0
    task_completion_time: float = 0.0  # 平均任务完成时间（秒）
    error_count: int = 0

    # 偏好
    preferred_tasks: list[str] = field(default_factory=list)

    # GEP 基因组（可选）
    genome: Optional[GEPChromosome] = None

    # GEP 评估上下文历史（用于 fitness 评估）
    _fitness_contexts: list = field(default_factory=list, repr=False)

    def calculate_fitness(self) -> float:
        """计算适应度

        如果 BeeDNA 有 genome（树结构 GEP 染色体），使用 GEPFitnessEvaluator；
        否则使用传统的加权求和方式。

        Returns:
            适应度分数（0.0-1.0）
        """
        # GEP 模式：使用基因表达式树计算 fitness
        if self.genome is not None and GEP_AVAILABLE:
            return self._calculate_gep_fitness()

        # 传统模式：基于成功率和完成时间计算适应度
        success_factor = self.success_rate

        # 完成时间越短越好（归一化到0-1）
        time_factor = 1.0 / (1.0 + self.task_completion_time)

        # 错误越少越好
        error_factor = 1.0 / (1.0 + self.error_count)

        # 加权平均
        self.fitness = (success_factor * 0.5 + time_factor * 0.3 + error_factor * 0.2)

        return self.fitness

    def _calculate_gep_fitness(self) -> float:
        """使用 GEP 基因组计算适应度"""
        if not GEP_AVAILABLE or self.genome is None:
            return self.fitness

        # 构建评估上下文
        context = {
            "task_type": self.role,
            "task_complexity": getattr(self, 'task_complexity', 0.5),
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "avg_time": self.task_completion_time,
            "memory_usage": getattr(self, 'memory_usage', 0.5),
            "cpu_usage": getattr(self, 'cpu_usage', 0.5),
        }

        evaluator = GEPFitnessEvaluator()
        contexts = self._fitness_contexts if self._fitness_contexts else [context]

        try:
            fitness = evaluator.evaluate_chromosome(self.genome, contexts)
            self.fitness = fitness
        except Exception as e:
            logger.warning(f"[BeeDNA] GEP fitness evaluation failed: {e}, falling back to legacy")
            self.fitness = (self.success_rate * 0.5 +
                          (1.0 / (1.0 + self.task_completion_time)) * 0.3 +
                          (1.0 / (1.0 + self.error_count)) * 0.2)

        return self.fitness

    def add_fitness_context(self, context: dict) -> None:
        """添加评估上下文（用于 GEP 多样本评估）"""
        if not hasattr(self, '_fitness_contexts'):
            self._fitness_contexts = []
        self._fitness_contexts.append(context)
        # 保留最近 100 个上下文防止无界增长
        if len(self._fitness_contexts) > 100:
            self._fitness_contexts = self._fitness_contexts[-100:]

    def init_gep_genome(self,
                        num_genes: int = 3,
                        max_depth: int = 5,
                        linking_function: str = "AND") -> "BeeDNA":
        """
        为当前 BeeDNA 初始化一个 GEP 基因组。

        Args:
            num_genes: 基因数量
            max_depth: 基因树最大深度
            linking_function: 基因链接函数 ("AND" 或 "OR")

        Returns:
            self（方便链式调用）
        """
        if not GEP_AVAILABLE:
            logger.warning("[BeeDNA] GEP not available, cannot init genome")
            return self

        factory = GEPGeneFactory(max_depth=max_depth, max_genes=num_genes)
        linking = FunctionType.AND if linking_function == "AND" else FunctionType.OR

        self.genome = factory.create_random_chromosome(num_genes=num_genes)
        self.genome.linking_function = linking

        logger.debug(f"[BeeDNA] GEP genome initialized: {self.genome}")
        return self

    def set_genome(self, chromosome: GEPChromosome) -> "BeeDNA":
        """设置 GEP 基因组"""
        self.genome = chromosome
        return self

    def get_genome_expressions(self) -> list[str]:
        """获取基因组的可读表达式"""
        if self.genome is None:
            return []
        return self.genome.get_gene_expressions()

    def mutate_gep(self, mutation_rate: float = 0.1) -> "BeeDNA":
        """
        对 GEP 基因组进行突变（不改变传统 capability 字段）。

        Returns:
            新的 BeeDNA（突变后）
        """
        if self.genome is None or not GEP_AVAILABLE:
            return self

        operators = GEPOperators(max_gene_depth=10, max_genes_per_chromosome=self.genome.max_genes)
        mutated_genes = [operators.point_mutation(gene) for gene in self.genome.genes]
        mutated_chromosome = GEPChromosome(
            genes=mutated_genes,
            linking_function=self.genome.linking_function,
            max_genes=self.genome.max_genes
        )

        new_dna = BeeDNA(
            species=self.species,
            generation=self.generation + 1,
            version=self.version,
            fitness=0.0,
            mining_capability=self.mining_capability,
            gathering_capability=self.gathering_capability,
            architectural_capability=self.architectural_capability,
            librarian_capability=self.librarian_capability,
            sentinel_capability=self.sentinel_capability,
            immune_response=self.immune_response,
            learning_rate=self.learning_rate,
            mutation_rate=self.mutation_rate,
            metadata=self.metadata.copy(),
            id=f"{self.id}_gep_mut_{random.randint(1000, 9999)}",
            role=self.role,
            target_name=self.target_name,
            target_url=self.target_url,
            mutation_gene=self.mutation_gene,
            can_merge_back=self.can_merge_back,
            success_rate=self.success_rate * 0.9,
            task_completion_time=0.0,
            error_count=0,
            preferred_tasks=list(self.preferred_tasks),
            genome=mutated_chromosome,
        )
        return new_dna

    def crossover_gep(self, other: "BeeDNA") -> tuple["BeeDNA", "BeeDNA"]:
        """
        两个 BeeDNA 的 GEP 基因组进行交叉。

        Args:
            other: 另一个 BeeDNA

        Returns:
            (子代1, 子代2)
        """
        if self.genome is None or other.genome is None or not GEP_AVAILABLE:
            return self.mutate(), other.mutate()

        operators = GEPOperators()
        chrom1, chrom2 = operators.gene_crossover(self.genome, other.genome)

        child1 = BeeDNA(
            species=self.species,
            generation=max(self.generation, other.generation) + 1,
            version=self.version,
            fitness=0.0,
            mining_capability=(self.mining_capability + other.mining_capability) / 2,
            gathering_capability=(self.gathering_capability + other.gathering_capability) / 2,
            architectural_capability=(self.architectural_capability + other.architectural_capability) / 2,
            librarian_capability=(self.librarian_capability + other.librarian_capability) / 2,
            sentinel_capability=(self.sentinel_capability + other.sentinel_capability) / 2,
            immune_response=(self.immune_response + other.immune_response) / 2,
            learning_rate=(self.learning_rate + other.learning_rate) / 2,
            mutation_rate=self.mutation_rate,
            metadata={**self.metadata, **other.metadata},
            id=f"{self.id}_gep_x_{random.randint(1000, 9999)}",
            role=self.role,
            target_name=self.target_name,
            target_url=self.target_url,
            mutation_gene=self.mutation_gene,
            can_merge_back=self.can_merge_back,
            success_rate=self.success_rate,
            task_completion_time=self.task_completion_time,
            error_count=self.error_count,
            preferred_tasks=list(self.preferred_tasks),
            genome=chrom1,
        )

        child2 = BeeDNA(
            species=self.species,
            generation=max(self.generation, other.generation) + 1,
            version=self.version,
            fitness=0.0,
            mining_capability=(self.mining_capability + other.mining_capability) / 2,
            gathering_capability=(self.gathering_capability + other.gathering_capability) / 2,
            architectural_capability=(self.architectural_capability + other.architectural_capability) / 2,
            librarian_capability=(self.librarian_capability + other.librarian_capability) / 2,
            sentinel_capability=(self.sentinel_capability + other.sentinel_capability) / 2,
            immune_response=(self.immune_response + other.immune_response) / 2,
            learning_rate=(self.learning_rate + other.learning_rate) / 2,
            mutation_rate=other.mutation_rate,
            metadata={**other.metadata, **self.metadata},
            id=f"{other.id}_gep_x_{random.randint(1000, 9999)}",
            role=other.role,
            target_name=other.target_name,
            target_url=other.target_url,
            mutation_gene=other.mutation_gene,
            can_merge_back=other.can_merge_back,
            success_rate=other.success_rate,
            task_completion_time=other.task_completion_time,
            error_count=other.error_count,
            preferred_tasks=list(other.preferred_tasks),
            genome=chrom2,
        )

        return child1, child2

    def mutate(self, mutation_rate: Optional[float] = None) -> "BeeDNA":
        """生成突变的BeeDNA副本 - 覆盖基类方法"""
        # 先调用父类逻辑获取基础突变 (注意：父类返回的是 DNA 实例)
        import random
        base_mutated = super().mutate(mutation_rate)
        
        # 转换回 BeeDNA 并处理特有字段
        # 生成新 ID
        new_id = f"{self.id}_mut_{int(random.random()*1000)}"
        
        mutated = BeeDNA(
            # 基础字段
            species=base_mutated.species,
            generation=base_mutated.generation,
            version=base_mutated.version,
            fitness=base_mutated.fitness,
            mining_capability=base_mutated.mining_capability,
            gathering_capability=base_mutated.gathering_capability,
            architectural_capability=base_mutated.architectural_capability,
            librarian_capability=base_mutated.librarian_capability,
            sentinel_capability=base_mutated.sentinel_capability,
            immune_response=base_mutated.immune_response,
            learning_rate=base_mutated.learning_rate,
            mutation_rate=base_mutated.mutation_rate,
            metadata=base_mutated.metadata,
            
            # BeeDNA 特有字段
            id=new_id,
            role=self.role,
            target_name=self.target_name,
            target_url=self.target_url,
            mutation_gene=self.mutation_gene,
            can_merge_back=self.can_merge_back,
            # 性能指标重置
            success_rate=self.success_rate * 0.9, # 突变后稍作衰减
            task_completion_time=0.0,
            error_count=0
        )
        return mutated

    def update_performance(
        self, success: bool, completion_time: float, error: bool = False
    ) -> None:
        """更新性能指标

        Args:
            success: 任务是否成功
            completion_time: 任务完成时间（秒）
            error: 是否发生错误
        """
        # 更新成功率（指数移动平均）
        alpha = 0.1  # 学习因子
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate

        # 更新平均完成时间
        if self.task_completion_time == 0:
            self.task_completion_time = completion_time
        else:
            self.task_completion_time = alpha * completion_time + (1 - alpha) * self.task_completion_time

        # 更新错误计数
        if error:
            self.error_count += 1

        # 重新计算适应度
        self.calculate_fitness()


class DNAPool:
    """DNA池

    管理多个DNA实例，用于进化和选择。使用 LRU 策略防止无界增长。
    """

    MAX_SIZE = 10000  # 最多保留 10000 个 DNA 对象

    def __init__(self):
        """初始化DNA池"""
        self._pool: OrderedDict[str, DNA] = OrderedDict()
        self._generation = 1

    def add(self, dna: DNA) -> None:
        """添加DNA到池中，超出上限时淘汰最旧条目

        Args:
            dna: DNA实例
        """
        if not dna.validate():
            logger.warning("Invalid DNA not added to pool", species=dna.species)
            return

        key = f"{dna.species}_{dna.generation}_{uuid.uuid4().hex[:8]}"
        self._pool[key] = dna

        # LRU 淘汰：超出上限时移除最旧的条目
        while len(self._pool) > self.MAX_SIZE:
            self._pool.popitem(last=False)

    def get_best(self, species: str, count: int = 1) -> list[DNA]:
        """获取指定物种的最佳DNA

        Args:
            species: 物种名称
            count: 返回数量

        Returns:
            最佳DNA列表（按适应度降序）
        """
        species_dnas = [dna for dna in self._pool.values() if dna.species == species]
        species_dnas.sort(key=lambda x: x.fitness, reverse=True)
        return species_dnas[:count]

    def _breed_new_individuals(self, best_dnas: list, population_size: int) -> list[DNA]:
        import random
        new_generation = []
        while len(new_generation) < population_size:
            parents = random.sample(best_dnas, min(2, len(best_dnas)))
            child = parents[0].crossover(parents[1]) if len(parents) == 2 else parents[0]
            new_generation.append(child.mutate())
        return new_generation

    def evolve_generation(self, species: str, population_size: int = 10) -> list[DNA]:
        best_dnas = self.get_best(species, count=min(5, population_size))
        if not best_dnas:
            logger.warning("No DNA found for evolution", species=species)
            return []

        # 精英数量不超过 population_size，新个体补足至 population_size
        elite_count = min(len(best_dnas), population_size)
        elites = list(best_dnas[:elite_count])
        offspring_count = max(0, population_size - elite_count)
        new_generation = elites + self._breed_new_individuals(best_dnas, offspring_count)
        self._generation += 1

        logger.info(
            "DNA evolution completed",
            species=species,
            generation=self._generation,
            population_size=len(new_generation),
        )
        return new_generation

    def get_pool_size(self) -> int:
        """获取池中DNA数量

        Returns:
            DNA数量
        """
        return len(self._pool)

    def clear(self) -> None:
        """清空DNA池"""
        self._pool.clear()
        self._generation = 1

        logger.info("DNA pool cleared")
