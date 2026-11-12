# -*- coding: utf-8 -*-
"""
EvolutionEngine - DNA 进化引擎

专注负责 DNA 的繁殖、突变和进化。
支持两种进化模式：
1. 传统 DNA 进化：基于 BeeDNA 的浮点向量繁殖
2. GEP 基因进化：基于树结构表达式的基因编程
"""

import logging
import random
import time
from typing import Any, Optional

from hive.core.dna import BeeDNA

logger = logging.getLogger("Hive.EvolutionEngine")

# GEP imports - lazy loading to avoid hard dependency
try:
    from hive.core.gep_gene import (
        FunctionType,
        GEPChromosome,
        GEPFitnessEvaluator,
        GEPGene,
        GEPGeneFactory,
        GEPOperators,
    )

    GEP_AVAILABLE = True
except ImportError:
    GEP_AVAILABLE = False
    logger.warning("GEP gene module not available - GEP evolution disabled")


class EvolutionEngine:
    """DNA 进化引擎

    支持两种进化模式：
    - reproduce(): 传统 BeeDNA 繁殖（浮点向量）
    - reproduce_gep(): GEP 基因表达式繁殖（树结构）
    """

    MAX_GENERATION = 10
    MUTATION_ROLES = ["darwin", "miner", "seeker", "architect", "librarian", "sentinel"]

    def __init__(
        self,
        max_generation: int = MAX_GENERATION,
        mutation_roles: list[str] | None = None,
        use_gep: bool = False,
        gep_population_size: int = 20,
        gep_max_genes: int = 5,
        gep_max_depth: int = 10,
    ):
        self.max_generation = max_generation
        self.mutation_roles = mutation_roles or self.MUTATION_ROLES.copy()
        self.generation_count = 0
        self.reproduction_count = 0

        # GEP evolution settings
        self.use_gep = use_gep and GEP_AVAILABLE
        self.gep_population_size = gep_population_size
        self.gep_max_genes = gep_max_genes
        self.gep_max_depth = gep_max_depth

        # GEP components (lazy initialization)
        self._gep_factory: Optional[GEPGeneFactory] = None
        self._gep_operators: Optional[GEPOperators] = None
        self._gep_evaluator: Optional[GEPFitnessEvaluator] = None
        self._gep_population: list[GEPChromosome] = []
        self._gep_generation = 0

    @property
    def gep_factory(self) -> Optional[GEPGeneFactory]:
        """Lazy initialization of GEP factory."""
        if self._gep_factory is None and GEP_AVAILABLE:
            self._gep_factory = GEPGeneFactory(
                max_depth=self.gep_max_depth, max_genes=self.gep_max_genes
            )
        return self._gep_factory

    @property
    def gep_operators(self) -> Optional[GEPOperators]:
        """Lazy initialization of GEP operators."""
        if self._gep_operators is None and GEP_AVAILABLE:
            self._gep_operators = GEPOperators(
                max_gene_depth=self.gep_max_depth, max_genes_per_chromosome=self.gep_max_genes
            )
        return self._gep_operators

    @property
    def gep_evaluator(self) -> Optional[GEPFitnessEvaluator]:
        """Lazy initialization of GEP evaluator."""
        if self._gep_evaluator is None and GEP_AVAILABLE:
            self._gep_evaluator = GEPFitnessEvaluator()
        return self._gep_evaluator

    def reproduce(self, parent_dna_dict: dict[str, Any]) -> BeeDNA | None:
        """分裂繁殖"""
        if not parent_dna_dict:
            return None

        parent_id = parent_dna_dict.get("id", "unknown")
        generation = self._calculate_generation(parent_id)

        if not self._check_resources(generation, parent_dna_dict.get("target_name", "unknown")):
            return None

        new_id = self._generate_new_id(parent_id, generation)
        target_name = parent_dna_dict.get("target_name", "unknown")

        self.reproduction_count += 1
        return self._create_offspring_dna(generation, new_id, target_name)

    def _calculate_generation(self, parent_id: str) -> int:
        """计算代数"""
        if "Gen" in parent_id:
            try:
                parts = parent_id.split("_")
                for p in parts:
                    if p.startswith("Gen") and p[3:].isdigit():
                        return int(p[3:]) + 1
            except (ValueError, IndexError):
                pass
        return 1

    def _generate_new_id(self, parent_id: str, generation: int) -> str:
        """生成新 ID（使用纳秒时间戳避免碰撞）"""
        import uuid

        short_hash = uuid.uuid4().hex[:6]
        # 使用纳秒时间戳后6位，避免同一秒内碰撞
        nano_part = str(time.time_ns())[-6:]
        return f"Gen{generation}_{short_hash}_{nano_part}"

    def _check_resources(self, generation: int, target_name: str) -> bool:
        """检查繁殖资源"""
        if generation > self.max_generation:
            logger.info(f"[EvolutionEngine] 基因链过长 (Gen {generation}), 停止分裂: {target_name}")
            return False
        return True

    def _create_offspring_dna(self, generation: int, new_id: str, target_name: str) -> BeeDNA:
        """创建子代 DNA"""
        if random.random() < 0.5:
            new_role = random.choice(self.mutation_roles)
            logger.info(f"[EvolutionEngine] Gen{generation} 突变: {new_role.upper()}")
            return BeeDNA(species="mutant", role=new_role, id=new_id, target_name=target_name)
        else:
            logger.info(f"[EvolutionEngine] Gen{generation} 正常分裂")
            return BeeDNA(species="bee", role="worker", id=new_id, target_name=target_name)

    def evolve_batch(self, results: list[dict[str, Any]]) -> list[BeeDNA]:
        """批量进化"""
        offspring = []
        for res in results:
            if res.get("status") == "SUCCESS":
                child = self.reproduce(res.get("dna", {}))
                if child:
                    offspring.append(child)

        if offspring:
            logger.info(f"[EvolutionEngine] 批量进化产生 {len(offspring)} 个子代")

        return offspring

    # =========================================================================
    # GEP Gene Expression Programming Methods
    # =========================================================================

    def reproduce_gep(
        self,
        parent_chromosome: Optional[GEPChromosome] = None,
        context: Optional[dict[str, Any]] = None,
        expected_result: Optional[Any] = None,
    ) -> Optional[GEPChromosome]:
        """
        GEP gene expression programming reproduction.

        Creates offspring from a parent chromosome using:
        - Point mutation
        - Subtree crossover
        - Gene crossover (between chromosomes)

        Args:
            parent_chromosome: Parent chromosome to reproduce from
            context: Optional task context for fitness evaluation
            expected_result: Optional expected output for supervised evaluation

        Returns:
            New offspring chromosome, or None if reproduction fails
        """
        if not GEP_AVAILABLE:
            logger.warning("[EvolutionEngine] GEP not available")
            return None

        if self.gep_factory is None or self.gep_operators is None:
            logger.error("[EvolutionEngine] GEP components not initialized")
            return None

        self.reproduction_count += 1

        # If no parent, create random chromosome
        if parent_chromosome is None:
            new_chromosome = self.gep_factory.create_random_chromosome()
            logger.info(f"[EvolutionEngine] GEP: Created random chromosome {new_chromosome.id}")
            return new_chromosome

        # Determine operation: mutation or crossover
        operation = random.random()

        if operation < 0.6:
            # Point mutation
            mutated_genes = []
            for gene in parent_chromosome.genes:
                mutated_gene = self.gep_operators.point_mutation(gene)
                mutated_genes.append(mutated_gene)

            new_chromosome = GEPChromosome(
                genes=mutated_genes,
                linking_function=parent_chromosome.linking_function,
                max_genes=parent_chromosome.max_genes,
            )
            logger.info(
                f"[EvolutionEngine] GEP: Point mutation on chromosome {parent_chromosome.id}"
            )

        elif operation < 0.8:
            # Subtree crossover within chromosome
            if len(parent_chromosome.genes) >= 2:
                gene1_idx = random.randint(0, len(parent_chromosome.genes) - 1)
                gene2_idx = (gene1_idx + 1) % len(parent_chromosome.genes)

                new_gene1, new_gene2 = self.gep_operators.subtree_crossover(
                    parent_chromosome.genes[gene1_idx], parent_chromosome.genes[gene2_idx]
                )

                new_genes = list(parent_chromosome.genes)
                new_genes[gene1_idx] = new_gene1
                new_genes[gene2_idx] = new_gene2

                new_chromosome = GEPChromosome(
                    genes=new_genes,
                    linking_function=parent_chromosome.linking_function,
                    max_genes=parent_chromosome.max_genes,
                )
                logger.info(
                    f"[EvolutionEngine] GEP: Subtree crossover on chromosome {parent_chromosome.id}"
                )
            else:
                # Fallback to point mutation
                mutated_genes = [
                    self.gep_operators.point_mutation(gene) for gene in parent_chromosome.genes
                ]
                new_chromosome = GEPChromosome(
                    genes=mutated_genes,
                    linking_function=parent_chromosome.linking_function,
                    max_genes=parent_chromosome.max_genes,
                )

        else:
            # Add new gene via gene crossover (requires population)
            if self._gep_population and len(self._gep_population) > 1:
                # Select random partner
                partner = random.choice(
                    [p for p in self._gep_population if p.id != parent_chromosome.id]
                )
                new_chrom1, new_chrom2 = self.gep_operators.gene_crossover(
                    parent_chromosome, partner
                )
                new_chromosome = new_chrom1
                logger.info(
                    f"[EvolutionEngine] GEP: Gene crossover {parent_chromosome.id} x {partner.id}"
                )
            else:
                # Fallback: add a new random gene
                new_genes = list(parent_chromosome.genes)
                if len(new_genes) < parent_chromosome.max_genes:
                    new_genes.append(self.gep_factory.create_random_gene())
                    new_chromosome = GEPChromosome(
                        genes=new_genes,
                        linking_function=parent_chromosome.linking_function,
                        max_genes=parent_chromosome.max_genes,
                    )
                else:
                    # Replace existing gene with random
                    replace_idx = random.randint(0, len(new_genes) - 1)
                    new_genes[replace_idx] = self.gep_factory.create_random_gene()
                    new_chromosome = GEPChromosome(
                        genes=new_genes,
                        linking_function=parent_chromosome.linking_function,
                        max_genes=parent_chromosome.max_genes,
                    )

        # Evaluate fitness if context provided
        if context is not None and self.gep_evaluator is not None:
            self.gep_evaluator.evaluate_chromosome(
                new_chromosome,
                [context],
                [expected_result] if expected_result is not None else None,
            )

        return new_chromosome

    def evolve_gep_generation(
        self,
        contexts: Optional[list[dict[str, Any]]] = None,
        expected_results: Optional[list[Any]] = None,
        elite_count: int = 2,
    ) -> list[GEPChromosome]:
        """
        Evolve a complete GEP generation.

        Uses tournament selection and elitism to create next generation.

        Args:
            contexts: List of task contexts for evaluation
            expected_results: Expected outputs for supervised evaluation
            elite_count: Number of top individuals to preserve

        Returns:
            New generation of chromosomes
        """
        if not GEP_AVAILABLE or self.gep_factory is None or self.gep_operators is None:
            logger.warning("[EvolutionEngine] GEP not available for evolution")
            return []

        if not self._gep_population:
            # Initialize population
            self._gep_population = [
                self.gep_factory.create_random_chromosome() for _ in range(self.gep_population_size)
            ]
            self._gep_generation = 0
            logger.info(
                f"[EvolutionEngine] GEP: Initialized population with {len(self._gep_population)} chromosomes"
            )

        # Evaluate all chromosomes
        if contexts and self.gep_evaluator is not None:
            for chrom in self._gep_population:
                self.gep_evaluator.evaluate_chromosome(chrom, contexts, expected_results)

        # Sort by fitness
        self._gep_population.sort(key=lambda c: c.fitness, reverse=True)

        # Apply elitism: preserve top N
        elites = self._gep_population[:elite_count]

        # Create offspring through reproduction
        offspring = []
        while len(offspring) < self.gep_population_size - elite_count:
            # Tournament selection
            parent = self._tournament_select(self._gep_population)

            # Reproduce
            child = self.reproduce_gep(parent)

            if child is not None:
                offspring.append(child)

        # Combine elites and offspring
        new_generation = elites + offspring

        self._gep_population = new_generation
        self._gep_generation += 1

        logger.info(
            f"[EvolutionEngine] GEP: Generation {self._gep_generation} complete, "
            f"best fitness: {new_generation[0].fitness if new_generation else 0.0:.4f}"
        )

        return new_generation

    def _tournament_select(
        self, population: list[GEPChromosome], tournament_size: int = 3
    ) -> GEPChromosome:
        """
        Tournament selection for GEP chromosomes.

        Args:
            population: Population to select from
            tournament_size: Number of individuals in tournament

        Returns:
            Selected chromosome
        """
        if len(population) <= tournament_size:
            return random.choice(population)

        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda c: c.fitness)

    def get_best_gep_chromosome(self, count: int = 1) -> list[GEPChromosome]:
        """
        Get the best GEP chromosomes by fitness.

        Args:
            count: Number of chromosomes to return

        Returns:
            List of top chromosomes
        """
        if not self._gep_population:
            return []

        self._gep_population.sort(key=lambda c: c.fitness, reverse=True)
        return self._gep_population[:count]

    def get_gep_status(self) -> dict[str, Any]:
        """Get GEP evolution status."""
        return {
            "gep_available": GEP_AVAILABLE,
            "gep_enabled": self.use_gep,
            "gep_generation": self._gep_generation,
            "gep_population_size": len(self._gep_population),
            "gep_best_fitness": max((c.fitness for c in self._gep_population), default=0.0),
            "gep_avg_fitness": sum(c.fitness for c in self._gep_population)
            / max(len(self._gep_population), 1),
        }

    def get_status(self) -> dict[str, Any]:
        """获取进化状态"""
        status = {
            "generation_count": self.generation_count,
            "reproduction_count": self.reproduction_count,
            "max_generation": self.max_generation,
        }

        # Include GEP status if enabled
        if self.use_gep:
            status.update(self.get_gep_status())

        return status
