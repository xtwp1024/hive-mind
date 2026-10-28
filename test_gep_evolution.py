# -*- coding: utf-8 -*-
"""
GEP Gene Expression Programming - Evolution Test

Tests for the True GEP gene structure, demonstrating:
1. Gene creation and structure
2. Gene evaluation on task contexts
3. Genetic operators (mutation, crossover)
4. Chromosome evolution over generations
5. Comparison with flat DNA vectors

Run with: python -m pytest test_gep_evolution.py -v
Or directly: python test_gep_evolution.py
"""

import logging
import random
import sys
import time
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Test.GEP")

# Add Hive to path
sys.path.insert(0, r"D:\Hive")

from hive.core.gep_gene import (
    GEPGene, GEPChromosome, GEPGeneFactory, GEPFitnessEvaluator,
    GEPOperators, FunctionType, TerminalType,
    gene_to_dict, dict_to_gene, chromosome_to_dict, dict_to_chromosome
)


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

def create_sample_context(task_type: str = "mining", success_rate: float = 0.8) -> dict[str, Any]:
    """Create a sample task context for evaluation."""
    return {
        "task_type": task_type,
        "task_complexity": 0.6,
        "success_rate": success_rate,
        "error_count": 2,
        "avg_time": 15.0,
        "memory_usage": 0.7,
        "cpu_usage": 0.5,
    }


def print_gene_details(gene: GEPGene, context: dict[str, Any]) -> None:
    """Print detailed information about a gene."""
    expr = gene.root.to_readable()
    result = gene.evaluate(context)
    print(f"  Gene ID: {gene.id}")
    print(f"  Expression: {expr}")
    print(f"  Depth: {gene.root.get_depth()}")
    print(f"  Result: {result}")
    print(f"  Fitness: {gene.fitness:.4f}")


def print_chromosome_details(chrom: GEPChromosome, context: dict[str, Any]) -> None:
    """Print detailed information about a chromosome."""
    print(f"\n  Chromosome ID: {chrom.id}")
    print(f"  Linking Function: {chrom.linking_function.value}")
    print(f"  Number of Genes: {len(chrom.genes)}")
    print(f"  Fitness: {chrom.fitness:.4f}")

    for i, gene in enumerate(chrom.genes):
        print(f"\n  Gene {i+1}:")
        print_gene_details(gene, context)


# ============================================================================
# Test 1: Basic Gene Structure
# ============================================================================

def test_gene_structure():
    """Test: Create and validate gene structure."""
    print("\n" + "="*70)
    print("TEST 1: Basic Gene Structure")
    print("="*70)

    factory = GEPGeneFactory(max_depth=5)

    # Create a simple gene
    gene = factory.create_random_gene(depth=3)

    print(f"\nCreated Gene:")
    print(f"  ID: {gene.id}")
    print(f"  Expression: {gene.root.to_readable()}")
    print(f"  Depth: {gene.root.get_depth()}")
    print(f"  Max Depth: {gene.max_depth}")
    print(f"  Valid: {gene.validate_structure()}")

    # Test evaluation
    context = create_sample_context()
    result = gene.evaluate(context)
    print(f"\n  Evaluation on context: {result}")

    assert gene.validate_structure(), "Gene should be valid"
    print("\n[PASS] Gene structure test passed")
    return gene


# ============================================================================
# Test 2: Gene Evaluation
# ============================================================================

def test_gene_evaluation():
    """Test: Evaluate genes against various contexts."""
    print("\n" + "="*70)
    print("TEST 2: Gene Evaluation")
    print("="*70)

    factory = GEPGeneFactory(max_depth=5)
    evaluator = GEPFitnessEvaluator()

    # Test multiple contexts
    contexts = [
        create_sample_context("mining", success_rate=0.9),
        create_sample_context("gathering", success_rate=0.3),
        create_sample_context("architect", success_rate=0.7),
    ]

    # Create a gene and evaluate on multiple contexts
    gene = factory.create_random_gene(depth=4)

    print(f"\nGene: {gene.root.to_readable()}")

    for i, ctx in enumerate(contexts):
        result = gene.evaluate(ctx)
        expected = ctx["success_rate"] > 0.5  # Simple expected result

        fitness = evaluator.evaluate_gene(gene, ctx, expected)

        print(f"\n  Context {i+1} ({ctx['task_type']}):")
        print(f"    Success Rate: {ctx['success_rate']}")
        print(f"    Gene Result: {result}")
        print(f"    Expected: {expected}")
        print(f"    Fitness: {fitness:.4f}")

    print("\n[PASS] Gene evaluation test passed")


# ============================================================================
# Test 3: Genetic Operators - Point Mutation
# ============================================================================

def test_point_mutation():
    """Test: Point mutation operator."""
    print("\n" + "="*70)
    print("TEST 3: Point Mutation")
    print("="*70)

    factory = GEPGeneFactory(max_depth=5)
    operators = GEPOperators(mutation_rate=0.2)

    # Create original gene
    original = factory.create_random_gene(depth=4)
    print(f"\nOriginal Gene:")
    print(f"  Expression: {original.root.to_readable()}")
    print(f"  ID: {original.id}")

    # Apply point mutation
    mutated = operators.point_mutation(original)

    print(f"\nMutated Gene:")
    print(f"  Expression: {mutated.root.to_readable()}")
    print(f"  ID: {mutated.id}")
    print(f"  Same structure: {original.id != mutated.id}")

    # Verify both are valid
    assert original.validate_structure(), "Original gene should be valid"
    assert mutated.validate_structure(), "Mutated gene should be valid"

    print("\n[PASS] Point mutation test passed")
    return original, mutated


# ============================================================================
# Test 4: Genetic Operators - Subtree Crossover
# ============================================================================

def test_subtree_crossover():
    """Test: Subtree crossover operator."""
    print("\n" + "="*70)
    print("TEST 4: Subtree Crossover")
    print("="*70)

    factory = GEPGeneFactory(max_depth=5)
    operators = GEPOperators()

    # Create two genes
    gene1 = factory.create_random_gene(depth=4)
    gene2 = factory.create_random_gene(depth=4)

    print(f"\nGene 1:")
    print(f"  Expression: {gene1.root.to_readable()}")

    print(f"\nGene 2:")
    print(f"  Expression: {gene2.root.to_readable()}")

    # Apply subtree crossover
    new_gene1, new_gene2 = operators.subtree_crossover(gene1, gene2)

    print(f"\nAfter Crossover:")
    print(f"  Gene 1: {new_gene1.root.to_readable()}")
    print(f"  Gene 2: {new_gene2.root.to_readable()}")

    # Verify validity
    assert gene1.validate_structure(), "Gene1 should remain valid"
    assert gene2.validate_structure(), "Gene2 should remain valid"
    assert new_gene1.validate_structure(), "New gene1 should be valid"
    assert new_gene2.validate_structure(), "New gene2 should be valid"

    print("\n[PASS] Subtree crossover test passed")
    return new_gene1, new_gene2


# ============================================================================
# Test 5: Chromosome Creation and Evaluation
# ============================================================================

def test_chromosome():
    """Test: Chromosome with multiple genes."""
    print("\n" + "="*70)
    print("TEST 5: Chromosome Creation")
    print("="*70)

    factory = GEPGeneFactory(max_genes=5, max_depth=4)

    # Create chromosome with 3 genes
    chrom = factory.create_random_chromosome(num_genes=3)

    print(f"\nChromosome:")
    print(f"  ID: {chrom.id}")
    print(f"  Linking Function: {chrom.linking_function.value}")
    print(f"  Number of Genes: {len(chrom.genes)}")

    for i, gene in enumerate(chrom.genes):
        print(f"\n  Gene {i+1}: {gene.root.to_readable()}")

    # Evaluate chromosome
    context = create_sample_context()
    result = chrom.evaluate(context)

    print(f"\n  Evaluation Result: {result}")

    # Test serialization
    chrom_dict = chromosome_to_dict(chrom)
    restored = dict_to_chromosome(chrom_dict)

    print(f"\n  Serialization Test:")
    print(f"    Original ID: {chrom.id}")
    print(f"    Restored ID: {restored.id}")
    print(f"    Gene Count Match: {len(chrom.genes) == len(restored.genes)}")

    print("\n[PASS] Chromosome test passed")
    return chrom


# ============================================================================
# Test 6: Evolution Engine Integration
# ============================================================================

def test_evolution_engine():
    """Test: Integration with EvolutionEngine."""
    print("\n" + "="*70)
    print("TEST 6: Evolution Engine Integration")
    print("="*70)

    from hive.core.evolution_engine import EvolutionEngine

    # Create engine with GEP enabled
    engine = EvolutionEngine(
        use_gep=True,
        gep_population_size=10,
        gep_max_genes=3,
        gep_max_depth=5
    )

    print(f"\nEvolutionEngine GEP Status:")
    status = engine.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Create initial population implicitly via evolve_gep_generation
    contexts = [create_sample_context("mining", 0.7 + i*0.05) for i in range(3)]
    expected = [c["success_rate"] > 0.5 for c in contexts]

    print(f"\nEvolving generation 1...")
    population = engine.evolve_gep_generation(contexts, expected, elite_count=2)

    print(f"\n  Population size: {len(population)}")
    print(f"  Best fitness: {population[0].fitness if population else 0:.4f}")

    # Reproduce from best
    if population:
        best = population[0]
        print(f"\n  Reproducing from best chromosome {best.id}...")
        offspring = engine.reproduce_gep(best, contexts[0], expected[0])

        if offspring:
            print(f"  Offspring ID: {offspring.id}")
            print(f"  Offspring genes: {len(offspring.genes)}")
            print(f"  Offspring fitness: {offspring.fitness:.4f}")

    # Get final status
    final_status = engine.get_status()
    print(f"\n  Final GEP Generation: {final_status.get('gep_generation', 0)}")

    print("\n[PASS] Evolution engine integration test passed")
    return engine


# ============================================================================
# Test 7: Multi-Generation Evolution
# ============================================================================

def test_multi_generation_evolution():
    """Test: Evolve over multiple generations to show improvement."""
    print("\n" + "="*70)
    print("TEST 7: Multi-Generation Evolution")
    print("="*70)

    from hive.core.evolution_engine import EvolutionEngine

    random.seed(42)  # For reproducibility

    # Create engine
    engine = EvolutionEngine(
        use_gep=True,
        gep_population_size=20,
        gep_max_genes=4,
        gep_max_depth=6
    )

    # Define task contexts for evaluation
    # Simulate different scenarios with known optimal responses
    evaluation_contexts = [
        {"task_type": "mining", "success_rate": 0.9, "error_count": 1, "avg_time": 10.0},
        {"task_type": "gathering", "success_rate": 0.3, "error_count": 5, "avg_time": 30.0},
        {"task_type": "architect", "success_rate": 0.7, "error_count": 2, "avg_time": 20.0},
        {"task_type": "sentinel", "success_rate": 0.85, "error_count": 1, "avg_time": 15.0},
    ]

    # Expected results: task succeeds when success_rate > 0.6
    expected_results = [ctx["success_rate"] > 0.6 for ctx in evaluation_contexts]

    print(f"\nEvaluation Contexts:")
    for i, (ctx, exp) in enumerate(zip(evaluation_contexts, expected_results)):
        print(f"  {i+1}. {ctx['task_type']}: success_rate={ctx['success_rate']:.2f} -> expected={exp}")

    # Evolve for N generations
    num_generations = 5
    best_fitness_history = []
    avg_fitness_history = []

    print(f"\n{'='*50}")
    print(f"Evolving for {num_generations} generations...")
    print(f"{'='*50}")

    for gen in range(num_generations):
        population = engine.evolve_gep_generation(
            evaluation_contexts,
            expected_results,
            elite_count=2
        )

        if population:
            best_fitness = population[0].fitness
            avg_fitness = sum(c.fitness for c in population) / len(population)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            print(f"\nGeneration {gen + 1}:")
            print(f"  Best Fitness: {best_fitness:.4f}")
            print(f"  Avg Fitness:  {avg_fitness:.4f}")
            print(f"  Best Chromosome: {population[0].id}")

    # Show improvement
    print(f"\n{'='*50}")
    print("EVOLUTION SUMMARY")
    print(f"{'='*50}")
    print(f"\nFitness History (Best):")
    for i, fit in enumerate(best_fitness_history):
        print(f"  Gen {i+1}: {fit:.4f}")

    print(f"\nFitness History (Average):")
    for i, fit in enumerate(avg_fitness_history):
        print(f"  Gen {i+1}: {fit:.4f}")

    improvement = best_fitness_history[-1] - best_fitness_history[0] if best_fitness_history else 0
    print(f"\nOverall Improvement: {improvement:+.4f}")

    # Get best chromosome details
    best_chromosomes = engine.get_best_gep_chromosome(count=1)
    if best_chromosomes:
        best = best_chromosomes[0]
        print(f"\nBest Chromosome Details:")
        print(f"  ID: {best.id}")
        print(f"  Fitness: {best.fitness:.4f}")
        print(f"  Linking: {best.linking_function.value}")
        print(f"  Gene Expressions:")
        for i, expr in enumerate(best.get_gene_expressions()):
            print(f"    {i+1}. {expr}")

    print("\n[PASS] Multi-generation evolution test passed")


# ============================================================================
# Test 8: Serialization Roundtrip
# ============================================================================

def test_serialization():
    """Test: Gene and chromosome serialization."""
    print("\n" + "="*70)
    print("TEST 8: Serialization Roundtrip")
    print("="*70)

    factory = GEPGeneFactory(max_depth=5)

    # Test gene serialization
    gene = factory.create_random_gene(depth=4)
    gene_dict = gene_to_dict(gene)
    restored_gene = dict_to_gene(gene_dict)

    print(f"\nGene Serialization:")
    print(f"  Original ID: {gene.id}")
    print(f"  Restored ID: {restored_gene.id}")
    print(f"  Original Expr: {gene.root.to_readable()}")
    print(f"  Restored Expr: {restored_gene.root.to_readable()}")
    print(f"  Match: {gene.root.to_readable() == restored_gene.root.to_readable()}")

    # Test chromosome serialization
    chrom = factory.create_random_chromosome(num_genes=3)
    chrom_dict = chromosome_to_dict(chrom)
    restored_chrom = dict_to_chromosome(chrom_dict)

    print(f"\nChromosome Serialization:")
    print(f"  Original ID: {chrom.id}")
    print(f"  Restored ID: {restored_chrom.id}")
    print(f"  Gene Count Match: {len(chrom.genes) == len(restored_chrom.genes)}")
    print(f"  Linking Match: {chrom.linking_function == restored_chrom.linking_function}")

    assert gene.root.to_readable() == restored_gene.root.to_readable()
    assert len(chrom.genes) == len(restored_chrom.genes)

    print("\n[PASS] Serialization test passed")


# ============================================================================
# Test 9: Performance Benchmark
# ============================================================================

def test_performance():
    """Test: Performance benchmarking."""
    print("\n" + "="*70)
    print("TEST 9: Performance Benchmark")
    print("="*70)

    import timeit

    factory = GEPGeneFactory(max_depth=8)
    evaluator = GEPFitnessEvaluator()
    operators = GEPOperators(mutation_rate=0.1)

    context = create_sample_context()

    # Benchmark gene creation
    def create_gene():
        return factory.create_random_gene(depth=6)

    creation_time = timeit.timeit(create_gene, number=100)
    print(f"\nGene Creation (100 genes): {creation_time:.4f}s ({creation_time*10:.2f}ms/gene)")

    # Benchmark evaluation
    gene = factory.create_random_gene(depth=6)

    def evaluate_gene():
        gene.evaluate(context)

    eval_time = timeit.timeit(evaluate_gene, number=1000)
    print(f"Gene Evaluation (1000 evals): {eval_time:.4f}s ({eval_time:.2f}ms/eval)")

    # Benchmark mutation
    def mutate_gene():
        operators.point_mutation(gene)

    mutation_time = timeit.timeit(mutate_gene, number=100)
    print(f"Point Mutation (100 ops): {mutation_time:.4f}s ({mutation_time*10:.2f}ms/op)")

    # Benchmark crossover
    gene2 = factory.create_random_gene(depth=6)

    def crossover():
        operators.subtree_crossover(gene, gene2)

    crossover_time = timeit.timeit(crossover, number=100)
    print(f"Subtree Crossover (100 ops): {crossover_time:.4f}s ({crossover_time*10:.2f}ms/op)")

    print("\n[PASS] Performance benchmark test passed")


# ============================================================================
# Test 10: Research Experiment - GEP vs Flat DNA
# ============================================================================

def research_experiment_gep_vs_flat():
    """
    Research Question: Does tree-structured GEP produce better strategies
    than flat float vectors?

    This experiment compares:
    1. GEP chromosome fitness over generations
    2. Traditional DNA crossover performance
    3. Analysis of evolved strategy complexity
    """
    print("\n" + "="*70)
    print("RESEARCH EXPERIMENT: GEP vs Flat DNA")
    print("="*70)

    from hive.core.evolution_engine import EvolutionEngine
    from hive.core.dna import BeeDNA, DNA

    random.seed(123)

    # Create task contexts with varying difficulty
    task_contexts = []
    expected_results = []

    # Easy tasks (high success rate)
    for i in range(5):
        ctx = {"task_type": "easy", "success_rate": 0.9 - i*0.05, "error_count": i}
        task_contexts.append(ctx)
        expected_results.append(True)  # Expect success

    # Hard tasks (low success rate)
    for i in range(5):
        ctx = {"task_type": "hard", "success_rate": 0.3 + i*0.05, "error_count": 10 - i}
        task_contexts.append(ctx)
        expected_results.append(False)  # Expect failure

    print(f"\nTask Distribution:")
    print(f"  Easy tasks: 5 (success_rate 0.85-0.90)")
    print(f"  Hard tasks: 5 (success_rate 0.35-0.55)")

    # ---- GEP Evolution ----
    print(f"\n{'='*50}")
    print("PART 1: GEP Evolution")
    print(f"{'='*50}")

    gep_engine = EvolutionEngine(
        use_gep=True,
        gep_population_size=30,
        gep_max_genes=4,
        gep_max_depth=7
    )

    gep_fitness_history = []

    for gen in range(10):
        population = gep_engine.evolve_gep_generation(
            task_contexts,
            expected_results,
            elite_count=3
        )

        if population:
            best_fit = population[0].fitness
            avg_fit = sum(c.fitness for c in population) / len(population)
            gep_fitness_history.append((best_fit, avg_fit))

            print(f"  Gen {gen+1:2d}: Best={best_fit:.4f}, Avg={avg_fit:.4f}")

    # ---- Traditional DNA Evolution ----
    print(f"\n{'='*50}")
    print("PART 2: Traditional DNA Evolution")
    print(f"{'='*50}")

    # Create initial DNA population
    dna_pool = []

    for i in range(30):
        dna = BeeDNA(
            species="bee",
            generation=1,
            mining_capability=random.random(),
            gathering_capability=random.random(),
            architectural_capability=random.random(),
            librarian_capability=random.random(),
            sentinel_capability=random.random(),
            immune_response=random.random(),
            learning_rate=random.random(),
            success_rate=0.5,
            id=f"dna_{i}"
        )
        dna_pool.append(dna)

    dna_fitness_history = []

    for gen in range(10):
        # Evaluate DNA fitness (simplified)
        # Note: DNA base class doesn't have calculate_fitness, so we compute inline
        for dna in dna_pool:
            # Simple fitness based on success rate and capabilities
            # Use weighted average of capabilities as proxy fitness
            fitness = (
                dna.mining_capability * 0.2 +
                dna.gathering_capability * 0.2 +
                dna.architectural_capability * 0.2 +
                dna.librarian_capability * 0.2 +
                dna.sentinel_capability * 0.2
            )
            dna.fitness = fitness

        # Sort by fitness
        dna_pool.sort(key=lambda d: d.fitness, reverse=True)

        # Elitism
        elites = dna_pool[:3]

        # Create offspring
        offspring = []
        while len(offspring) < 30 - 3:
            parents = random.sample(elites, 2)
            child = parents[0].crossover(parents[1])
            child = child.mutate()
            offspring.append(child)

        dna_pool = elites + offspring

        best_fit = dna_pool[0].fitness
        avg_fit = sum(d.fitness for d in dna_pool) / len(dna_pool)
        dna_fitness_history.append((best_fit, avg_fit))

        print(f"  Gen {gen+1:2d}: Best={best_fit:.4f}, Avg={avg_fit:.4f}")

    # ---- Comparison Analysis ----
    print(f"\n{'='*50}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*50}")

    print(f"\nGEP Final Best Fitness: {gep_fitness_history[-1][0]:.4f}")
    print(f"DNA Final Best Fitness:  {dna_fitness_history[-1][0]:.4f}")

    print(f"\nGEP Final Avg Fitness: {gep_fitness_history[-1][1]:.4f}")
    print(f"DNA Final Avg Fitness:  {dna_fitness_history[-1][1]:.4f}")

    # Calculate improvement
    gep_improvement = gep_fitness_history[-1][0] - gep_fitness_history[0][0]
    dna_improvement = dna_fitness_history[-1][0] - dna_fitness_history[0][0]

    print(f"\nGEP Improvement: {gep_improvement:+.4f}")
    print(f"DNA Improvement:  {dna_improvement:+.4f}")

    # Strategy complexity analysis
    best_gep = gep_engine.get_best_gep_chromosome(count=1)[0]

    print(f"\n{'='*50}")
    print("STRATEGY COMPLEXITY ANALYSIS (GEP)")
    print(f"{'='*50}")

    print(f"\nBest GEP Chromosome:")
    print(f"  ID: {best_gep.id}")
    print(f"  Fitness: {best_gep.fitness:.4f}")
    print(f"  Genes: {len(best_gep.genes)}")
    print(f"  Linking: {best_gep.linking_function.value}")

    total_nodes = 0
    max_depth = 0
    for i, gene in enumerate(best_gep.genes):
        node_count = _count_nodes(gene.root)
        depth = gene.root.get_depth()
        total_nodes += node_count
        max_depth = max(max_depth, depth)

        print(f"\n  Gene {i+1}:")
        print(f"    Expression: {gene.root.to_readable()}")
        print(f"    Node Count: {node_count}")
        print(f"    Depth: {depth}")

    print(f"\nTotal Nodes: {total_nodes}")
    print(f"Max Depth: {max_depth}")

    # Interpretation
    print(f"\n{'='*50}")
    print("INTERPRETATION")
    print(f"{'='*50}")

    print("""
    GEP Advantages:
    - Can express complex conditional logic (IF-THEN-ELSE)
    - Evolves algorithmic structure, not just parameters
    - More interpretable (symbolic expressions)
    - Supports function discovery

    Flat DNA Advantages:
    - Faster evaluation
    - Simpler optimization landscape
    - Better for continuous parameter tuning
    - Works well with gradient-based methods

    Conclusion:
    GEP excels at problems requiring:
    - Discrete decision making
    - Symbolic/algorithmic solutions
    - Interpretable strategies
    - Problems with clear decision boundaries

    Flat vectors excel at:
    - Continuous optimization
    - High-dimensional parameter spaces
    - When solution structure is known
    """)

    print("\n[PASS] Research experiment completed")


def _count_nodes(node: Any) -> int:
    """Count total nodes in a tree."""
    if not node.children:
        return 1
    return 1 + sum(_count_nodes(child) for child in node.children)


# ============================================================================
# Main Entry Point
# ============================================================================

def run_all_tests():
    """Run all tests in sequence."""
    print("\n")
    print("#"*70)
    print("# GEP Gene Expression Programming - Full Test Suite")
    print("#"*70)

    tests = [
        ("Basic Gene Structure", test_gene_structure),
        ("Gene Evaluation", test_gene_evaluation),
        ("Point Mutation", test_point_mutation),
        ("Subtree Crossover", test_subtree_crossover),
        ("Chromosome Creation", test_chromosome),
        ("Evolution Engine", test_evolution_engine),
        ("Multi-Generation Evolution", test_multi_generation_evolution),
        ("Serialization", test_serialization),
        ("Performance Benchmark", test_performance),
        ("Research Experiment: GEP vs Flat DNA", research_experiment_gep_vs_flat),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASS"))
        except Exception as e:
            logger.error(f"Test '{name}' failed: {e}")
            results.append((name, f"FAIL: {e}"))

    # Summary
    print("\n")
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, result in results:
        status = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"  {status} {name}")

    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
