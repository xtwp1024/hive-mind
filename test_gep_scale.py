# -*- coding: utf-8 -*-
"""
GEP Scale Experiment

系统性测试 GEP 在不同规模参数下的性能表现：
- population_size: 10 / 50 / 100 / 200 / 500
- max_depth: 3 / 5 / 10 / 15 / 20
- max_genes: 1 / 3 / 5 / 10

测量指标：
- 每代耗时 (ms)
- 内存使用
- 树节点数量
- 适应度收敛速度

Run: python test_gep_scale.py
"""

import gc
import os
import sys
import time
import psutil
import random
from typing import Any

# Setup
sys.path.insert(0, r"D:\Hive")

from hive.core.gep_gene import GEPGeneFactory, GEPFitnessEvaluator, GEPOperators
from hive.core.evolution_engine import EvolutionEngine


def get_memory_mb() -> float:
    """获取当前进程内存使用 (MB)"""
    return psutil.Process().memory_info().rss / 1024 / 1024


def create_contexts(count: int = 10) -> tuple[list[dict], list[bool]]:
    """创建测试用上下文"""
    contexts = []
    expected = []
    for i in range(count):
        ctx = {
            "task_type": random.choice(["mining", "gathering", "architect", "sentinel"]),
            "success_rate": random.uniform(0.3, 0.95),
            "error_count": random.randint(0, 10),
            "avg_time": random.uniform(5.0, 60.0),
            "memory_usage": random.uniform(0.2, 0.9),
            "cpu_usage": random.uniform(0.1, 0.8),
        }
        contexts.append(ctx)
        expected.append(ctx["success_rate"] > 0.6)
    return contexts, expected


def run_scale_experiment(
    population_size: int,
    max_depth: int,
    max_genes: int,
    generations: int = 5,
    contexts: list[dict] | None = None,
    expected_results: list[bool] | None = None,
) -> dict[str, Any]:
    """
    运行单组规模实验

    Returns:
        {
            "population_size": ...,
            "max_depth": ...,
            "max_genes": ...,
            "generation_times": [...],  # 每代耗时 (ms)
            "memory_delta_mb": ...,    # 内存增长
            "best_fitness_history": [...],
            "avg_fitness_history": [...],
            "final_tree_nodes": ...,
        }
    """
    if contexts is None or expected_results is None:
        contexts, expected_results = create_contexts(10)

    engine = EvolutionEngine(
        use_gep=True,
        gep_population_size=population_size,
        gep_max_genes=max_genes,
        gep_max_depth=max_depth,
    )

    gc.collect()
    mem_before = get_memory_mb()

    gen_times = []
    best_fit_history = []
    avg_fit_history = []

    for gen in range(generations):
        gc.collect()
        t0 = time.perf_counter()

        population = engine.evolve_gep_generation(
            contexts, expected_results, elite_count=max(1, population_size // 10)
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        gen_times.append(elapsed_ms)

        if population:
            best_fit_history.append(population[0].fitness)
            avg_fit_history.append(sum(c.fitness for c in population) / len(population))

    gc.collect()
    mem_after = get_memory_mb()
    mem_delta = mem_after - mem_before

    # 统计最终种群树节点数
    total_nodes = 0
    if population:
        for chrom in population[:3]:  # 只统计前3个
            for gene in chrom.genes:
                total_nodes += _count_nodes(gene.root)

    return {
        "population_size": population_size,
        "max_depth": max_depth,
        "max_genes": max_genes,
        "generation_times": gen_times,
        "memory_delta_mb": mem_delta,
        "best_fitness_history": best_fit_history,
        "avg_fitness_history": avg_fit_history,
        "final_tree_nodes": total_nodes,
    }


def _count_nodes(node: Any) -> int:
    """统计树节点数"""
    if not hasattr(node, "children") or not node.children:
        return 1
    return 1 + sum(_count_nodes(child) for child in node.children)


def print_result(r: dict[str, Any]) -> None:
    """打印单组实验结果"""
    times = r["generation_times"]
    avg_time = sum(times) / len(times) if times else 0
    best = r["best_fitness_history"]
    avg = r["avg_fitness_history"]

    final_fit = best[-1] if best else 0.0
    print(
        f"  pop={r['population_size']:4d} | depth={r['max_depth']:2d} | "
        f"genes={r['max_genes']:2d} | "
        f"avg_time={avg_time:7.2f}ms | "
        f"final_fit={final_fit:.4f} | "
        f"nodes={r['final_tree_nodes']:5d} | "
        f"mem={r['memory_delta_mb']:+.1f}MB"
    )


def run_all_experiments() -> None:
    """运行全部实验"""
    print("#" * 90)
    print("# GEP Scale Experiment")
    print("#" * 90)

    # 准备共享上下文（所有实验用同一组，避免随机性干扰）
    print("\n[Setup] Generating shared contexts...")
    shared_contexts, shared_expected = create_contexts(20)
    print(f"[Setup] {len(shared_contexts)} contexts ready\n")

    # ---- 实验1: population_size 规模 ----
    print("=" * 90)
    print("EXPERIMENT 1: population_size scaling (max_depth=10, max_genes=5, 5 gens)")
    print("=" * 90)
    print(f"  {'Config':<45} | {'Avg Time/gen':>12} | {'Final Fit':>10} | {'Nodes':>6} | {'Mem':>7}")
    print("-" * 90)

    pop_sizes = [10, 50, 100, 200, 500]
    pop_results = []
    for ps in pop_sizes:
        r = run_scale_experiment(ps, 10, 5, 5, shared_contexts, shared_expected)
        pop_results.append(r)
        print_result(r)

    # ---- 实验2: max_depth 规模 ----
    print("\n" + "=" * 90)
    print("EXPERIMENT 2: max_depth scaling (population_size=100, max_genes=5, 5 gens)")
    print("=" * 90)
    print(f"  {'Config':<45} | {'Avg Time/gen':>12} | {'Final Fit':>10} | {'Nodes':>6} | {'Mem':>7}")
    print("-" * 90)

    depths = [3, 5, 10, 15, 20]
    depth_results = []
    for d in depths:
        r = run_scale_experiment(100, d, 5, 5, shared_contexts, shared_expected)
        depth_results.append(r)
        print_result(r)

    # ---- 实验3: max_genes 规模 ----
    print("\n" + "=" * 90)
    print("EXPERIMENT 3: max_genes scaling (population_size=100, max_depth=10, 5 gens)")
    print("=" * 90)
    print(f"  {'Config':<45} | {'Avg Time/gen':>12} | {'Final Fit':>10} | {'Nodes':>6} | {'Mem':>7}")
    print("-" * 90)

    gene_counts = [1, 3, 5, 10, 20]
    gene_results = []
    for g in gene_counts:
        r = run_scale_experiment(100, 10, g, 5, shared_contexts, shared_expected)
        gene_results.append(r)
        print_result(r)

    # ---- 总结 ----
    print("\n" + "#" * 90)
    print("# SUMMARY")
    print("#" * 90)

    print("\n1. population_size 影响:")
    for r in pop_results:
        times = r["generation_times"]
        avg_t = sum(times) / len(times)
        print(f"   {r['population_size']:4d} pop -> {avg_t:7.2f}ms/gen, {r['final_tree_nodes']} nodes")

    print("\n2. max_depth 影响:")
    for r in depth_results:
        times = r["generation_times"]
        avg_t = sum(times) / len(times)
        print(f"   depth={r['max_depth']:2d} -> {avg_t:7.2f}ms/gen, {r['final_tree_nodes']} nodes")

    print("\n3. max_genes 影响:")
    for r in gene_results:
        times = r["generation_times"]
        avg_t = sum(times) / len(times)
        print(f"   genes={r['max_genes']:2d} -> {avg_t:7.2f}ms/gen, {r['final_tree_nodes']} nodes")

    # 时间复杂度分析
    print("\n4. 时间复杂度估算:")
    if len(pop_results) >= 3:
        t10 = sum(pop_results[0]["generation_times"]) / len(pop_results[0]["generation_times"])
        t100 = sum(pop_results[2]["generation_times"]) / len(pop_results[2]["generation_times"])
        ratio = t100 / t10 if t10 > 0 else 0
        print(f"   pop 10->100 (10x): {ratio:.2f}x time (expect ~10x if O(n))")

    if len(gene_results) >= 3:
        t1 = sum(gene_results[0]["generation_times"]) / len(gene_results[0]["generation_times"])
        t10 = sum(gene_results[3]["generation_times"]) / len(gene_results[3]["generation_times"])
        ratio = t10 / t1 if t1 > 0 else 0
        print(f"   genes 1->10 (10x): {ratio:.2f}x time (expect ~10x if O(g))")


if __name__ == "__main__":
    run_all_experiments()
