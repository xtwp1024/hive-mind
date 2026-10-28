# -*- coding: utf-8 -*-
"""
Benchmark Script: AsyncBaseBee vs BaseBee

Compares throughput, latency, memory usage, and task completion rate
between the asyncio-based AsyncBaseBee and multiprocessing-based BaseBee.

Usage:
    python benchmark_async.py [--tasks N] [--workers N] [--mode {sync,async,both}]

Example:
    python benchmark_async.py --tasks 100 --workers 4 --mode both
"""

import argparse
import asyncio
import multiprocessing
import os
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any

# Add Hive to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hive.core.dna import BeeDNA
from hive.core.interfaces import Task
from hive.bees.base_bee import BaseBee
from hive.bees.async_base_bee import AsyncBaseBee
from hive.utils.logger import get_logger

logger = get_logger("Hive.Benchmark")


@dataclass
class BenchmarkResult:
    """Benchmark results container"""
    mode: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_time: float
    throughput: float  # tasks/sec
    avg_latency: float  # sec/task
    peak_memory_mb: float
    cpu_percent: float
    task_results: List[Dict[str, Any]]


def create_test_tasks(count: int, task_type: str = "mixed") -> List[Task]:
    """Create test tasks for benchmarking"""
    tasks = []
    for i in range(count):
        task = Task(
            id=f"bench_task_{i}",
            type=task_type,
            priority=0,
            payload={
                "iterations": 2000,  # CPU-bound work iterations
                "data": b"x" * 1000,  # Data for hashing
            },
            timeout=60.0
        )
        tasks.append(task)
    return tasks


def run_basebee_benchmark(
    tasks: List[Task],
    num_workers: int
) -> BenchmarkResult:
    """
    Run benchmark with BaseBee (multiprocessing.Process).

    Each BaseBee runs in its own process.
    """
    start_time = time.time()
    tracemalloc.start()

    # Create result queue
    result_queue = multiprocessing.Queue()

    # Create bees
    bees = []
    for i, task in enumerate(tasks[:num_workers]):
        dna = BeeDNA(
            id=f"basebee_{i}",
            species="BenchmarkBee",
            role="standard",
            target_name=f"bench_target_{i}"
        )
        bee = BaseBee(dna, f"BaseBee-{i}", result_queue)
        bees.append(bee)

    # Start all bees
    for bee in bees:
        bee.start()

    # Wait for completion with timeout
    completed = 0
    failed = 0
    task_results = []
    timeout = 120  # seconds

    while completed + failed < len(tasks) and (time.time() - start_time) < timeout:
        try:
            result = result_queue.get(timeout=1)
            task_results.append(result)
            if result.get("status") == "SUCCESS":
                completed += 1
            else:
                failed += 1
        except:
            pass

    # Join all bees
    for bee in bees:
        bee.join(timeout=5)
        if bee.is_alive():
            bee.terminate()

    total_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Drain remaining results
    while True:
        try:
            result = result_queue.get_nowait()
            task_results.append(result)
        except:
            break

    throughput = completed / total_time if total_time > 0 else 0
    avg_latency = total_time / completed if completed > 0 else 0

    return BenchmarkResult(
        mode="BaseBee (multiprocessing)",
        total_tasks=len(tasks),
        completed_tasks=completed,
        failed_tasks=failed,
        total_time=total_time,
        throughput=throughput,
        avg_latency=avg_latency,
        peak_memory_mb=peak / (1024 * 1024),
        cpu_percent=0,  # Not easily measured on Windows
        task_results=task_results
    )


async def run_async_bee_task(bee: AsyncBaseBee, task: Task) -> Dict[str, Any]:
    """Execute a single task with AsyncBaseBee"""
    try:
        result = await bee.execute_task(task)
        return {
            "task_id": task.id,
            "success": result.success,
            "execution_time": result.execution_time,
            "error": result.error
        }
    except Exception as e:
        return {
            "task_id": task.id,
            "success": False,
            "error": str(e)
        }


async def run_asyncbee_benchmark(
    tasks: List[Task],
    num_workers: int
) -> BenchmarkResult:
    """
    Run benchmark with AsyncBaseBee (asyncio + ProcessPoolExecutor).

    Uses asyncio for task distribution and ProcessPoolExecutor for CPU work.
    """
    start_time = time.time()
    tracemalloc.start()

    result_queue = multiprocessing.Queue()

    # Create async bees
    bees = []
    for i in range(num_workers):
        dna = BeeDNA(
            id=f"asyncbee_{i}",
            species="BenchmarkBee",
            role="async",  # Important: triggers AsyncBaseBee routing
            target_name=f"bench_target_{i}"
        )
        bee = AsyncBaseBee(dna, f"AsyncBee-{i}", result_queue, max_workers=4)
        bees.append(bee)

    # Create async task distribution
    task_results = []

    async def distribute_tasks():
        """Distribute tasks across bees using round-robin"""
        bee_index = 0
        for task in tasks:
            bee = bees[bee_index % len(bees)]
            result = await run_async_bee_task(bee, task)
            task_results.append(result)
            bee_index += 1

    # Run all tasks
    try:
        await asyncio.wait_for(distribute_tasks(), timeout=120)
    except asyncio.TimeoutError:
        logger.warning("Async benchmark timed out")

    total_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    completed = sum(1 for r in task_results if r.get("success"))
    failed = len(task_results) - completed

    throughput = completed / total_time if total_time > 0 else 0
    avg_latency = sum(r.get("execution_time", 0) for r in task_results) / completed if completed > 0 else 0

    # Cleanup
    for bee in bees:
        bee.shutdown()

    return BenchmarkResult(
        mode="AsyncBaseBee (asyncio+ProcessPool)",
        total_tasks=len(tasks),
        completed_tasks=completed,
        failed_tasks=failed,
        total_time=total_time,
        throughput=throughput,
        avg_latency=avg_latency,
        peak_memory_mb=peak / (1024 * 1024),
        cpu_percent=0,
        task_results=task_results
    )


def run_sync_baseline_benchmark(
    tasks: List[Task],
    num_workers: int
) -> BenchmarkResult:
    """
    Run baseline benchmark with ThreadPoolExecutor.

    For comparison: threads vs processes vs asyncio
    """
    start_time = time.time()
    tracemalloc.start()

    def cpu_work(data: bytes, iterations: int) -> int:
        """CPU-bound work function"""
        result = 0
        for i in range(iterations):
            result += hash(data + str(i).encode())
        return result

    def task_worker(task: Task) -> Dict[str, Any]:
        """Process a single task"""
        task_start = time.time()
        try:
            data = task.payload.get("data", b"x" * 1000)
            iterations = task.payload.get("iterations", 2000)
            cpu_work(data, iterations)
            return {
                "task_id": task.id,
                "success": True,
                "execution_time": time.time() - task_start
            }
        except Exception as e:
            return {
                "task_id": task.id,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - task_start
            }

    # Use ThreadPoolExecutor for I/O-bound async comparison
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(task_worker, tasks))

    total_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    completed = sum(1 for r in results if r.get("success"))
    failed = len(results) - completed

    throughput = completed / total_time if total_time > 0 else 0
    avg_latency = sum(r.get("execution_time", 0) for r in results) / completed if completed > 0 else 0

    return BenchmarkResult(
        mode="ThreadPoolExecutor (baseline)",
        total_tasks=len(tasks),
        completed_tasks=completed,
        failed_tasks=failed,
        total_time=total_time,
        throughput=throughput,
        avg_latency=avg_latency,
        peak_memory_mb=peak / (1024 * 1024),
        cpu_percent=0,
        task_results=results
    )


def print_benchmark_report(results: List[BenchmarkResult]):
    """Print formatted benchmark report"""
    print("\n" + "=" * 80)
    print("ASYNC-HYBRID WORKER ARCHITECTURE BENCHMARK REPORT")
    print("=" * 80)
    print()

    for result in results:
        print(f"--- {result.mode} ---")
        print(f"  Total Tasks:        {result.total_tasks}")
        print(f"  Completed:           {result.completed_tasks}")
        print(f"  Failed:               {result.failed_tasks}")
        print(f"  Total Time:           {result.total_time:.2f}s")
        print(f"  Throughput:          {result.throughput:.2f} tasks/sec")
        print(f"  Avg Latency:          {result.avg_latency:.4f}s/task")
        print(f"  Peak Memory:          {result.peak_memory_mb:.2f} MB")
        print()

    if len(results) >= 2:
        print("-" * 80)
        print("COMPARISON")

        # Find the best throughput
        best = max(results, key=lambda r: r.throughput)
        baseline = results[0]

        print(f"  Best Mode:           {best.mode}")
        print(f"  Best Throughput:     {best.throughput:.2f} tasks/sec")
        print()

        for result in results:
            speedup = result.throughput / baseline.throughput if baseline.throughput > 0 else 0
            memory_ratio = result.peak_memory_mb / baseline.peak_memory_mb if baseline.peak_memory_mb > 0 else 0
            print(f"  {result.mode}:")
            print(f"    Speedup vs baseline: {speedup:.2f}x")
            print(f"    Memory ratio:        {memory_ratio:.2f}x")
            print()

        # Throughput improvement analysis
        if len(results) >= 2:
            async_result = next((r for r in results if "AsyncBaseBee" in r.mode), None)
            base_result = next((r for r in results if "BaseBee (multiprocessing)" in r.mode), None)

            if async_result and base_result:
                print("=" * 80)
                print("KEY FINDINGS: AsyncBaseBee vs BaseBee (multiprocessing)")
                print("=" * 80)

                throughput_ratio = async_result.throughput / base_result.throughput if base_result.throughput > 0 else 0
                memory_ratio = async_result.peak_memory_mb / base_result.peak_memory_mb if base_result.peak_memory_mb > 0 else 0

                print(f"  Throughput Ratio:    {throughput_ratio:.2f}x ({async_result.mode} vs {base_result.mode})")
                print(f"  Memory Ratio:        {memory_ratio:.2f}x")

                if throughput_ratio > 1:
                    print(f"\n  RESULT: AsyncBaseBee shows {throughput_ratio:.2f}x THROUGHPUT IMPROVEMENT")
                    print(f"  This aligns with the 3-5x improvement potential from async I/O overlap.")
                else:
                    print(f"\n  NOTE: BaseBee showed better throughput in this workload.")
                    print(f"  AsyncBaseBee advantages: I/O-bound tasks, lower memory footprint")
                    print(f"  BaseBee advantages: CPU-bound tasks, true process isolation")

                print()
                print("BOTTLENECK ANALYSIS:")
                print("  - AsyncBaseBee: Event loop overhead, GIL contention in ProcessPool")
                print("  - BaseBee: Process spawn overhead, IPC latency")
                print("  - Recommendation: Use AsyncBaseBee for I/O-bound, BaseBee for CPU-bound")
                print()

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="AsyncBaseBee vs BaseBee Benchmark")
    parser.add_argument("--tasks", type=int, default=50, help="Number of tasks to run")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument(
        "--mode",
        choices=["sync", "async", "both", "threads"],
        default="both",
        help="Benchmark mode: sync (BaseBee), async (AsyncBaseBee), both, or threads"
    )
    args = parser.parse_args()

    print(f"Starting benchmark with {args.tasks} tasks and {args.workers} workers...")
    print(f"Mode: {args.mode}")

    # Create test tasks
    tasks = create_test_tasks(args.tasks, task_type="cpu_bound")

    results = []

    # Run selected benchmarks
    if args.mode in ("sync", "both"):
        print("\nRunning BaseBee (multiprocessing) benchmark...")
        try:
            result = run_basebee_benchmark(tasks, args.workers)
            results.append(result)
            print(f"  Completed: {result.completed_tasks}/{result.total_tasks}")
            print(f"  Throughput: {result.throughput:.2f} tasks/sec")
        except Exception as e:
            print(f"  BaseBee benchmark failed: {e}")

    if args.mode in ("async", "both"):
        print("\nRunning AsyncBaseBee (asyncio+ProcessPool) benchmark...")
        try:
            result = asyncio.run(run_asyncbee_benchmark(tasks, args.workers))
            results.append(result)
            print(f"  Completed: {result.completed_tasks}/{result.total_tasks}")
            print(f"  Throughput: {result.throughput:.2f} tasks/sec")
        except Exception as e:
            print(f"  AsyncBaseBee benchmark failed: {e}")

    if args.mode == "threads":
        print("\nRunning ThreadPoolExecutor baseline benchmark...")
        try:
            result = run_sync_baseline_benchmark(tasks, args.workers)
            results.append(result)
            print(f"  Completed: {result.completed_tasks}/{result.total_tasks}")
            print(f"  Throughput: {result.throughput:.2f} tasks/sec")
        except Exception as e:
            print(f"  ThreadPool benchmark failed: {e}")

    # Print final report
    if results:
        print_benchmark_report(results)
    else:
        print("No benchmark results to report.")


if __name__ == "__main__":
    main()
