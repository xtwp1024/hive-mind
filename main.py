# -*- coding: utf-8 -*-
"""
蜂巢系统主入口 - 蜂群本质版

回归蜂群核心：
- QueenBee: 蜂后，统筹调度
- Worker Bees: 35种工蜂，协同工作
- DNA System: 遗传进化
- 简洁的主循环
"""

import asyncio
import os
import sys
import multiprocessing
from pathlib import Path
from typing import Any, List

# 强制 UTF-8 环境
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from hive.config.settings import settings
from hive.utils.logger import get_logger, setup_logging
from hive.utils.monitor import get_monitor
from hive.utils.metrics import get_metrics
from hive.utils.metrics_server import start_metrics_server

# 导入核心组件
from hive.core.queen import QueenBee
from hive.ministers.engineer import get_engineer
from hive.core.dna import BeeDNA
from hive.core.warmup import get_warmup
from hive.utils.event_stream import emit, init_events_file

logger = get_logger(__name__)


async def initialize_system():
    """初始化系统组件"""
    logger.info("=" * 60)
    logger.info("Initializing Hive Mind Swarm System")
    logger.info("=" * 60)

    # 配置日志
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_file=None if settings.is_development() else Path(settings.LOG_DIR) / "hive.log",
        json_output=settings.LOG_FORMAT == "json",
        enable_colors=settings.is_development(),
        shared_brain_enabled=settings.SHARED_BRAIN_ENABLED,
        shared_brain_dir=settings.SHARED_BRAIN_EVENTS_DIR,
        shared_brain_source="hive",
    )

    # 获取全局实例
    monitor = get_monitor()
    metrics = get_metrics()

    # 启动 Prometheus 指标端点
    start_metrics_server(port=9090)

    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Max workers: {settings.MAX_WORKERS}")
    logger.info(f"Mutation rate: {settings.MUTATION_RATE}")

    # 获取系统信息
    sys_info = monitor.get_system_info()
    logger.info(f"CPU: {sys_info.get('cpu_count', 'unknown')} cores")
    logger.info(f"Memory: {sys_info.get('memory_total_gb', 'unknown')} GB")

    return monitor, metrics


def _handle_warmup_phase(cycle_count: int, engineer: Any) -> List[BeeDNA]:
    """处理预热阶段任务分配

    策略：
    - Cycle 1：用 Warmup 预热 40 种蜂种（保证所有蜂种登场）
    - Cycle 2+：先用 engineer.assign_tasks 找真实任务；如果没活，
                 回退到 warmup 模式生成随机 idle 任务，保证蜂群持续运转
    """
    if cycle_count == 1:
        logger.info("Warming up - Ensuring all bee species are present...")
        try:
            warmup = get_warmup()
            dna_batch = warmup.create_active_swarm(capacity=30)
            species_status = warmup.get_species_status()
            logger.info(f"Warmup complete: {species_status['total_species']} species")
            return dna_batch
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    # Cycle 2+：先试真实任务，没活就回退到 warmup idle
    logger.info("Assigning tasks...")
    try:
        dna_batch = engineer.assign_tasks(capacity=30)
        if dna_batch:
            logger.info(f"Assigned {len(dna_batch)} real tasks")
            return dna_batch
        logger.info("Engineer 暂无真实任务，回退 warmup idle 模式")
    except Exception as e:
        logger.error(f"Failed to assign tasks: {e}")

    # 回退 warmup
    try:
        warmup = get_warmup()
        dna_batch = warmup.create_active_swarm(capacity=30)
        logger.info(f"Warmup fallback: {len(dna_batch)} idle tasks")
        return dna_batch
    except Exception as e:
        logger.error(f"Warmup fallback failed: {e}")
        return []


async def _handle_deployment_phase(queen: QueenBee, dna_batch: List[BeeDNA]) -> None:
    """处理工蜂部署与结果回收"""
    if not dna_batch:
        logger.info("No tasks to deploy")
        return

    logger.info(f"Deploying {len(dna_batch)} bees...")

    # 优先尝试 V2 异步调度器（需要预注册 Actor）
    results = await queen.deploy_swarm_v2(dna_batch)

    if results:
        # V2 成功：直接走 V2 进化路径
        logger.info(f"[V2 Dispatcher] Processed {len(results)} tasks")
        offspring_dna = queen.process_results_v2(results)
        if offspring_dna:
            logger.info(f"Executing {len(offspring_dna)} offspring tasks...")
            queen.process_results_v2(await queen.deploy_swarm_v2(offspring_dna))
        return

    # V2 无 Actor 或空转 → 回退 V1 WorkerManager 多进程路径
    logger.info("[V2 Dispatcher] No registered actors, falling back to V1 WorkerManager...")
    queen.deploy_swarm(dna_batch)

    # 等待工蜂进程完成
    await _wait_for_v1_workers(queen)

    # 回收队列结果 + 进化繁育
    logger.info("Evolution & Digestion...")
    queen.consume_honey()


async def _wait_for_v1_workers(queen: QueenBee, poll_interval: float = 0.3, max_wait: float = 45) -> None:
    """等待 V1 WorkerManager 中所有活跃工蜂完成

    配合 WorkerManager.cleanup_dead_workers 的僵死检测：
    - 正常 worker 秒返回 → 下一轮 poll 就被清理
    - 僵死 worker 超过 ZOMBIE_THRESHOLD (20s) → 被 force_timeout 强杀
    - max_wait 只是最终兜底（需要大于 force_timeout 才有意义）
    """
    from hive.core.service_locator import BeeServiceLocator
    worker_manager = BeeServiceLocator.get_service("WorkerManager")
    if not worker_manager:
        await asyncio.sleep(2)
        return

    ZOMBIE_THRESHOLD = 20.0  # 工蜂存活超过 20s 就判定为僵死

    elapsed = 0.0
    while elapsed < max_wait:
        removed = worker_manager.cleanup_dead_workers(force_timeout=ZOMBIE_THRESHOLD)
        active = worker_manager.active_workers
        if not active:
            if removed > 0:
                logger.info(f"[_wait] 本轮清理 {removed} 只 Worker，蜂巢空了 ✓ (花了 {elapsed:.1f}s)")
            break
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    # 最终清场
    worker_manager.cleanup_dead_workers(force_timeout=5.0)  # 最后给 5s 阈值收漏网之鱼
    active_count = len(worker_manager.active_workers)
    if active_count > 0:
        logger.warning(f"[{max_wait}s 兜底超时] 仍有 {active_count} 只工蜂活跃，强制 shutdown")
        worker_manager.shutdown(timeout=5)


def _update_metrics_and_health(queen: QueenBee, monitor: Any, metrics: Any) -> None:
    """更新监控指标与健康检查"""
    status = queen.get_colony_status()
    logger.info(f"Honey collected: {status.get('honey_collected', 0)}")

    metrics.increment("cycles_processed")
    metrics.set_gauge("active_workers", status.get("active_workers", 0))
    metrics.set_gauge("honey_collected", status.get("honey_collected", 0))

    is_healthy, snapshot, issues = monitor.check_health()
    if not is_healthy:
        logger.warning(f"Health issues detected: {issues}")


async def _execute_single_cycle(cycle_count: int, max_cycles: int, queen: QueenBee, engineer: Any, monitor: Any, metrics: Any) -> None:
    """执行单个周期"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Cycle {cycle_count}/{max_cycles}")
    logger.info("=" * 60)

    # ── 每轮 cycle 开始前，先清理 WorkerManager 残留 ──
    # （Evolution 阶段 Queen 可能 spawn 子代 worker 但没人等它们，残留占满 MAX_WORKERS 容量）
    from hive.core.service_locator import BeeServiceLocator
    wm = BeeServiceLocator.get_service("WorkerManager")
    if wm and wm.active_workers:
        pre_clean = len(wm.active_workers)
        wm.cleanup_dead_workers(force_timeout=3.0)
        if wm.active_workers:
            # 还活的一律清场（这是新 cycle，旧 worker 不应留）
            logger.info(f"[Cycle 预热] 强制清场 {len(wm.active_workers)} 只残留 worker")
            wm.shutdown(timeout=2)
        if pre_clean:
            logger.info(f"[Cycle {cycle_count}] WorkerManager 清场: {pre_clean} → {len(wm.active_workers)}")

    emit("cycle_start", cycle=cycle_count, total=max_cycles)

    # 1. 任务分配
    dna_batch = _handle_warmup_phase(cycle_count, engineer)

    # 2. 部署蜂群
    emit("bees_deploying", count=len(dna_batch), cycle=cycle_count)
    await _handle_deployment_phase(queen, dna_batch)

    # 3. 更新指标
    _update_metrics_and_health(queen, monitor, metrics)

    # 4. 事件收尾
    status = queen.get_colony_status()
    emit("cycle_end", cycle=cycle_count, honey=status.get("honey_collected", 0), active=status.get("active_workers", 0))

    # 5. 等待下一轮
    await asyncio.sleep(1)


def _cleanup_hive(queen: QueenBee) -> None:
    """清理蜂巢"""
    logger.info("Cleaning up Hive...")
    try:
        queen.shutdown(wait=True, timeout=5)
    except Exception as e:
        logger.warning(f"关闭蜂群时出现异常: {e}")
    logger.info("Hive cleanup complete")


def _log_final_metrics(metrics: Any) -> None:
    """记录最终指标"""
    logger.info("\nFinal Metrics Summary:")
    metrics.log_summary()
    all_metrics = metrics.get_all_metrics()
    for key, value in all_metrics.items():
        logger.info(f"  {key}: {value}")


async def run_hive():
    """运行蜂巢主循环"""
    logger.info("\nStarting Hive Mind Swarm")
    
    # 初始化事件流（前端动画数据源）
    init_events_file()
    emit("system_init", message="Hive Mind Swarm online")
    
    metrics = get_metrics()
    monitor = get_monitor()
    engineer = get_engineer()

    # 初始化蜂后
    result_queue = multiprocessing.Queue()
    queen = QueenBee(result_queue)
    logger.info("QueenBee initialized")
    emit("queen_born")

    # 主循环
    # HIVE_SMOKE_CYCLES: 有界运行时验证用（tools/smoke_runtime.py），
    # 未设置时保持默认 10 周期，不改变正常启动行为。
    try:
        max_cycles = max(1, int(os.environ.get("HIVE_SMOKE_CYCLES", "10")))
    except ValueError:
        logger.warning("HIVE_SMOKE_CYCLES 非整数，回退默认 10")
        max_cycles = 10

    try:
        for i in range(max_cycles):
            await _execute_single_cycle(i + 1, max_cycles, queen, engineer, monitor, metrics)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        _cleanup_hive(queen)
        emit("system_shutdown")

    _log_final_metrics(metrics)


async def main():
    """主函数"""
    try:
        # 初始化系统
        monitor, metrics = await initialize_system()

        # 运行蜂巢
        await run_hive()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
