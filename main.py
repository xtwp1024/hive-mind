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
    """处理预热阶段任务分配"""
    if cycle_count != 1:
        logger.info("Assigning tasks...")
        try:
            dna_batch = engineer.assign_tasks(capacity=30)
            logger.info(f"Assigned {len(dna_batch)} tasks")
            return dna_batch
        except Exception as e:
            logger.error(f"Failed to assign tasks: {e}")
            return []

    # 周期 1：预热
    logger.info("Warming up - Ensuring all bee species are present...")
    try:
        warmup = get_warmup()
        dna_batch = warmup.create_active_swarm(capacity=30)
        species_status = warmup.get_species_status()
        logger.info(f"Warmup complete: {species_status['total_species']} species")
        return dna_batch
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        dna_batch = engineer.assign_tasks(capacity=30)
        return dna_batch


async def _handle_deployment_phase(queen: QueenBee, dna_batch: List[BeeDNA]) -> None:
    """处理工蜂部署与结果回收"""
    if not dna_batch:
        logger.info("No tasks to deploy")
        return

    logger.info(f"Deploying {len(dna_batch)} bees...")
    
    # V2 Execution
    results = await queen.deploy_swarm_v2(dna_batch)
    logger.info(f"Processed {len(results)} tasks")
    
    # 进化处理
    logger.info("Evolution & Digestion...")
    offspring_dna = queen.process_results_v2(results)
    
    # 执行子代任务
    if offspring_dna:
        logger.info(f"Executing {len(offspring_dna)} offspring tasks...")
        offspring_results = await queen.deploy_swarm_v2(offspring_dna)
        queen.process_results_v2(offspring_results)


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

    # 1. 任务分配
    dna_batch = _handle_warmup_phase(cycle_count, engineer)

    # 2. 部署蜂群
    await _handle_deployment_phase(queen, dna_batch)

    # 3. 更新指标
    _update_metrics_and_health(queen, monitor, metrics)

    # 4. 等待下一轮
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
    
    metrics = get_metrics()
    monitor = get_monitor()
    engineer = get_engineer()

    # 初始化蜂后
    result_queue = multiprocessing.Queue()
    queen = QueenBee(result_queue)
    logger.info("QueenBee initialized")

    # 主循环
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
