# -*- coding: utf-8 -*-
"""
SwarmEvolutionCoordinator - 蜂群进化协调器

Phase 3: 群体进化
让进化真正发生——不是复制，是有方向的进化。

核心职责:
1. 适应度追踪 (Fitness Tracking): 实时收集每只蜂的任务表现
2. 选择压力 (Selection Pressure): 适应度低于阈值的蜂自动淘汰
3. GEP 进化协调: 连接 DarwinBee + EvolutionEngine + SwarmCommunication
4. 精英保留: 最优染色体不参与变异，直接复制

闭环流程:
  TaskResult → FitnessTracker.update() → SelectionPressure.cull()
    → SwarmEvolutionCoordinator.evolve() → DarwinBee.generate()
    → Incubator.hatch() → promoted to hive/bees/mutants/
    → TaskResult (回到起点)

使用方式:
    coordinator = SwarmEvolutionCoordinator()
    coordinator.start()
"""

import logging
import threading
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger("Hive.SwarmEvolution")


# ===== 适应度记录 =====

@dataclass
class BeeFitnessRecord:
    """单只蜂的适应度记录"""
    bee_id: str
    bee_type: str
    birth_time: float = field(default_factory=time.time)
    task_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_duration: float = 0.0
    fitness: float = 0.0  # 0-1 适应度分数
    generation: int = 0  # 代数（用于追踪进化深度）
    chromosome_id: Optional[str] = None  # 关联的 GEP 染色体 ID
    last_update: float = field(default_factory=time.time)
    status: str = "active"  # active, promoted, culled

    def success_rate(self) -> float:
        return self.success_count / max(self.task_count, 1)

    def avg_duration(self) -> float:
        return self.total_duration / max(self.task_count, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bee_id": self.bee_id,
            "bee_type": self.bee_type,
            "status": self.status,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate(),
            "avg_duration": self.avg_duration(),
            "fitness": self.fitness,
            "generation": self.generation,
            "chromosome_id": self.chromosome_id,
            "birth_time": self.birth_time,
            "last_update": self.last_update,
            "age_seconds": time.time() - self.birth_time,
        }


class FitnessTracker:
    """
    适应度追踪器

    实时收集所有 WorkerBee 的任务结果，
    计算适应度分数并更新记录。
    """

    # 适应度权重
    WEIGHT_SUCCESS_RATE = 0.5    # 成功率权重
    WEIGHT_EFFICIENCY = 0.3      # 效率（速度）权重
    WEIGHT_STABILITY = 0.2        # 稳定性（低错误率）权重

    def __init__(self):
        self._records: Dict[str, BeeFitnessRecord] = {}
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []
        self._max_history = 10000

    def register_bee(self, bee_id: str, bee_type: str, generation: int = 0, chromosome_id: Optional[str] = None):
        """注册新蜂到追踪器"""
        with self._lock:
            if bee_id not in self._records:
                self._records[bee_id] = BeeFitnessRecord(
                    bee_id=bee_id,
                    bee_type=bee_type,
                    generation=generation,
                    chromosome_id=chromosome_id,
                )
                logger.debug(f"📊 [FitnessTracker] 注册蜂: {bee_id} ({bee_type})")

    def update(self, bee_id: str, success: bool, duration: float, error: Optional[str] = None):
        """
        更新蜂的适应度数据

        Args:
            bee_id: 蜂 ID
            success: 任务是否成功
            duration: 任务耗时（秒）
            error: 错误信息（如果有）
        """
        with self._lock:
            if bee_id not in self._records:
                self._records[bee_id] = BeeFitnessRecord(
                    bee_id=bee_id,
                    bee_type="unknown",
                )

            record = self._records[bee_id]
            record.task_count += 1
            record.total_duration += duration
            record.last_update = time.time()

            if success:
                record.success_count += 1
            else:
                record.error_count += 1

            # 计算适应度
            record.fitness = self._calculate_fitness(record)

    def _calculate_fitness(self, record: BeeFitnessRecord) -> float:
        """
        计算适应度分数（0-1）

        公式:
        fitness = w1 * success_rate + w2 * efficiency_score + w3 * stability_score
        """
        # 1. 成功率 (0-1)
        success_rate = record.success_rate()

        # 2. 效率分数：假设 10s 以内为满分，每增加 1s 减 0.05 分
        avg_dur = record.avg_duration()
        efficiency_score = max(0.0, 1.0 - (avg_dur / 200.0))  # 200s 以上效率分为 0

        # 3. 稳定性分数（无错误率）
        stability_score = 1.0 - min(1.0, record.error_count / max(record.task_count, 1))

        fitness = (
            self.WEIGHT_SUCCESS_RATE * success_rate +
            self.WEIGHT_EFFICIENCY * efficiency_score +
            self.WEIGHT_STABILITY * stability_score
        )
        return min(1.0, max(0.0, fitness))

    def get_record(self, bee_id: str) -> Optional[BeeFitnessRecord]:
        return self._records.get(bee_id)

    def get_all_records(self) -> List[BeeFitnessRecord]:
        return list(self._records.values())

    def get_top_bees(self, count: int = 5, active_only: bool = True) -> List[BeeFitnessRecord]:
        """获取适应度最高的蜂"""
        records = self.get_all_records()
        if active_only:
            records = [r for r in records if r.status == "active"]
        records.sort(key=lambda r: r.fitness, reverse=True)
        return records[:count]

    def get_fittest_chromosome_id(self) -> Optional[str]:
        """获取最优适应度蜂关联的染色体 ID"""
        top = self.get_top_bees(count=1)
        if top:
            return top[0].chromosome_id
        return None

    def mark_culled(self, bee_id: str):
        """标记蜂已被淘汰"""
        if bee_id in self._records:
            self._records[bee_id].status = "culled"
            logger.info(f"🔪 [FitnessTracker] 蜂 {bee_id} 已被淘汰")

    def mark_promoted(self, bee_id: str):
        """标记蜂已晋升"""
        if bee_id in self._records:
            self._records[bee_id].status = "promoted"
            logger.info(f"🎖️ [FitnessTracker] 蜂 {bee_id} 已晋升为成虫")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        records = self.get_all_records()
        active = [r for r in records if r.status == "active"]
        return {
            "total_bees": len(records),
            "active_bees": len(active),
            "avg_fitness": sum(r.fitness for r in active) / max(len(active), 1),
            "best_fitness": max((r.fitness for r in active), default=0.0),
            "culled_count": sum(1 for r in records if r.status == "culled"),
            "promoted_count": sum(1 for r in records if r.status == "promoted"),
        }


# ===== 选择压力 =====

@dataclass
class SelectionPressure:
    """
    自然选择压力

    控制何时淘汰低适应度蜂，以及精英保留策略。
    """
    min_fitness_threshold: float = 0.3    # 最低适应度阈值，低于此值直接淘汰
    cull_interval: float = 300.0            # 淘汰检查间隔（秒）
    min_tasks_before_cull: int = 3           # 至少执行多少任务后才开始考虑淘汰
    max_age_seconds: float = 3600.0          # 最大存活时间（秒），超时强制淘汰

    # 精英保留
    elite_count: int = 3                    # 保留最优 N 只蜂不参与淘汰
    elite_min_fitness: float = 0.7          # 精英最低适应度

    def should_cull(self, record: BeeFitnessRecord, is_elite: bool = False) -> bool:
        """
        判断蜂是否应该被淘汰

        Returns:
            True 如果应该淘汰
        """
        if is_elite and record.fitness >= self.elite_min_fitness:
            return False

        # 任务数不足，不淘汰
        if record.task_count < self.min_tasks_before_cull:
            return False

        # 适应度过低，直接淘汰
        if record.fitness < self.min_fitness_threshold:
            return True

        # 超时淘汰
        age = time.time() - record.birth_time
        if age > self.max_age_seconds:
            return True

        return False

    def get_cull_candidates(self, tracker: FitnessTracker) -> List[BeeFitnessRecord]:
        """获取应该淘汰的蜂列表"""
        records = tracker.get_all_records()
        active = [r for r in records if r.status == "active"]

        # 按适应度排序
        active.sort(key=lambda r: r.fitness)

        # 精英不参与淘汰
        elite = tracker.get_top_bees(count=self.elite_count, active_only=True)
        elite_ids = {r.bee_id for r in elite}

        cull_list = []
        for record in active:
            if record.bee_id in elite_ids:
                continue
            if self.should_cull(record):
                cull_list.append(record)

        return cull_list


# ===== 蜂群进化协调器 =====

class SwarmEvolutionCoordinator:
    """
    蜂群进化协调器

    Phase 3 核心组件。协调以下组件形成闭环进化系统:
    - FitnessTracker: 适应度追踪
    - SelectionPressure: 自然选择压力
    - EvolutionEngine: GEP 进化引擎
    - SwarmCommunication: 蜂群通信（接收 DarwinBee 的进化结果）

    闭环流程:
    1. FitnessTracker 收集任务结果 → 计算适应度
    2. SelectionPressure 识别低适应度蜂 → 触发淘汰
    3. Coordinator 根据适应度 → 选择父代染色体
    4. DarwinBee 执行 GEP 进化 → 生成新幼虫
    5. Incubator 孵化试炼 → 晋升或淘汰
    """

    def __init__(self):
        self._tracker = FitnessTracker()
        self._pressure = SelectionPressure()
        self._evolution_engine = None  # 懒加载
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # SwarmCommunication 集成
        self._swarm = None

        # 进化历史
        self._evolution_log: List[Dict[str, Any]] = []
        self._evolution_count = 0

        # GEP 配置
        self._gep_population_size = 20
        self._gep_elite_count = 3

        logger.info("🧬 [SwarmEvolution] 蜂群进化协调器已初始化")

    def _get_evolution_engine(self):
        """懒加载 EvolutionEngine"""
        if self._evolution_engine is None:
            from hive.core.evolution_engine import EvolutionEngine
            self._evolution_engine = EvolutionEngine(
                use_gep=True,
                gep_population_size=self._gep_population_size,
                gep_max_genes=5,
                gep_max_depth=10,
            )
            logger.info("🧬 [SwarmEvolution] GEP 进化引擎已启动")
        return self._evolution_engine

    def start(self):
        """启动协调器（启动后台选择压力线程）"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._pressure_loop, daemon=True, name="swarm_evolution")
        self._thread.start()

        # 注册到蜂群通信
        self._register_to_swarm()

        logger.info("🧬 [SwarmEvolution] 协调器已启动，选择压力线程运行中")

    def stop(self):
        """停止协调器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("🧬 [SwarmEvolution] 协调器已停止")

    def _register_to_swarm(self):
        """注册到蜂群通信系统"""
        try:
            from hive.core.swarm_communication import get_swarm_communication, MessageType, SwarmMessage
            self._swarm = get_swarm_communication()
            self._swarm.register_bee("SwarmEvolutionCoordinator", self)
            self._swarm.subscribe(MessageType.KNOWLEDGE, self._on_swarm_message)
            logger.info("🧬 [SwarmEvolution] 已注册到蜂群通信网络")
        except ImportError:
            logger.warning("🧬 [SwarmEvolution] SwarmCommunication 未安装，跳过注册")

    def _on_swarm_message(self, message):
        """处理蜂群消息（接收 DarwinBee 的进化结果）"""
        content = message.content
        topic = content.get("topic", "")

        if topic == "bee_born":
            # 新蜂诞生
            self._tracker.register_bee(
                bee_id=content.get("bee_id"),
                bee_type=content.get("bee_type", "unknown"),
                generation=content.get("generation", 0),
                chromosome_id=content.get("chromosome_id"),
            )
        elif topic == "task_result":
            # 任务结果更新适应度
            self._tracker.update(
                bee_id=content.get("bee_id"),
                success=content.get("success", False),
                duration=content.get("duration", 0.0),
                error=content.get("error"),
            )

    def _pressure_loop(self):
        """后台选择压力循环"""
        while self._running:
            try:
                time.sleep(self._pressure.cull_interval)
                self._run_selection_cycle()
            except Exception as e:
                logger.error(f"🧬 [SwarmEvolution] 选择压力循环错误: {e}")

    def _run_selection_cycle(self):
        """执行一轮选择压力检查"""
        candidates = self._pressure.get_cull_candidates(self._tracker)

        if not candidates:
            return

        logger.info(f"🔪 [SwarmEvolution] 选择压力: {len(candidates)} 只蜂将被淘汰")
        for record in candidates:
            self._tracker.mark_culled(record.bee_id)
            self._emit_cull_alert(record)

    def _emit_cull_alert(self, record: BeeFitnessRecord):
        """发出淘汰警告（通过蜂群网络广播）"""
        if self._swarm:
            try:
                from hive.core.swarm_communication import SwarmMessage, MessageType
                msg = SwarmMessage(
                    id=f"cull_{record.bee_id}_{int(time.time())}",
                    type=MessageType.ALERT,
                    from_bee="SwarmEvolutionCoordinator",
                    content={
                        "topic": "bee_culled",
                        "bee_id": record.bee_id,
                        "bee_type": record.bee_type,
                        "fitness": record.fitness,
                        "reason": "fitness_too_low" if record.fitness < self._pressure.min_fitness_threshold else "age_timeout",
                    },
                    priority=9,
                )
                self._swarm.broadcast(msg)
            except Exception as e:
                logger.warning(f"🧬 [SwarmEvolution] 广播淘汰警告失败: {e}")

    # ===== 对外接口 =====

    def report_task_result(self, bee_id: str, success: bool, duration: float,
                          error: Optional[str] = None):
        """
        上报任务结果（WorkerBee 完成任务后调用此方法）

        Args:
            bee_id: 蜂 ID
            success: 是否成功
            duration: 耗时（秒）
            error: 错误信息
        """
        self._tracker.update(bee_id, success, duration, error)

    def register_bee(self, bee_id: str, bee_type: str, generation: int = 0,
                    chromosome_id: Optional[str] = None):
        """注册新蜂"""
        self._tracker.register_bee(bee_id, bee_type, generation, chromosome_id)

    def evolve_next_generation(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        触发下一代进化

        使用当前最优染色体的适应度作为父代，
        通过 GEP 进化产生新染色体。

        Returns:
            包含进化结果的字典
        """
        engine = self._get_evolution_engine()
        self._evolution_count += 1

        # 获取最优染色体作为父代
        best_chrom = None
        best_record = self._tracker.get_top_bees(count=1)
        if best_record and best_record[0].chromosome_id:
            # 找到对应染色体（需要从 GEP 种群中查找）
            status = engine.get_gep_status()
            if status.get("gep_population_size", 0) > 0:
                best_chromosomes = engine.get_best_gep_chromosome(count=1)
                if best_chromosomes:
                    best_chrom = best_chromosomes[0]

        # 执行 GEP 进化
        new_chromosomes = engine.evolve_gep_generation(
            contexts=[context] if context else None,
            elite_count=self._gep_elite_count,
        )

        result = {
            "evolution_id": self._evolution_count,
            "generation": engine._gep_generation,
            "population_size": len(new_chromosomes),
            "best_fitness": new_chromosomes[0].fitness if new_chromosomes else 0.0,
            "parent_chromosome": best_chrom.id if best_chrom else None,
            "timestamp": datetime.now().isoformat(),
        }

        self._evolution_log.append(result)
        logger.info(f"🧬 [SwarmEvolution] 进化 #{self._evolution_count}: best_fitness={result['best_fitness']:.4f}")

        return result

    def get_status(self) -> Dict[str, Any]:
        """获取协调器状态"""
        return {
            "running": self._running,
            "evolution_count": self._evolution_count,
            "fitness_stats": self._tracker.get_stats(),
            "evolution_log_size": len(self._evolution_log),
            "gep_status": self._get_evolution_engine().get_gep_status() if self._evolution_engine else {},
        }


# 全局单例
_coordinator: Optional[SwarmEvolutionCoordinator] = None
_coordinator_lock = threading.Lock()


def get_swarm_evolution_coordinator() -> SwarmEvolutionCoordinator:
    """获取蜂群进化协调器单例"""
    global _coordinator
    if _coordinator is None:
        with _coordinator_lock:
            if _coordinator is None:
                _coordinator = SwarmEvolutionCoordinator()
    return _coordinator
