"""
BrainQueenAdapter - 大脑-蜂后适配器
===================================

将 HiveBrain 深度集成到 QueenBee，使蜂群拥有统一的大脑皮层。

集成架构:
    HiveBrain (大脑皮层)
         ↓ 认知决策
    QueenBee (运动系统)
         ↓ 执行
    WorkerManager (肌肉)
         ↓ 操作
    Swarm (蜂群)

Usage:
    adapter = BrainQueenAdapter(queen_bee, hive_brain)
    result = await adapter.brainful_think("部署分析项目架构的蜂群")
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .hive_brain import BrainConfig, HiveBrain, ThinkResult
from .queen_adapter import QueenAdapter
from .sandbox import SandboxConfig, SandboxEnvironment, SimulationResult
from .strategy_selector import CognitiveMode, ExecutionMode, Strategy

# Import at file level for better performance
try:
    from hive.core.dna import BeeDNA
    from hive.core.service_locator import BeeServiceLocator
except ImportError:
    BeeServiceLocator = None
    BeeDNA = None

logger = logging.getLogger("Hive.Brain.BrainQueenAdapter")


@dataclass
class BrainfulResult:
    """
    带大脑的执行结果

    包含完整思考+执行闭环的结果
    """

    success: bool
    message: str
    think_result: Optional[ThinkResult] = None
    execution_results: List[Any] = field(default_factory=list)
    strategy_used: Optional[Strategy] = None
    execution_duration_ms: float = 0.0
    cognitive_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sandbox_result: Optional[SimulationResult] = None  # 沙箱模拟结果


@dataclass
class BrainfulSandboxConfig:
    """带沙箱的大脑配置"""

    sandbox_enabled: bool = True  # 是否启用沙箱
    allow_file_delete: bool = False
    allow_network: bool = False
    allow_execute: List[str] = field(default_factory=list)
    max_iterations: int = 100
    timeout_seconds: float = 30.0
    mock_fs: Optional[Dict[str, str]] = None  # 模拟文件系统


class BrainQueenAdapter:
    """
    大脑-蜂后适配器

    将 HiveBrain 作为蜂群的"大脑皮层"，负责:
    1. 接收高层指令
    2. 分解为目标
    3. 生成执行计划
    4. 协调 QueenBee 执行
    5. 反思学习

    消息流:
        User → HiveBrain.think() → 目标分解 → QueenAdapter.execute()
                                           ↓
                                     QueenBee 执行
                                           ↓
                                HiveBrain.store_experience() → 反思
    """

    def __init__(
        self,
        queen_bee,
        hive_brain: Optional[HiveBrain] = None,
        config: Optional[BrainConfig] = None,
        sandbox_config: Optional[BrainfulSandboxConfig] = None,
    ):
        """
        Args:
            queen_bee: QueenBee 实例
            hive_brain: HiveBrain 实例（如果为None则创建新的）
            config: 大脑配置
            sandbox_config: 沙箱配置（如果为None则使用默认配置）
        """
        self.queen = queen_bee
        self.hive_brain = hive_brain
        self.queen_adapter = None  # 延迟初始化

        self._config = config
        self._sandbox_config = sandbox_config
        self._initialized = False

        # 沙箱
        self._sandbox: Optional[SandboxEnvironment] = None

        # 统计
        self._stats = {
            "brainful_thinks": 0,
            "successful_thinks": 0,
            "failed_thinks": 0,
            "strategies_used": {},
            "sandbox_blocks": 0,
            "sandbox_simulations": 0,
        }

        # 执行历史
        self._execution_history: List[BrainfulResult] = []
        self._max_history = 100

        logger.info("🧠 BrainQueenAdapter created")

    async def initialize(self):
        """初始化适配器"""
        if self._initialized:
            return

        # 初始化 HiveBrain
        if self.hive_brain is None:
            self.hive_brain = HiveBrain(config=self._config or BrainConfig(name="QueenBrain"))
            await self.hive_brain.initialize()

        # 初始化 QueenAdapter
        self.queen_adapter = QueenAdapter(self.queen, self.hive_brain)

        # 初始化沙箱
        if self._sandbox_config and self._sandbox_config.sandbox_enabled:
            sandbox_cfg = SandboxConfig(
                allow_file_delete=self._sandbox_config.allow_file_delete,
                allow_network=self._sandbox_config.allow_network,
                allow_execute=self._sandbox_config.allow_execute,
                max_iterations=self._sandbox_config.max_iterations,
                timeout_seconds=self._sandbox_config.timeout_seconds,
            )
            self._sandbox = SandboxEnvironment(
                config=sandbox_cfg,
                mock_fs=self._sandbox_config.mock_fs,
            )
            logger.info("🧠 Sandbox enabled")

        self._initialized = True
        logger.info("🧠 BrainQueenAdapter initialized")

    async def brainful_think(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> BrainfulResult:
        """
        带大脑的思考和执行

        这是主要的入口方法，完成:
        1. 思考 (HiveBrain.think)
        2. 执行 (QueenAdapter.execute_plan)
        3. 学习 (store_experience + reflect)

        Args:
            instruction: 指令文本
            context: 可选的上下文

        Returns:
            BrainfulResult: 包含思考和执行结果的综合体
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        self._stats["brainful_thinks"] += 1

        result = BrainfulResult(
            success=False,
            message="",
        )

        try:
            # ===== 阶段1: 思考 =====
            think_result = await self.hive_brain.think(instruction)

            if not think_result.success:
                result.message = f"思考失败: {think_result.message}"
                return result

            result.think_result = think_result

            # 记录使用的策略
            if think_result.metadata.get("strategy"):
                strategy_mode = think_result.metadata["strategy"].get("mode", "unknown")
                self._stats["strategies_used"][strategy_mode] = (
                    self._stats["strategies_used"].get(strategy_mode, 0) + 1
                )

            # ===== 阶段2: 执行 =====
            execution_results = []
            if think_result.plan and think_result.sub_goals:
                execution_results = await self._execute_with_queen(
                    think_result,
                    context or {},
                )

                # 判断是否成功
                success = (
                    all(r.get("success", False) for r in execution_results if isinstance(r, dict))
                    if execution_results
                    else True
                )

                result.success = success
                result.message = "执行完成" if success else "部分执行失败"
            else:
                result.success = True
                result.message = "思考完成（无需执行）"

            result.execution_results = execution_results

            # ===== 阶段3: 学习 =====
            await self._learn_from_execution(result)

            self._stats["successful_thinks"] += 1

        except Exception as e:
            self._stats["failed_thinks"] += 1
            result.message = f"执行异常: {str(e)}"
            logger.error(f"❌ Brainful think failed: {e}", exc_info=True)

        # 记录执行历史
        result.execution_duration_ms = (time.time() - start_time) * 1000
        self._execution_history.append(result)
        if len(self._execution_history) > self._max_history:
            self._execution_history.pop(0)

        # 获取认知指标
        status = await self.hive_brain.get_status()
        result.cognitive_metrics = status.get("health_status", {})

        return result

    async def brainful_think_with_sandbox(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
        force_sandbox: bool = False,
    ) -> BrainfulResult:
        """
        带沙箱的思考和执行

        在执行前先通过沙箱模拟评估风险，高风险操作会被阻止。

        Args:
            instruction: 指令文本
            context: 可选的上下文
            force_sandbox: 是否强制使用沙箱（即使未启用）

        Returns:
            BrainfulResult: 包含沙箱模拟结果的综合体
        """
        if not self._initialized:
            await self.initialize()

        self._stats["sandbox_simulations"] += 1

        # 如果沙箱未初始化且未强制使用，直接走普通流程
        if self._sandbox is None and not force_sandbox:
            return await self.brainful_think(instruction, context)

        # 使用沙箱模拟
        sandbox = self._sandbox
        if sandbox is None:
            # 创建临时沙箱
            sandbox = SandboxEnvironment(
                config=SandboxConfig(),
                mock_fs=self._sandbox_config.mock_fs if self._sandbox_config else None,
            )

        sandbox_result = await sandbox.simulate_think(instruction, self.hive_brain)

        # 检查安全合规性
        if not sandbox_result.safety_compliance:
            self._stats["sandbox_blocks"] += 1
            logger.warning(
                f"🚫 Sandbox blocked: {instruction[:50]}... "
                f"risks={len(sandbox_result.risks_identified)}"
            )

            # 返回被阻止的结果
            blocked_result = BrainfulResult(
                success=False,
                message=f"沙箱阻止执行: {', '.join(sandbox_result.risks_identified[:3])}",
                sandbox_result=sandbox_result,
            )
            return blocked_result

        # 沙箱通过，继续执行
        result = await self.brainful_think(instruction, context)
        result.sandbox_result = sandbox_result
        return result

    async def _execute_with_queen(
        self,
        think_result: ThinkResult,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """使用 QueenAdapter 执行计划"""
        if not self.queen_adapter:
            return []

        execution_results = []

        for step in think_result.plan.steps:
            goal = step.sub_goal

            try:
                # 根据策略选择执行方式
                strategy = think_result.metadata.get("strategy")
                if strategy and strategy.get("mode") == CognitiveMode.SYSTEM1.value:
                    # System1: 快速执行
                    exec_result = await self.queen_adapter.execute_simple(goal.description)
                else:
                    # System2: 标准执行
                    exec_result = await self._execute_step_via_queen(goal)

                execution_results.append(
                    {
                        "success": exec_result.get("success", False),
                        "step_id": goal.id,
                        "description": goal.description,
                        "result": exec_result,
                    }
                )

            except Exception as e:
                logger.error(f"❌ Step execution error: {e}")
                execution_results.append(
                    {
                        "success": False,
                        "step_id": goal.id,
                        "description": goal.description,
                        "error": str(e),
                    }
                )

        return execution_results

    async def _execute_step_via_queen(self, step) -> Dict[str, Any]:
        """通过 QueenBee 执行单个步骤"""
        # 获取服务
        worker_manager = None
        if BeeServiceLocator:
            try:
                worker_manager = BeeServiceLocator.get_service("WorkerManager")
            except Exception:
                pass

        if not worker_manager:
            return {
                "success": False,
                "error": "WorkerManager 不可用",
            }

        # 创建 DNA 并部署
        if not BeeDNA:
            return {
                "success": False,
                "error": "BeeDNA 不可用",
            }

        try:
            dna = BeeDNA(
                capabilities=[step.sub_goal.goal_type.value],
                mutation_gene="standard",
            )

            # 异步部署
            if hasattr(self.queen, "deploy_swarm_v2"):
                results = await self.queen.deploy_swarm_v2([dna])
                return {
                    "success": True,
                    "deployed": len(results),
                }

            return {
                "success": True,
                "message": "部署完成",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _learn_from_execution(self, result: BrainfulResult):
        """从执行结果中学习"""
        if not result.think_result:
            return

        # 收集执行信息
        success = result.success
        description = (
            result.think_result.perception.original_text if result.think_result.perception else ""
        )

        # 计算成功率
        total_steps = len(result.execution_results)
        successful_steps = (
            sum(
                1
                for r in result.execution_results
                if isinstance(r, dict) and r.get("success", False)
            )
            if result.execution_results
            else 1
        )

        success_rate = (
            successful_steps / total_steps if total_steps > 0 else (1.0 if success else 0.0)
        )

        # 存储经验
        importance = 0.5 + success_rate * 0.3  # 0.8 for full success, 0.5 for failure

        await self.hive_brain.store_experience(
            content=f"{description[:100]}... -> {'成功' if success else '失败'} ({successful_steps}/{total_steps})",
            success=success,
            importance=importance,
            tags=[
                (
                    result.think_result.perception.intent.value
                    if result.think_result.perception
                    else "unknown"
                ),
                "brainful_execution",
            ],
        )

        # 更新策略选择器的历史
        if result.think_result.metadata.get("strategy"):
            strategy = result.think_result.metadata["strategy"]
            await self.hive_brain.strategy_selector.adapt_strategy(
                Strategy(
                    mode=CognitiveMode(strategy.get("mode", "system2")),
                    execution_mode=(
                        getattr(result.strategy_used, "execution_mode", None)
                        or ExecutionMode.HYBRID
                        if result.strategy_used is not None
                        else ExecutionMode.HYBRID
                    ),
                ),
                {"success": success, "duration": result.execution_duration_ms},
            )

    async def introspect(self) -> Dict[str, Any]:
        """
        自我反思

        让 HiveBrain 进行深度自我分析
        """
        if not self._initialized:
            await self.initialize()

        status = await self.hive_brain.get_status()

        # 获取执行历史分析
        recent_success_rate = 0.0
        if self._execution_history:
            recent = self._execution_history[-10:]
            successes = sum(1 for r in recent if r.success)
            recent_success_rate = successes / len(recent)

        return {
            "cognitive_state": status.get("state"),
            "health_status": status.get("health_status"),
            "recent_success_rate": recent_success_rate,
            "execution_history_size": len(self._execution_history),
            "strategies_used": self._stats["strategies_used"],
            "recommendations": self._generate_recommendations(status),
        }

    def _generate_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于认知状态建议
        state = status.get("state", {})
        attention_load = state.get("attention_load", 0)
        memory_pressure = state.get("memory_pressure", 0)

        if attention_load > 0.7:
            recommendations.append("注意力负载较高，建议减少并行思考")

        if memory_pressure > 0.7:
            recommendations.append("记忆压力较大，建议清理不活跃目标")

        # 基于策略使用建议
        strategies = self._stats.get("strategies_used", {})
        system1_count = strategies.get("system1", 0)
        system2_count = strategies.get("system2", 0)

        if system1_count > system2_count * 5:
            recommendations.append("System1 使用过多，建议增加深度分析任务")

        if not recommendations:
            recommendations.append("系统运行正常，继续保持")

        return recommendations

    async def shutdown(self):
        """关闭适配器"""
        if self.hive_brain:
            await self.hive_brain.shutdown()
        self._initialized = False
        logger.info("🧠 BrainQueenAdapter shutdown")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "initialized": self._initialized,
            "execution_history_size": len(self._execution_history),
            "brain_stats": self.hive_brain.get_stats() if self.hive_brain else {},
        }
