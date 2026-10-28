"""
Cognitive Diversity - 认知多样性系统
====================================

实现多个大脑的协调思考，通过不同性格配置产生多样化的思考结果。

核心组件:
- BrainPersonality: 性格配置，影响思考风格
- CognitiveSwarm: 多大脑协调器
- CollectiveResult: 集体思考结果

Usage:
    swarm = CognitiveSwarm(size=3)
    result = await swarm.collective_think("分析这个项目的架构")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hive_brain import HiveBrain, BrainConfig, ThinkResult

logger = logging.getLogger("Hive.Brain.CognitiveDiversity")


# ===== 性格配置 =====


@dataclass
class BrainPersonality:
    """
    大脑性格配置

    影响大脑的思考风格和行为模式
    """

    name: str
    risk_tolerance: float = 0.5  # 0-1, 风险承受度
    exploration_rate: float = 0.5  # 探索 vs 利用
    reflection_tendency: float = 0.5  # 反思频率
    patience: float = 0.5  # 耐心程度
    boldness: float = 0.5  # 激进 vs 保守

    def to_config(self) -> BrainConfig:
        """转换为 BrainConfig"""
        # 根据性格调整配置
        enable_reflection = self.reflection_tendency > 0.3

        return BrainConfig(
            name=f"Brain-{self.name}",
            enable_reflection=enable_reflection,
            enable_self_monitor=True,
            llm_for_decomposition=self.exploration_rate > 0.5,
            max_parallel_execution=3 if self.boldness > 0.5 else 2,
            reflection_threshold=1.0 - self.reflection_tendency,  # 反思倾向越高，阈值越低
        )

    def prefers_exploration(self) -> bool:
        """是否偏好探索"""
        return self.exploration_rate > 0.5

    def is_patient(self) -> bool:
        """是否有耐心"""
        return self.patience > 0.5

    def is_bold(self) -> bool:
        """是否激进"""
        return self.boldness > 0.5


# ===== 预设性格 =====


PERSONALITIES: Dict[str, BrainPersonality] = {
    "aggressive": BrainPersonality(
        name="激进型",
        risk_tolerance=0.8,
        exploration_rate=0.7,
        reflection_tendency=0.3,
        patience=0.3,
        boldness=0.9,
    ),
    "conservative": BrainPersonality(
        name="保守型",
        risk_tolerance=0.2,
        exploration_rate=0.3,
        reflection_tendency=0.8,
        patience=0.8,
        boldness=0.2,
    ),
    "explorer": BrainPersonality(
        name="探索型",
        risk_tolerance=0.6,
        exploration_rate=0.9,
        reflection_tendency=0.5,
        patience=0.7,
        boldness=0.5,
    ),
    "balanced": BrainPersonality(
        name="平衡型",
        risk_tolerance=0.5,
        exploration_rate=0.5,
        reflection_tendency=0.5,
        patience=0.5,
        boldness=0.5,
    ),
    "methodical": BrainPersonality(
        name="严谨型",
        risk_tolerance=0.3,
        exploration_rate=0.4,
        reflection_tendency=0.9,
        patience=0.9,
        boldness=0.2,
    ),
    "creative": BrainPersonality(
        name="创造型",
        risk_tolerance=0.7,
        exploration_rate=0.95,
        reflection_tendency=0.4,
        patience=0.4,
        boldness=0.7,
    ),
}


def get_personality(name: str) -> BrainPersonality:
    """获取指定名称的性格配置"""
    return PERSONALITIES.get(name, PERSONALITIES["balanced"])


def suggest_personality_for_task(task: str) -> BrainPersonality:
    """
    根据任务类型建议适合的性格

    Args:
        task: 任务描述

    Returns:
        BrainPersonality: 建议的性格
    """
    task_lower = task.lower()

    # 分析型任务 -> 严谨型
    if any(kw in task_lower for kw in ["分析", "评估", "审查", "review"]):
        return get_personality("methodical")

    # 探索型任务 -> 探索型
    if any(kw in task_lower for kw in ["探索", "发现", "研究", "search", "find"]):
        return get_personality("explorer")

    # 快速决策任务 -> 激进型
    if any(kw in task_lower for kw in ["快速", "紧急", "立即", "quick", "urgent"]):
        return get_personality("aggressive")

    # 安全优先任务 -> 保守型
    if any(kw in task_lower for kw in ["安全", "稳定", "验证", "verify"]):
        return get_personality("conservative")

    # 创造型任务 -> 创造型
    if any(kw in task_lower for kw in ["创造", "设计", "发明", "create", "design"]):
        return get_personality("creative")

    return get_personality("balanced")


# ===== 集体思考结果 =====


@dataclass
class CollectiveResult:
    """
    集体思考结果

    包含所有大脑的独立思考结果和集体共识
    """

    individual_results: List[ThinkResult]
    consensus: Any  # 共识结果
    confidence: float  # 置信度 0-1
    disagreement_points: List[str] = field(default_factory=list)
    voting_summary: Dict[str, int] = field(default_factory=dict)
    best_brain_index: int = -1  # 最佳大脑索引

    @property
    def success_count(self) -> int:
        """成功的思考数量"""
        return sum(1 for r in self.individual_results if r.success)

    @property
    def total_count(self) -> int:
        """总思考数量"""
        return len(self.individual_results)

    def has_consensus(self, threshold: float = 0.6) -> bool:
        """是否有共识（超过threshold比例的大脑得出相似结论）"""
        return self.confidence >= threshold


# ===== CognitiveSwarm =====


class CognitiveSwarm:
    """
    多大脑协调器

    管理多个具有不同性格的HiveBrain，通过投票机制产生集体决策。

    工作流程:
        1. 分配任务给所有大脑
        2. 每个大脑独立思考
        3. 收集结果并投票
        4. 返回集体决策
    """

    def __init__(
        self,
        size: int = 3,
        personality_names: Optional[List[str]] = None,
        enable_collective: bool = True,
    ):
        """
        Args:
            size: 大脑数量（如果未指定personality_names）
            personality_names: 性格名称列表，长度应等于size
            enable_collective: 是否启用集体思考
        """
        self.size = size
        self.enable_collective = enable_collective
        self.brains: List[HiveBrain] = []
        self.personalities: List[BrainPersonality] = []
        self._initialized = False
        self._lock = asyncio.Lock()

        # 如果提供了性格名称，使用它们；否则使用默认分配
        if personality_names:
            self._setup_personalities(personality_names)
        else:
            # 默认分配：平衡、探索、严谨
            default_names = ["balanced", "explorer", "methodical"]
            self._setup_personalities(default_names[:size])

    def _setup_personalities(self, names: List[str]):
        """设置性格列表"""
        self.personalities = [get_personality(name) for name in names]

        # 如果性格数量少于size，填充平衡型
        while len(self.personalities) < self.size:
            self.personalities.append(get_personality("balanced"))

    async def initialize(self):
        """初始化所有大脑"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info(f"🧠 Initializing CognitiveSwarm with {self.size} brains")

            # 创建并初始化每个大脑
            for i, personality in enumerate(self.personalities):
                config = personality.to_config()
                brain = HiveBrain(config)
                await brain.initialize()
                self.brains.append(brain)

                logger.info(
                    f"  Brain {i}: {personality.name} "
                    f"(risk={personality.risk_tolerance:.1f}, "
                    f"exploration={personality.exploration_rate:.1f})"
                )

            self._initialized = True
            logger.info("🧠 CognitiveSwarm initialized")

    async def collective_think(self, task: str) -> CollectiveResult:
        """
        集体思考

        所有大脑独立思考同一任务，然后通过投票产生集体决策。

        Args:
            task: 任务描述

        Returns:
            CollectiveResult: 集体思考结果
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"🧠 CognitiveSwarm: collective think for '{task[:50]}...'")

        # 1. 所有大脑并行思考
        think_tasks = [brain.think(task) for brain in self.brains]
        results = await asyncio.gather(*think_tasks, return_exceptions=True)

        # 处理异常结果
        processed_results: List[ThinkResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"  Brain {i} raised exception: {result}")
                # 创建失败结果
                processed_results.append(
                    ThinkResult(
                        success=False,
                        message=str(result),
                    )
                )
            else:
                processed_results.append(result)

        # 2. 投票产生共识
        consensus, confidence, disagreement_points, voting_summary = self._vote(
            processed_results
        )

        # 3. 找出最佳大脑
        best_idx = self._find_best_brain(processed_results)

        return CollectiveResult(
            individual_results=processed_results,
            consensus=consensus,
            confidence=confidence,
            disagreement_points=disagreement_points,
            voting_summary=voting_summary,
            best_brain_index=best_idx,
        )

    def _vote(self, results: List[ThinkResult]) -> Tuple[Any, float, List[str], Dict[str, int]]:
        """
        投票机制

        通过分析思考结果的相似性来产生共识。

        Args:
            results: 所有大脑的思考结果

        Returns:
            Tuple: (共识, 置信度, 分歧点, 投票摘要)
        """
        if not results:
            return None, 0.0, ["No results"], {}

        successful_results = [r for r in results if r.success]

        if not successful_results:
            return None, 0.0, ["All brains failed"], {}

        # 基于意图投票
        intent_votes: Dict[str, int] = {}
        for result in successful_results:
            if result.perception:
                intent = result.perception.intent.value
                intent_votes[intent] = intent_votes.get(intent, 0) + 1

        # 找到最多票数的意图
        if intent_votes:
            best_intent = max(intent_votes, key=intent_votes.get)
            intent_confidence = intent_votes[best_intent] / len(successful_results)
        else:
            best_intent = None
            intent_confidence = 0.0

        # 分析分歧点
        disagreement_points = []
        all_intents = [r.perception.intent.value for r in successful_results if r.perception]
        if len(set(all_intents)) > 1:
            disagreement_points.append(f"意图分歧: {all_intents}")

        # 检查子目标数量差异
        subgoal_counts = [len(r.sub_goals) for r in successful_results]
        if max(subgoal_counts) - min(subgoal_counts) > 2:
            disagreement_points.append(f"子目标数量差异大: {subgoal_counts}")

        # 检查计划步骤数量差异
        plan_steps = [
            len(r.plan.steps) if r.plan else 0 for r in successful_results
        ]
        if max(plan_steps) - min(plan_steps) > 3:
            disagreement_points.append(f"计划步骤数量差异大: {plan_steps}")

        # 计算置信度：基于共识程度和成功数量
        base_confidence = intent_confidence
        success_ratio = len(successful_results) / len(results)
        confidence = base_confidence * success_ratio

        # 构建共识对象
        consensus = {
            "primary_intent": best_intent,
            "intent_votes": intent_votes,
            "avg_subgoal_count": sum(subgoal_counts) / len(subgoal_counts) if subgoal_counts else 0,
            "consensus_quality": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
        }

        return consensus, confidence, disagreement_points, intent_votes

    def _find_best_brain(self, results: List[ThinkResult]) -> int:
        """找出最佳大脑的索引"""
        best_idx = -1
        best_score = -1

        for i, result in enumerate(results):
            if not result.success:
                continue

            score = 0.0

            # 基础分
            score += 0.5

            # 置信度加分
            if result.cognitive_state:
                score += result.cognitive_state.confidence * 0.3

            # 感知置信度加分
            if result.perception:
                score += result.perception.confidence * 0.2

            # 子目标数量（适度为佳）
            subgoal_count = len(result.sub_goals)
            if 1 <= subgoal_count <= 5:
                score += 0.2

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def assign_by_personality(self, task: str) -> Tuple[HiveBrain, BrainPersonality]:
        """
        根据任务类型分配最适合的大脑

        Args:
            task: 任务描述

        Returns:
            Tuple: (分配的大脑, 使用的性格)
        """
        if not self._initialized:
            # 返回第一个大脑的性格（延迟初始化）
            personality = suggest_personality_for_task(task)
            return self.brains[0] if self.brains else None, personality

        # 根据任务建议性格
        suggested = suggest_personality_for_task(task)

        # 找到最匹配的大脑
        for i, personality in enumerate(self.personalities):
            if personality.name == suggested.name:
                return self.brains[i], personality

        # 回退到第一个大脑
        return self.brains[0], self.personalities[0]

    async def think_single(
        self, task: str, personality_name: Optional[str] = None
    ) -> ThinkResult:
        """
        使用单个大脑思考

        Args:
            task: 任务描述
            personality_name: 可选，指定性格名称

        Returns:
            ThinkResult: 思考结果
        """
        if personality_name:
            personality = get_personality(personality_name)
            for i, p in enumerate(self.personalities):
                if p.name == personality.name:
                    return await self.brains[i].think(task)

        # 默认使用第一个大脑
        return await self.brains[0].think(task)

    async def think_with_personality_adjustment(
        self, task: str, adjustment: Dict[str, float]
    ) -> ThinkResult:
        """
        使用性格调整后的单一大脑思考

        Args:
            task: 任务描述
            adjustment: 性格调整值，如 {"boldness": 0.8, "patience": 0.3}

        Returns:
            ThinkResult: 思考结果
        """
        if not self._initialized:
            await self.initialize()

        # 选择一个大脑作为基础
        brain_idx = 0
        base_personality = self.personalities[brain_idx]

        # 创建临时调整后的性格
        adjusted = BrainPersonality(
            name=f"{base_personality.name}-adjusted",
            risk_tolerance=adjustment.get("risk_tolerance", base_personality.risk_tolerance),
            exploration_rate=adjustment.get("exploration_rate", base_personality.exploration_rate),
            reflection_tendency=adjustment.get("reflection_tendency", base_personality.reflection_tendency),
            patience=adjustment.get("patience", base_personality.patience),
            boldness=adjustment.get("boldness", base_personality.boldness),
        )

        # 临时应用调整（修改配置）- 需要锁保护防止并发竞态
        async with self._lock:
            original_config = self.brains[brain_idx].config
            temp_config = adjusted.to_config()

            # 交换配置
            self.brains[brain_idx].config = temp_config

            try:
                return await self.brains[brain_idx].think(task)
            finally:
                # 恢复原始配置
                self.brains[brain_idx].config = original_config

    def get_brain_info(self) -> List[Dict[str, Any]]:
        """获取所有大脑的信息"""
        return [
            {
                "index": i,
                "personality": p.name,
                "risk_tolerance": p.risk_tolerance,
                "exploration_rate": p.exploration_rate,
                "reflection_tendency": p.reflection_tendency,
                "patience": p.patience,
                "boldness": p.boldness,
            }
            for i, p in enumerate(self.personalities)
        ]

    async def shutdown(self):
        """关闭所有大脑"""
        logger.info("🧠 CognitiveSwarm shutting down")
        for brain in self.brains:
            await brain.shutdown()
        self._initialized = False
