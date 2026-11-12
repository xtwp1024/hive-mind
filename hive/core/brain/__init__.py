"""
HiveBrain - 统一认知大脑
========================

核心架构:
- CognitiveBus: 消息总线，连接各认知模块
- WorkingMemory: 工作记忆，当前思考上下文
- MemoryHierarchy: 记忆层次，统一接口
- Perceiver: 感知器，意图识别
- GoalDecomposer: 目标分解器
- HiveBrain: 主大脑类

Usage:
    brain = HiveBrain()
    result = await brain.think("帮我分析这个项目")
"""

from .attention_economy import AttentionBudget, AttentionEconomy
from .brain_queen_adapter import (
    BrainfulResult,
    BrainfulSandboxConfig,
    BrainQueenAdapter,
)
from .cognitive_bus import CognitiveBus, Message, MessageType
from .cognitive_diversity import (
    PERSONALITIES,
    BrainPersonality,
    CognitiveSwarm,
    CollectiveResult,
    get_personality,
    suggest_personality_for_task,
)
from .goal_decomposer import GoalDecomposer, SubGoal
from .hive_brain import HiveBrain
from .memory_archival import MemoryArchive, MemoryArchiver, StorageTier
from .memory_hierarchy import Experience, MemoryHierarchy, MemoryItem
from .meta_cognition import (
    AUTOTUNABLE_PARAMS,
    Adjustment,
    AttentionAllocation,
    CognitiveModel,
    MetaCognitionLayer,
    OptimizationResult,
    PerformanceMetrics,
    SelfImprovementEngine,
)
from .perception import Perceiver, Perception
from .planner import Plan, Planner
from .queen_adapter import QueenAdapter, QueenAdapterConfig
from .sandbox import (
    SandboxConfig,
    SandboxEnvironment,
    SimulatedOp,
    SimulationResult,
)
from .working_memory import WorkingMemory

__all__ = [
    # Cognitive Diversity
    "BrainPersonality",
    "CognitiveSwarm",
    "CollectiveResult",
    "PERSONALITIES",
    "get_personality",
    "suggest_personality_for_task",
    # Sandbox
    "SandboxEnvironment",
    "SandboxConfig",
    "SimulatedOp",
    "SimulationResult",
    # Core
    "CognitiveBus",
    "Message",
    "MessageType",
    # Memory
    "WorkingMemory",
    "MemoryHierarchy",
    "MemoryItem",
    "Experience",
    "MemoryArchiver",
    "MemoryArchive",
    "StorageTier",
    # Perception
    "Perceiver",
    "Perception",
    "Intent",
    # Goal & Plan
    "GoalDecomposer",
    "SubGoal",
    "Planner",
    "Plan",
    # Reflection & Critique
    "Reflector",
    "ReflectionResult",
    "Insight",
    "Critic",
    "Evaluation",
    "Issue",
    # Strategy
    "StrategySelector",
    "Strategy",
    "CognitiveMode",
    # Monitor
    "SelfMonitor",
    "CognitiveMetrics",
    "HealthStatus",
    "Alert",
    # Phase 3 - Queen Integration
    "BrainQueenAdapter",
    "BrainfulResult",
    "BrainfulSandboxConfig",
    "MetaCognitionLayer",
    "AttentionAllocation",
    "CognitiveModel",
    "OptimizationResult",
    "PerformanceMetrics",
    "Adjustment",
    "SelfImprovementEngine",
    "AUTOTUNABLE_PARAMS",
    # Main
    "HiveBrain",
    # Attention Economy
    "AttentionEconomy",
    "AttentionBudget",
    "QueenAdapter",
    "QueenAdapterConfig",
    "ThinkResult",
    "BrainConfig",
    "CognitiveState",
    "get_hive_brain",
]
