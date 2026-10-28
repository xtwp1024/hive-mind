"""
HiveBrain 测试脚本
==================

测试 HiveBrain 的基本功能。

Usage:
    python -m hive.core.brain.test_brain
"""

import asyncio
import logging
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("Hive.Brain.Test")


async def test_working_memory():
    """测试工作记忆"""
    logger.info("=" * 50)
    logger.info("Testing WorkingMemory")
    logger.info("=" * 50)

    from .working_memory import WorkingMemory

    wm = WorkingMemory(max_items=10)

    # 添加注意力项
    item_id = await wm.add("测试内容", tags=["test"], attention_level=0.8)
    logger.info(f"✅ Added item: {item_id}")

    # 设置焦点
    await wm.set_focus(item_id)
    logger.info(f"✅ Set focus to: {item_id}")

    # 添加目标
    goal1 = await wm.push_goal("完成测试任务", priority=3)
    logger.info(f"✅ Pushed goal: {goal1.description}")

    # 获取上下文
    context = wm.get_context()
    logger.info(f"✅ Context: {context}")

    assert len(wm._items) == 1
    assert len(wm._goal_stack) == 1
    logger.info("✅ WorkingMemory test passed")


async def test_cognitive_bus():
    """测试认知消息总线"""
    logger.info("=" * 50)
    logger.info("Testing CognitiveBus")
    logger.info("=" * 50)

    from .cognitive_bus import CognitiveBus, Message, MessageType

    bus = CognitiveBus()
    await bus.start()

    received = []

    async def handler(message):
        received.append(message)
        logger.info(f"📨 Received: {message.type.value}")

    bus.subscribe(MessageType.PERCEPTION, handler)

    # 发布消息
    msg = Message(type=MessageType.PERCEPTION, data={"text": "hello"})
    await bus.publish(msg, source="test")

    # 等待处理
    await asyncio.sleep(0.2)

    await bus.stop()

    assert len(received) == 1
    assert received[0].data["text"] == "hello"
    logger.info("✅ CognitiveBus test passed")


async def test_perceiver():
    """测试感知器"""
    logger.info("=" * 50)
    logger.info("Testing Perceiver")
    logger.info("=" * 50)

    from .perception import Perceiver, Intent

    perceiver = Perceiver()

    # 测试意图识别
    test_cases = [
        ("帮我分析这个项目", Intent.ANALYZE),
        ("创建一个新的模块", Intent.CREATE),
        ("查看一下代码", Intent.READ),
        ("修复这个bug", Intent.UPDATE),
        ("你好", Intent.GREETING),
    ]

    for text, expected_intent in test_cases:
        perception = await perceiver.process(text)
        logger.info(f"  '{text}' -> {perception.intent.value} (expected: {expected_intent.value})")
        assert perception.intent == expected_intent, f"Expected {expected_intent}, got {perception.intent}"

    logger.info("✅ Perceiver test passed")


async def test_goal_decomposer():
    """测试目标分解器"""
    logger.info("=" * 50)
    logger.info("Testing GoalDecomposer")
    logger.info("=" * 50)

    from .goal_decomposer import GoalDecomposer

    decomposer = GoalDecomposer(use_llm=False)  # 不使用LLM，使用模板

    # 测试分解
    sub_goals = await decomposer.decompose("帮我分析项目架构")
    logger.info(f"  Decomposed into {len(sub_goals)} sub-goals:")

    for sg in sub_goals:
        logger.info(f"    - {sg.description} (priority={sg.priority}, deps={sg.dependencies})")

    assert len(sub_goals) > 0
    logger.info("✅ GoalDecomposer test passed")


async def test_planner():
    """测试规划器"""
    logger.info("=" * 50)
    logger.info("Testing Planner")
    logger.info("=" * 50)

    from .goal_decomposer import GoalDecomposer
    from .planner import Planner

    decomposer = GoalDecomposer(use_llm=False)
    planner = Planner()

    # 分解目标
    sub_goals = await decomposer.decompose("帮我分析项目架构")

    # 生成计划
    plan = await planner.plan(sub_goals)

    logger.info(f"  Plan: {len(plan.steps)} steps")
    logger.info(f"  Total duration: ~{plan.total_estimated_duration:.0f}s")
    logger.info(f"  Overall risk: {plan.overall_risk:.2f}")

    for i, step in enumerate(plan.steps):
        logger.info(f"    {i+1}. {step.sub_goal.description} (strategy={step.strategy.value})")

    assert len(plan.steps) > 0
    logger.info("✅ Planner test passed")


async def test_hive_brain():
    """测试完整的HiveBrain"""
    logger.info("=" * 50)
    logger.info("Testing HiveBrain (Full)")
    logger.info("=" * 50)

    from .hive_brain import HiveBrain, BrainConfig

    config = BrainConfig(
        name="TestBrain",
        enable_reflection=False,  # 简化测试
        llm_for_decomposition=False,
    )

    brain = HiveBrain(config)
    await brain.initialize()

    # 测试思考
    result = await brain.think("帮我分析这个项目")

    logger.info(f"  Success: {result.success}")
    logger.info(f"  Message: {result.message}")
    if result.perception:
        logger.info(f"  Intent: {result.perception.intent.value}")
        logger.info(f"  Entities: {len(result.perception.entities)}")
        logger.info(f"  Confidence: {result.perception.confidence:.2f}")
    if result.plan:
        logger.info(f"  Plan steps: {len(result.plan.steps)}")
    logger.info(f"  Sub-goals: {len(result.sub_goals)}")

    # 获取状态
    status = await brain.get_status()
    logger.info(f"  State: {status['state']}")

    await brain.shutdown()

    assert result.success
    logger.info("✅ HiveBrain test passed")


async def test_memory_hierarchy():
    """测试记忆层次"""
    logger.info("=" * 50)
    logger.info("Testing MemoryHierarchy")
    logger.info("=" * 50)

    from .memory_hierarchy import MemoryHierarchy, Experience

    hierarchy = MemoryHierarchy()
    await hierarchy.initialize()

    # 存储经验
    exp = Experience(
        content="测试经验：分析项目架构",
        success=True,
        importance=0.7,
        tags=["test", "analysis"],
    )
    success = await hierarchy.store(exp)
    logger.info(f"  Stored experience: {success}")

    # 检索
    context = await hierarchy.recall("项目架构")
    logger.info(f"  Recalled context: {context.to_dict()}")

    stats = hierarchy.get_stats()
    logger.info(f"  Stats: {stats}")

    assert success
    logger.info("✅ MemoryHierarchy test passed")


async def test_memory_archival():
    """测试记忆归档系统"""
    logger.info("=" * 50)
    logger.info("Testing MemoryArchival")
    logger.info("=" * 50)

    from .memory_archival import MemoryArchiver, MemoryArchive, StorageTier
    from .memory_hierarchy import MemoryItem
    import time

    archiver = MemoryArchiver()

    # 测试1: 创建MemoryItem并归档
    memory = MemoryItem(
        id="test_memory_001",
        content="测试记忆内容：分析项目架构",
        memory_type="episodic",
        importance=0.3,
        created_at=time.time(),
        last_accessed=time.time() - (60 * 24 * 3600),  # 60天前
        access_count=2,
    )

    # 手动归档
    archive = await archiver._archive_memory(memory)
    logger.info(f"  Archived memory: {archive.id}")
    assert archive.id == "test_memory_001"
    assert archive.storage_tier in [StorageTier.COLD, StorageTier.WARM]

    # 测试2: 检索归档
    retrieved = await archiver.retrieve_archived("test_memory_001")
    assert retrieved is not None
    assert retrieved.access_count == 3  # 增加了一次访问

    # 测试3: 复活归档记忆
    rehydrated = await archiver.rehydrate(retrieved)
    assert rehydrated.id == "test_memory_001"
    assert rehydrated.memory_type == "episodic"
    assert rehydrated.metadata.get("rehydrated") is True

    # 测试4: 层级迁移
    archive2 = await archiver._archive_memory(MemoryItem(
        id="test_memory_002",
        content="另一个测试记忆",
        memory_type="semantic",
        importance=0.6,
        created_at=time.time(),
        last_accessed=time.time(),
        access_count=10,
    ))
    await archiver.migrate_tier("test_memory_002", StorageTier.HOT)
    assert archiver.archives["test_memory_002"].storage_tier == StorageTier.HOT

    # 测试5: 获取统计
    stats = archiver.get_stats()
    logger.info(f"  Stats: {stats}")
    assert stats["total_archives"] >= 1

    # 测试6: StorageTier枚举
    assert StorageTier.HOT.value == "hot"
    assert StorageTier.DEEP_FREEZE.value == "deep_freeze"

    logger.info("✅ MemoryArchival test passed")


async def test_attention_economy():
    """测试注意力经济系统"""
    logger.info("=" * 50)
    logger.info("Testing AttentionEconomy")
    logger.info("=" * 50)

    from .attention_economy import AttentionEconomy, AttentionBudget

    # 测试1: 基本消耗
    economy = AttentionEconomy()
    budget_status = economy.get_budget_status()
    logger.info(f"  Initial budget: {budget_status['available']:.1f}/{budget_status['total']:.1f}")

    assert budget_status["available"] == 100.0
    assert budget_status["is_low"] is False

    # 测试2: 消耗感知
    assert economy.can_afford("perception") is True
    consumed = economy.consume("perception")
    assert consumed is True

    budget_status = economy.get_budget_status()
    logger.info(f"  After perception: {budget_status['available']:.1f}")
    assert budget_status["available"] == 98.0  # 100 - 2

    # 测试3: 消耗规划
    consumed = economy.consume("planning")
    assert consumed is True
    budget_status = economy.get_budget_status()
    logger.info(f"  After planning: {budget_status['available']:.1f}")
    assert budget_status["available"] == 83.0  # 98 - 15

    # 测试4: 预算不足
    economy.budget.available = 5.0
    assert economy.can_afford("planning") is False
    assert economy.can_afford("perception") is True  # perception 只消耗 2

    # 测试5: 恢复
    economy.budget.available = 10.0
    economy.recharge(20.0)
    assert economy.budget.available == 30.0

    # 测试6: 自动恢复
    economy2 = AttentionEconomy()
    economy2.budget.available = 50.0
    economy2.auto_recharge(elapsed_seconds=10.0)  # 10秒 * 0.5 = 5
    assert economy2.budget.available > 50.0

    # 测试7: 耗尽检查
    budget = AttentionBudget(total=100.0, available=5.0)
    assert budget.is_depleted(10.0) is True
    assert budget.is_depleted(3.0) is False

    # 测试8: 预算状态
    economy3 = AttentionEconomy()
    economy3.consume("planning")
    economy3.consume("reflection")
    status = economy3.get_budget_status()
    logger.info(f"  Budget status: {status}")
    assert "can_afford" in status
    assert status["can_afford"]["planning"] is True
    assert status["can_afford"]["reflection"] is True  # 需要 25，有 60，所以可以

    logger.info("✅ AttentionEconomy test passed")


async def test_sandbox_environment():
    """测试沙箱环境"""
    logger.info("=" * 50)
    logger.info("Testing SandboxEnvironment")
    logger.info("=" * 50)

    from .sandbox import SandboxEnvironment, SandboxConfig, SimulationResult

    # 测试1: 基本沙箱配置
    config = SandboxConfig(
        allow_file_delete=False,
        allow_network=False,
        allow_execute=["python", "ls"],
        max_iterations=100,
        timeout_seconds=30.0,
    )
    logger.info(f"  Config: allow_delete={config.allow_file_delete}, allow_network={config.allow_network}")

    # 测试2: 创建带 mock_fs 的沙箱
    mock_fs = {
        "readme.md": "# Project Readme",
        "src/main.py": "print('hello')",
        "config.json": '{"key": "value"}',
    }
    sandbox = SandboxEnvironment(config=config, mock_fs=mock_fs)
    logger.info(f"  Sandbox created with {len(sandbox.fs)} mock files")

    # 测试3: 低风险任务 - 应该通过
    safe_task = "帮我分析这个项目"
    result = await sandbox.simulate_think(safe_task)
    logger.info(f"  Safe task result: compliant={result.safety_compliance}, risks={len(result.risks_identified)}")
    assert isinstance(result, SimulationResult)
    assert result.safety_compliance is True
    assert len(result.risks_identified) == 0

    # 测试4: 高风险任务 - 删除操作（不允许）
    delete_task = "删除所有文件"
    result = await sandbox.simulate_think(delete_task)
    logger.info(f"  Delete task: compliant={result.safety_compliance}, blocked={sum(1 for op in result.operations if not op.allowed)}")
    assert result.safety_compliance is False
    assert any("delete" in op.type.lower() for op in result.operations)

    # 测试5: 网络操作（不允许）
    network_task = "curl http://example.com"
    result = await sandbox.simulate_think(network_task)
    logger.info(f"  Network task: compliant={result.safety_compliance}, risks={result.risks_identified}")
    assert result.safety_compliance is False
    assert any(op.type == "network" and not op.allowed for op in result.operations)

    # 测试6: 命令执行（不在白名单）
    exec_task = "执行 rm -rf /"
    result = await sandbox.simulate_think(exec_task)
    logger.info(f"  Exec task: compliant={result.safety_compliance}, blocked={sum(1 for op in result.operations if not op.allowed)}")
    assert result.safety_compliance is False

    # 测试7: 白名单命令应该允许
    sandbox_allowed = SandboxEnvironment(
        config=SandboxConfig(allow_execute=["python", "ls", "git"]),
        mock_fs=mock_fs,
    )
    exec_task2 = "执行 python script.py"
    result = await sandbox_allowed.simulate_think(exec_task2)
    logger.info(f"  Whitelisted exec: compliant={result.safety_compliance}")
    # 白名单命令会被检查，但由于是 simulate_think，不会真正执行

    # 测试8: 恢复被阻止的操作
    sandbox_allow_delete = SandboxEnvironment(
        config=SandboxConfig(allow_file_delete=True),
        mock_fs=mock_fs,
    )
    result = await sandbox_allow_delete.simulate_think("删除 readme.md")
    logger.info(f"  Allow delete task: compliant={result.safety_compliance}")
    # 允许删除时，风险模式仍会识别，但操作会被允许

    # 测试9: 操作统计摘要
    summary = sandbox.get_operations_summary()
    logger.info(f"  Operations summary: {summary}")
    assert isinstance(summary, dict)

    # 测试10: 重置沙箱
    sandbox.reset()
    logger.info(f"  Sandbox reset: operations={len(sandbox.operations_log)}")
    assert len(sandbox.operations_log) == 0

    logger.info("✅ SandboxEnvironment test passed")


async def test_cognitive_diversity():
    """测试认知多样性系统"""
    logger.info("=" * 50)
    logger.info("Testing Cognitive Diversity")
    logger.info("=" * 50)

    from .cognitive_diversity import (
        BrainPersonality,
        CognitiveSwarm,
        CollectiveResult,
        PERSONALITIES,
        get_personality,
        suggest_personality_for_task,
    )

    # 测试预设性格
    logger.info("Testing preset personalities:")
    for name, personality in PERSONALITIES.items():
        logger.info(f"  {name}: risk={personality.risk_tolerance:.1f}, "
                    f"exploration={personality.exploration_rate:.1f}, "
                    f"boldness={personality.boldness:.1f}")
    assert len(PERSONALITIES) >= 6, "Should have at least 6 preset personalities"
    logger.info("✅ Preset personalities test passed")

    # 测试获取性格
    p = get_personality("aggressive")
    assert p.name == "激进型"
    assert p.risk_tolerance == 0.8
    logger.info("✅ get_personality test passed")

    # 测试任务建议性格
    suggested = suggest_personality_for_task("帮我分析这个项目的架构")
    logger.info(f"  Suggested personality for 'analyze': {suggested.name}")
    assert suggested.name in ["严谨型", "平衡型", "探索型"]
    logger.info("✅ suggest_personality_for_task test passed")

    # 测试 CognitiveSwarm
    swarm = CognitiveSwarm(size=3)
    await swarm.initialize()

    # 获取大脑信息
    brain_info = swarm.get_brain_info()
    logger.info(f"  Swarm has {len(brain_info)} brains:")
    for info in brain_info:
        logger.info(f"    Brain {info['index']}: {info['personality']} "
                    f"(risk={info['risk_tolerance']:.1f})")
    assert len(brain_info) == 3
    logger.info("✅ CognitiveSwarm initialization test passed")

    # 测试集体思考
    collective_result = await swarm.collective_think("帮我分析这个项目")
    logger.info(f"  Collective think result:")
    logger.info(f"    Success count: {collective_result.success_count}/{collective_result.total_count}")
    logger.info(f"    Confidence: {collective_result.confidence:.2f}")
    logger.info(f"    Best brain index: {collective_result.best_brain_index}")
    if collective_result.disagreement_points:
        logger.info(f"    Disagreements: {collective_result.disagreement_points}")
    logger.info(f"    Consensus: {collective_result.consensus}")

    assert isinstance(collective_result, CollectiveResult)
    assert collective_result.total_count == 3
    assert 0.0 <= collective_result.confidence <= 1.0
    logger.info("✅ CognitiveSwarm.collective_think test passed")

    # 测试单一大脑思考
    single_result = await swarm.think_single("你好", personality_name="explorer")
    logger.info(f"  Single think result: success={single_result.success}")
    assert single_result.success
    logger.info("✅ CognitiveSwarm.think_single test passed")

    # 测试根据任务分配大脑
    brain, personality = swarm.assign_by_personality("帮我深入分析这个问题")
    logger.info(f"  Assigned brain: {personality.name} for task")
    assert brain is not None
    logger.info("✅ CognitiveSwarm.assign_by_personality test passed")

    # 测试性格调整
    adjusted_result = await swarm.think_with_personality_adjustment(
        "快速决策",
        {"boldness": 0.9, "patience": 0.2}
    )
    logger.info(f"  Adjusted personality think: success={adjusted_result.success}")
    assert adjusted_result.success
    logger.info("✅ CognitiveSwarm.think_with_personality_adjustment test passed")

    # 清理
    await swarm.shutdown()
    logger.info("✅ CognitiveSwarm shutdown test passed")

    logger.info("✅ Cognitive Diversity test passed")


async def main():
    """运行所有测试"""
    logger.info("\n" + "=" * 60)
    logger.info("🧠 HiveBrain Test Suite")
    logger.info("=" * 60 + "\n")

    try:
        await test_working_memory()
        print()

        await test_cognitive_bus()
        print()

        await test_perceiver()
        print()

        await test_goal_decomposer()
        print()

        await test_planner()
        print()

        await test_memory_hierarchy()
        print()

        await test_attention_economy()
        print()

        await test_hive_brain()
        print()

        await test_sandbox_environment()
        print()

        await test_cognitive_diversity()
        print()

        logger.info("\n" + "=" * 60)
        logger.info("✅ All tests passed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
