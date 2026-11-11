# -*- coding: utf-8 -*-
"""
蜂巢进化组件集成测试

测试 CognitiveBee 和 SwarmCommunication 的集成
"""

import unittest
import sys
from pathlib import Path
import multiprocessing
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hive.bees.cognitive_bee import CognitiveBee
from hive.core.dna import BeeDNA
from hive.core.swarm_communication import (
    get_swarm_communication,
    SwarmMessage,
    MessageType,
    TaskBid
)


class TestCognitiveBeeIntegration(unittest.TestCase):
    """测试 CognitiveBee 集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.result_queue = multiprocessing.Queue()
        self.dna = BeeDNA(species="cognitive")
        self.cognitive = CognitiveBee(self.dna, "cognitive_test", self.result_queue)
    
    def test_cognitive_bee_initialization(self):
        """测试 CognitiveBee 初始化"""
        self.assertEqual(self.cognitive.id, "cognitive_test")
        self.assertEqual(self.cognitive.dna.species, "cognitive")
        self.assertIsInstance(self.cognitive.capability_history, dict)
        self.assertIsInstance(self.cognitive.blind_spots, list)
        self.assertIsInstance(self.cognitive.learning_plans, list)
    
    def test_assess_capabilities(self):
        """测试能力评估"""
        capabilities = self.cognitive.assess_capabilities()
        
        # 验证返回10项能力
        self.assertEqual(len(capabilities), 10)
        
        # 验证每项能力都有评分
        for name, assessment in capabilities.items():
            self.assertIsInstance(assessment.score, float)
            self.assertGreaterEqual(assessment.score, 0.0)
            self.assertLessEqual(assessment.score, 1.0)
            self.assertIsInstance(assessment.evidence, list)
    
    def test_identify_blind_spots(self):
        """测试盲区识别"""
        blind_spots = self.cognitive.identify_blind_spots()
        
        # 应该能识别出盲区
        self.assertIsInstance(blind_spots, list)
        self.assertGreater(len(blind_spots), 0)
        
        # 验证盲区结构
        for spot in blind_spots:
            self.assertIsNotNone(spot.domain)
            self.assertIn(spot.severity, ["critical", "high", "medium", "low"])
            self.assertIsInstance(spot.suggested_learning_path, list)
    
    def test_create_learning_plan(self):
        """测试学习计划创建"""
        plan = self.cognitive.create_learning_plan()
        
        # 验证计划结构
        self.assertIsInstance(plan.targets, list)
        self.assertGreater(len(plan.targets), 0)
        self.assertIsNotNone(plan.priority)
        self.assertIsNotNone(plan.method)
        self.assertGreater(plan.estimated_time, 0)
        self.assertIsInstance(plan.success_criteria, dict)
    
    def test_optimize_learning_strategy(self):
        """测试学习策略优化"""
        strategies = self.cognitive.optimize_learning_strategy()
        
        # 验证返回策略效率
        self.assertIsInstance(strategies, dict)
        self.assertGreater(len(strategies), 0)
        
        # 验证每个策略都有效率分数
        for strategy, efficiency in strategies.items():
            self.assertIsInstance(efficiency, float)
            self.assertGreaterEqual(efficiency, 0.0)
            self.assertLessEqual(efficiency, 1.0)
    
    def test_get_cognitive_status(self):
        """测试获取认知状态"""
        status = self.cognitive.get_cognitive_status()
        
        # 验证状态结构
        self.assertEqual(status["bee_id"], "cognitive_test")
        self.assertEqual(status["bee_type"], "CognitiveBee")
        self.assertIn("capabilities", status)
        self.assertIn("blind_spots", status)
        self.assertIn("learning_plans", status)
        self.assertIn("overall_intelligence", status)
        
        # 验证总体智能分数
        self.assertIsInstance(status["overall_intelligence"], float)
        self.assertGreaterEqual(status["overall_intelligence"], 0.0)
        self.assertLessEqual(status["overall_intelligence"], 1.0)
    
    def test_specialized_task(self):
        """测试专门任务执行"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir)
            report = self.cognitive.specialized_task(target_path)
            
            # 验证报告结构
            self.assertEqual(report["assessment_type"], "cognitive_evaluation")
            self.assertIn("capabilities", report)
            self.assertIn("blind_spots", report)
            self.assertIn("learning_plan", report)
            self.assertIn("optimal_strategies", report)
            self.assertIn("overall_intelligence", report)


class TestSwarmCommunicationIntegration(unittest.TestCase):
    """测试 SwarmCommunication 集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.swarm = get_swarm_communication()
        
        # 创建测试蜂
        self.result_queue = multiprocessing.Queue()
        self.dna = BeeDNA(species="test")
        self.cognitive = CognitiveBee(self.dna, "test_bee_1", self.result_queue)

        # --- R7 / P1-9③ fixture 隔离 BEGIN ---
        # get_swarm_communication() 是进程级单例，registered_bees 跨用例累积：
        # 按用例名字典序 test_broadcast_discovery 先注册 "test_bee_1"，轮到
        # test_register_bee 时同名注册不再 +1，断言 initial_count + 1 变 1 != 2。
        # 快照 -> 清空 -> addCleanup 复原，让每个用例从空注册表出发，且不把
        # 本类状态泄漏给其它测试模块。只改测试，不改产品行为。
        _registered_bees_snapshot = dict(self.swarm.registered_bees)
        self.swarm.registered_bees.clear()

        def _restore_registered_bees():
            self.swarm.registered_bees.clear()
            self.swarm.registered_bees.update(_registered_bees_snapshot)

        self.addCleanup(_restore_registered_bees)
        # --- R7 / P1-9③ fixture 隔离 END ---
    
    def test_swarm_initialization(self):
        """测试蜂群通信初始化"""
        self.assertIsNotNone(self.swarm)
        self.assertTrue(self.swarm.running)
        self.assertIsInstance(self.swarm.registered_bees, dict)
        self.assertIsInstance(self.swarm.message_history, list)
    
    def test_register_bee(self):
        """测试注册蜂"""
        initial_count = len(self.swarm.registered_bees)
        
        self.swarm.register_bee("test_bee_1", self.cognitive)
        
        self.assertEqual(len(self.swarm.registered_bees), initial_count + 1)
        self.assertIn("test_bee_1", self.swarm.registered_bees)
    
    def test_broadcast_message(self):
        """测试广播消息"""
        message = SwarmMessage(
            id="test_msg_1",
            type=MessageType.DISCOVERY,
            from_bee="test_bee_1",
            content={"discovery": "test_pattern"}
        )
        
        initial_count = len(self.swarm.message_history)
        
        self.swarm.broadcast(message)
        
        # 等待消息处理
        time.sleep(0.5)
        
        # 验证消息已处理
        self.assertGreater(len(self.swarm.message_history), initial_count)
    
    def test_broadcast_discovery(self):
        """测试广播发现"""
        self.swarm.register_bee("test_bee_1", self.cognitive)
        
        discovery = {
            "title": "New Pattern",
            "pattern": "singleton",
            "confidence": 0.95
        }
        
        self.swarm.broadcast_discovery("test_bee_1", discovery)
        
        # 等待处理
        time.sleep(0.5)
        
        # 验证消息已记录
        self.assertGreater(len(self.swarm.message_history), 0)
    
    def test_task_negotiation(self):
        """测试任务协商"""
        task = {
            "type": "code_analysis",
            "complexity": "medium",
            "description": "Analyze code structure"
        }
        
        bids = self.swarm.task_negotiation(task)
        
        # 验证返回投标列表
        self.assertIsInstance(bids, list)
    
    def test_share_knowledge(self):
        """测试知识分享"""
        self.swarm.register_bee("test_bee_1", self.cognitive)
        
        knowledge = {
            "topic": "Design Patterns",
            "content": "Singleton pattern implementation",
            "confidence": 0.9
        }
        
        self.swarm.share_knowledge("test_bee_1", knowledge)
        
        # 等待处理
        time.sleep(0.5)
        
        # 验证消息已记录
        self.assertGreater(len(self.swarm.message_history), 0)
    
    def test_get_swarm_status(self):
        """测试获取蜂群状态"""
        status = self.swarm.get_swarm_status()
        
        # 验证状态结构
        self.assertIn("registered_bees", status)
        self.assertIn("active_bees", status)
        self.assertIn("message_count", status)
        self.assertIn("active_tasks", status)
        self.assertIn("running", status)
        
        self.assertTrue(status["running"])


class TestCognitiveBeeSwarmIntegration(unittest.TestCase):
    """测试 CognitiveBee 与 SwarmCommunication 的集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.swarm = get_swarm_communication()
        self.result_queue = multiprocessing.Queue()
        
        # 创建多个认知蜂
        self.cognitive1 = CognitiveBee(
            BeeDNA(species="cognitive"),
            "cognitive_1",
            self.result_queue
        )
        self.cognitive2 = CognitiveBee(
            BeeDNA(species="cognitive"),
            "cognitive_2",
            self.result_queue
        )
        
        # 注册到蜂群
        self.swarm.register_bee("cognitive_1", self.cognitive1)
        self.swarm.register_bee("cognitive_2", self.cognitive2)
    
    def test_cognitive_bees_can_share_assessments(self):
        """测试认知蜂可以分享评估结果"""
        # Bee 1 执行评估
        status1 = self.cognitive1.get_cognitive_status()
        
        # Bee 1 分享评估结果
        self.swarm.share_knowledge("cognitive_1", {
            "topic": "self_assessment",
            "overall_intelligence": status1["overall_intelligence"],
            "blind_spots": status1["blind_spots"]
        })
        
        # 等待消息传播
        time.sleep(0.5)
        
        # 验证消息已广播
        self.assertGreater(len(self.swarm.message_history), 0)
        
        # 查找分享的消息
        knowledge_messages = [
            msg for msg in self.swarm.message_history
            if msg.type == MessageType.KNOWLEDGE
        ]
        self.assertGreater(len(knowledge_messages), 0)
    
    def test_cognitive_bees_can_collaborate_on_learning(self):
        """测试认知蜂可以协作学习"""
        # Bee 1 识别盲区
        blind_spots1 = self.cognitive1.identify_blind_spots()
        
        # Bee 2 识别盲区
        blind_spots2 = self.cognitive2.identify_blind_spots()
        
        # 两个蜂应该识别出相似的盲区
        domains1 = {spot.domain for spot in blind_spots1}
        domains2 = {spot.domain for spot in blind_spots2}
        
        # 应该有重叠
        overlap = domains1 & domains2
        self.assertGreater(len(overlap), 0)
    
    def test_swarm_can_make_collective_decision_on_learning_priority(self):
        """测试蜂群可以集体决策学习优先级"""
        # 准备决策问题
        question = "下一步应该优先学习哪个领域？"
        options = [
            "security_audit",
            "quantum_computing",
            "blockchain_development"
        ]
        
        # 集体决策
        decision = self.swarm.collective_decision(question, options, timeout=2)
        
        # 验证决策结果
        self.assertIn(decision, options)


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.result_queue = multiprocessing.Queue()
        self.dna = BeeDNA(species="cognitive")
        self.cognitive = CognitiveBee(self.dna, "perf_test", self.result_queue)
        self.swarm = get_swarm_communication()
    
    def test_cognitive_bee_assessment_performance(self):
        """测试 CognitiveBee 评估性能"""
        start_time = time.time()
        
        capabilities = self.cognitive.assess_capabilities()
        
        elapsed = time.time() - start_time
        
        # 应该在100ms内完成
        self.assertLess(elapsed, 0.1, f"评估耗时 {elapsed:.3f}s，超过100ms")
        
        print(f"\n✅ 能力评估性能: {elapsed*1000:.2f}ms")
    
    def test_blind_spot_identification_performance(self):
        """测试盲区识别性能"""
        start_time = time.time()
        
        blind_spots = self.cognitive.identify_blind_spots()
        
        elapsed = time.time() - start_time
        
        # 应该在150ms内完成（包含评估）
        self.assertLess(elapsed, 0.15, f"盲区识别耗时 {elapsed:.3f}s，超过150ms")
        
        print(f"✅ 盲区识别性能: {elapsed*1000:.2f}ms")
    
    def test_learning_plan_creation_performance(self):
        """测试学习计划创建性能"""
        start_time = time.time()
        
        plan = self.cognitive.create_learning_plan()
        
        elapsed = time.time() - start_time
        
        # 应该在200ms内完成
        self.assertLess(elapsed, 0.2, f"计划创建耗时 {elapsed:.3f}s，超过200ms")
        
        print(f"✅ 学习计划创建性能: {elapsed*1000:.2f}ms")
    
    def test_message_broadcast_performance(self):
        """测试消息广播性能"""
        message = SwarmMessage(
            id="perf_test",
            type=MessageType.DISCOVERY,
            from_bee="perf_test",
            content={"test": "data"}
        )
        
        start_time = time.time()
        
        self.swarm.broadcast(message)
        
        elapsed = time.time() - start_time
        
        # 应该在10ms内完成
        self.assertLess(elapsed, 0.01, f"消息广播耗时 {elapsed:.3f}s，超过10ms")
        
        print(f"✅ 消息广播性能: {elapsed*1000:.2f}ms")


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
