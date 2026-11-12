# -*- coding: utf-8 -*-
"""
Ministers单元测试

测试IntelBee、EngineerBee和AlchemistBee的所有功能。
"""

import unittest
import sys
from pathlib import Path
import multiprocessing
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hive.ministers import BaseMinister, IntelBee, EngineerBee, AlchemistBee
from hive.core.dna import BeeDNA
from hive.core.interfaces import MinisterInterface


class TestBaseMinister(unittest.TestCase):
    """测试BaseMinister基类"""
    
    def setUp(self):
        """设置测试环境"""
        self.result_queue = multiprocessing.Queue()
        self.dna = BeeDNA(species="base_minister")
        # 使用IntelBee作为BaseMinister的具体实现来测试
        self.minister = IntelBee(self.dna, "test_minister", self.result_queue)
    
    def test_base_minister_initialization(self):
        """测试BaseMinister初始化"""
        self.assertEqual(self.minister.id, "test_minister")
        self.assertEqual(self.minister.dna, self.dna)
        self.assertIsInstance(self.minister.strategies, list)
        self.assertIsInstance(self.minister.decisions, list)
        self.assertIsInstance(self.minister.knowledge_base, dict)
    
    def test_advise_method_exists(self):
        """测试advise方法存在"""
        context = {
            "situation": "test",
            "goals": ["goal1"],
            "constraints": [],
            "resources": {}
        }
        advice = self.minister.advise(context)
        
        self.assertIsInstance(advice, dict)
        self.assertIn("minister", advice)
        self.assertIn("recommendations", advice)
        self.assertIn("priority", advice)
        self.assertIn("risks", advice)
        self.assertIn("timestamp", advice)
    
    def test_execute_strategy_method_exists(self):
        """测试execute_strategy方法存在"""
        strategy = {
            "name": "test_strategy",
            "objectives": ["objective1"],
            "actions": ["action1", "action2"]
        }
        result = self.minister.execute_strategy(strategy)
        
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        # IntelBee返回evaluation字段而非total_tasks/completed_tasks
        self.assertIn("evaluation", result)
    
    def test_execute_strategy_validation(self):
        """测试execute_strategy验证逻辑"""
        # 缺少必需字段
        invalid_strategy = {"name": "test"}
        result = self.minister.execute_strategy(invalid_strategy)
        
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)
    
    def test_get_minister_status(self):
        """测试get_minister_status方法"""
        status = self.minister.get_minister_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("minister_type", status)
        self.assertIn("strategies_executed", status)
        self.assertIn("decisions_made", status)
        self.assertIn("knowledge_items", status)
    
    def test_specialized_task(self):
        """测试specialized_task方法"""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir)
            result = self.minister.specialized_task(target_path)
            
            self.assertIsInstance(result, dict)
            self.assertIn("project_path", result)
            self.assertIn("project_name", result)


class TestIntelBee(unittest.TestCase):
    """测试IntelBee"""
    
    def setUp(self):
        """设置测试环境"""
        self.result_queue = multiprocessing.Queue()
        self.dna = BeeDNA(species="intel")
        self.intel = IntelBee(self.dna, "intel_1", self.result_queue)
    
    def test_intel_initialization(self):
        """测试IntelBee初始化"""
        self.assertEqual(self.intel.id, "intel_1")
        self.assertIn("projects", self.intel.intelligence_db)
        self.assertIn("patterns", self.intel.intelligence_db)
        self.assertIn("insights", self.intel.intelligence_db)
    
    def test_advise_method(self):
        """测试IntelBee.advise()方法"""
        context = {
            "project_info": "Test Project",
            "current_status": "in_progress",
            "goals": ["complete_phase1"]
        }
        
        advice = self.intel.advise(context)
        
        # 验证返回结构
        self.assertEqual(advice["minister"], "IntelBee")
        self.assertIn("intelligence", advice)
        self.assertIn("analysis", advice)
        self.assertIn("patterns", advice)
        self.assertIn("risks", advice)
        self.assertIn("opportunities", advice)
        self.assertIn("recommendations", advice)
        self.assertIn("confidence", advice)
        self.assertIn("priority", advice)
        self.assertIn("timestamp", advice)
        
        # 验证数据类型
        self.assertIsInstance(advice["intelligence"], dict)
        self.assertIsInstance(advice["risks"], list)
        self.assertIsInstance(advice["opportunities"], list)
        self.assertIsInstance(advice["recommendations"], list)
        self.assertIsInstance(advice["patterns"], list)
        self.assertIsInstance(advice["confidence"], float)
        
        # 验证决策被记录
        self.assertEqual(len(self.intel.decisions), 1)
    
    def test_advise_with_complex_context(self):
        """测试复杂上下文下的advise"""
        context = {
            "project_info": {
                "name": "Complex Project",
                "size": "large",
                "team": 10
            },
            "current_status": "critical",
            "goals": ["fix_bugs", "improve_performance"],
            "constraints": ["budget", "time"],
            "resources": {"developers": 5, "time": "2 weeks"}
        }
        
        advice = self.intel.advise(context)
        
        # 验证能够处理复杂上下文
        self.assertEqual(advice["minister"], "IntelBee")
        self.assertGreater(len(advice["intelligence"].get("sources", [])), 0)
    
    def test_execute_strategy_method(self):
        """测试IntelBee.execute_strategy()方法"""
        strategy = {
            "name": "market_expansion",
            "objectives": ["expand_to_new_markets"],
            "actions": ["research", "planning", "execution"]
        }
        
        result = self.intel.execute_strategy(strategy)
        
        # 验证返回结构
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
        self.assertIn("strategy_name", result)
        self.assertEqual(result["strategy_name"], "market_expansion")
        self.assertIn("execution_plan", result)
        self.assertIn("resource_allocation", result)
        self.assertIn("results", result)
        self.assertIn("evaluation", result)
        self.assertIn("lessons_learned", result)
        self.assertIn("timestamp", result)
        
        # 验证执行计划
        execution_plan = result["execution_plan"]
        self.assertIn("phases", execution_plan)
        self.assertEqual(len(execution_plan["phases"]), 3)
        
        # 验证资源分配
        resource_alloc = result["resource_allocation"]
        self.assertEqual(resource_alloc["workers_needed"], 3)
        self.assertEqual(resource_alloc["estimated_effort"], 24)  # 3 * 8
        
        # 验证战略被记录
        self.assertEqual(len(self.intel.strategies), 1)
    
    def test_execute_strategy_validation(self):
        """测试策略验证"""
        # 缺少objectives
        invalid_strategy = {"name": "test", "actions": []}
        result = self.intel.execute_strategy(invalid_strategy)
        
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)
    
    def test_specialized_task(self):
        """测试specialized_task方法"""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir)
            
            # 创建测试文件
            (target_path / "requirements.txt").write_text("pytest\nrequests")
            (target_path / "README.md").write_text("# Test Project")
            
            intelligence = self.intel.specialized_task(target_path)
            
            # 验证返回结构
            self.assertIn("project_path", intelligence)
            self.assertIn("project_name", intelligence)
            self.assertIn("structure", intelligence)
            self.assertIn("technologies", intelligence)
            self.assertIn("quality_metrics", intelligence)
            self.assertIn("potential", intelligence)
            
            # 验证分析结果
            self.assertEqual(intelligence["project_name"], target_path.name)
            self.assertIn("Python", intelligence["technologies"])
            self.assertEqual(intelligence["structure"]["is_directory"], True)
            self.assertEqual(intelligence["structure"]["has_readme"], True)
            
            # 验证情报被存储
            self.assertEqual(len(self.intel.intelligence_db["projects"]), 1)
    
    async def test_assign_task_bee_interface(self):
        """测试BeeInterface.assign_task()方法"""
        from hive.core.interfaces import Task
        
        task = Task(
            id="task_001",
            type="analysis",
            priority=1
        )
        
        result = await self.intel.assign_task(task)
        
        # assign_task 返回 Optional[str] (bee_id 或 None)
        self.assertIsNone(result)  # Ministers 默认不直接执行
    
    def test_get_bee_status_bee_interface(self):
        """测试BeeInterface.get_bee_status()方法"""
        status = self.intel.get_bee_status()
        
        self.assertIsInstance(status, dict)
        self.assertEqual(status["id"], "intel_1")
        self.assertEqual(status["species"], "IntelBee")
        self.assertEqual(status["role"], "minister")
        self.assertEqual(status["status"], "active")
        self.assertIn("intelligence_db_size", status)
        self.assertIn("strategies_executed", status)
    
    def test_get_statistics_bee_interface(self):
        """测试BeeInterface.get_statistics()方法"""
        # 先执行一些操作以生成数据
        context = {"project_info": "test"}
        self.intel.advise(context)
        
        stats = self.intel.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_insights", stats)
        self.assertIn("total_strategies", stats)
        self.assertIn("projects_analyzed", stats)
        self.assertIn("patterns_identified", stats)


class TestEngineerBee(unittest.TestCase):
    """测试EngineerBee"""
    
    def setUp(self):
        """设置测试环境"""
        self.result_queue = multiprocessing.Queue()
        self.dna = BeeDNA(species="engineer")
        self.engineer = EngineerBee(self.dna, "engineer_1", self.result_queue)
    
    def test_engineer_initialization(self):
        """测试EngineerBee初始化"""
        self.assertEqual(self.engineer.id, "engineer_1")
        self.assertIn("architectures", self.engineer.engineering_db)
        self.assertIn("designs", self.engineer.engineering_db)
        self.assertIn("implementations", self.engineer.engineering_db)
    
    def test_advise_method(self):
        """测试EngineerBee.advise()方法"""
        context = {
            "system_requirements": ["requirement1", "requirement2"],
            "constraints": ["constraint1"],
            "current_architecture": {
                "style": "modular",
                "components": ["component1"]
            }
        }
        
        advice = self.engineer.advise(context)
        
        # R11 契约核对：advise() 由 BaseMinister 统一实现，返回分析结果 + 建议 + 风险
        self.assertEqual(advice["minister"], "EngineerBee")
        for key in ("analysis", "recommendations", "priority", "risks",
                    "confidence", "timestamp"):
            self.assertIn(key, advice)
        # 旧契约字段已移除（Ministers 只做战略决策，不再直接做工程评估）
        for legacy in ("requirements_analysis", "architecture_assessment",
                       "technical_debt", "improvements", "feasibility"):
            self.assertNotIn(legacy, advice)
        
        # 验证数据类型
        self.assertIsInstance(advice["analysis"], dict)
        self.assertIsInstance(advice["recommendations"], list)
        self.assertIsInstance(advice["risks"], list)
        self.assertIsInstance(advice["priority"], str)
        
        # 验证决策被记录
        self.assertEqual(len(self.engineer.decisions), 1)
    
    def test_advise_with_technical_debt(self):
        """R11 契约核对：技术债务评估已从 advise() 移除，风险统一走 risks 通道。"""
        context = {
            "system_requirements": [],
            "constraints": [],
            "current_architecture": {
                "style": "monolithic",
                "modularity_score": 0.5  # 低模块化分数
            }
        }
        
        advice = self.engineer.advise(context)
        
        self.assertNotIn("technical_debt", advice)
        self.assertIsInstance(advice["risks"], list)
        self.assertTrue(advice["risks"])
        risk = advice["risks"][0]
        for key in ("type", "severity", "description"):
            self.assertIn(key, risk)
    
    def test_execute_strategy_method(self):
        """R11 契约核对：execute_strategy() 现返回 _aggregate_results 的汇总结果。"""
        strategy = {
            "name": "architecture_refactor",
            "objectives": ["improve_modularity", "reduce_coupling"],
            "architecture": {
                "components": ["comp1", "comp2"]
            }
        }
        
        result = self.engineer.execute_strategy(strategy)
        
        self.assertEqual(result["status"], "success")
        for key in ("total_tasks", "completed_tasks", "success_rate"):
            self.assertIn(key, result)
        # 旧契约字段属于「Minister 直接实现架构」的旧版本，现只做任务分解汇总
        for legacy in ("strategy_name", "architecture", "implementation_plan",
                       "implementation_results", "validation", "quality_metrics",
                       "timestamp"):
            self.assertNotIn(legacy, result)
        
        # 无 tasks 字段时分解出 0 个任务，仍判 success / success_rate=0
        self.assertEqual(result["total_tasks"], 0)
        self.assertEqual(result["completed_tasks"], 0)
        
        # 验证战略被记录
        self.assertEqual(len(self.engineer.strategies), 1)
    
    def test_specialized_task(self):
        """R11 契约核对：Ministers 不直接执行任务，统一返回 delegated 委托语义。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir)
            
            src_dir = target_path / "src"
            src_dir.mkdir()
            tests_dir = target_path / "tests"
            tests_dir.mkdir()
            
            analysis = self.engineer.specialized_task(target_path)
            
            self.assertEqual(analysis["status"], "delegated")
            self.assertEqual(analysis["message"], "Ministers delegate tasks to Workers")
            for legacy in ("project_path", "architecture_type", "components",
                           "dependencies", "quality_assessment", "recommendations"):
                self.assertNotIn(legacy, analysis)
    
    async def test_assign_task_bee_interface(self):
        """测试BeeInterface.assign_task()方法"""
        from hive.core.interfaces import Task
        
        task = Task(
            id="task_002",
            type="engineering",
            priority=1
        )
        
        result = await self.engineer.assign_task(task)
        
        # assign_task 返回 Optional[str] (bee_id 或 None)
        self.assertIsNone(result)  # Ministers 默认不直接执行
    
    def test_get_bee_status_bee_interface(self):
        """R11 契约核对：EngineerBee 重写 get_bee_status()，返回工程蜂运行态。"""
        status = self.engineer.get_bee_status()
        
        self.assertIsInstance(status, dict)
        self.assertEqual(status["id"], "engineer_1")
        for key in ("name", "active", "mutation_pool_size"):
            self.assertIn(key, status)
        # 旧契约字段来自 MinisterInterface 默认实现，EngineerBee 已重写
        for legacy in ("species", "role", "engineering_db_size"):
            self.assertNotIn(legacy, status)
    
    def test_get_statistics_bee_interface(self):
        """R11 契约核对：get_statistics() = get_minister_status() + 工程统计。"""
        context = {"system_requirements": []}
        self.engineer.advise(context)
        
        stats = self.engineer.get_statistics()
        
        self.assertIsInstance(stats, dict)
        for key in ("total_assessments", "total_strategies",
                    "history_cache_size", "minister_type", "species"):
            self.assertIn(key, stats)
        self.assertEqual(stats["minister_type"], "EngineerBee")
        self.assertEqual(stats["decisions_made"], 1)
        for legacy in ("projects_analyzed", "architectures_designed"):
            self.assertNotIn(legacy, stats)


class TestAlchemistBee(unittest.TestCase):
    """测试AlchemistBee"""
    
    def setUp(self):
        """设置测试环境"""
        self.result_queue = multiprocessing.Queue()
        self.dna = BeeDNA(species="alchemist")
        self.alchemist = AlchemistBee(self.dna, "alchemist_1", self.result_queue)
    
    def test_alchemist_initialization(self):
        """测试AlchemistBee初始化"""
        self.assertEqual(self.alchemist.id, "alchemist_1")
        self.assertIn("recipes", self.alchemist.alchemy_db)
        self.assertIn("transformations", self.alchemist.alchemy_db)
        self.assertIn("optimizations", self.alchemist.alchemy_db)
    
    def test_advise_method(self):
        """测试AlchemistBee.advise()方法"""
        context = {
            "code_analysis": {
                "complexity": "high",
                "maintainability": 0.6,
                "duplication": 0.2,
                "test_coverage": 0.5
            },
            "optimization_goals": ["performance", "maintainability"],
            "constraints": []
        }
        
        advice = self.alchemist.advise(context)
        
        # 验证返回结构
        self.assertEqual(advice["minister"], "AlchemistBee")
        self.assertIn("code_quality", advice)
        self.assertIn("optimization_opportunities", advice)
        self.assertIn("transformation_options", advice)
        self.assertIn("optimization_plan", advice)
        self.assertIn("estimated_benefits", advice)
        self.assertIn("recommendations", advice)
        self.assertIn("confidence", advice)
        self.assertIn("timestamp", advice)
        
        # 验证数据类型
        self.assertIsInstance(advice["code_quality"], dict)
        self.assertIsInstance(advice["optimization_opportunities"], list)
        self.assertIsInstance(advice["transformation_options"], list)
        self.assertIsInstance(advice["recommendations"], list)
        
        # 验证决策被记录
        self.assertEqual(len(self.alchemist.decisions), 1)
    
    def test_advise_identifies_optimizations(self):
        """测试识别优化机会"""
        context = {
            "code_analysis": {
                "complexity": "high",
                "maintainability": 0.5,
                "duplication": 0.2,
                "test_coverage": 0.4
            },
            "optimization_goals": [],
            "constraints": []
        }
        
        advice = self.alchemist.advise(context)
        
        # 应该识别出多个优化机会
        opportunities = advice["optimization_opportunities"]
        self.assertGreater(len(opportunities), 0)
        
        # 验证识别出的问题
        opportunity_types = [opp["type"] for opp in opportunities]
        self.assertIn("reduce_complexity", opportunity_types)
        self.assertIn("remove_duplication", opportunity_types)
        self.assertIn("increase_coverage", opportunity_types)
    
    def test_execute_strategy_method(self):
        """测试AlchemistBee.execute_strategy()方法"""
        strategy = {
            "name": "code_optimization",
            "objectives": ["improve_performance"],
            "transformations": ["refactor", "optimize", "test"]
        }
        
        result = self.alchemist.execute_strategy(strategy)
        
        # 验证返回结构
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
        self.assertIn("strategy_name", result)
        self.assertEqual(result["strategy_name"], "code_optimization")
        self.assertIn("transformation_plan", result)
        self.assertIn("transformation_results", result)
        self.assertIn("validation", result)
        self.assertIn("optimized_output", result)
        self.assertIn("quality_improvement", result)
        self.assertIn("timestamp", result)
        
        # 验证转换结果
        validation = result["validation"]
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["total_transformations"], 3)
        
        # 验证优化输出
        optimized = result["optimized_output"]
        self.assertIn("optimized_code", optimized)
        self.assertIn("quality_metrics", optimized)
        
        # 验证改进程度
        improvement = result["quality_improvement"]
        self.assertIn("code_quality_delta", improvement)
        self.assertIn("performance_delta", improvement)
        
        # 验证战略被记录
        self.assertEqual(len(self.alchemist.strategies), 1)
    
    def test_specialized_task(self):
        """测试specialized_task方法"""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir)
            
            # 创建Python文件
            (target_path / "script.py").write_text("print('hello')")
            
            optimization = self.alchemist.specialized_task(target_path)
            
            # 验证返回结构
            self.assertIn("project_path", optimization)
            self.assertIn("code_metrics", optimization)
            self.assertIn("optimization_opportunities", optimization)
            self.assertIn("refactoring_suggestions", optimization)
            self.assertIn("estimated_improvement", optimization)
            
            # 验证代码指标
            metrics = optimization["code_metrics"]
            self.assertIn("total_lines", metrics)
            self.assertIn("code_lines", metrics)
            self.assertIn("complexity_score", metrics)
            
            # 验证优化被存储
            self.assertEqual(len(self.alchemist.alchemy_db["optimizations"]), 1)
    
    async def test_assign_task_bee_interface(self):
        """测试BeeInterface.assign_task()方法"""
        from hive.core.interfaces import Task
        
        task = Task(
            id="task_003",
            type="optimization",
            priority=1
        )
        
        result = await self.alchemist.assign_task(task)
        
        # assign_task 返回 Optional[str] (bee_id 或 None)
        self.assertIsNone(result)  # Ministers 默认不直接执行
    
    def test_get_bee_status_bee_interface(self):
        """测试BeeInterface.get_bee_status()方法"""
        status = self.alchemist.get_bee_status()
        
        self.assertIsInstance(status, dict)
        self.assertEqual(status["id"], "alchemist_1")
        self.assertEqual(status["species"], "AlchemistBee")
        self.assertEqual(status["role"], "minister")
        self.assertIn("alchemy_db_size", status)
    
    def test_get_statistics_bee_interface(self):
        """测试BeeInterface.get_statistics()方法"""
        # 先执行一些操作
        context = {"code_analysis": {}}
        self.alchemist.advise(context)
        
        stats = self.alchemist.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_optimizations", stats)
        self.assertIn("total_strategies", stats)
        self.assertIn("projects_analyzed", stats)
        self.assertIn("code_transformations", stats)


class TestMinistersIntegration(unittest.TestCase):
    """测试Ministers集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.result_queue = multiprocessing.Queue()
        self.dna = BeeDNA(species="minister")
        self.intel = IntelBee(self.dna, "intel_1", self.result_queue)
        self.engineer = EngineerBee(self.dna, "engineer_1", self.result_queue)
        self.alchemist = AlchemistBee(self.dna, "alchemist_1", self.result_queue)
    
    async def test_all_ministers_implement_interface(self):
        """测试所有Ministers都实现了BeeInterface"""
        from hive.core.interfaces import Task
        ministers = [self.intel, self.engineer, self.alchemist]
        
        for minister in ministers:
            # 测试assign_task
            task = Task(id="test", type="test", priority=1)
            result = await minister.assign_task(task)
            self.assertIsNone(result)  # Ministers默认不直接执行
            
            # 测试get_bee_status
            status = minister.get_bee_status()
            self.assertIn("id", status)
            self.assertIn("species", status)
            self.assertIn("role", status)
            
            # 测试get_statistics
            stats = minister.get_statistics()
            self.assertIsInstance(stats, dict)
    
    def test_all_ministers_advise_with_different_contexts(self):
        """测试所有Ministers能够处理不同的上下文"""
        ministers = [self.intel, self.engineer, self.alchemist]
        
        for minister in ministers:
            context = {
                "test_data": "data",
                "test_value": 123,
                "test_list": [1, 2, 3]
            }
            
            advice = minister.advise(context)
            
            self.assertEqual(advice["minister"], minister.__class__.__name__)
            self.assertIn("recommendations", advice)
            self.assertIn("confidence", advice)
    
    def test_all_ministers_execute_strategy(self):
        """测试所有Ministers能够执行策略"""
        ministers = [self.intel, self.engineer, self.alchemist]
        
        # 为每个Minister提供适当的策略参数
        strategies = [
            {  # IntelBee策略
                "name": "test_strategy",
                "objectives": ["objective1"],
                "actions": ["action1", "action2"]
            },
            {  # EngineerBee策略
                "name": "test_strategy",
                "objectives": ["objective1"],
                "architecture": {
                    "components": ["comp1"]
                }
            },
            {  # AlchemistBee策略
                "name": "test_strategy",
                "objectives": ["objective1"],
                "transformations": ["transformation1"]
            }
        ]
        
        for minister, strategy in zip(ministers, strategies):
            result = minister.execute_strategy(strategy)
            
            self.assertIn("status", result)
            # 接受success或partial作为有效状态（IntelBee可能返回partial）
            self.assertIn(result["status"], ["success", "partial"])
            self.assertEqual(len(minister.strategies), 1)
    
    def test_ministers_inheritance(self):
        """测试Ministers的继承关系"""
        ministers = [self.intel, self.engineer, self.alchemist]
        
        for minister in ministers:
            # 验证继承自BaseMinister
            self.assertIsInstance(minister, BaseMinister)
            
            # 验证实现MinisterInterface
            self.assertIsInstance(minister, MinisterInterface)
            
            # 验证重写的方法
            self.assertTrue(hasattr(minister, 'advise'))
            self.assertTrue(hasattr(minister, 'execute_strategy'))
            self.assertTrue(hasattr(minister, 'specialized_task'))
    
    def test_ministers_knowledge_accumulation(self):
        """测试Ministers的知识积累"""
        ministers = [(self.intel, {"name": "test", "objectives": ["obj1"], "actions": ["a1", "a2"]}),
                     (self.engineer, {"name": "test", "objectives": ["obj1"], "architecture": {"components": ["c1"]}}),
                     (self.alchemist, {"name": "test", "objectives": ["obj1"], "transformations": ["t1"]})]
        
        for minister, strategy in ministers:
            # 初始状态
            self.assertEqual(len(minister.decisions), 0)
            self.assertEqual(len(minister.strategies), 0)
            
            # 执行advise
            minister.advise({"test": "data"})
            self.assertEqual(len(minister.decisions), 1)
            
            # 执行execute_strategy
            minister.execute_strategy(strategy)
            self.assertEqual(len(minister.strategies), 1)


class TestMinistersEdgeCases(unittest.TestCase):
    """测试Ministers边界情况"""
    
    def setUp(self):
        """设置测试环境"""
        self.result_queue = multiprocessing.Queue()
        self.dna = BeeDNA(species="test")
        self.intel = IntelBee(self.dna, "intel_1", self.result_queue)
    
    def test_advise_with_empty_context(self):
        """测试空上下文"""
        advice = self.intel.advise({})
        
        self.assertEqual(advice["minister"], "IntelBee")
        self.assertIn("recommendations", advice)
    
    def test_advise_with_large_context(self):
        """测试大型上下文"""
        large_context = {f"key_{i}": f"value_{i}" for i in range(100)}
        
        advice = self.intel.advise(large_context)
        
        self.assertEqual(advice["minister"], "IntelBee")
        self.assertIn("intelligence", advice)
    
    def test_execute_strategy_with_invalid_strategy(self):
        """测试无效策略"""
        invalid_strategies = [
            {},  # 空策略
            {"name": "test"},  # 缺少objectives
            {"objectives": []},  # 缺少name
        ]
        
        for strategy in invalid_strategies:
            result = self.intel.execute_strategy(strategy)
            self.assertEqual(result["status"], "failed")
    
    def test_specialized_task_with_nonexistent_path(self):
        """测试不存在的路径"""
        nonexistent_path = Path("/nonexistent/path")
        
        intelligence = self.intel.specialized_task(nonexistent_path)
        
        self.assertIn("project_path", intelligence)
    
    def test_multiple_strategies_execution(self):
        """测试多次策略执行"""
        strategies = [
            {"name": f"strategy_{i}", "objectives": ["obj1"], "tasks": []}
            for i in range(5)
        ]
        
        for strategy in strategies:
            self.intel.execute_strategy(strategy)
        
        self.assertEqual(len(self.intel.strategies), 5)


if __name__ == '__main__':
    # 运行测试并生成覆盖率报告
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印测试统计
    print("\n" + "="*70)
    print("测试统计:")
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    print("="*70)
    
    # 退出码
    sys.exit(0 if result.wasSuccessful() else 1)
