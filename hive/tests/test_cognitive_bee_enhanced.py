# -*- coding: utf-8 -*-
"""测试 CognitiveBee 增强功能

测试内容:
1. 历史性能分析 (compare_with_history, get_all_historical_trends)
2. 改进建议生成 (analyze_improvement_recommendations)
3. 主动学习触发器 (check_active_learning_triggers, execute_active_learning)
4. 元学习能力 (record_learning_outcome, recommend_learning_method)
5. 与进化系统集成 (report_to_evolution_coordinator)
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil
import json
import time

from hive.bees.cognitive_bee import (
    CognitiveBee,
    CapabilityAssessment,
    HistoricalComparison,
    ImprovementRecommendation,
    ActiveLearningTrigger,
    BlindSpot,
    LearningPlan,
)
from hive.core.dna import BeeDNA


class TestCapabilityAssessment:
    """测试能力评估数据类"""

    def test_capability_assessment_defaults(self):
        """测试默认值"""
        assessment = CapabilityAssessment(
            capability_name="test_capability",
            score=0.85,
        )
        
        assert assessment.capability_name == "test_capability"
        assert assessment.score == 0.85
        assert assessment.confidence == 1.0
        assert assessment.is_uncertain == False
        assert assessment.evidence == []
        assert assessment.last_updated is not None
        assert assessment.improvement_rate == 0.0

    def test_capability_assessment_with_evidence(self):
        """测试带有证据的评估"""
        assessment = CapabilityAssessment(
            capability_name="code_generation",
            score=0.90,
            confidence=0.95,
            is_uncertain=False,
            evidence=["成功生成 500+ 任务", "代码质量良好"],
            improvement_rate=0.05,
        )
        
        assert len(assessment.evidence) == 2
        assert assessment.improvement_rate == 0.05


class TestHistoricalComparison:
    """测试历史对比功能"""

    def test_compare_with_history_no_history(self):
        """测试无历史数据时返回 None"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_001")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_001", queue)
            
            result = bee.compare_with_history("code_generation")
            assert result is None

    def test_compare_with_history_with_data(self):
        """测试有历史数据时的对比"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_002")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_002", queue)
            
            older = CapabilityAssessment(
                capability_name="code_generation",
                score=0.70,
                confidence=0.80,
                last_updated=datetime.now() - timedelta(days=7),
            )
            
            newer = CapabilityAssessment(
                capability_name="code_generation",
                score=0.80,
                confidence=0.85,
                last_updated=datetime.now(),
            )
            
            bee.capability_history["code_generation"] = [older, newer]
            
            comparison = bee.compare_with_history("code_generation")
            
            assert comparison is not None
            assert comparison.capability_name == "code_generation"
            assert comparison.current_score == 0.80
            assert comparison.previous_score == 0.70
            assert comparison.score_delta == pytest.approx(0.10)
            assert comparison.improvement_trend == "improving"

    def test_compare_with_history_declining(self):
        """测试下降趋势"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_003")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_003", queue)
            
            older = CapabilityAssessment(
                capability_name="bug_detection",
                score=0.80,
                confidence=0.80,
                last_updated=datetime.now() - timedelta(days=7),
            )
            
            newer = CapabilityAssessment(
                capability_name="bug_detection",
                score=0.60,
                confidence=0.65,
                last_updated=datetime.now(),
            )
            
            bee.capability_history["bug_detection"] = [older, newer]
            
            comparison = bee.compare_with_history("bug_detection")
            
            assert comparison.improvement_trend == "declining"
            assert comparison.score_delta == pytest.approx(-0.20)

    def test_get_all_historical_trends(self):
        """测试获取所有历史趋势"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_004")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_004", queue)
            
            bee.capability_history["code_generation"] = [
                CapabilityAssessment("code_generation", 0.70, 0.80, last_updated=datetime.now() - timedelta(days=7)),
                CapabilityAssessment("code_generation", 0.80, 0.85, last_updated=datetime.now()),
            ]
            
            bee.capability_history["code_analysis"] = [
                CapabilityAssessment("code_analysis", 0.75, 0.80, last_updated=datetime.now() - timedelta(days=7)),
                CapabilityAssessment("code_analysis", 0.76, 0.81, last_updated=datetime.now()),
            ]
            
            trends = bee.get_all_historical_trends()
            
            assert "code_generation" in trends
            assert trends["code_generation"].improvement_trend == "improving"


class TestImprovementRecommendations:
    """测试改进建议生成"""

    def test_analyze_improvement_recommendations_generates_list(self):
        """测试生成改进建议列表"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_005")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_005", queue)
            
            bee.memory["capabilities"]["low_score"] = {"score": 0.35, "improvement_rate": 0.01}
            bee.memory["capabilities"]["mid_score"] = {"score": 0.55, "improvement_rate": 0.02}
            bee.memory["capabilities"]["high_score"] = {"score": 0.85, "improvement_rate": 0.03}
            
            recommendations = bee.analyze_improvement_recommendations()
            
            assert len(recommendations) > 0
            for rec in recommendations:
                assert isinstance(rec, ImprovementRecommendation)
                assert rec.priority in ["critical", "high", "medium", "low"]

    def test_recommendations_sorted_by_priority(self):
        """测试建议按优先级排序"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_006")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_006", queue)
            
            recommendations = bee.analyze_improvement_recommendations()
            
            if len(recommendations) > 1:
                priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
                for i in range(len(recommendations) - 1):
                    current_prio = priority_order[recommendations[i].priority]
                    next_prio = priority_order[recommendations[i + 1].priority]
                    assert current_prio <= next_prio


class TestActiveLearningTriggers:
    """测试主动学习触发器"""

    def test_check_active_learning_triggers_finds_triggers(self):
        """测试检查主动学习触发器"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_007")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_007", queue)
            
            triggers = bee.check_active_learning_triggers()
            
            for trigger in triggers:
                assert isinstance(trigger, ActiveLearningTrigger)
                assert trigger.trigger_type in [
                    "low_score", "high_uncertainty", "knowledge_gap", "trend_down"
                ]
                assert trigger.severity in ["critical", "high", "medium"]

    def test_missing_domains_trigger_knowledge_gap(self):
        """测试缺失领域触发知识缺口"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_008")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_008", queue)
            
            bee.memory["missing_domains"] = ["quantum_computing", "blockchain"]
            
            triggers = bee.check_active_learning_triggers()
            
            knowledge_gap_triggers = [t for t in triggers if t.trigger_type == "knowledge_gap"]
            assert len(knowledge_gap_triggers) >= 2

    def test_execute_active_learning(self):
        """测试执行主动学习"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_009")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_009", queue)
            
            trigger = ActiveLearningTrigger(
                trigger_id="trigger_test_001",
                trigger_type="low_score",
                capability_name="test_capability",
                current_score=0.35,
                confidence=0.50,
                severity="critical",
                triggered_at=datetime.now(),
                suggested_learning_task={
                    "type": "capability_improvement",
                    "actions": ["深度学习", "实践任务"],
                    "goal": "提升分数",
                },
            )
            
            result = bee.execute_active_learning(trigger)
            
            assert result["trigger_id"] == "trigger_test_001"
            assert result["capability"] == "test_capability"
            assert result["status"] == "completed"
            assert len(result["actions_executed"]) == 2


class TestMetaLearning:
    """测试元学习能力"""

    def test_record_learning_outcome(self):
        """测试记录学习结果"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_010")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_010", queue)
            
            bee.record_learning_outcome(
                capability_name="code_generation",
                success=True,
                effort_hours=2.0,
                improvement=0.10,
            )
            
            efficiencies = bee.meta_learning_stats.get("learning_efficiency", {})
            assert "code_generation" in efficiencies

    def test_recommend_learning_method(self):
        """测试推荐学习方法"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_011")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_011", queue)
            
            bee.meta_learning_stats["learning_efficiency"] = {
                "code_generation": 0.90,
                "pattern_mining": 0.85,
                "code_reading": 0.70,
            }
            
            method, efficiency = bee.recommend_learning_method("test_capability")
            
            assert method in ["pattern_mining", "code_reading", "guided_learning", "trial_error", "code_generation"]
            assert isinstance(efficiency, float)


class TestEnhancedReport:
    """测试增强版报告"""

    def test_generate_enhanced_report(self):
        """测试生成增强版报告"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_012")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_012", queue)
            
            test_path = Path(tempfile.mkdtemp())
            
            try:
                report = bee.generate_enhanced_report(test_path)
                
                assert report["report_version"] == "2.0"
                assert "enhanced" in report
                assert "historical_trends" in report["enhanced"]
                assert "improvement_recommendations" in report["enhanced"]
                assert "active_learning_triggers" in report["enhanced"]
                assert "meta_learning" in report["enhanced"]
                
            finally:
                if test_path.exists():
                    shutil.rmtree(test_path)


class TestReportToEvolution:
    """测试报告到进化协调器"""

    def test_report_to_evolution_coordinator_import_error(self):
        """测试进化协调器不可用时的处理"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_013")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_013", queue)
            
            result = bee.report_to_evolution_coordinator()
            
            assert result["reported"] == False
            assert result["reason"] == "coordinator_unavailable"


class TestCognitiveBeeIntegration:
    """测试 CognitiveBee 集成场景"""

    def test_full_cognitive_flow(self):
        """测试完整的认知流程"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_014")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_014", queue)
            
            capabilities = bee.assess_capabilities()
            assert len(capabilities) > 0
            
            blind_spots = bee.identify_blind_spots()
            assert isinstance(blind_spots, list)
            
            plan = bee.create_learning_plan()
            assert isinstance(plan, LearningPlan)
            assert len(plan.targets) > 0
            
            status = bee.get_cognitive_status()
            assert status["bee_type"] == "CognitiveBee"
            assert "overall_intelligence" in status

    def test_update_capability(self):
        """测试更新能力分数"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_015")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_015", queue)
            
            original_score = bee.memory["capabilities"].get("test_domain", {}).get("score", 0.0)
            
            bee.update_capability("test_domain", score_delta=0.1)
            
            new_score = bee.memory["capabilities"]["test_domain"]["score"]
            assert new_score == original_score + 0.1

    def test_get_cognitive_status_includes_uncertainty(self):
        """测试认知状态包含不确定性"""
        with patch('hive.bees.base_bee.get_logger'):
            dna = BeeDNA(species="CognitiveBee", id="test_cog_016")
            queue = MagicMock()
            bee = CognitiveBee(dna, "cog_016", queue)
            
            status = bee.get_cognitive_status()
            
            assert "uncertain_capabilities" in status
            assert "cognitive_uncertainty" in status
            assert "capabilities" in status
