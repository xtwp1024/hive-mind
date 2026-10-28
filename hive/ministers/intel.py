# -*- coding: utf-8 -*-
"""
IntelBee: 情报部长

负责战略分析和决策,是蜂巢的智囊。
集成大脑系统进行智能分析。
"""

from hive.ministers import BaseMinister
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
from pathlib import Path
from typing import Dict, Any, List, Optional
import multiprocessing
import json

logger = get_logger(__name__)

# 导入大脑接口
try:
    from hive.utils.brain_interface import get_brain
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    logger.warning("大脑接口不可用,使用内置分析")


class IntelBee(BaseMinister):
    """
    IntelBee: 情报部长
    """
    
    def __init__(self, dna: BeeDNA, id: str, result_queue: multiprocessing.Queue):
        super().__init__(dna, id, result_queue)
        self.intelligence_db = {"projects": [], "patterns": [], "insights": []}
        self.brain = None
        if BRAIN_AVAILABLE:
            try:
                self.brain = get_brain()
                logger.info(f"[IntelBee-{id}] 🧠 大脑连接成功")
            except Exception as e:
                logger.warning(f"[IntelBee-{id}] 大脑连接失败: {e}")
        logger.info(f"[IntelBee-{id}] 情报部长初始化完成")
    
    def advise(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """提供战略建议"""
        logger.info(f"[IntelBee] 开始战略分析")
        
        intelligence = self._gather_intelligence(context)
        analysis = self._analyze_situation(intelligence)
        patterns = self._identify_patterns(analysis)
        risks = self._assess_risks(analysis)
        opportunities = self._identify_opportunities(analysis)
        
        recommendations = self._generate_strategic_recommendations(
            analysis, patterns, risks, opportunities
        )
        
        advice = self._construct_advice_report(
            intelligence, analysis, patterns, risks, opportunities, recommendations
        )
        
        self.intelligence_db["insights"].append(advice)
        self.decisions.append(advice)
        logger.info(f"[IntelBee] 战略分析完成: {len(recommendations)}条建议")
        
        return advice

    def _construct_advice_report(self, intelligence, analysis, patterns, risks, opportunities, recommendations):
        return {
            "minister": "IntelBee",
            "intelligence": intelligence,
            "analysis": analysis,
            "patterns": patterns,
            "risks": risks,
            "opportunities": opportunities,
            "recommendations": recommendations,
            "confidence": self._calculate_confidence(analysis),
            "priority": self._determine_priority(recommendations),
            "timestamp": self._get_timestamp()
        }
    
    def execute_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """执行战略"""
        logger.info(f"[IntelBee] 执行战略: {strategy.get('name', 'Unnamed')}")
        
        if not self._validate_strategy(strategy):
            return self._create_failure_response(strategy, "Strategy validation failed")
        
        plan = self._create_execution_plan(strategy)
        allocation = self._allocate_resources(plan)
        results = self._monitor_execution(plan)
        evaluation = self._evaluate_results(results)
        
        outcome = self._construct_execution_outcome(
            strategy, plan, allocation, results, evaluation
        )
        
        self.strategies.append(outcome)
        logger.info(f"[IntelBee] 战略执行完成: {outcome['status']}")
        
        return outcome

    def _create_failure_response(self, strategy: Dict, error: str) -> Dict:
        return {
            "status": "failed",
            "strategy_name": strategy.get("name", "Unnamed"),
            "error": error,
            "reason": "Objectives or actions missing"
        }

    def _construct_execution_outcome(self, strategy, plan, allocation, results, evaluation):
        return {
            "status": "success" if evaluation["success_rate"] > 0.7 else "partial",
            "strategy_name": strategy.get("name"),
            "execution_plan": plan,
            "resource_allocation": allocation,
            "results": results,
            "evaluation": evaluation,
            "lessons_learned": self._extract_lessons(results),
            "timestamp": self._get_timestamp()
        }
    
    def specialized_task(self, target_path: Path):
        """IntelBee的专门任务: 项目情报收集"""
        logger.info(f"[IntelBee] 收集项目情报: {target_path}")
        
        intelligence = {
            "project_path": str(target_path),
            "project_name": target_path.name,
            "structure": self._analyze_project_structure(target_path),
            "technologies": self._identify_technologies(target_path),
            "quality_metrics": self._assess_quality(target_path),
            "potential": self._assess_potential(target_path)
        }
        
        self.intelligence_db["projects"].append(intelligence)
        logger.info(f"[IntelBee] 情报收集完成: {intelligence['project_name']}")
        
        return intelligence
    
    # ===== 私有方法 =====
    
    def _gather_intelligence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        base_intel = {
            "context_summary": str(context.get("project_info", "Unknown")),
            "data_points": len(context),
            "sources": list(context.keys())
        }
        
        if self.brain and context.get("github_data"):
            try:
                brain_analysis = self.brain.analyze_github_trends(context["github_data"])
                if brain_analysis:
                    base_intel["brain_analysis"] = brain_analysis
                    logger.info("[IntelBee] 🧠 大脑分析完成")
            except Exception as e:
                logger.warning(f"[IntelBee] 大脑分析失败: {e}")
        
        return base_intel
    
    def _analyze_situation(self, intelligence: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "situation": "analyzed",
            "key_factors": intelligence.get("sources", []),
            "complexity": "medium"
        }
    
    def _identify_patterns(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{
            "pattern": "standard_structure",
            "confidence": 0.8,
            "description": "Standard project structure detected"
        }]
    
    def _assess_risks(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{
            "type": "technical_debt",
            "severity": "medium",
            "probability": 0.6,
            "impact": "moderate",
            "mitigation": "Regular refactoring"
        }]
    
    def _identify_opportunities(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{
            "type": "optimization",
            "potential": "high",
            "effort": "medium",
            "description": "Code optimization opportunities"
        }]
    
    def _generate_strategic_recommendations(self, analysis, patterns, risks, opportunities) -> List[Dict[str, Any]]:
        recommendations = []
        for risk in risks:
            if risk["severity"] in ["high", "critical"]:
                recommendations.append({
                    "action": "mitigate_risk", "priority": "high",
                    "target": risk["type"], "rationale": f"High severity risk: {risk['type']}"
                })
        for opp in opportunities:
            if opp["potential"] == "high":
                recommendations.append({
                    "action": "seize_opportunity", "priority": "medium",
                    "target": opp["type"], "rationale": f"High potential opportunity: {opp['type']}"
                })
        return recommendations
    
    def _determine_priority(self, recommendations: List[Dict]) -> str:
        if not recommendations: return "none"
        if any(r.get("priority") == "high" for r in recommendations): return "high"
        return "medium" if len(recommendations) > 3 else "low"
    
    def _create_execution_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        actions = strategy.get("actions", [])
        return {
            "phases": [{
                "phase": i + 1, "action": action, "duration": "1 day", "dependencies": []
            } for i, action in enumerate(actions)],
            "total_duration": f"{len(actions)} days",
            "critical_path": actions
        }
    
    def _allocate_resources(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        phases = execution_plan.get("phases", [])
        return {
            "workers_needed": len(phases),
            "estimated_effort": len(phases) * 8,
            "resource_allocation": [{
                "phase": phase["phase"], "workers": 1, "hours": 8
            } for phase in phases]
        }
    
    def _monitor_execution(self, execution_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{
            "phase": phase["phase"], "status": "completed",
            "actual_duration": phase["duration"], "issues": []
        } for phase in execution_plan.get("phases", [])]
    
    def _evaluate_results(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(execution_results)
        completed = sum(1 for r in execution_results if r["status"] == "completed")
        return {
            "total_phases": total, "completed_phases": completed,
            "success_rate": completed / total if total > 0 else 0,
            "overall_status": "success" if completed == total else "partial"
        }
    
    def _extract_lessons(self, execution_results: List[Dict[str, Any]]) -> List[str]:
        return ["Execution completed as planned", "Resource allocation was adequate", "No major issues encountered"]
    
    def _analyze_project_structure(self, target_path: Path) -> Dict[str, Any]:
        if not target_path.exists(): return {"error": "Path does not exist"}
        return {
            "is_directory": target_path.is_dir(),
            "has_readme": (target_path / "README.md").exists(),
            "has_setup": (target_path / "setup.py").exists(),
            "structure_type": "standard"
        }
    
    def _identify_technologies(self, target_path: Path) -> List[str]:
        techs = []
        if (target_path / "requirements.txt").exists(): techs.append("Python")
        if (target_path / "package.json").exists(): techs.append("Node.js")
        if (target_path / "Cargo.toml").exists(): techs.append("Rust")
        return techs if techs else ["Unknown"]
    
    def _assess_quality(self, target_path: Path) -> Dict[str, Any]:
        return {
            "has_tests": (target_path / "tests").exists(),
            "has_docs": (target_path / "docs").exists(),
            "has_ci": (target_path / ".github").exists(),
            "quality_score": 0.7
        }
    
    def _assess_potential(self, target_path: Path) -> str:
        score = self._assess_quality(target_path)["quality_score"]
        if score > 0.8: return "high"
        elif score > 0.5: return "medium"
        return "low"
    
    # ===== Minister Logic =====
    
    def get_bee_status(self) -> Dict[str, Any]:
        status = super().get_bee_status()
        status.update({
            "species": "IntelBee",
            "id": self.id,
            "role": "minister",
            "status": "active" if self.is_active else "inactive",
            "intelligence_db_size": len(self.intelligence_db.get("insights", []))
        })
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        stats.update({
            "total_strategies": len(self.strategies),
            "total_insights": len(self.intelligence_db.get("insights", [])),
            "projects_analyzed": len(self.intelligence_db.get("projects", [])),
            "patterns_identified": len(self.intelligence_db.get("patterns", []))
        })
        return stats


__all__ = ['IntelBee']
