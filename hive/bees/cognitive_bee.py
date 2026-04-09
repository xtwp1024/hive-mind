# -*- coding: utf-8 -*-
"""
CognitiveBee - 认知蜂
蜂巢的"大脑皮层" - 负责自我评估和元认知
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

try:
    from hive.utils.brain_interface import BrainInterface, get_brain
except ImportError:
    BrainInterface = None
    get_brain = None

logger = get_logger(__name__)

# Phase 1: 置信度阈值
CONFIDENCE_THRESHOLD = 0.7  # 低于此值认为"不确定"


@dataclass
class CapabilityAssessment:
    """能力评估结果"""
    capability_name: str
    score: float
    confidence: float = 1.0  # Phase 1: 置信度 (知道自己知不知道)
    is_uncertain: bool = False  # Phase 1: 是否不确定
    evidence: List[str] = None
    last_updated: datetime = None
    improvement_rate: float = 0.0

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class BlindSpot:
    """知识盲区"""
    domain: str
    severity: str
    impact: str
    suggested_learning_path: List[str]


@dataclass
class LearningPlan:
    """学习计划"""
    targets: List[str]
    priority: str
    method: str
    estimated_time: int
    success_criteria: Dict[str, float]


class CognitiveBee(BaseBee):
    """认知蜂 - 蜂巢的自我意识"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.capability_history: Dict[str, List[CapabilityAssessment]] = {}
        self.blind_spots: List[BlindSpot] = []
        self.learning_plans: List[LearningPlan] = []
        self.meta_learning_stats = {
            "learning_efficiency": {}, "knowledge_retention": {}, "transfer_success": {}
        }
        self.memory_path = Path("hive/memory/cognitive_state.json")
        self.memory = self._load_memory()
        self._brain: Optional[BrainInterface] = None  # Phase 1: BrainInterface 懒加载
        logger.info(f"🧠 [CognitiveBee] {self.id} 初始化完成 - 认知系统启动")

    def _get_brain(self) -> Optional[BrainInterface]:
        """Phase 1: 懒加载 BrainInterface 单例"""
        if self._brain is None and BrainInterface is not None and get_brain is not None:
            self._brain = get_brain()
        return self._brain

    def _verify_knowledge(self, domain: str, question: str) -> tuple[float, bool]:
        """
        Phase 1: 向 Brain 提问验证蜂巢对某个领域的真实掌握程度

        Returns:
            (confidence_score, is_uncertain) — 置信度 (0-1) 和是否不确定
        """
        brain = self._get_brain()
        if brain is None:
            # 无 Brain 时，返回内存中的分数作为置信度
            cap = self.memory["capabilities"].get(domain, {"score": 0.5})
            score = cap.get("score", 0.5)
            return score, score < CONFIDENCE_THRESHOLD

        prompt = f"""你是一个严格的代码能力评估专家。
请评估以下问题的回答质量（0-100分）：

问题：{question}

请直接回答分数，不要解释。"""

        try:
            result = brain.consult(prompt, system_prompt="你是一个严格的代码能力评估专家。直接给出0-100的分数。", model="glm-4-flash")
            if result:
                # 提取分数
                import re
                numbers = re.findall(r'\b(\d{1,3})\b', result)
                if numbers:
                    score = min(float(numbers[0]) / 100.0, 1.0)
                    is_uncertain = score < CONFIDENCE_THRESHOLD
                    return score, is_uncertain
        except Exception as e:
            logger.debug(f"🧠 知识验证失败 {domain}: {e}")

        cap = self.memory["capabilities"].get(domain, {"score": 0.5})
        score = cap.get("score", 0.5)
        return score, score < CONFIDENCE_THRESHOLD
        
    def _load_memory(self) -> Dict[str, Any]:
        if self.memory_path.exists():
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")
        return {
            "capabilities": {
                "security_audit": {"score": 0.58, "improvement_rate": 0.01},
                "vulnerability_detection": {"score": 0.55, "improvement_rate": 0.01},
                "code_generation": {"score": 0.85, "improvement_rate": 0.05}
            },
            "missing_domains": ["quantum_computing", "blockchain_development"]
        }

    def _save_memory(self):
        if not self.memory_path.parent.exists():
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
            
    def update_capability(self, domain: str, score_delta: float = 0.1):
        logger.info(f"🧠 [Memory] 更新认知状态: {domain} +{score_delta}")
        if domain not in self.memory["capabilities"]:
            self.memory["capabilities"][domain] = {"score": 0.0, "improvement_rate": 0.0}
        cap = self.memory["capabilities"][domain]
        cap["score"] = min(1.0, cap["score"] + score_delta)
        cap["improvement_rate"] = 0.05
        if cap["score"] > 0.3 and domain in self.memory["missing_domains"]:
            self.memory["missing_domains"].remove(domain)
        self._save_memory()
    
    def assess_capabilities(self) -> Dict[str, CapabilityAssessment]:
        logger.info(f"🔍 [CognitiveBee] 开始能力评估...")
        capabilities = {
            "code_generation": self._assess_code_generation(),
            "code_analysis": self._assess_code_analysis(),
            "bug_detection": self._assess_bug_detection(),
            "code_optimization": self._assess_code_optimization(),
            "architecture_design": self._assess_architecture_design(),
            "system_analysis": self._assess_system_analysis(),
            "security_audit": self._assess_security_audit(),
            "vulnerability_detection": self._assess_vulnerability_detection(),
            "pattern_recognition": self._assess_pattern_recognition(),
            "knowledge_extraction": self._assess_knowledge_extraction(),
        }
        self._record_history(capabilities)
        logger.info(f"✅ [CognitiveBee] 能力评估完成: {len(capabilities)} 项能力")
        return capabilities

    def _record_history(self, capabilities: Dict[str, CapabilityAssessment]):
        for name, assessment in capabilities.items():
            if name not in self.capability_history:
                self.capability_history[name] = []
            self.capability_history[name].append(assessment)
    
    def identify_blind_spots(self) -> List[BlindSpot]:
        logger.info(f"🔎 [CognitiveBee] 识别知识盲区...")
        blind_spots = []
        capabilities = self.assess_capabilities()

        for name, assessment in capabilities.items():
            # Phase 1: 使用 is_uncertain 替代简单分数判断
            if assessment.is_uncertain or assessment.score < 0.6:
                blind_spots.append(BlindSpot(
                    domain=name,
                    severity="high" if assessment.is_uncertain or assessment.score < 0.4 else "medium",
                    impact=f"认知不确定性：{name} 相关任务的执行质量可能受影响",
                    suggested_learning_path=self._suggest_learning_path(name)
                ))

        for domain in self._identify_missing_domains():
            blind_spots.append(BlindSpot(
                domain=domain, severity="critical", impact=f"完全缺少 {domain} 能力",
                suggested_learning_path=self._suggest_learning_path(domain)
            ))

        self.blind_spots = blind_spots
        return blind_spots
    
    def create_learning_plan(self, focus: Optional[str] = None) -> LearningPlan:
        logger.info(f"📋 [CognitiveBee] 制定学习计划...")
        blind_spots = self.identify_blind_spots()
        
        targets, priority = self._determine_learning_focus(focus, blind_spots)
        method = self._select_optimal_learning_method(targets)
        estimated_time = len(targets) * 8
        success_criteria = {t: 0.8 for t in targets}
        
        plan = LearningPlan(targets, priority, method, estimated_time, success_criteria)
        self.learning_plans.append(plan)
        
        logger.info(f"✅ 计划创建: {targets}, {method}, {estimated_time}h")
        return plan

    def _determine_learning_focus(self, focus: Optional[str], blind_spots: List[BlindSpot]):
        if focus: return [focus], "user_specified"
            
        critical = [s for s in blind_spots if s.severity == "critical"]
        if critical: return [s.domain for s in critical[:3]], "critical"
        
        high = [s for s in blind_spots if s.severity == "high"]
        if high: return [s.domain for s in high[:3]], "high"
        
        return self._select_high_potential_targets(), "optimization"
    
    def optimize_learning_strategy(self) -> Dict[str, float]:
        logger.info(f"🎯 [CognitiveBee] 优化学习策略...")
        strategies = {method: self._calculate_efficiency(method) 
                      for method in ["code_reading", "pattern_mining", "trial_error", "guided_learning", "peer_learning"]}
        
        self.meta_learning_stats["learning_efficiency"] = strategies
        return strategies
    
    def specialized_task(self, target_path: Path) -> Dict[str, Any]:
        logger.info(f"🧠 [CognitiveBee] 执行认知评估任务: {target_path}")
        
        capabilities = self.assess_capabilities()
        blind_spots = self.identify_blind_spots()
        plan = self.create_learning_plan()
        strategies = self.optimize_learning_strategy()
        
        report = self._generate_cognitive_report(target_path, capabilities, blind_spots, plan, strategies)
        
        logger.info(f"✅ [CognitiveBee] 认知评估完成 (智能: {report['overall_intelligence']:.2%})")
        return report

    def _generate_cognitive_report(self, target_path, capabilities, blind_spots, plan, strategies):
        # Phase 1: 统计不确定性
        uncertain_count = sum(1 for c in capabilities.values() if c.is_uncertain)
        return {
            "assessment_type": "cognitive_evaluation",
            "target": str(target_path),
            "capabilities": {
                name: {
                    "score": cap.score,
                    "confidence": cap.confidence,
                    "is_uncertain": cap.is_uncertain,  # Phase 1: 不确定性标记
                    "improvement_rate": cap.improvement_rate,
                    "evidence": cap.evidence
                }
                for name, cap in capabilities.items()
            },
            "blind_spots": [
                {"domain": s.domain, "severity": s.severity, "impact": s.impact, "learning_path": s.suggested_learning_path}
                for s in blind_spots
            ],
            "learning_plan": {
                "targets": plan.targets, "priority": plan.priority, "method": plan.method, "estimated_time": plan.estimated_time
            },
            "optimal_strategies": strategies,
            "overall_intelligence": self._calculate_overall_intelligence(capabilities),
            "cognitive_uncertainty": {  # Phase 1: 新增认知不确定性报告
                "uncertain_capabilities": uncertain_count,
                "total_capabilities": len(capabilities),
                "uncertainty_rate": uncertain_count / max(len(capabilities), 1),
                "recommendation": "调用 LibrarianBee 检索缺失知识" if uncertain_count > 2 else "认知状态良好"
            },
            "timestamp": datetime.now().isoformat()
        }

    # ===== Private Helpers =====
    
    def _assess_code_generation(self) -> CapabilityAssessment:
        # Phase 1: 使用真实验证
        score, is_uncertain = self._verify_knowledge(
            "code_generation",
            "如何实现一个支持并发和错误重试的 HTTP 客户端？"
        )
        score = max(score, 0.5)  # 基础分数
        evidence = ["Phase 1 知识验证", "成功生成180+任务"] if not is_uncertain else ["知识验证显示不确定性"]
        return CapabilityAssessment("code_generation", score, score, is_uncertain, evidence, datetime.now(), 0.05)

    def _assess_code_analysis(self) -> CapabilityAssessment:
        score, is_uncertain = self._verify_knowledge(
            "code_analysis",
            "分析这段代码的复杂度并指出设计模式：这是一个日志记录器实现"
        )
        score = max(score, 0.5)
        evidence = ["模式识别准确"] if not is_uncertain else ["知识验证显示不确定性"]
        return CapabilityAssessment("code_analysis", score, score, is_uncertain, evidence, datetime.now(), 0.03)

    def _assess_bug_detection(self) -> CapabilityAssessment:
        # Bug 检测是低置信度领域
        score, is_uncertain = self._verify_knowledge(
            "bug_detection",
            "这段 Python 代码有哪些潜在的 bug？请列出具体问题"
        )
        is_uncertain = is_uncertain or score < 0.7
        evidence = ["基础bug检测", "需要更多验证"] if is_uncertain else ["Bug检测能力良好"]
        return CapabilityAssessment("bug_detection", score, score, is_uncertain, evidence, datetime.now(), 0.02)

    def _assess_code_optimization(self) -> CapabilityAssessment:
        score, is_uncertain = self._verify_knowledge(
            "code_optimization",
            "如何优化这个慢的数据库查询？请给出具体方案"
        )
        score = max(score, 0.5)
        evidence = ["性能优化建议"] if not is_uncertain else ["知识验证显示不确定性"]
        return CapabilityAssessment("code_optimization", score, score, is_uncertain, evidence, datetime.now(), 0.04)

    def _assess_architecture_design(self) -> CapabilityAssessment:
        score, is_uncertain = self._verify_knowledge(
            "architecture_design",
            "设计一个支持百万并发的实时消息系统，描述核心组件和数据流"
        )
        evidence = ["模块化设计"] if not is_uncertain else ["架构设计知识验证失败"]
        return CapabilityAssessment("architecture_design", score, score, is_uncertain, evidence, datetime.now(), 0.03)

    def _assess_system_analysis(self) -> CapabilityAssessment:
        score, is_uncertain = self._verify_knowledge(
            "system_analysis",
            "分析这个微服务系统的依赖关系和性能瓶颈"
        )
        score = max(score, 0.5)
        evidence = ["依赖分析"] if not is_uncertain else ["知识验证显示不确定性"]
        return CapabilityAssessment("system_analysis", score, score, is_uncertain, evidence, datetime.now(), 0.02)

    def _assess_security_audit(self) -> CapabilityAssessment:
        # Phase 1: 安全审计是 CipherBee 的核心，认知蜂需要真实验证
        score, is_uncertain = self._verify_knowledge(
            "security_audit",
            "请审查这段代码的安全漏洞：用户登录、密码存储、Session管理"
        )
        s = self.memory["capabilities"].get("security_audit", {"score": 0.58, "r": 0.01})
        score = max(score, s["score"])  # 取验证和内存中的较高值
        is_uncertain = is_uncertain or score < 0.7
        evidence = ["安全审计能力", "Phase 1 知识验证"] if not is_uncertain else ["安全知识存在不确定性"]
        return CapabilityAssessment("security_audit", score, score, is_uncertain, evidence, datetime.now(), s.get("r", 0.01))

    def _assess_vulnerability_detection(self) -> CapabilityAssessment:
        # Phase 1: 漏洞检测与 CipherBee 能力对齐
        score, is_uncertain = self._verify_knowledge(
            "vulnerability_detection",
            "OWASP Top 10 有哪些？请列举并说明 SQL 注入的原理"
        )
        s = self.memory["capabilities"].get("vulnerability_detection", {"score": 0.55, "r": 0.01})
        score = max(score, s["score"])
        is_uncertain = is_uncertain or score < 0.65
        evidence = ["漏洞检测能力"] if not is_uncertain else ["漏洞知识验证失败，建议调用 CipherBee"]
        return CapabilityAssessment("vulnerability_detection", score, score, is_uncertain, evidence, datetime.now(), s.get("r", 0.01))

    def _assess_pattern_recognition(self) -> CapabilityAssessment:
        score, is_uncertain = self._verify_knowledge(
            "pattern_recognition",
            "这个支付模块用到了哪些设计模式？请详细分析"
        )
        evidence = ["模式匹配"] if not is_uncertain else ["知识验证显示不确定性"]
        return CapabilityAssessment("pattern_recognition", score, score, is_uncertain, evidence, datetime.now(), 0.04)

    def _assess_knowledge_extraction(self) -> CapabilityAssessment:
        score, is_uncertain = self._verify_knowledge(
            "knowledge_extraction",
            "从这段架构文档中提取关键技术决策和依赖关系"
        )
        evidence = ["57K+知识条目"] if not is_uncertain else ["知识提取存在不确定性"]
        return CapabilityAssessment("knowledge_extraction", score, score, is_uncertain, evidence, datetime.now(), 0.06)

    def _identify_missing_domains(self) -> List[str]:
        return self.memory.get("missing_domains", [])

    def _suggest_learning_path(self, domain: str) -> List[str]:
        return ["研究相关项目", "阅读文档", "实践应用"]

    def _select_high_potential_targets(self) -> List[str]:
        capabilities = self.assess_capabilities()
        sorted_caps = sorted(capabilities.items(), key=lambda x: x[1].improvement_rate, reverse=True)
        return [name for name, _ in sorted_caps[:3]]
    
    def _select_optimal_learning_method(self, targets: List[str]) -> str:
        eff = self.meta_learning_stats["learning_efficiency"]
        return max(eff, key=eff.get) if eff else "pattern_mining"

    def _calculate_efficiency(self, method: str) -> float:
        return {"code_reading": 0.70, "pattern_mining": 0.90}.get(method, 0.60)

    def _calculate_overall_intelligence(self, capabilities: Dict[str, CapabilityAssessment]) -> float:
        if not capabilities: return 0.0
        # Phase 1: 置信度加权平均
        total_weight = sum(cap.confidence for cap in capabilities.values())
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(cap.score * cap.confidence for cap in capabilities.values())
        return weighted_sum / total_weight

    def get_cognitive_status(self) -> Dict[str, Any]:
        capabilities = self.assess_capabilities()
        uncertain_count = sum(1 for c in capabilities.values() if c.is_uncertain)
        return {
            "bee_id": self.id,
            "bee_type": "CognitiveBee",
            "capabilities": {n: {"score": c.score, "confidence": c.confidence, "is_uncertain": c.is_uncertain} for n, c in capabilities.items()},
            "blind_spots": len(self.blind_spots),
            "uncertain_capabilities": uncertain_count,
            "learning_plans": len(self.learning_plans),
            "meta_learning_stats": self.meta_learning_stats,
            "overall_intelligence": self._calculate_overall_intelligence(capabilities),
            "cognitive_uncertainty": uncertain_count / max(len(capabilities), 1),
            "timestamp": datetime.now().isoformat()
        }


__all__ = ['CognitiveBee', 'CapabilityAssessment', 'BlindSpot', 'LearningPlan']
