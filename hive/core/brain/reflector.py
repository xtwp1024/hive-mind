"""
Reflector - 反思器
==================

功能:
- 经验总结 (Experience Summarization)
- 策略反思 (Strategy Reflection)
- 知识提炼 (Knowledge Distillation)
- 更新 Cerebellum 反射

Usage:
    reflector = Reflector()
    insights = await reflector.reflect(execution_result, context)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .memory_hierarchy import MemoryHierarchy, Experience
from .perception import Perception

logger = logging.getLogger("Hive.Brain.Reflector")

try:
    from hive.core.cerebellum import Cerebellum
except ImportError:
    Cerebellum = None


class ReflectionType(str, Enum):
    """反思类型"""

    SUCCESS_ANALYSIS = "success_analysis"  # 成功因素分析
    FAILURE_ANALYSIS = "failure_analysis"  # 失败根因挖掘
    PATTERN_RECOGNITION = "pattern_recognition"  # 模式识别
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"  # 知识提炼
    AXIOM_EXTRACTION = "axiom_extraction"  # 公理提取


@dataclass
class Insight:
    """反思洞察"""

    id: str
    type: ReflectionType
    content: str
    confidence: float = 0.5  # 0-1
    source_experience: str = ""
    actionable: bool = True
    priority: int = 3  # 1-5
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "confidence": self.confidence,
            "actionable": self.actionable,
            "priority": self.priority,
            "created_at": self.created_at,
        }


@dataclass
class ReflectionResult:
    """反思结果"""

    success: bool
    insights: List[Insight] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    axioms: List[str] = field(default_factory=list)
    reflex_updated: bool = False
    knowledge_consolidated: bool = False
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class Reflector:
    """
    反思器

    从执行结果中提取洞察、模式和公理。

    工作流程:
    1. 分析成功/失败因素
    2. 识别反复出现的模式
    3. 提炼为可操作的洞察
    4. 更新 Cerebellum 反射
    5. 整合到长期记忆
    """

    # 成功/失败关键词
    SUCCESS_INDICATORS = {
        "完成", "成功", "通过", "正常", "有效",
        "completed", "success", "passed", "done", "worked",
    }
    FAILURE_INDICATORS = {
        "失败", "错误", "异常", "问题", "无法", "超时",
        "failed", "error", "exception", "failed", "timeout",
    }

    # 模式关键词
    PATTERN_KEYWORDS = [
        "总是", "经常", "每次", "通常",
        "always", "often", "every time", "usually",
        "repeated", "again", "same",
    ]

    def __init__(self, memory_hierarchy: Optional[MemoryHierarchy] = None):
        self.memory = memory_hierarchy
        self.cerebellum = None
        self._insights_history: List[Insight] = []
        self._stats = {
            "reflections": 0,
            "successes": 0,
            "failures": 0,
            "patterns_found": 0,
            "axioms_extracted": 0,
        }

        # 尝试连接Cerebellum
        if Cerebellum:
            try:
                self.cerebellum = Cerebellum()
            except Exception as e:
                logger.warning(f"⚠️ Failed to connect Cerebellum: {e}")

    async def reflect(
        self,
        result: Dict[str, Any],
        perception: Optional[Perception] = None,
    ) -> ReflectionResult:
        """
        执行反思

        Args:
            result: 执行结果
            perception: 原始感知

        Returns:
            ReflectionResult: 反思结果
        """
        self._stats["reflections"] += 1

        is_success = result.get("success", False)
        if is_success:
            self._stats["successes"] += 1
        else:
            self._stats["failures"] += 1

        insights = []
        patterns = []
        axioms = []

        # 1. 分析成功/失败因素
        if is_success:
            factor_insights = await self._analyze_success(result, perception)
        else:
            factor_insights = await self._analyze_failure(result, perception)
        insights.extend(factor_insights)

        # 2. 模式识别
        detected_patterns = await self._recognize_patterns(result)
        patterns.extend(detected_patterns)
        self._stats["patterns_found"] += len(detected_patterns)

        # 3. 知识提炼（如果经验重要）
        if result.get("importance", 0) > 0.7 or not is_success:
            distilled = await self._distill_knowledge(result, perception)
            insights.extend(distilled)

        # 4. 公理提取（高置信度洞察）
        high_conf_insights = [i for i in insights if i.confidence > 0.8]
        for insight in high_conf_insights:
            axiom = self._extract_axiom(insight)
            if axiom:
                axioms.append(axiom)
                self._stats["axioms_extracted"] += 1

        # 5. 更新反射（如果发现高置信度模式）
        reflex_updated = False
        for insight in insights:
            if insight.confidence > 0.85 and insight.actionable:
                if await self._update_cerebellum(insight):
                    reflex_updated = True

        # 生成总结
        summary = self._generate_summary(is_success, len(insights), len(patterns))

        result_obj = ReflectionResult(
            success=True,
            insights=insights,
            patterns=patterns,
            axioms=axioms,
            reflex_updated=reflex_updated,
            summary=summary,
        )

        logger.info(
            f"🧠 Reflector: {len(insights)} insights, "
            f"{len(patterns)} patterns, {len(axioms)} axioms"
        )

        return result_obj

    async def _analyze_success(
        self,
        result: Dict[str, Any],
        perception: Optional[Perception],
    ) -> List[Insight]:
        """分析成功因素"""
        insights = []
        base_id = f"insight_{int(time.time() * 1000) % 10000}"

        # 分析是什么导致了成功
        description = result.get("description", "")
        details = result.get("details", "")

        # 检查是否有可复用的模式
        if "分析" in description:
            insights.append(Insight(
                id=f"{base_id}_1",
                type=ReflectionType.SUCCESS_ANALYSIS,
                content="分析类任务通常可以通过分解为子任务来提高成功率",
                confidence=0.75,
                source_experience=description,
                priority=2,
            ))

        if "创建" in description or "实现" in description:
            insights.append(Insight(
                id=f"{base_id}_2",
                type=ReflectionType.SUCCESS_ANALYSIS,
                content="实现类任务需要充分的设计阶段",
                confidence=0.7,
                source_experience=description,
                priority=3,
            ))

        # 检查效率
        duration = result.get("duration", 0)
        if duration > 0 and duration < 60:
            insights.append(Insight(
                id=f"{base_id}_3",
                type=ReflectionType.SUCCESS_ANALYSIS,
                content=f"快速完成（{duration:.0f}s），可能是重复任务或简单任务",
                confidence=0.6,
                source_experience=description,
                priority=1,
            ))

        return insights

    async def _analyze_failure(
        self,
        result: Dict[str, Any],
        perception: Optional[Perception],
    ) -> List[Insight]:
        """分析失败根因"""
        insights = []
        base_id = f"insight_{int(time.time() * 1000) % 10000}"

        error = result.get("error", "")
        description = result.get("description", "")

        # 根因分类
        if "超时" in error:
            insights.append(Insight(
                id=f"{base_id}_1",
                type=ReflectionType.FAILURE_ANALYSIS,
                content="任务超时，可能是复杂度超出预期或资源不足",
                confidence=0.8,
                source_experience=description,
                actionable=True,
                priority=5,
            ))

        if "权限" in error or "permission" in error.lower():
            insights.append(Insight(
                id=f"{base_id}_2",
                type=ReflectionType.FAILURE_ANALYSIS,
                content="权限不足，需要检查执行环境或降级操作",
                confidence=0.85,
                source_experience=description,
                actionable=True,
                priority=5,
            ))

        if "找不到" in error or "not found" in error.lower():
            insights.append(Insight(
                id=f"{base_id}_3",
                type=ReflectionType.FAILURE_ANALYSIS,
                content="依赖不存在，应先检查前置条件",
                confidence=0.8,
                source_experience=description,
                actionable=True,
                priority=4,
            ))

        if "内存" in error or "memory" in error.lower():
            insights.append(Insight(
                id=f"{base_id}_4",
                type=ReflectionType.FAILURE_ANALYSIS,
                content="内存问题，可能存在泄漏或大数据处理",
                confidence=0.75,
                source_experience=description,
                actionable=True,
                priority=4,
            ))

        return insights

    async def _recognize_patterns(self, result: Dict[str, Any]) -> List[str]:
        """识别反复出现的模式"""
        patterns = []
        description = result.get("description", "")
        error = result.get("error", "")

        # 检查反复失败
        recent_similar = self._check_recent_similar(description)
        if recent_similar > 2:
            patterns.append(f"'{description[:30]}...' 类型的任务反复失败")

        # 检查特定错误模式
        if error:
            error_type = self._classify_error(error)
            patterns.append(f"错误类型: {error_type}")

        # 检查是否是已知问题
        if self._is_known_issue(error):
            patterns.append("这是一个已知问题，可能已有解决方案")

        return patterns

    def _check_recent_similar(self, description: str) -> int:
        """检查最近类似任务的数量"""
        count = 0
        desc_start = description[:20] if len(description) > 20 else description

        for insight in self._insights_history[-10:]:
            if desc_start in insight.source_experience:
                count += 1

        return count

    def _classify_error(self, error: str) -> str:
        """分类错误类型"""
        error_lower = error.lower()

        if any(k in error_lower for k in ["timeout", "超时"]):
            return "超时"
        if any(k in error_lower for k in ["memory", "内存"]):
            return "资源"
        if any(k in error_lower for k in ["permission", "权限", "denied"]):
            return "权限"
        if any(k in error_lower for k in ["not found", "找不到", "不存在"]):
            return "缺失"
        if any(k in error_lower for k in ["syntax", "语法"]):
            return "语法"
        if any(k in error_lower for k in ["null", "none", "空"]):
            return "空值"

        return "其他"

    def _is_known_issue(self, error: str) -> bool:
        """检查是否是已知问题"""
        known_patterns = [
            "路径太长",
            "文件名非法",
            "端口被占用",
            "数据库锁定",
        ]
        return any(p in error for p in known_patterns)

    async def _distill_knowledge(
        self,
        result: Dict[str, Any],
        perception: Optional[Perception],
    ) -> List[Insight]:
        """提炼知识"""
        insights = []
        base_id = f"insight_{int(time.time() * 1000) % 10000}"

        description = result.get("description", "")
        is_success = result.get("success", False)

        # 从感知中提取知识
        if perception:
            for entity in perception.entities:
                if entity.type == "file":
                    insights.append(Insight(
                        id=f"{base_id}_file_{entity.value}",
                        type=ReflectionType.KNOWLEDGE_DISTILLATION,
                        content=f"涉及文件操作: {entity.value}",
                        confidence=0.6,
                        source_experience=description,
                        actionable=False,
                        priority=2,
                    ))

        # 从结果中提炼
        if "分析" in description and is_success:
            insights.append(Insight(
                id=f"{base_id}_analysis_knowledge",
                type=ReflectionType.KNOWLEDGE_DISTILLATION,
                content="分析类任务的关键是理解问题的本质结构",
                confidence=0.65,
                source_experience=description,
                actionable=False,
                priority=2,
            ))

        return insights

    def _extract_axiom(self, insight: Insight) -> Optional[str]:
        """从洞察中提取公理"""
        content = insight.content.strip()
        if not content:
            return None

        # 转换为断言形式，确保语法通顺
        if "需要" in content or "应该" in content or "可以" in content:
            # 移除"可能"等不确定词汇，并确保语法正确
            axiom = content
            axiom = axiom.replace("可能 ", "")  # 移除"可能"
            axiom = axiom.replace("可能需要", "需要")  # 清理重复
            axiom = axiom.replace("  ", " ")  # 清理多余空格

            # 添加适当的修饰词形成公理
            if axiom.startswith("需要") or axiom.startswith("应该"):
                if not axiom.startswith("总是"):
                    axiom = f"总是{axiom[0]}{axiom[1:]}"  # "需要" -> "总是需要"
            elif axiom.startswith("可以"):
                axiom = axiom.replace("可以", "推荐")

            return axiom

        # 如果置信度很高，直接返回但确保语法完整
        if insight.confidence > 0.9:
            # 确保以句号结尾
            if not content.endswith(("。", ".", "!", "?")):
                content = content + "。"
            return content

        return None

    async def _update_cerebellum(self, insight: Insight) -> bool:
        """更新小脑反射"""
        if not self.cerebellum:
            return False

        try:
            if hasattr(self.cerebellum, "imprint"):
                # 创建一个触发-响应对
                trigger = insight.source_experience[:50] if insight.source_experience else "general"
                response = insight.content

                self.cerebellum.imprint(trigger, response)

                # 保存到历史
                self._insights_history.append(insight)

                logger.info(f"🧠 Updated cerebellum: {trigger[:30]}... -> {response[:30]}...")
                return True

        except Exception as e:
            logger.warning(f"⚠️ Failed to update cerebellum: {e}")

        return False

    def _generate_summary(
        self,
        is_success: bool,
        num_insights: int,
        num_patterns: int,
    ) -> str:
        """生成反思总结"""
        if is_success:
            base = "任务成功完成"
        else:
            base = "任务执行失败"

        if num_insights > 0:
            base += f"，产生 {num_insights} 个洞察"
        if num_patterns > 0:
            base += f"，识别 {num_patterns} 个模式"

        return base

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "insights_in_history": len(self._insights_history),
            "cerebellum_connected": self.cerebellum is not None,
        }
