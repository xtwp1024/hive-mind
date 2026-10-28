"""
Critic - 批评器
===============

功能:
- 结果评估 (Outcome Evaluation)
- 自我批评 (Self-Criticism)
- 异常检测 (Anomaly Detection)
- 改进建议 (Improvement Suggestions)

Usage:
    critic = Critic()
    evaluation = await critic.evaluate(result, expected)
    suggestions = await critic.suggest_improvements(evaluation)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .perception import Perception

logger = logging.getLogger("Hive.Brain.Critic")


class Severity(str, Enum):
    """严重程度"""

    INFO = "info"  # 信息
    WARNING = "warning"  # 警告
    ERROR = "error"  # 错误
    CRITICAL = "critical"  # 严重


@dataclass
class Issue:
    """发现的问题"""

    id: str
    type: str  # performance, quality, safety, reliability
    description: str
    severity: Severity
    location: Optional[str] = None
    root_cause: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.5  # 0-1
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "severity": self.severity.value,
            "location": self.location,
            "root_cause": self.root_cause,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
        }


@dataclass
class Evaluation:
    """
    评估结果

    Attributes:
        success: 是否成功
        score: 综合评分 (0-100)
        quality_score: 质量评分
        efficiency_score: 效率评分
        safety_score: 安全性评分
        issues: 发现的问题列表
        improvements: 改进建议列表
        overall_assessment: 整体评估
    """

    success: bool
    score: float = 0.0
    quality_score: float = 0.0
    efficiency_score: float = 0.0
    safety_score: float = 0.0
    issues: List[Issue] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    overall_assessment: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_critical_issues(self) -> bool:
        return any(i.severity == Severity.CRITICAL for i in self.issues)

    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "score": self.score,
            "quality_score": self.quality_score,
            "efficiency_score": self.efficiency_score,
            "safety_score": self.safety_score,
            "issue_count": len(self.issues),
            "critical_issues": sum(1 for i in self.issues if i.severity == Severity.CRITICAL),
            "warnings": sum(1 for i in self.issues if i.severity == Severity.WARNING),
            "improvements": self.improvements,
            "overall_assessment": self.overall_assessment,
        }


class Critic:
    """
    批评器

    对执行结果进行深度评估:
    1. 质量检查
    2. 效率评估
    3. 安全性检查
    4. 异常检测
    5. 生成改进建议
    """

    # 性能阈值
    DURATION_THRESHOLDS = {
        "fast": 10,  # < 10s
        "normal": 60,  # < 60s
        "slow": 300,  # < 5min
        "timeout": 600,  # < 10min
    }

    # 安全关键词
    SAFETY_CRITICAL = {
        "rm -rf", "delete.*all", "drop table", "truncate",
        "format", "fdisk", "mkfs",
    }

    # 评分权重（可配置）
    DEFAULT_WEIGHTS = {
        "quality": 0.4,
        "efficiency": 0.3,
        "safety": 0.3,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: 可选的评分权重，覆盖默认值
                    包含 quality, efficiency, safety 三个键
        """
        self._weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self._stats = {
            "evaluations": 0,
            "issues_found": 0,
            "critical_found": 0,
            "suggestions_generated": 0,
        }
        self._issue_history: List[Issue] = []

    async def evaluate(
        self,
        result: Dict[str, Any],
        expected: Optional[Dict[str, Any]] = None,
    ) -> Evaluation:
        """
        评估执行结果

        Args:
            result: 执行结果
            expected: 期望结果（如果有）

        Returns:
            Evaluation: 评估结果
        """
        self._stats["evaluations"] += 1

        issues = []
        improvements = []

        # 1. 质量评估
        quality_score = await self._evaluate_quality(result)
        issues.extend(await self._check_quality_issues(result))

        # 2. 效率评估
        efficiency_score = await self._evaluate_efficiency(result)
        issues.extend(await self._check_efficiency_issues(result))

        # 3. 安全性评估
        safety_score = await self._evaluate_safety(result)
        issues.extend(await self._check_safety_issues(result))

        # 4. 异常检测
        anomalies = await self._detect_anomalies(result)
        issues.extend(anomalies)

        # 5. 生成改进建议
        improvements = await self.suggest_improvements(quality_score, efficiency_score, safety_score, issues)

        # 计算综合评分（使用可配置权重）
        score = (
            quality_score * self._weights.get("quality", 0.4)
            + efficiency_score * self._weights.get("efficiency", 0.3)
            + safety_score * self._weights.get("safety", 0.3)
        )

        # 整体评估
        assessment = self._generate_assessment(score, issues)

        self._stats["issues_found"] += len(issues)
        self._stats["critical_found"] += sum(1 for i in issues if i.severity == Severity.CRITICAL)
        self._stats["suggestions_generated"] += len(improvements)

        # 保存到历史
        self._issue_history.extend(issues)

        evaluation = Evaluation(
            success=result.get("success", False),
            score=score,
            quality_score=quality_score,
            efficiency_score=efficiency_score,
            safety_score=safety_score,
            issues=issues,
            improvements=improvements,
            overall_assessment=assessment,
        )

        logger.info(
            f"🧠 Critic: score={score:.1f}, "
            f"issues={len(issues)}, "
            f"critical={sum(1 for i in issues if i.severity == Severity.CRITICAL)}"
        )

        return evaluation

    async def _evaluate_quality(self, result: Dict[str, Any]) -> float:
        """评估质量 (0-100)"""
        score = 100.0

        # 检查是否有错误
        if not result.get("success", True):
            score -= 30

        # 检查返回数据完整性
        if "data" in result:
            data = result["data"]
            if not data:
                score -= 10

        # 检查是否有警告
        if result.get("warnings"):
            score -= 5 * len(result.get("warnings", []))

        # 检查部分成功
        if result.get("partial", False):
            score -= 20

        return max(0.0, min(100.0, score))

    async def _evaluate_efficiency(self, result: Dict[str, Any]) -> float:
        """评估效率 (0-100)"""
        score = 100.0

        duration = result.get("duration", 0)

        if duration == 0:
            # 即时完成，可能有问题
            score = 70
        elif duration < self.DURATION_THRESHOLDS["fast"]:
            score = 100  # 快速完成
        elif duration < self.DURATION_THRESHOLDS["normal"]:
            score = 90
        elif duration < self.DURATION_THRESHOLDS["slow"]:
            score = 70
            score -= (duration - self.DURATION_THRESHOLDS["slow"]) / 60 * 5  # 每分钟扣5分
        elif duration < self.DURATION_THRESHOLDS["timeout"]:
            score = 50
        else:
            score = 20  # 超时

        # 检查是否有不必要的重复
        if result.get("retries", 0) > 0:
            score -= min(20, result["retries"] * 5)

        return max(0.0, min(100.0, score))

    async def _evaluate_safety(self, result: Dict[str, Any]) -> float:
        """评估安全性 (0-100)"""
        score = 100.0

        description = result.get("description", "").lower()
        error = result.get("error", "").lower()

        # 检查危险操作
        for danger in self.SAFETY_CRITICAL:
            if danger in description or danger in error:
                score -= 50

        # 检查权限问题
        if "permission" in error or "权限" in error:
            score -= 20

        # 检查数据泄露风险
        if result.get("exposed_secrets", False):
            score -= 40

        return max(0.0, min(100.0, score))

    async def _check_quality_issues(self, result: Dict[str, Any]) -> List[Issue]:
        """检查质量问题"""
        issues = []
        base_id = f"issue_{int(time.time() * 1000) % 10000}"

        if not result.get("success", True):
            issues.append(Issue(
                id=f"{base_id}_1",
                type="quality",
                description="任务执行失败",
                severity=Severity.ERROR,
                root_cause=result.get("error", "Unknown"),
                suggestions=["分析失败原因", "检查前置条件", "考虑降级方案"],
                confidence=0.9,
            ))

        if result.get("partial", False):
            issues.append(Issue(
                id=f"{base_id}_2",
                type="quality",
                description="任务部分成功",
                severity=Severity.WARNING,
                suggestions=["分析未完成部分", "重试或跳过失败部分"],
                confidence=0.8,
            ))

        return issues

    async def _check_efficiency_issues(self, result: Dict[str, Any]) -> List[Issue]:
        """检查效率问题"""
        issues = []
        base_id = f"issue_{int(time.time() * 1000) % 10000}"

        duration = result.get("duration", 0)

        if duration > self.DURATION_THRESHOLDS["slow"]:
            issues.append(Issue(
                id=f"{base_id}_1",
                type="performance",
                description=f"执行时间过长 ({duration:.0f}s)",
                severity=Severity.WARNING,
                root_cause="可能是任务复杂度高或资源不足",
                suggestions=["考虑任务分解", "增加并行度", "优化算法"],
                confidence=0.8,
            ))

        retries = result.get("retries", 0)
        if retries > 2:
            issues.append(Issue(
                id=f"{base_id}_2",
                type="performance",
                description=f"重试次数过多 ({retries})",
                severity=Severity.WARNING,
                root_cause="可能是网络不稳定或资源竞争",
                suggestions=["增加重试间隔", "检查依赖服务稳定性"],
                confidence=0.75,
            ))

        return issues

    async def _check_safety_issues(self, result: Dict[str, Any]) -> List[Issue]:
        """检查安全问题"""
        issues = []
        base_id = f"issue_{int(time.time() * 1000) % 10000}"

        description = result.get("description", "").lower()
        error = result.get("error", "").lower()

        # 危险操作
        for danger in self.SAFETY_CRITICAL:
            if danger in description:
                issues.append(Issue(
                    id=f"{base_id}_safety_1",
                    type="safety",
                    description=f"检测到危险操作: {danger}",
                    severity=Severity.CRITICAL,
                    suggestions=["确认操作必要性", "使用更安全的替代方案", "增加确认步骤"],
                    confidence=0.95,
                ))

        # 权限问题
        if "permission" in error or "权限" in error:
            issues.append(Issue(
                id=f"{base_id}_safety_2",
                type="safety",
                description="权限不足",
                severity=Severity.ERROR,
                suggestions=["检查执行用户权限", "使用适当的权限级别", "降级到只读操作"],
                confidence=0.9,
            ))

        return issues

    async def _detect_anomalies(self, result: Dict[str, Any]) -> List[Issue]:
        """检测异常"""
        issues = []
        base_id = f"issue_{int(time.time() * 1000) % 10000}"

        # 检查是否是反复出现的问题
        error = result.get("error", "")
        if error:
            similar_count = sum(1 for i in self._issue_history[-10:] if error in i.description)

            if similar_count >= 3:
                issues.append(Issue(
                    id=f"{base_id}_anomaly_1",
                    type="reliability",
                    description=f"同样的错误反复出现 ({similar_count + 1}次)",
                    severity=Severity.WARNING,
                    root_cause="可能是系统性问题",
                    suggestions=["深入分析根因", "检查依赖组件", "建立监控告警"],
                    confidence=0.85,
                ))

        # 检查资源泄漏迹象
        memory = result.get("memory_delta", 0)
        if memory > 500:  # 内存增长超过500MB
            issues.append(Issue(
                id=f"{base_id}_anomaly_2",
                type="reliability",
                description=f"疑似内存泄漏 (增长 {memory:.0f}MB)",
                severity=Severity.WARNING,
                suggestions=["检查对象引用", "使用弱引用", "添加内存监控"],
                confidence=0.7,
            ))

        return issues

    async def suggest_improvements(
        self,
        quality_score: float,
        efficiency_score: float,
        safety_score: float,
        issues: List[Issue],
    ) -> List[str]:
        """生成改进建议"""
        suggestions = []

        # 基于评分建议
        if quality_score < 70:
            suggestions.append("质量需要改进：加强验证和错误处理")
        if efficiency_score < 70:
            suggestions.append("效率需要优化：考虑并行化或缓存")
        if safety_score < 70:
            suggestions.append("安全性需要加强：增加权限检查和操作审计")

        # 基于具体问题建议
        for issue in issues:
            if issue.suggestions:
                suggestions.extend(issue.suggestions)

        # 去重
        seen = set()
        unique = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return unique[:5]  # 最多5条建议

    def _generate_assessment(self, score: float, issues: List[Issue]) -> str:
        """生成整体评估"""
        if score >= 90 and not issues:
            return "优秀 - 任务圆满完成，无明显问题"
        elif score >= 80:
            return "良好 - 任务完成，有小的改进空间"
        elif score >= 70:
            return "合格 - 任务完成但有明显问题需要解决"
        elif score >= 60:
            return "及格 - 勉强完成，建议改进后重试"
        else:
            critical = sum(1 for i in issues if i.severity == Severity.CRITICAL)
            if critical > 0:
                return "不合格 - 存在严重问题，必须修复后才能继续"
            return "不合格 - 需要重大改进"

    async def self_critique(
        self,
        process_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        自我批评

        反思Critic本身的评估是否准确
        """
        assessment = {
            "self_aware": False,
            "potential_bias": None,
            "accuracy": 0.0,
            "improvements": [],
        }

        # 检查是否过度批评
        if len(self._issue_history) > 10:
            critical_ratio = sum(
                1 for i in self._issue_history[-10:] if i.severity == Severity.CRITICAL
            ) / 10

            if critical_ratio > 0.5:
                assessment["potential_bias"] = "可能过度批评"
                assessment["improvements"].append("考虑适当宽容，给出更多建设性建议")

        return assessment

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "issues_in_history": len(self._issue_history),
            "recent_critical_count": sum(
                1 for i in self._issue_history[-20:] if i.severity == Severity.CRITICAL
            ),
        }
