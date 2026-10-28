# -*- coding: utf-8 -*-
"""
Capability Gap Analyzer - 能力缺口分析器

Phase 3: 自主进化能力
分析历史任务数据，发现能力缺口，为自动蜂种生成提供输入。

核心功能:
1. 任务历史追踪 - 收集所有任务执行数据
2. 能力缺口检测 - 分析成功率、耗时、错误模式发现缺口
3. 缺口严重度评估 - 基于业务影响和频率评估优先级
4. 缺口报告生成 - 为进化系统提供结构化输入
5. 趋势分析 - 识别能力随时间的变化趋势
"""

import logging
import threading
import time
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger("Hive.CapabilityGapAnalyzer")


class GapSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapType(Enum):
    SUCCESS_RATE = "success_rate"
    EFFICIENCY = "efficiency"
    ERROR_PATTERN = "error_pattern"
    KNOWLEDGE_GAP = "knowledge_gap"
    WORKLOAD_BOTTLENECK = "workload_bottleneck"
    CAPABILITY_MISMATCH = "capability_mismatch"


@dataclass
class TaskExecutionRecord:
    task_id: str
    bee_id: str
    bee_type: str
    task_category: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    capability_used: Optional[str] = None
    expected_duration: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "bee_id": self.bee_id,
            "bee_type": self.bee_type,
            "task_category": self.task_category,
            "success": self.success,
            "duration": self.duration,
            "error_message": self.error_message,
            "capability_used": self.capability_used,
            "expected_duration": self.expected_duration,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class CapabilityGap:
    gap_id: str
    gap_type: GapType
    severity: GapSeverity
    capability_name: str
    task_category: str

    affected_tasks_count: int = 0
    success_rate: float = 0.0
    avg_duration: float = 0.0
    expected_duration: Optional[float] = None
    efficiency_ratio: Optional[float] = None

    error_patterns: List[str] = field(default_factory=list)
    root_cause_hypothesis: List[str] = field(default_factory=list)

    trend_direction: str = "stable"
    trend_confidence: float = 0.0

    first_detected: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)

    business_impact_score: float = 0.0
    remediation_priority: float = 0.0

    suggested_remediation: List[str] = field(default_factory=list)
    estimated_improvement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type.value,
            "severity": self.severity.value,
            "capability_name": self.capability_name,
            "task_category": self.task_category,
            "affected_tasks_count": self.affected_tasks_count,
            "success_rate": self.success_rate,
            "avg_duration": self.avg_duration,
            "expected_duration": self.expected_duration,
            "efficiency_ratio": self.efficiency_ratio,
            "error_patterns": self.error_patterns,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "trend_direction": self.trend_direction,
            "trend_confidence": self.trend_confidence,
            "first_detected": self.first_detected,
            "last_observed": self.last_observed,
            "business_impact_score": self.business_impact_score,
            "remediation_priority": self.remediation_priority,
            "suggested_remediation": self.suggested_remediation,
            "estimated_improvement": self.estimated_improvement,
        }


@dataclass
class GapAnalysisReport:
    report_id: str
    generated_at: float
    analysis_window_hours: float

    total_tasks_analyzed: int
    total_gaps_detected: int

    gaps_by_severity: Dict[str, int]
    gaps_by_type: Dict[str, int]
    gaps_by_capability: Dict[str, int]

    critical_gaps: List[CapabilityGap]
    high_gaps: List[CapabilityGap]
    all_gaps: List[CapabilityGap]

    remediation_plan: List[Dict[str, Any]]
    evolutionary_insights: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "analysis_window_hours": self.analysis_window_hours,
            "total_tasks_analyzed": self.total_tasks_analyzed,
            "total_gaps_detected": self.total_gaps_detected,
            "gaps_by_severity": self.gaps_by_severity,
            "gaps_by_type": self.gaps_by_type,
            "gaps_by_capability": self.gaps_by_capability,
            "critical_gaps": [g.to_dict() for g in self.critical_gaps],
            "high_gaps": [g.to_dict() for g in self.high_gaps],
            "all_gaps": [g.to_dict() for g in self.all_gaps],
            "remediation_plan": self.remediation_plan,
            "evolutionary_insights": self.evolutionary_insights,
        }


class CapabilityGapAnalyzer:
    """
    能力缺口分析器

    Phase 3 核心组件。分析历史任务数据，发现能力缺口，
    为自动蜂种生成和进化系统提供输入。

    分析维度:
    1. 成功率缺口 - 低于阈值的任务类型
    2. 效率缺口 - 耗时超过预期的任务
    3. 错误模式 - 特定错误类型的聚集
    4. 工作负载瓶颈 - 特定蜂种过载
    5. 能力不匹配 - 分配的任务与能力不符

    使用方式:
        analyzer = CapabilityGapAnalyzer()
        analyzer.record_task(TaskExecutionRecord(...))
        report = analyzer.analyze(analysis_window_hours=24)
    """

    SUCCESS_RATE_THRESHOLDS = {
        GapSeverity.CRITICAL: 0.3,
        GapSeverity.HIGH: 0.5,
        GapSeverity.MEDIUM: 0.7,
        GapSeverity.LOW: 0.9,
    }

    EFFICIENCY_RATIO_THRESHOLDS = {
        GapSeverity.CRITICAL: 3.0,
        GapSeverity.HIGH: 2.0,
        GapSeverity.MEDIUM: 1.5,
        GapSeverity.LOW: 1.2,
    }

    MIN_SAMPLE_SIZE = 5

    def __init__(self, max_records: int = 100000):
        self._records: List[TaskExecutionRecord] = []
        self._max_records = max_records
        self._lock = threading.Lock()

        self._gap_history: List[CapabilityGap] = []
        self._detected_gaps: Dict[str, CapabilityGap] = {}

        self._persistence_path = Path("hive/data/gap_analyzer")
        self._persistence_path.mkdir(parents=True, exist_ok=True)

        self._capability_benchmarks: Dict[str, Dict[str, float]] = {}
        self._load_benchmarks()

        logger.info("🔍 [CapabilityGapAnalyzer] 能力缺口分析器已初始化")

    def _load_benchmarks(self):
        """加载能力基准"""
        self._capability_benchmarks = {
            "mining": {"expected_success_rate": 0.85, "expected_duration": 30.0},
            "gathering": {"expected_success_rate": 0.90, "expected_duration": 20.0},
            "architectural": {"expected_success_rate": 0.75, "expected_duration": 60.0},
            "librarian": {"expected_success_rate": 0.95, "expected_duration": 15.0},
            "sentinel": {"expected_success_rate": 0.90, "expected_duration": 10.0},
            "cognitive": {"expected_success_rate": 0.80, "expected_duration": 45.0},
            "code_generation": {"expected_success_rate": 0.70, "expected_duration": 90.0},
            "security": {"expected_success_rate": 0.95, "expected_duration": 25.0},
        }

    def record_task(self, record: TaskExecutionRecord) -> None:
        """记录任务执行"""
        with self._lock:
            self._records.append(record)

            while len(self._records) > self._max_records:
                self._records.pop(0)

            logger.debug(f"📊 [CapabilityGapAnalyzer] 记录任务: {record.task_id} ({record.bee_type})")

    def record_task_simple(
        self,
        task_id: str,
        bee_id: str,
        bee_type: str,
        task_category: str,
        success: bool,
        duration: float,
        error_message: Optional[str] = None,
        capability_used: Optional[str] = None,
        expected_duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """简化的任务记录接口"""
        if metadata is None:
            metadata = {}
        if expected_duration is not None:
            metadata["expected_duration"] = expected_duration
            
        record = TaskExecutionRecord(
            task_id=task_id,
            bee_id=bee_id,
            bee_type=bee_type,
            task_category=task_category,
            success=success,
            duration=duration,
            error_message=error_message,
            capability_used=capability_used,
            metadata=metadata,
        )
        self.record_task(record)

    def get_records(self, hours: Optional[float] = None) -> List[TaskExecutionRecord]:
        """获取记录（可选时间窗口）"""
        with self._lock:
            if hours is None:
                return list(self._records)

            cutoff = time.time() - (hours * 3600)
            return [r for r in self._records if r.timestamp >= cutoff]

    def analyze(self, analysis_window_hours: float = 24.0) -> GapAnalysisReport:
        """
        执行能力缺口分析

        Args:
            analysis_window_hours: 分析时间窗口（小时）

        Returns:
            GapAnalysisReport - 完整的缺口分析报告
        """
        logger.info(f"🔍 [CapabilityGapAnalyzer] 开始分析最近 {analysis_window_hours} 小时的任务数据")

        records = self.get_records(hours=analysis_window_hours)

        if not records:
            logger.warning("🔍 [CapabilityGapAnalyzer] 没有任务记录可供分析")
            return self._create_empty_report(analysis_window_hours)

        gaps: List[CapabilityGap] = []

        gaps.extend(self._analyze_success_rate_gaps(records))
        gaps.extend(self._analyze_efficiency_gaps(records))
        gaps.extend(self._analyze_error_patterns(records))
        gaps.extend(self._analyze_capability_mismatch(records))

        gaps = self._deduplicate_and_merge_gaps(gaps)

        for gap in gaps:
            self._calculate_severity(gap)
            self._calculate_priority(gap, records)
            self._generate_remediation_suggestions(gap)

        for gap in gaps:
            existing = self._detected_gaps.get(gap.gap_id)
            if existing:
                existing.last_observed = time.time()
                existing.affected_tasks_count = gap.affected_tasks_count
                existing.success_rate = gap.success_rate
                existing.avg_duration = gap.avg_duration
            else:
                self._detected_gaps[gap.gap_id] = gap

        critical_gaps = [g for g in gaps if g.severity == GapSeverity.CRITICAL]
        high_gaps = [g for g in gaps if g.severity == GapSeverity.HIGH]

        report = GapAnalysisReport(
            report_id=f"gap_report_{int(time.time())}",
            generated_at=time.time(),
            analysis_window_hours=analysis_window_hours,
            total_tasks_analyzed=len(records),
            total_gaps_detected=len(gaps),
            gaps_by_severity=self._count_by_severity(gaps),
            gaps_by_type=self._count_by_type(gaps),
            gaps_by_capability=self._count_by_capability(gaps),
            critical_gaps=critical_gaps,
            high_gaps=high_gaps,
            all_gaps=gaps,
            remediation_plan=self._generate_remediation_plan(gaps),
            evolutionary_insights=self._generate_evolutionary_insights(gaps, records),
        )

        self._save_report(report)

        logger.info(f"🔍 [CapabilityGapAnalyzer] 分析完成: 检测到 {len(gaps)} 个缺口")
        logger.info(f"   - 严重: {len(critical_gaps)}, 高优先级: {len(high_gaps)}")

        return report

    def _analyze_success_rate_gaps(self, records: List[TaskExecutionRecord]) -> List[CapabilityGap]:
        """分析成功率缺口"""
        gaps = []

        by_category = defaultdict(list)
        for r in records:
            by_category[r.task_category].append(r)

        for category, tasks in by_category.items():
            if len(tasks) < self.MIN_SAMPLE_SIZE:
                continue

            success_count = sum(1 for t in tasks if t.success)
            success_rate = success_count / len(tasks)

            if success_rate < self.SUCCESS_RATE_THRESHOLDS[GapSeverity.LOW]:
                capabilities_used = Counter(
                    t.capability_used or "unknown" for t in tasks
                )
                primary_capability = capabilities_used.most_common(1)[0][0]

                gap = CapabilityGap(
                    gap_id=f"success_{category}_{primary_capability}",
                    gap_type=GapType.SUCCESS_RATE,
                    severity=GapSeverity.LOW,
                    capability_name=primary_capability,
                    task_category=category,
                    affected_tasks_count=len(tasks),
                    success_rate=success_rate,
                    avg_duration=statistics.mean(t.duration for t in tasks),
                    error_patterns=self._extract_error_patterns(tasks),
                )
                gaps.append(gap)

        return gaps

    def _analyze_efficiency_gaps(self, records: List[TaskExecutionRecord]) -> List[CapabilityGap]:
        """分析效率缺口"""
        gaps = []

        by_bee_type = defaultdict(list)
        for r in records:
            by_bee_type[r.bee_type].append(r)

        for bee_type, tasks in by_bee_type.items():
            if len(tasks) < self.MIN_SAMPLE_SIZE:
                continue

            for task in tasks:
                capability = task.capability_used or "unknown"
                benchmark = self._capability_benchmarks.get(capability, {})
                expected = benchmark.get("expected_duration", 60.0)

                if task.expected_duration:
                    expected = task.expected_duration

                if task.duration > expected * self.EFFICIENCY_RATIO_THRESHOLDS[GapSeverity.LOW]:
                    gap_id = f"efficiency_{bee_type}_{capability}"

                    existing = next((g for g in gaps if g.gap_id == gap_id), None)
                    if existing:
                        existing.affected_tasks_count += 1
                        existing.avg_duration = (
                            (existing.avg_duration * (existing.affected_tasks_count - 1) + task.duration)
                            / existing.affected_tasks_count
                        )
                    else:
                        gap = CapabilityGap(
                            gap_id=gap_id,
                            gap_type=GapType.EFFICIENCY,
                            severity=GapSeverity.LOW,
                            capability_name=capability,
                            task_category=task.task_category,
                            affected_tasks_count=1,
                            success_rate=1.0 if task.success else 0.0,
                            avg_duration=task.duration,
                            expected_duration=expected,
                            efficiency_ratio=task.duration / expected if expected > 0 else 0.0,
                        )
                        gaps.append(gap)

        return gaps

    def _analyze_error_patterns(self, records: List[TaskExecutionRecord]) -> List[CapabilityGap]:
        """分析错误模式"""
        gaps = []

        failed_tasks = [r for r in records if not r.success and r.error_message]

        if not failed_tasks:
            return gaps

        error_clusters = self._cluster_errors(failed_tasks)

        for pattern, tasks in error_clusters.items():
            if len(tasks) < 3:
                continue

            capabilities = Counter(t.capability_used or "unknown" for t in tasks)
            primary_capability = capabilities.most_common(1)[0][0]
            categories = Counter(t.task_category for t in tasks)
            primary_category = categories.most_common(1)[0][0]

            gap = CapabilityGap(
                gap_id=f"error_{pattern[:20]}_{primary_capability}",
                gap_type=GapType.ERROR_PATTERN,
                severity=GapSeverity.MEDIUM,
                capability_name=primary_capability,
                task_category=primary_category,
                affected_tasks_count=len(tasks),
                success_rate=0.0,
                avg_duration=statistics.mean(t.duration for t in tasks),
                error_patterns=[pattern],
                root_cause_hypothesis=self._hypothesize_root_cause(pattern),
            )
            gaps.append(gap)

        return gaps

    def _analyze_capability_mismatch(self, records: List[TaskExecutionRecord]) -> List[CapabilityGap]:
        """分析能力不匹配"""
        gaps = []

        by_bee_task = defaultdict(lambda: {"success": 0, "total": 0, "durations": []})

        for r in records:
            key = (r.bee_type, r.task_category, r.capability_used or "unknown")
            by_bee_task[key]["total"] += 1
            by_bee_task[key]["durations"].append(r.duration)
            if r.success:
                by_bee_task[key]["success"] += 1

        for (bee_type, task_category, capability), stats in by_bee_task.items():
            if stats["total"] < self.MIN_SAMPLE_SIZE:
                continue

            success_rate = stats["success"] / stats["total"]

            benchmark = self._capability_benchmarks.get(capability, {})
            expected_rate = benchmark.get("expected_success_rate", 0.8)

            if success_rate < expected_rate * 0.7:
                gap = CapabilityGap(
                    gap_id=f"mismatch_{bee_type}_{task_category}",
                    gap_type=GapType.CAPABILITY_MISMATCH,
                    severity=GapSeverity.MEDIUM,
                    capability_name=capability,
                    task_category=task_category,
                    affected_tasks_count=stats["total"],
                    success_rate=success_rate,
                    avg_duration=statistics.mean(stats["durations"]),
                    root_cause_hypothesis=[
                        f"{bee_type} 可能不适合执行 {task_category} 类型任务",
                        f"考虑创建专门处理 {task_category} 的新蜂种",
                    ],
                )
                gaps.append(gap)

        return gaps

    def _cluster_errors(self, failed_tasks: List[TaskExecutionRecord]) -> Dict[str, List[TaskExecutionRecord]]:
        """聚类错误消息"""
        clusters = defaultdict(list)

        for task in failed_tasks:
            pattern = self._extract_error_signature(task.error_message or "")
            clusters[pattern].append(task)

        return clusters

    def _extract_error_signature(self, error_msg: str) -> str:
        """提取错误签名"""
        import re

        error_msg = re.sub(r'\b[A-Fa-f0-9]{8,}\b', 'HEX', error_msg)
        error_msg = re.sub(r'\b\d+\.\d+\.\d+\.\d+\b', 'IP', error_msg)
        error_msg = re.sub(r'\b\w+@\w+\.\w+\b', 'EMAIL', error_msg)
        error_msg = re.sub(r'\b\d+\b', 'NUM', error_msg)

        keywords = []
        for word in error_msg.split():
            if len(word) > 3 and word.isalpha():
                keywords.append(word.lower())

        return "_".join(keywords[:5]) if keywords else "unknown_error"

    def _extract_error_patterns(self, tasks: List[TaskExecutionRecord]) -> List[str]:
        """从任务中提取错误模式"""
        failed = [t for t in tasks if not t.success and t.error_message]
        if not failed:
            return []

        signatures = [self._extract_error_signature(t.error_message or "") for t in failed]
        counter = Counter(signatures)
        return [sig for sig, count in counter.most_common(5) if count >= 2]

    def _hypothesize_root_cause(self, error_pattern: str) -> List[str]:
        """假设根本原因"""
        pattern = error_pattern.lower()

        hypotheses = []

        if any(kw in pattern for kw in ["timeout", "connection", "network"]):
            hypotheses.append("网络连接问题或超时")
            hypotheses.append("考虑增加超时配置或实现重试机制")

        if any(kw in pattern for kw in ["permission", "access", "denied", "unauthorized"]):
            hypotheses.append("权限不足或认证问题")
            hypotheses.append("检查 API 密钥或访问令牌配置")

        if any(kw in pattern for kw in ["parse", "json", "decode", "format"]):
            hypotheses.append("数据格式解析错误")
            hypotheses.append("考虑增强数据验证和错误处理")

        if any(kw in pattern for kw in ["memory", "oom", "heap"]):
            hypotheses.append("内存资源不足")
            hypotheses.append("考虑分批处理或增加资源配额")

        if not hypotheses:
            hypotheses.append("需要更多数据确定根本原因")
            hypotheses.append("建议增加详细的错误日志")

        return hypotheses

    def _deduplicate_and_merge_gaps(self, gaps: List[CapabilityGap]) -> List[CapabilityGap]:
        """去重和合并缺口"""
        merged: Dict[str, CapabilityGap] = {}

        for gap in gaps:
            key = f"{gap.gap_type.value}_{gap.capability_name}_{gap.task_category}"

            if key in merged:
                existing = merged[key]
                existing.affected_tasks_count += gap.affected_tasks_count
                existing.success_rate = (
                    existing.success_rate * 0.7 + gap.success_rate * 0.3
                )
                existing.avg_duration = (
                    (existing.avg_duration * existing.affected_tasks_count +
                     gap.avg_duration * gap.affected_tasks_count) /
                    (existing.affected_tasks_count + gap.affected_tasks_count)
                )
                existing.error_patterns = list(
                    set(existing.error_patterns + gap.error_patterns)
                )
            else:
                merged[key] = gap

        return list(merged.values())

    def _calculate_severity(self, gap: CapabilityGap) -> None:
        """计算缺口严重度"""
        if gap.gap_type == GapType.SUCCESS_RATE:
            if gap.success_rate <= self.SUCCESS_RATE_THRESHOLDS[GapSeverity.CRITICAL]:
                gap.severity = GapSeverity.CRITICAL
            elif gap.success_rate <= self.SUCCESS_RATE_THRESHOLDS[GapSeverity.HIGH]:
                gap.severity = GapSeverity.HIGH
            elif gap.success_rate <= self.SUCCESS_RATE_THRESHOLDS[GapSeverity.MEDIUM]:
                gap.severity = GapSeverity.MEDIUM
            else:
                gap.severity = GapSeverity.LOW

        elif gap.gap_type == GapType.EFFICIENCY and gap.efficiency_ratio:
            if gap.efficiency_ratio >= self.EFFICIENCY_RATIO_THRESHOLDS[GapSeverity.CRITICAL]:
                gap.severity = GapSeverity.CRITICAL
            elif gap.efficiency_ratio >= self.EFFICIENCY_RATIO_THRESHOLDS[GapSeverity.HIGH]:
                gap.severity = GapSeverity.HIGH
            elif gap.efficiency_ratio >= self.EFFICIENCY_RATIO_THRESHOLDS[GapSeverity.MEDIUM]:
                gap.severity = GapSeverity.MEDIUM
            else:
                gap.severity = GapSeverity.LOW

        elif gap.gap_type == GapType.ERROR_PATTERN:
            if gap.affected_tasks_count >= 20:
                gap.severity = GapSeverity.CRITICAL
            elif gap.affected_tasks_count >= 10:
                gap.severity = GapSeverity.HIGH
            elif gap.affected_tasks_count >= 5:
                gap.severity = GapSeverity.MEDIUM
            else:
                gap.severity = GapSeverity.LOW

        elif gap.gap_type == GapType.CAPABILITY_MISMATCH:
            if gap.success_rate < 0.3:
                gap.severity = GapSeverity.CRITICAL
            elif gap.success_rate < 0.5:
                gap.severity = GapSeverity.HIGH
            else:
                gap.severity = GapSeverity.MEDIUM

    def _calculate_priority(self, gap: CapabilityGap, records: List[TaskExecutionRecord]) -> None:
        """计算修复优先级"""
        severity_weights = {
            GapSeverity.CRITICAL: 1.0,
            GapSeverity.HIGH: 0.7,
            GapSeverity.MEDIUM: 0.4,
            GapSeverity.LOW: 0.1,
        }

        total_capability_tasks = sum(
            1 for r in records
            if (r.capability_used == gap.capability_name or
                r.task_category == gap.task_category)
        )

        impact_score = gap.affected_tasks_count / max(total_capability_tasks, 1)

        gap.business_impact_score = (
            severity_weights[gap.severity] * 0.6 +
            impact_score * 0.4
        )

        gap.remediation_priority = gap.business_impact_score

    def _generate_remediation_suggestions(self, gap: CapabilityGap) -> None:
        """生成修复建议"""
        suggestions = []

        if gap.gap_type == GapType.SUCCESS_RATE:
            suggestions.extend([
                f"增强 {gap.capability_name} 能力的训练数据",
                f"审查 {gap.task_category} 任务的处理逻辑",
                f"增加错误处理和重试机制",
            ])

        elif gap.gap_type == GapType.EFFICIENCY:
            suggestions.extend([
                f"优化 {gap.capability_name} 相关算法",
                f"考虑缓存或批量处理",
                f"增加并行处理能力",
            ])

        elif gap.gap_type == GapType.ERROR_PATTERN:
            for pattern in gap.error_patterns:
                suggestions.append(f"针对错误模式 '{pattern}' 添加专门处理")
            suggestions.extend([
                "增加详细的错误日志以便调试",
                "考虑添加输入验证",
            ])

        elif gap.gap_type == GapType.CAPABILITY_MISMATCH:
            suggestions.extend([
                f"考虑创建专门处理 {gap.task_category} 的新蜂种",
                f"重新评估任务分配策略",
                f"为现有蜂种增加交叉训练",
            ])

        gap.suggested_remediation = suggestions
        gap.estimated_improvement = self._estimate_improvement(gap)

    def _estimate_improvement(self, gap: CapabilityGap) -> float:
        """估计改进幅度"""
        if gap.severity == GapSeverity.CRITICAL:
            return 0.4
        elif gap.severity == GapSeverity.HIGH:
            return 0.3
        elif gap.severity == GapSeverity.MEDIUM:
            return 0.2
        else:
            return 0.1

    def _count_by_severity(self, gaps: List[CapabilityGap]) -> Dict[str, int]:
        """按严重度计数"""
        counter = Counter(g.severity.value for g in gaps)
        return {
            "critical": counter.get("critical", 0),
            "high": counter.get("high", 0),
            "medium": counter.get("medium", 0),
            "low": counter.get("low", 0),
        }

    def _count_by_type(self, gaps: List[CapabilityGap]) -> Dict[str, int]:
        """按类型计数"""
        counter = Counter(g.gap_type.value for g in gaps)
        return dict(counter)

    def _count_by_capability(self, gaps: List[CapabilityGap]) -> Dict[str, int]:
        """按能力计数"""
        counter = Counter(g.capability_name for g in gaps)
        return dict(counter.most_common(10))

    def _generate_remediation_plan(self, gaps: List[CapabilityGap]) -> List[Dict[str, Any]]:
        """生成修复计划"""
        if not gaps:
            return []

        sorted_gaps = sorted(gaps, key=lambda g: g.remediation_priority, reverse=True)

        plan = []
        for i, gap in enumerate(sorted_gaps[:10]):
            plan.append({
                "priority_rank": i + 1,
                "gap_id": gap.gap_id,
                "gap_type": gap.gap_type.value,
                "severity": gap.severity.value,
                "capability": gap.capability_name,
                "task_category": gap.task_category,
                "affected_tasks": gap.affected_tasks_count,
                "success_rate": gap.success_rate,
                "remediation_priority": gap.remediation_priority,
                "suggested_actions": gap.suggested_remediation,
                "estimated_improvement": gap.estimated_improvement,
            })

        return plan

    def _generate_evolutionary_insights(
        self,
        gaps: List[CapabilityGap],
        records: List[TaskExecutionRecord]
    ) -> Dict[str, Any]:
        """生成进化洞察（供 AutoSpeciesGenerator 使用）"""
        insights = {
            "needs_new_species": [],
            "needs_dna_adaptation": [],
            "needs_task_reassignment": [],
            "capability_demand_trends": {},
            "recommended_mutation_targets": [],
        }

        for gap in gaps:
            if gap.gap_type == GapType.CAPABILITY_MISMATCH:
                insights["needs_new_species"].append({
                    "task_category": gap.task_category,
                    "capability_needed": gap.capability_name,
                    "severity": gap.severity.value,
                    "success_rate": gap.success_rate,
                })

            if gap.gap_type == GapType.SUCCESS_RATE and gap.severity in [GapSeverity.CRITICAL, GapSeverity.HIGH]:
                insights["needs_dna_adaptation"].append({
                    "capability": gap.capability_name,
                    "current_success_rate": gap.success_rate,
                    "target_improvement": gap.estimated_improvement,
                })

            if gap.gap_type == GapType.EFFICIENCY:
                insights["needs_task_reassignment"].append({
                    "bee_type": gap.task_category,
                    "capability": gap.capability_name,
                    "efficiency_ratio": gap.efficiency_ratio,
                })

        by_category = defaultdict(list)
        for r in records:
            by_category[r.task_category].append(r)

        for category, tasks in by_category.items():
            if len(tasks) > 10:
                insights["capability_demand_trends"][category] = {
                    "task_count": len(tasks),
                    "avg_success_rate": sum(1 for t in tasks if t.success) / len(tasks),
                    "avg_duration": statistics.mean(t.duration for t in tasks),
                }

        for gap in gaps:
            if gap.severity in [GapSeverity.CRITICAL, GapSeverity.HIGH]:
                insights["recommended_mutation_targets"].append({
                    "capability": gap.capability_name,
                    "gap_type": gap.gap_type.value,
                    "priority": gap.remediation_priority,
                })

        return insights

    def _create_empty_report(self, window_hours: float) -> GapAnalysisReport:
        """创建空报告"""
        return GapAnalysisReport(
            report_id=f"gap_report_{int(time.time())}",
            generated_at=time.time(),
            analysis_window_hours=window_hours,
            total_tasks_analyzed=0,
            total_gaps_detected=0,
            gaps_by_severity={"critical": 0, "high": 0, "medium": 0, "low": 0},
            gaps_by_type={},
            gaps_by_capability={},
            critical_gaps=[],
            high_gaps=[],
            all_gaps=[],
            remediation_plan=[],
            evolutionary_insights={},
        )

    def _save_report(self, report: GapAnalysisReport) -> None:
        """保存报告"""
        try:
            report_path = self._persistence_path / f"report_{report.report_id}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"🔍 [CapabilityGapAnalyzer] 报告已保存: {report_path}")
        except Exception as e:
            logger.warning(f"🔍 [CapabilityGapAnalyzer] 保存报告失败: {e}")

    def get_latest_report(self) -> Optional[GapAnalysisReport]:
        """获取最新报告"""
        reports = sorted(
            self._persistence_path.glob("report_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if not reports:
            return None

        try:
            with open(reports[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"🔍 [CapabilityGapAnalyzer] 加载最新报告: {reports[0].name}")
            return self._dict_to_report(data)
        except Exception as e:
            logger.warning(f"🔍 [CapabilityGapAnalyzer] 加载报告失败: {e}")
            return None

    def _dict_to_report(self, data: Dict[str, Any]) -> GapAnalysisReport:
        """从字典恢复报告"""
        return GapAnalysisReport(
            report_id=data["report_id"],
            generated_at=data["generated_at"],
            analysis_window_hours=data["analysis_window_hours"],
            total_tasks_analyzed=data["total_tasks_analyzed"],
            total_gaps_detected=data["total_gaps_detected"],
            gaps_by_severity=data["gaps_by_severity"],
            gaps_by_type=data["gaps_by_type"],
            gaps_by_capability=data["gaps_by_capability"],
            critical_gaps=[],
            high_gaps=[],
            all_gaps=[],
            remediation_plan=data["remediation_plan"],
            evolutionary_insights=data["evolutionary_insights"],
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_records": len(self._records),
            "detected_gaps_count": len(self._detected_gaps),
            "gaps_by_severity": self._count_by_severity(list(self._detected_gaps.values())),
        }

    def integrate_with_swarm_evolution(self, coordinator) -> Dict[str, Any]:
        """
        与蜂群进化系统集成

        将能力缺口分析结果提供给进化系统，
        指导下一代蜂的进化方向。
        """
        report = self.analyze(analysis_window_hours=24)

        if not report.evolutionary_insights.get("needs_new_species"):
            return {
                "status": "no_action_needed",
                "message": "未检测到需要进化的能力缺口",
            }

        insights = report.evolutionary_insights

        result = {
            "status": "evolution_guided",
            "report_id": report.report_id,
            "critical_gaps_count": len(report.critical_gaps),
            "high_gaps_count": len(report.high_gaps),
            "guided_mutations": [],
            "recommended_species": [],
        }

        for adaptation in insights.get("needs_dna_adaptation", []):
            result["guided_mutations"].append({
                "capability": adaptation["capability"],
                "target_improvement": adaptation["target_improvement"],
                "mutation_strategy": "increase_capability_weight",
            })

        for new_species in insights.get("needs_new_species", []):
            result["recommended_species"].append({
                "task_category": new_species["task_category"],
                "capability_focus": new_species["capability_needed"],
                "justification": f"现有蜂种成功率仅 {new_species['success_rate']:.2%}",
            })

        logger.info(f"🔍 [CapabilityGapAnalyzer] 已向进化系统提供指导: {len(result['guided_mutations'])} 个突变目标, {len(result['recommended_species'])} 个新蜂种建议")

        return result


_analyzer: Optional[CapabilityGapAnalyzer] = None
_analyzer_lock = threading.Lock()


def get_capability_gap_analyzer() -> CapabilityGapAnalyzer:
    """获取能力缺口分析器单例"""
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = CapabilityGapAnalyzer()
    return _analyzer
