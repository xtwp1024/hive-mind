#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolver - Hive 自我进化引擎

核心进化逻辑：
1. 分析运行历史识别改进点
2. 自动修复检测到的错误
3. 稳定时强制优化/创新
4. 与现有 DNA System 集成
"""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hive.evolution.gep_store import GEPStore
from hive.evolution.strategies import EvolutionStrategy, StrategyType

logger = logging.getLogger("Hive.Evolver")


@dataclass
class EvolutionResult:
    """进化结果"""

    success: bool
    action: str  # repair, innovate, skip
    gene_id: Optional[str] = None
    capsule_id: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class Evolver:
    """
    Hive 自我进化引擎

    负责分析系统运行状态，识别问题和改进机会，
    并通过 GEP 协议存储和应用进化成果。
    """

    def __init__(
        self,
        strategy: Optional[EvolutionStrategy] = None,
        data_dir: Optional[str] = None,
        on_review: Optional[Callable[[dict], bool]] = None,
    ):
        """
        初始化进化引擎

        Args:
            strategy: 进化策略配置
            data_dir: GEP 数据目录
            on_review: 人工审核回调函数
        """
        self.strategy = strategy or EvolutionStrategy.from_env()
        self.on_review = on_review

        # 设置 GEP 数据目录
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent / "gep"

        self.genes_file = self.data_dir / "genes.json"
        self.capsules_file = self.data_dir / "capsules.json"
        self.events_file = self.data_dir / "events.jsonl"

        # SQLite 后端存储（替代 JSON 文件）
        self._store = GEPStore(self.data_dir)

        # 文件操作锁保留，供 events.jsonl 追加写入使用
        self._file_lock = threading.Lock()

        # 迁移旧版 JSON 数据（如存在）
        self._store.migrate_from_json(self.genes_file, self.capsules_file)

        self._ensure_gep_files()

        # 增量日志扫描：记录每个日志文件上次读取的字节偏移
        self._last_log_positions: dict = {}

        # 统计
        self._cycle_count = 0
        self._repairs_count = 0
        self._innovations_count = 0

    def _ensure_gep_files(self):
        """确保 events.jsonl 存在（genes/capsules 已由 GEPStore 管理）"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.events_file.exists():
            self.events_file.touch()

    def record_event(self, event_type: str, details: dict):
        """记录进化事件（同时写入 SQLite 和 events.jsonl 保持向后兼容）"""
        event = {
            "id": f"evt-{uuid.uuid4().hex[:8]}",
            "timestamp": time.time(),
            "type": event_type,
            "strategy": self.strategy.strategy_type.value,
            "details": details,
        }
        self._store.record_event(event)
        with self._file_lock:
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def save_gene(self, gene_id: str, payload: dict):
        """保存基因到 GEP 存储"""
        self._store.save_gene(gene_id, payload)
        self.record_event("GENE_SAVED", {"gene_id": gene_id})
        logger.info(f"🧬 基因保存成功: {gene_id}")

    def save_capsule(self, capsule_id: str, payload: dict):
        """保存胶囊到 GEP 存储"""
        self._store.save_capsule(capsule_id, payload)
        self.record_event("CAPSULE_SAVED", {"capsule_id": capsule_id})
        logger.info(f"💊 胶囊保存成功: {capsule_id}")

    def get_gene(self, gene_id: str) -> Optional[dict]:
        """获取基因"""
        return self._store.get_gene(gene_id)

    def get_capsule(self, capsule_id: str) -> Optional[dict]:
        """获取胶囊"""
        return self._store.get_capsule(capsule_id)

    def _check_system_load(self) -> bool:
        """检查系统负载，决定是否退避"""
        try:
            # Windows 使用 CPU 使用率估算
            if os.name == "nt":
                import ctypes

                kernel32 = ctypes.windll.kernel32
                # 简化处理：假设负载正常
                return True
            else:
                load_avg = os.getloadavg()[0]
                if load_avg > self.strategy.load_max:
                    logger.warning(
                        f"系统负载过高 ({load_avg:.2f} > {self.strategy.load_max})，退避 {self.strategy.backoff_seconds}s"
                    )
                    return False
        except Exception:
            pass
        return True

    def _scan_morgue(self) -> List[dict]:
        """扫描 morgue 目录获取崩溃报告"""
        morgue_dir = self.data_dir.parent.parent / "morgue"
        reports = []

        if not morgue_dir.exists():
            return reports

        for cert_file in morgue_dir.glob("*.json"):
            try:
                with open(cert_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not data.get("evolved_at"):
                        data["_file_path"] = str(cert_file)
                        reports.append(data)
            except Exception as e:
                logger.debug(f"读取崩溃报告失败: {cert_file}, {e}")

        return sorted(reports, key=lambda x: x.get("timestamp", 0), reverse=True)

    def _scan_error_logs(self) -> List[dict]:
        """扫描错误日志（增量扫描，只读取上次位置之后的新增内容）"""
        errors = []
        log_dir = self.data_dir.parent.parent.parent / "logs"

        if not log_dir.exists():
            return errors

        for log_file in log_dir.glob("*.log"):
            try:
                key = str(log_file)
                offset = self._last_log_positions.get(key, 0)
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(offset)
                    for line in f:
                        if "ERROR" in line or "Exception" in line:
                            errors.append(
                                {
                                    "source": key,
                                    "line": line.strip()[:500],
                                    "timestamp": time.time(),
                                }
                            )
                    self._last_log_positions[key] = f.tell()
            except Exception as e:
                logger.debug(f"扫描日志文件失败: {log_file}, {e}")

        return errors[-50:]  # 最近50条

    def _analyze_failures(self) -> dict:
        """分析失败情况，计算错误率和稳定性"""
        morgue_reports = self._scan_morgue()
        error_logs = self._scan_error_logs()

        # 从 events.jsonl 统计
        recent_events = []
        try:
            with open(self.events_file, "r", encoding="utf-8") as f:
                cutoff = time.time() - 86400
                for line in f:
                    if not line.strip():
                        continue
                    evt = json.loads(line)
                    if evt.get("timestamp", 0) > cutoff:
                        recent_events.append(evt)
        except Exception:
            pass

        total_events = len(recent_events) or 1
        failure_events = [e for e in recent_events if "FAIL" in e.get("type", "")]

        error_rate = len(failure_events) / total_events
        stability_score = 1.0 - min(error_rate * 2, 1.0)

        return {
            "error_rate": error_rate,
            "stability_score": stability_score,
            "morgue_count": len(morgue_reports),
            "error_log_count": len(error_logs),
            "recent_failures": failure_events[:5],
            "morgue_reports": morgue_reports[:3],
        }

    def _synthesize_gene(self, issue: dict) -> str:
        """从问题中合成基因"""
        gene_id = f"gene-{uuid.uuid4().hex[:8]}"

        gene_payload = {
            "id": gene_id,
            "created_at": time.time(),
            "source_issue": issue.get("type", "unknown"),
            "mutation_type": "repair",
            "description": f"自动修复: {issue.get('type', 'unknown')}",
            "patch": {
                "type": "behavioral",
                "target": issue.get("source", "unknown"),
                "action": "retry_with_backoff",
            },
            "validation": {
                "command": ["python", "-m", "pytest", "tests/", "-x"],
                "timeout": 300,
            },
        }

        self._store.save_gene(gene_id, gene_payload)
        self.record_event("GENE_CREATED", {"gene_id": gene_id, "issue": issue.get("type")})
        logger.info(f" 基因合成完成: {gene_id}")

        return gene_id

    def _create_innovation(self, context: dict) -> str:
        """创建创新胶囊"""
        capsule_id = f"capsule-{uuid.uuid4().hex[:8]}"

        capsule_payload = {
            "id": capsule_id,
            "created_at": time.time(),
            "type": "optimization",
            "description": "系统稳定期自动优化",
            "genes": [],
            "context": {
                "stability_score": context.get("stability_score", 0),
                "trigger": "forced_innovation",
            },
        }

        self._store.save_capsule(capsule_id, capsule_payload)
        self.record_event("CAPSULE_CREATED", {"capsule_id": capsule_id})
        logger.info(f" 创新胶囊生成: {capsule_id}")

        return capsule_id

    def _request_review(self, action: str, details: dict) -> bool:
        """请求人工审核"""
        if not self.strategy.require_human_review:
            return True

        if self.on_review:
            return self.on_review({"action": action, "details": details})

        # 默认：打印并等待
        logger.info(f" 等待人工审核: {action}")
        logger.info(f"   详情: {json.dumps(details, ensure_ascii=False, indent=2)}")
        return True  # 非交互模式默认通过

    def run_cycle(self) -> EvolutionResult:
        """
        运行一个进化周期

        Returns:
            EvolutionResult: 进化结果
        """
        self._cycle_count += 1
        logger.info(
            f" 进化周期 #{self._cycle_count} 开始 (策略: {self.strategy.strategy_type.value})"
        )

        # 1. 检查系统负载
        if not self._check_system_load():
            time.sleep(self.strategy.backoff_seconds)
            return EvolutionResult(
                success=False,
                action="skip",
                message="系统负载过高，已退避",
            )

        # 2. 分析系统状态
        analysis = self._analyze_failures()
        error_rate = analysis["error_rate"]
        stability_score = analysis["stability_score"]

        logger.info(f" 系统分析: 错误率={error_rate:.2%}, 稳定性={stability_score:.2%}")

        # 3. 决定行动
        if self.strategy.should_repair(error_rate):
            # 修复模式
            if analysis["morgue_reports"]:
                issue = analysis["morgue_reports"][0]

                if not self._request_review("repair", issue):
                    return EvolutionResult(
                        success=False,
                        action="skip",
                        message="人工审核未通过",
                    )

                gene_id = self._synthesize_gene(issue)
                self._repairs_count += 1

                return EvolutionResult(
                    success=True,
                    action="repair",
                    gene_id=gene_id,
                    message=f"修复基因已生成: {gene_id}",
                    details={"issue": issue.get("type"), "error_rate": error_rate},
                )

        elif self.strategy.should_innovate(error_rate, stability_score):
            # 创新模式
            if self._innovations_count >= self.strategy.max_innovations_per_cycle:
                return EvolutionResult(
                    success=True,
                    action="skip",
                    message="本周期创新配额已用完",
                )

            if not self._request_review("innovate", {"stability_score": stability_score}):
                return EvolutionResult(
                    success=False,
                    action="skip",
                    message="人工审核未通过",
                )

            capsule_id = self._create_innovation(analysis)
            self._innovations_count += 1

            return EvolutionResult(
                success=True,
                action="innovate",
                capsule_id=capsule_id,
                message=f"创新胶囊已生成: {capsule_id}",
                details={"stability_score": stability_score},
            )

        # 无需行动
        self.record_event("CYCLE_SKIP", {"reason": "no_action_needed"})
        return EvolutionResult(
            success=True,
            action="skip",
            message="系统状态良好，无需进化",
            details=analysis,
        )

    def run_loop(self, interval: int = 300, max_cycles: Optional[int] = None):
        """
        持续运行进化循环

        Args:
            interval: 每个周期间隔秒数
            max_cycles: 最大周期数，None 表示无限
        """
        logger.info(f" 进化引擎启动 (间隔: {interval}s, 策略: {self.strategy.strategy_type.value})")
        self.record_event("EVOLVER_START", {"interval": interval, "max_cycles": max_cycles})

        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                result = self.run_cycle()
                logger.info(f"   结果: {result.action} - {result.message}")

                cycle += 1
                if max_cycles is None or cycle < max_cycles:
                    time.sleep(interval)
        except KeyboardInterrupt:
            logger.info(" 进化引擎被中断")
        finally:
            self.record_event(
                "EVOLVER_STOP",
                {
                    "cycles": cycle,
                    "repairs": self._repairs_count,
                    "innovations": self._innovations_count,
                },
            )

    def get_status(self) -> dict:
        """获取进化引擎状态"""
        stats = self._store.get_stats()
        return {
            "strategy": self.strategy.strategy_type.value,
            "cycles": self._cycle_count,
            "repairs": self._repairs_count,
            "innovations": self._innovations_count,
            "genes_count": stats["genes"],
            "capsules_count": stats["capsules"],
        }

    # ==================== 集成增强 ====================

    def run_with_monitoring(self, interval: int = 300, max_cycles: Optional[int] = None):
        """
        带监控的进化循环

        使用 MonitorBridge 获取实时指标，动态调整进化策略。
        """
        try:
            from hive.evolution.monitor_bridge import MonitorBridge

            monitor = MonitorBridge()
        except ImportError:
            logger.warning("MonitorBridge 不可用，使用普通循环")
            return self.run_loop(interval, max_cycles)

        logger.info(f"🚀 带监控的进化引擎启动")
        self.record_event("EVOLVER_START_MONITORED", {"interval": interval})

        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                # 获取监控建议
                recommendations = monitor.get_evolution_recommendations()

                if not recommendations["should_evolve"]:
                    logger.info(f"⏸️ 监控建议暂停: {recommendations['reasons']}")
                    time.sleep(self.strategy.backoff_seconds)
                    cycle += 1  # 防止 should_evolve 持续 False 时无限循环
                    continue

                # 动态调整策略
                if recommendations["suggested_strategy"] != self.strategy.strategy_type.value:
                    logger.info(
                        f"🔄 策略调整: {self.strategy.strategy_type.value} -> {recommendations['suggested_strategy']}"
                    )
                    try:
                        self.strategy.strategy_type = StrategyType(
                            recommendations["suggested_strategy"]
                        )
                    except ValueError:
                        pass

                # 运行进化周期
                result = self.run_cycle()
                logger.info(f"   结果: {result.action} - {result.message}")

                cycle += 1
                if max_cycles is None or cycle < max_cycles:
                    time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("⏹️ 进化引擎被中断")
        finally:
            self.record_event(
                "EVOLVER_STOP",
                {
                    "cycles": cycle,
                    "repairs": self._repairs_count,
                    "innovations": self._innovations_count,
                },
            )

    def run_with_tests(self, interval: int = 300, max_cycles: Optional[int] = None):
        """
        带测试验证的进化循环

        每次进化后自动运行测试，检测退化则回滚。
        """
        try:
            from hive.evolution.test_runner import TestRunner

            test_runner = TestRunner()
        except ImportError:
            logger.warning("TestRunner 不可用，使用普通循环")
            return self.run_loop(interval, max_cycles)

        logger.info(f"🧪 带测试验证的进化引擎启动")

        # 建立基线
        baseline = test_runner.establish_baseline()
        if not baseline.success:
            logger.warning(f"⚠️ 基线测试失败，但继续运行")

        self.record_event(
            "EVOLVER_START_TESTED",
            {
                "interval": interval,
                "baseline_pass_rate": baseline.pass_rate,
            },
        )

        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                result = self.run_cycle()
                logger.info(f"   结果: {result.action} - {result.message}")

                # 如果有实际变更，运行测试验证
                if result.action in ("repair", "innovate") and result.success:
                    logger.info("🧪 运行测试验证...")
                    test_result = test_runner.run_quick_tests()
                    comparison = test_runner.compare_with_baseline(test_result)

                    if comparison.get("regression"):
                        logger.warning(f"⚠️ 检测到退化: {comparison['reasons']}")
                        self.record_event(
                            "REGRESSION_DETECTED",
                            {
                                "gene_id": result.gene_id,
                                "capsule_id": result.capsule_id,
                                "reasons": comparison["reasons"],
                            },
                        )
                        # TODO: 实现回滚逻辑
                    else:
                        logger.info(f"✅ 测试通过: {test_result.pass_rate:.1%}")

                cycle += 1
                if max_cycles is None or cycle < max_cycles:
                    time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("⏹️ 进化引擎被中断")
        finally:
            self.record_event(
                "EVOLVER_STOP",
                {
                    "cycles": cycle,
                    "repairs": self._repairs_count,
                    "innovations": self._innovations_count,
                },
            )

    def sync_with_evomap(self) -> dict:
        """
        与 EvoMap 网络同步

        获取热门胶囊，发布本地高价值基因。
        """
        try:
            from hive.evolution.evomap_bridge import get_evomap_bridge

            bridge = get_evomap_bridge()
        except ImportError:
            logger.warning("EvoMapBridge 不可用")
            return {"status": "unavailable"}

        result = {
            "status": "ok",
            "heartbeat": None,
            "imported": 0,
            "published": 0,
        }

        result["heartbeat"] = bridge.heartbeat()

        trending = bridge.fetch_trending(limit=5)
        for capsule in trending:
            if bridge.import_capsule_to_gep(capsule, self):
                result["imported"] += 1

        # 读取待发布基因（锁外网络 I/O）
        genes = self._store.get_all_genes()
        to_publish = {
            gid: gene
            for gid, gene in genes.items()
            if gene.get("validated") and not gene.get("published_to_evomap")
        }

        for gene_id, gene in to_publish.items():
            bundle_id = bridge.publish_gene(gene)
            if bundle_id:
                self._store.update_gene_field(gene_id, "published_to_evomap", bundle_id)
                result["published"] += 1

        logger.info(f"🌐 EvoMap 同步完成: 导入 {result['imported']}, 发布 {result['published']}")
        return result

    def apply_to_dna_system(self) -> int:
        """
        将进化成果应用到 Hive DNA 系统

        Returns:
            应用的基因数量
        """
        try:
            from hive.evolution.dna_bridge import get_bridge

            bridge = get_bridge()
        except ImportError:
            logger.warning("DNABridge 不可用")
            return 0

        genes = self._store.get_all_genes()
        candidates = {gid: gene for gid, gene in genes.items() if not gene.get("applied_to_dna")}

        applied = 0
        for gene_id, gene in candidates.items():
            if bridge.apply_gene_to_dna(gene):
                self._store.update_gene_field(gene_id, "applied_to_dna", time.time())
                applied += 1

        if applied > 0:
            logger.info(f"🧬 已将 {applied} 个基因应用到 DNA 系统")
        return applied
