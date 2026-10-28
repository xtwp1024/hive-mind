#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化蜜蜂评估器 - AutomatedBeeEvaluator
实现自动化的蜜蜂变种使用统计和定期评估机制

功能：
1. 定期自动评估蜜蜂变种质量（每72小时）
2. 自动清理长期未使用的蜜蜂
3. 生成评估报告和建议
4. 支持手动触发评估
5. 集成使用监控和质量评估
"""

import json
import schedule
import time
import threading
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# 添加项目根目录到sys.path，支持从任何位置运行
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 处理导入：尝试绝对导入，失败则使用相对导入
try:
    from hive.monitoring.bee_usage_monitor import BeeUsageMonitor, BeeUsageStatus
except ImportError:
    from .bee_usage_monitor import BeeUsageMonitor, BeeUsageStatus


@dataclass
class EvaluationResult:
    """评估结果"""
    timestamp: str
    total_bees: int
    evaluated_bees: int
    quality_stats: Dict[str, int]
    usage_stats: Dict[str, int]
    cleanup_recommendations: List[str]
    action_items: List[str]
    score: float  # 整体健康度评分 0-100


class AutomatedBeeEvaluator:
    """
    自动化蜜蜂评估器
    
    功能：
    1. 定期自动评估（每72小时）
    2. 整合使用统计和质量评估
    3. 生成可执行的建议
    4. 自动保存评估历史
    """
    
    def __init__(self, data_dir: str = "hive/monitoring/data"):
        """
        初始化评估器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化监控器
        self.usage_monitor = BeeUsageMonitor(data_dir)
        
        # 评估历史文件
        self.evaluation_history_file = self.data_dir / "evaluation_history.json"
        self.last_evaluation_file = self.data_dir / "last_evaluation.json"
        
        # 评估间隔（72小时）
        self.evaluation_interval_hours = 72
        
        # 加载上次评估时间
        self.last_evaluation_time = self._load_last_evaluation_time()
        
        # 调度器线程
        self.scheduler_thread: Optional[threading.Thread] = None
        self.is_running = False
    
    def _load_last_evaluation_time(self) -> Optional[datetime]:
        """加载上次评估时间"""
        if not self.last_evaluation_file.exists():
            return None
        
        try:
            with open(self.last_evaluation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            timestamp = data.get('last_evaluation_time')
            if timestamp:
                return datetime.fromisoformat(timestamp)
        except Exception as e:
            print(f"加载上次评估时间失败: {e}")
        
        return None
    
    def _save_last_evaluation_time(self, timestamp: datetime):
        """保存上次评估时间"""
        try:
            data = {
                'last_evaluation_time': timestamp.isoformat(),
                'evaluation_interval_hours': self.evaluation_interval_hours
            }
            
            with open(self.last_evaluation_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存上次评估时间失败: {e}")
    
    def should_run_evaluation(self) -> bool:
        """
        检查是否应该运行评估
        
        Returns:
            是否需要运行评估
        """
        if self.last_evaluation_time is None:
            return True
        
        hours_since_last = (datetime.now() - self.last_evaluation_time).total_seconds() / 3600
        
        return hours_since_last >= self.evaluation_interval_hours
    
    def evaluate_bees(self, force: bool = False) -> EvaluationResult:
        """
        评估蜜蜂变种
        
        Args:
            force: 是否强制运行评估
            
        Returns:
            评估结果
        """
        # 检查是否需要评估
        if not force and not self.should_run_evaluation():
            hours_since_last = (datetime.now() - self.last_evaluation_time).total_seconds() / 3600
            print(f"距离上次评估仅{hours_since_last:.1f}小时，无需再次评估（间隔{self.evaluation_interval_hours}小时）")
            return None
        
        print(f"\n{'='*80}")
        print(f"开始自动化蜜蜂评估 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # 获取使用统计
        usage_stats = self.usage_monitor.get_usage_statistics(stale_days=30)
        
        # 获取清理候选
        cleanup_candidates = self.usage_monitor.cleanup_candidates(stale_days=30)
        
        # 生成质量统计
        quality_stats = self._generate_quality_stats(usage_stats)
        
        # 生成清理建议
        cleanup_recommendations = self._generate_cleanup_recommendations(
            usage_stats, cleanup_candidates
        )
        
        # 生成行动项
        action_items = self._generate_action_items(
            usage_stats, cleanup_candidates
        )
        
        # 计算整体健康度评分
        health_score = self._calculate_health_score(usage_stats)
        
        # 创建评估结果
        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            total_bees=usage_stats['total_bees'],
            evaluated_bees=usage_stats['tracked_bees'],
            quality_stats=quality_stats,
            usage_stats=usage_stats['usage_distribution'],
            cleanup_recommendations=cleanup_recommendations,
            action_items=action_items,
            score=health_score
        )
        
        # 保存评估结果
        self._save_evaluation_result(result)
        
        # 更新上次评估时间
        self.last_evaluation_time = datetime.now()
        self._save_last_evaluation_time(self.last_evaluation_time)
        
        # 生成报告
        report = self._generate_evaluation_report(result, usage_stats)
        print(report)
        
        # 保存报告到文件
        self._save_report(report, result.timestamp)
        
        print(f"\n{'='*80}")
        print("自动化蜜蜂评估完成")
        print(f"{'='*80}\n")
        
        return result
    
    def _generate_quality_stats(self, usage_stats: Dict[str, Any]) -> Dict[str, int]:
        """生成质量统计"""
        total_bees = usage_stats['total_bees']
        
        # 基于使用情况的质量分类
        active_count = usage_stats['usage_distribution']['active']
        inactive_count = usage_stats['usage_distribution']['inactive']
        unused_count = usage_stats['usage_distribution']['unused']
        stale_count = usage_stats['usage_distribution']['stale']
        
        return {
            "high_quality": active_count,  # 活跃使用 = 高质量
            "medium_quality": inactive_count,  # 非活跃 = 中等质量
            "low_quality": unused_count,  # 未使用 = 低质量
            "deprecated": stale_count,  # 长期未使用 = 已废弃
            "quality_score": (active_count / total_bees * 100) if total_bees > 0 else 0
        }
    
    def _generate_cleanup_recommendations(
        self, 
        usage_stats: Dict[str, Any], 
        cleanup_candidates: List[str]
    ) -> List[str]:
        """生成清理建议"""
        recommendations = []
        
        total_bees = usage_stats['total_bees']
        stale_count = usage_stats['usage_distribution']['stale']
        unused_count = usage_stats['usage_distribution']['unused']
        
        # 分析清理优先级
        if unused_count > 0:
            recommendations.append(
                f"🔴 高优先级: 清理{unused_count}个完全未使用的蜜蜂: {', '.join(cleanup_candidates[:3])}{'...' if len(cleanup_candidates) > 3 else ''}"
            )
        
        if stale_count > 0:
            stale_bees = [b['bee_name'] for b in usage_stats['stale_bees']]
            recommendations.append(
                f"🟡 中优先级: 检查{stale_count}个长期未使用的蜜蜂: {', '.join(stale_bees[:3])}{'...' if len(stale_bees) > 3 else ''}"
            )
        
        # 计算清理比例
        cleanup_ratio = (unused_count + stale_count) / total_bees if total_bees > 0 else 0
        
        if cleanup_ratio > 0.5:
            recommendations.append(
                f"🔴 紧急: 超过50%的蜜蜂长期未使用，建议立即进行全面清理"
            )
        elif cleanup_ratio > 0.3:
            recommendations.append(
                f"🟡 警告: 超过30%的蜜蜂长期未使用，建议在下次评估前完成清理"
            )
        
        # 活跃蜜蜂建议
        active_ratio = usage_stats['usage_distribution']['active'] / total_bees if total_bees > 0 else 0
        if active_ratio < 0.3:
            recommendations.append(
                f"🔴 活跃蜜蜂占比过低({active_ratio*100:.1f}%)，建议精简变种，专注核心功能"
            )
        
        return recommendations
    
    def _generate_action_items(
        self, 
        usage_stats: Dict[str, Any], 
        cleanup_candidates: List[str]
    ) -> List[str]:
        """生成可执行的行动项"""
        actions = []
        
        # 立即行动项
        if cleanup_candidates:
            actions.append(f"[立即] 评估并移除未使用的蜜蜂: {cleanup_candidates[0]}")
        
        # 72小时行动项
        actions.append("[72小时内] 完成所有未使用蜜蜂的评估和清理")
        
        # 7天行动项
        actions.append("[7天内] 优化活跃蜜蜂，提高整体使用率")
        actions.append("[7天内] 建立蜜蜂变种使用指南")
        
        # 长期行动项
        actions.append("[长期] 建立蜜蜂变种生命周期管理流程")
        actions.append("[长期] 定期审查新蜜蜂变种的设计合理性")
        
        return actions
    
    def _calculate_health_score(self, usage_stats: Dict[str, Any]) -> float:
        """
        计算整体健康度评分
        
        评分标准：
        - 活跃蜜蜂占比: 40分
        - 总使用次数: 30分
        - 未跟踪蜜蜂比例: -10分
        - 清洁度（无未使用蜜蜂）: 20分
        
        Returns:
            健康度评分 0-100
        """
        total_bees = usage_stats['total_bees']
        if total_bees == 0:
            return 0.0
        
        score = 0.0
        
        # 1. 活跃蜜蜂占比 (0-40分)
        active_ratio = usage_stats['usage_distribution']['active'] / total_bees
        score += active_ratio * 40
        
        # 2. 总使用次数 (0-30分)
        total_uses = usage_stats['total_uses']
        uses_score = min(total_uses / 10, 1.0) * 30  # 每10次使用得3分，最高30分
        score += uses_score
        
        # 3. 未跟踪蜜蜂比例 (0-10分，扣分项)
        untracked_ratio = usage_stats['untracked_bees'] / total_bees
        score -= untracked_ratio * 10
        
        # 4. 清洁度 (0-20分)
        unused_count = usage_stats['usage_distribution']['unused']
        stale_count = usage_stats['usage_distribution']['stale']
        cleanliness_ratio = 1 - (unused_count + stale_count) / total_bees
        score += cleanliness_ratio * 20
        
        return max(0.0, min(100.0, score))
    
    def _generate_evaluation_report(
        self, 
        result: EvaluationResult, 
        usage_stats: Dict[str, Any]
    ) -> str:
        """生成评估报告"""
        report_lines = [
            "="*80,
            "蜜蜂变种自动化评估报告",
            "="*80,
            f"评估时间: {datetime.fromisoformat(result.timestamp).strftime('%Y-%m-%d %H:%M:%S')}",
            f"评估间隔: {self.evaluation_interval_hours}小时",
            "",
            "📊 整体健康度评分:",
            f"  {result.score:.1f}/100 {'🟢 优秀' if result.score >= 80 else '🟡 良好' if result.score >= 60 else '🔴 需要改进'}",
            "",
            "📈 质量统计:",
            f"  高质量: {result.quality_stats['high_quality']}个",
            f"  中等质量: {result.quality_stats['medium_quality']}个",
            f"  低质量: {result.quality_stats['low_quality']}个",
            f"  已废弃: {result.quality_stats['deprecated']}个",
            f"  质量得分: {result.quality_stats['quality_score']:.1f}%",
            "",
            "📊 使用统计:",
            f"  总蜜蜂数: {result.total_bees}",
            f"  已评估蜜蜂: {result.evaluated_bees}",
            f"  活跃使用: {usage_stats['usage_distribution']['active']}个",
            f"  非活跃: {usage_stats['usage_distribution']['inactive']}个",
            f"  未使用: {usage_stats['usage_distribution']['unused']}个",
            f"  长期未使用: {usage_stats['usage_distribution']['stale']}个",
            f"  总使用次数: {usage_stats['total_uses']}",
        ]
        
        if usage_stats['most_used']:
            report_lines.extend([
                "",
                "🏆 最常用的蜜蜂:"
            ])
            for i, bee in enumerate(usage_stats['most_used'][:3], 1):
                report_lines.append(
                    f"  {i}. {bee['bee_name']} - {bee['use_count']}次"
                )
        
        report_lines.extend([
            "",
            "="*80,
            "清理建议:",
            "="*80,
        ])
        
        for rec in result.cleanup_recommendations:
            report_lines.append(f"  {rec}")
        
        report_lines.extend([
            "",
            "="*80,
            "行动项:",
            "="*80,
        ])
        
        for i, action in enumerate(result.action_items, 1):
            report_lines.append(f"  {i}. {action}")
        
        report_lines.extend([
            "",
            "="*80,
            "下次评估时间:",
            f"{(datetime.now() + timedelta(hours=self.evaluation_interval_hours)).strftime('%Y-%m-%d %H:%M:%S')}",
            "="*80,
        ])
        
        return "\n".join(report_lines)
    
    def _save_evaluation_result(self, result: EvaluationResult):
        """保存评估结果到历史记录"""
        try:
            # 加载现有历史
            history = []
            if self.evaluation_history_file.exists():
                with open(self.evaluation_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # 添加新结果
            history.append({
                'timestamp': result.timestamp,
                'total_bees': result.total_bees,
                'evaluated_bees': result.evaluated_bees,
                'quality_stats': result.quality_stats,
                'usage_stats': result.usage_stats,
                'score': result.score
            })
            
            # 只保留最近10次评估
            history = history[-10:]
            
            # 保存历史
            with open(self.evaluation_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"保存评估历史失败: {e}")
    
    def _save_report(self, report: str, timestamp: str):
        """保存报告到文件"""
        try:
            report_file = self.data_dir / f"evaluation_report_{timestamp[:10]}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n评估报告已保存至: {report_file}")
        except Exception as e:
            print(f"保存报告失败: {e}")
    
    def start_scheduler(self):
        """启动定时调度器"""
        if self.is_running:
            print("调度器已在运行中")
            return
        
        self.is_running = True
        
        # 定义定时任务
        def run_scheduled_evaluation():
            try:
                print(f"\n📅 定时评估触发于: {datetime.now()}")
                self.evaluate_bees()
            except Exception as e:
                print(f"定时评估失败: {e}")
        
        # 每72小时运行一次评估
        schedule.every(72).hours.do(run_scheduled_evaluation)
        
        # 启动调度器线程
        def scheduler_loop():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        print(f"✅ 自动评估调度器已启动，每{self.evaluation_interval_hours}小时运行一次")
    
    def stop_scheduler(self):
        """停止定时调度器"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        print("✅ 自动评估调度器已停止")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self.evaluation_history_file.exists():
            return {"error": "没有评估历史记录"}
        
        try:
            with open(self.evaluation_history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            if not history:
                return {"error": "评估历史为空"}
            
            latest = history[-1]
            
            return {
                "total_evaluations": len(history),
                "last_evaluation": latest['timestamp'],
                "last_score": latest['score'],
                "trend": self._calculate_score_trend(history),
                "next_evaluation": (
                    datetime.fromisoformat(latest['timestamp']) + 
                    timedelta(hours=self.evaluation_interval_hours)
                ).isoformat() if self.last_evaluation_time else None
            }
        except Exception as e:
            return {"error": f"读取评估历史失败: {e}"}
    
    def _calculate_score_trend(self, history: List[Dict]) -> str:
        """计算分数趋势"""
        if len(history) < 2:
            return "N/A"
        
        scores = [item['score'] for item in history]
        avg = sum(scores) / len(scores)
        recent_avg = sum(scores[-3:]) / min(3, len(scores))
        
        if recent_avg > avg * 1.05:
            return "📈 上升"
        elif recent_avg < avg * 0.95:
            return "📉 下降"
        else:
            return "➡️ 稳定"


def main():
    """主函数：演示自动化评估器"""
    evaluator = AutomatedBeeEvaluator()
    
    # 运行评估
    result = evaluator.evaluate_bees(force=True)
    
    if result:
        # 显示评估摘要
        summary = evaluator.get_evaluation_summary()
        print(f"\n评估摘要:")
        print(f"  总评估次数: {summary.get('total_evaluations', 0)}")
        print(f"  最近评分: {summary.get('last_score', 0):.1f}/100")
        print(f"  趋势: {summary.get('trend', 'N/A')}")
        print(f"  下次评估: {summary.get('next_evaluation', 'N/A')}")
    
    # 询问是否启动调度器
    print("\n提示：在生产环境中，可以调用 evaluator.start_scheduler() 来启动自动评估")


if __name__ == "__main__":
    main()
