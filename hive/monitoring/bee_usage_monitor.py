#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蜜蜂使用情况监控器 - BeeUsageMonitor
用于监控蜜蜂变种的使用情况，自动统计使用频率和最后使用时间

功能：
1. 跟踪每个蜜蜂的使用次数
2. 记录最后使用时间
3. 识别长期未使用的蜜蜂
4. 生成使用统计报告
5. 自动标记需要清理的蜜蜂
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum


class BeeUsageStatus(Enum):
    """蜜蜂使用状态"""
    ACTIVE = "活跃使用"
    INACTIVE = "非活跃"
    UNUSED = "未使用"
    STALE = "长期未使用"


@dataclass
class BeeUsageRecord:
    """蜜蜂使用记录"""
    bee_name: str
    use_count: int = 0
    first_used: Optional[str] = None  # ISO格式时间戳
    last_used: Optional[str] = None  # ISO格式时间戳
    recent_uses: List[str] = field(default_factory=list)  # 最近10次使用时间
    
    def mark_used(self):
        """标记蜜蜂被使用"""
        self.use_count += 1
        now = datetime.now().isoformat()
        
        if self.first_used is None:
            self.first_used = now
        
        self.last_used = now
        
        # 只保留最近10次使用记录
        self.recent_uses.insert(0, now)
        if len(self.recent_uses) > 10:
            self.recent_uses = self.recent_uses[:10]
    
    def get_status(self, stale_days: int = 30) -> BeeUsageStatus:
        """
        获取使用状态
        
        Args:
            stale_days: 长期未使用的天数阈值
            
        Returns:
            使用状态
        """
        if self.use_count == 0:
            return BeeUsageStatus.UNUSED
        
        if self.last_used is None:
            return BeeUsageStatus.UNUSED
        
        last_used_time = datetime.fromisoformat(self.last_used)
        days_since_use = (datetime.now() - last_used_time).days
        
        if days_since_use <= 7:
            return BeeUsageStatus.ACTIVE
        elif days_since_use <= stale_days:
            return BeeUsageStatus.INACTIVE
        else:
            return BeeUsageStatus.STALE
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class BeeUsageMonitor:
    """
    蜜蜂使用监控器

    功能：
    1. 记录蜜蜂使用情况
    2. 生成使用统计报告
    3. 识别需要清理的蜜蜂
    4. 自动持久化数据
    """

    def __init__(self, data_dir: str = "hive/monitoring/data"):
        """
        初始化监控器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.usage_file = self.data_dir / "bee_usage_records.json"
        self.statistics_file = self.data_dir / "bee_usage_statistics.json"

        # 加载使用记录
        self.usage_records: Dict[str, BeeUsageRecord] = self._load_usage_records()

        # 蜜蜂变种目录
        self.variants_dir = Path("hive/bees/variants")

        # 线程安全锁
        self._lock = threading.Lock()
    
    def _load_usage_records(self) -> Dict[str, BeeUsageRecord]:
        """加载使用记录"""
        if not self.usage_file.exists():
            return {}
        
        try:
            with open(self.usage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = {}
            for bee_name, record_data in data.items():
                records[bee_name] = BeeUsageRecord(
                    bee_name=bee_name,
                    use_count=record_data.get('use_count', 0),
                    first_used=record_data.get('first_used'),
                    last_used=record_data.get('last_used'),
                    recent_uses=record_data.get('recent_uses', [])
                )
            
            return records
        except Exception as e:
            print(f"加载使用记录失败: {e}")
            return {}
    
    def _save_usage_records(self):
        """保存使用记录"""
        try:
            data = {
                bee_name: record.to_dict()
                for bee_name, record in self.usage_records.items()
            }
            
            with open(self.usage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存使用记录失败: {e}")
    
    def mark_bee_used(self, bee_name: str):
        """
        标记蜜蜂被使用

        Args:
            bee_name: 蜜蜂名称
        """
        with self._lock:
            if bee_name not in self.usage_records:
                self.usage_records[bee_name] = BeeUsageRecord(bee_name=bee_name)

            self.usage_records[bee_name].mark_used()
            self._save_usage_records()
    
    def get_usage_statistics(self, stale_days: int = 30) -> Dict[str, Any]:
        """
        获取使用统计

        Args:
            stale_days: 长期未使用的天数阈值

        Returns:
            统计信息
        """
        # 扫描所有蜜蜂变种
        all_bees = []
        if self.variants_dir.exists():
            bee_files = list(self.variants_dir.glob("*_bee.py"))
            all_bees = [f.stem for f in bee_files if f.name != "__init__.py"]

        with self._lock:
            # 统计信息
            stats = {
                "total_bees": len(all_bees),
                "tracked_bees": len(self.usage_records),
                "untracked_bees": len([b for b in all_bees if b not in self.usage_records]),
                "usage_distribution": {
                    "active": 0,
                    "inactive": 0,
                    "unused": 0,
                    "stale": 0
                },
                "most_used": [],
                "least_used": [],
                "stale_bees": [],
                "unused_bees": [],
                "total_uses": sum(record.use_count for record in self.usage_records.values())
            }

            # 分类统计
            for bee_name, record in self.usage_records.items():
                status = record.get_status(stale_days)
                # 使用status.name（英文）而不是status.value（中文）
                stats["usage_distribution"][status.name.lower()] = stats["usage_distribution"][status.name.lower()] + 1

                if status == BeeUsageStatus.STALE:
                    stats["stale_bees"].append({
                        "bee_name": bee_name,
                        "last_used": record.last_used,
                        "days_since_use": self._days_since_last_use(record.last_used)
                    })
                elif status == BeeUsageStatus.UNUSED:
                    stats["unused_bees"].append(bee_name)

            # 排序
            sorted_records = sorted(
                self.usage_records.items(),
                key=lambda x: x[1].use_count,
                reverse=True
            )

            # 最常用的5个蜜蜂
            stats["most_used"] = [
                {
                    "bee_name": bee_name,
                    "use_count": record.use_count,
                    "last_used": record.last_used
                }
                for bee_name, record in sorted_records[:5]
            ]

            # 最少使用的5个蜜蜂（至少使用过一次）
            used_records = [(b, r) for b, r in sorted_records if r.use_count > 0]
            stats["least_used"] = [
                {
                    "bee_name": bee_name,
                    "use_count": record.use_count,
                    "last_used": record.last_used
                }
                for bee_name, record in used_records[-5:]
            ]

            # 按使用频率排序的长期未使用蜜蜂
            stats["stale_bees"] = sorted(
                stats["stale_bees"],
                key=lambda x: x["days_since_use"],
                reverse=True
            )

        return stats
    
    def _days_since_last_use(self, last_used: Optional[str]) -> int:
        """计算距离上次使用的天数"""
        if last_used is None:
            return 999
        
        try:
            last_used_time = datetime.fromisoformat(last_used)
            return (datetime.now() - last_used_time).days
        except (ValueError, TypeError):
            return 999
    
    def generate_usage_report(self, stale_days: int = 30) -> str:
        """
        生成使用情况报告
        
        Args:
            stale_days: 长期未使用的天数阈值
            
        Returns:
            格式化的报告文本
        """
        stats = self.get_usage_statistics(stale_days)
        
        report_lines = [
            "=" * 80,
            "蜜蜂使用情况监控报告",
            "=" * 80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 总体统计:",
            f"  总蜜蜂数: {stats['total_bees']}",
            f"  已跟踪蜜蜂: {stats['tracked_bees']}",
            f"  未跟踪蜜蜂: {stats['untracked_bees']}",
            f"  总使用次数: {stats['total_uses']}",
            "",
            "📈 使用分布:",
            f"  活跃使用: {stats['usage_distribution']['active']}个",
            f"  非活跃: {stats['usage_distribution']['inactive']}个",
            f"  未使用: {stats['usage_distribution']['unused']}个",
            f"  长期未使用 (> {stale_days}天): {stats['usage_distribution']['stale']}个",
            "",
            "🏆 最常用的蜜蜂 (Top 5):"
        ]
        
        for i, bee in enumerate(stats["most_used"], 1):
            report_lines.append(
                f"  {i}. {bee['bee_name']} - {bee['use_count']}次 (最后使用: {bee['last_used'][:10]})"
            )
        
        report_lines.extend([
            "",
            "📉 最少使用的蜜蜂 (Top 5):"
        ])
        
        for i, bee in enumerate(stats["least_used"], 1):
            report_lines.append(
                f"  {i}. {bee['bee_name']} - {bee['use_count']}次 (最后使用: {bee['last_used'][:10]})"
            )
        
        if stats["stale_bees"]:
            report_lines.extend([
                "",
                f"⚠️  长期未使用的蜜蜂 (> {stale_days}天):"
            ])
            for i, bee in enumerate(stats["stale_bees"], 1):
                report_lines.append(
                    f"  {i}. {bee['bee_name']} - {bee['days_since_use']}天未使用"
                )
        
        if stats["unused_bees"]:
            report_lines.extend([
                "",
                "❌ 完全未使用的蜜蜂:"
            ])
            for i, bee in enumerate(stats["unused_bees"], 1):
                report_lines.append(f"  {i}. {bee}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "建议:",
            "=" * 80
        ])
        
        recommendations = self._generate_recommendations(stats)
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"  {i}. {rec}")
        
        return "\n".join(report_lines)
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """生成清理建议"""
        recommendations = []
        
        stale_count = len(stats["stale_bees"])
        unused_count = len(stats["unused_bees"])
        total_bees = stats["total_bees"]
        
        if unused_count > 0:
            recommendations.append(
                f"发现{unused_count}个完全未使用的蜜蜂，建议评估是否需要保留"
            )
        
        if stale_count > 0:
            recommendations.append(
                f"发现{stale_count}个长期未使用的蜜蜂，建议检查是否仍有价值"
            )
        
        if stats["usage_distribution"]["active"] < total_bees * 0.3:
            recommendations.append(
                f"活跃蜜蜂占比低于30%，建议精简蜜蜂变种，专注于核心功能"
            )
        
        if stale_count + unused_count > total_bees * 0.5:
            recommendations.append(
                f"警告：超过50%的蜜蜂长期未使用，建议进行全面清理"
            )
        
        if stats["untracked_bees"] > 0:
            recommendations.append(
                f"有{stats['untracked_bees']}个蜜蜂未在监控系统中，建议集成使用追踪"
            )
        
        recommendations.append(
            "定期审查蜜蜂使用情况（建议每72小时一次）"
        )
        
        return recommendations
    
    def save_statistics(self, stale_days: int = 30):
        """
        保存统计信息到文件
        
        Args:
            stale_days: 长期未使用的天数阈值
        """
        stats = self.get_usage_statistics(stale_days)
        
        with open(self.statistics_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def cleanup_candidates(self, stale_days: int = 30) -> List[str]:
        """
        获取清理候选列表
        
        Args:
            stale_days: 长期未使用的天数阈值
            
        Returns:
            建议清理的蜜蜂名称列表
        """
        candidates = []
        
        for bee_name, record in self.usage_records.items():
            status = record.get_status(stale_days)
            if status in [BeeUsageStatus.UNUSED, BeeUsageStatus.STALE]:
                candidates.append(bee_name)
        
        return candidates


def main():
    """主函数：演示使用监控器"""
    monitor = BeeUsageMonitor()
    
    # 生成报告
    report = monitor.generate_usage_report()
    print(report)
    
    # 保存统计信息
    monitor.save_statistics()
    print("\n统计信息已保存至:", monitor.statistics_file)


if __name__ == "__main__":
    main()
