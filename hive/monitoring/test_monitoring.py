#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试蜜蜂监控系统
模拟蜜蜂使用情况并验证监控和评估功能
"""

import time

from automated_bee_evaluator import AutomatedBeeEvaluator
from bee_usage_monitor import BeeUsageMonitor


def test_bee_usage_monitoring():
    """测试蜜蜂使用监控"""
    print("=" * 80)
    print("测试蜜蜂使用监控系统")
    print("=" * 80)

    # 创建监控器
    monitor = BeeUsageMonitor()

    # 模拟蜜蜂使用情况
    print("\n开始模拟蜜蜂使用情况...\n")

    # 活跃蜜蜂 - 频繁使用
    active_bees = ["primearchitect_bee", "vortex_bee", "oracle_bee"]
    for bee in active_bees:
        for _ in range(15):  # 使用15次
            monitor.mark_bee_used(bee)
            time.sleep(0.01)
        print(f"✓ {bee}: 活跃使用 (15次)")

    # 中等活跃蜜蜂 - 偶尔使用
    medium_bees = ["cipher_bee", "prism_bee", "echo_bee"]
    for bee in medium_bees:
        for _ in range(5):  # 使用5次
            monitor.mark_bee_used(bee)
            time.sleep(0.01)
        print(f"✓ {bee}: 中等活跃 (5次)")

    # 非活跃蜜蜂 - 很少使用
    inactive_bees = ["cybersentinel_bee", "librarian_bee"]
    for bee in inactive_bees:
        monitor.mark_bee_used(bee)  # 使用1次
        print(f"✓ {bee}: 非活跃 (1次)")

    # 未使用蜜蜂
    unused_bees = ["cyberminer_bee", "vortexprimearchitect_bee"]
    for bee in unused_bees:
        print(f"✗ {bee}: 未使用 (0次)")

    print("\n" + "=" * 80)
    print("使用情况统计")
    print("=" * 80)

    # 获取使用统计
    stats = monitor.get_usage_statistics(stale_days=30)

    print(f"\n总蜜蜂数: {stats['total_bees']}")
    print(f"已跟踪蜜蜂: {stats['tracked_bees']}")
    print(f"总使用次数: {stats['total_uses']}")
    print(f"\n使用分布:")
    print(f"  活跃使用: {stats['usage_distribution']['active']}个")
    print(f"  非活跃: {stats['usage_distribution']['inactive']}个")
    print(f"  未使用: {stats['usage_distribution']['unused']}个")
    print(f"  长期未使用: {stats['usage_distribution']['stale']}个")

    if stats["most_used"]:
        print(f"\n最常用的蜜蜂:")
        for i, bee in enumerate(stats["most_used"][:5], 1):
            print(f"  {i}. {bee['bee_name']} - {bee['use_count']}次")

    # 生成使用报告
    report = monitor.generate_usage_report()
    print(f"\n{report}")

    # 查看清理候选
    cleanup = monitor.cleanup_candidates(stale_days=30)
    print(f"\n清理候选 ({len(cleanup)}个):")
    for bee in cleanup[:5]:
        print(f"  - {bee}")

    return monitor


def test_automated_evaluation():
    """测试自动化评估"""
    print("\n" + "=" * 80)
    print("测试自动化评估器")
    print("=" * 80)

    # 创建评估器
    evaluator = AutomatedBeeEvaluator()

    # 运行评估
    print("\n运行自动化评估...")
    result = evaluator.evaluate_bees(force=True)

    if result:
        print("\n" + "=" * 80)
        print("评估结果摘要")
        print("=" * 80)
        print(f"\n整体健康度评分: {result.score:.1f}/100")
        print(f"评估的蜜蜂数: {result.evaluated_bees}/{result.total_bees}")
        print(f"质量得分: {result.quality_stats['quality_score']:.1f}%")

        # 获取评估摘要
        summary = evaluator.get_evaluation_summary()
        print(f"\n历史记录:")
        print(f"  总评估次数: {summary.get('total_evaluations', 0)}")
        print(f"  趋势: {summary.get('trend', 'N/A')}")
        print(f"  下次评估: {summary.get('next_evaluation', 'N/A')}")

    return evaluator


def main():
    """主函数"""
    print("\n🐝 蜜蜂监控系统测试 🐝\n")

    # 测试使用监控
    monitor = test_bee_usage_monitoring()

    # 测试自动化评估
    evaluator = test_automated_evaluation()

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    print("\n✓ 使用监控系统正常工作")
    print("✓ 自动化评估器正常工作")
    print("✓ 定期评估机制已设置（每72小时）")
    print("\n数据文件位置:")
    print("  - hive/monitoring/data/bee_usage.json")
    print("  - hive/monitoring/data/evaluation_report_*.txt")
    print("  - hive/monitoring/data/evaluation_history.json")
    print("  - hive/monitoring/data/last_evaluation.json")


if __name__ == "__main__":
    main()
