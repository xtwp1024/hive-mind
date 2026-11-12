#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive Evolution CLI

命令行入口，支持多种运行模式：
- 单次运行: python -m hive.evolution
- 审核模式: python -m hive.evolution --review
- 循环模式: python -m hive.evolution --loop
- 状态查看: python -m hive.evolution --status
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 确保可以导入 hive 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hive.evolution.evolver import EvolutionResult, Evolver
from hive.evolution.strategies import EvolutionStrategy, StrategyType


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def human_review_callback(proposal: dict) -> bool:
    """人工审核回调"""
    print("\n" + "=" * 60)
    print("[REVIEW] 进化提案需要人工审核")
    print("=" * 60)
    print(f"类型: {proposal.get('type', 'unknown')}")
    print(f"描述: {proposal.get('description', 'N/A')}")

    if "patch" in proposal:
        print("\n[PATCH] 补丁内容:")
        print("-" * 40)
        print(proposal["patch"][:500])
        if len(proposal.get("patch", "")) > 500:
            print("... (截断)")
        print("-" * 40)

    if "details" in proposal:
        print("\n[DETAILS] 详情:")
        print(json.dumps(proposal["details"], indent=2, ensure_ascii=False)[:300])

    print()
    while True:
        response = input("是否批准此提案? [y/n/s(跳过)]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        elif response in ("s", "skip"):
            print("已跳过")
            return False
        else:
            print("请输入 y(是), n(否) 或 s(跳过)")


def cmd_run(args):
    """执行单次进化周期"""
    strategy = EvolutionStrategy.from_env()

    # 命令行参数覆盖
    if args.strategy:
        try:
            strategy.strategy_type = StrategyType(args.strategy)
        except ValueError:
            print(f"[WARN] 未知策略: {args.strategy}, 使用默认策略")

    if args.review:
        strategy.require_human_review = True

    on_review = human_review_callback if strategy.require_human_review else None
    evolver = Evolver(strategy=strategy, on_review=on_review)

    print(f"[DNA] Hive Evolution Engine v1.0.0")
    print(f"      Strategy: {strategy.strategy_type.value}")
    print(f"      Review: {'enabled' if strategy.require_human_review else 'disabled'}")
    print()

    result = evolver.run_cycle()

    status_mark = "[OK]" if result.success else "[FAIL]"
    print(f"\n{status_mark} {result.action.upper()}: {result.message}")

    if result.gene_id:
        print(f"      Gene ID: {result.gene_id}")
    if result.capsule_id:
        print(f"      Capsule ID: {result.capsule_id}")

    return 0 if result.success else 1


def cmd_loop(args):
    """持续运行进化循环"""
    strategy = EvolutionStrategy.from_env()

    if args.strategy:
        try:
            strategy.strategy_type = StrategyType(args.strategy)
        except ValueError:
            pass

    if args.review:
        strategy.require_human_review = True

    on_review = human_review_callback if strategy.require_human_review else None
    evolver = Evolver(strategy=strategy, on_review=on_review)

    interval = args.interval or 300  # 默认5分钟
    max_cycles = args.max_cycles

    print(f"[DNA] Hive Evolution Engine v1.0.0 (Loop Mode)")
    print(f"      Strategy: {strategy.strategy_type.value}")
    print(f"      Interval: {interval}s")
    print(f"      Max Cycles: {max_cycles or 'unlimited'}")
    print()

    try:
        evolver.run_loop(interval=interval, max_cycles=max_cycles)
    except KeyboardInterrupt:
        print("\n[STOP] Stopped")

    return 0


def cmd_status(args):
    """查看进化引擎状态"""
    evolver = Evolver()
    status = evolver.get_status()

    print("[DNA] Hive Evolution Engine Status")
    print("=" * 40)
    print(f"Strategy:     {status['strategy']}")
    print(f"Cycles:       {status['cycles']}")
    print(f"Repairs:      {status['repairs']}")
    print(f"Innovations:  {status['innovations']}")
    print(f"Genes:        {status['genes_count']}")
    print(f"Capsules:     {status['capsules_count']}")
    print("=" * 40)

    # 显示最近事件
    events_file = evolver.events_file
    if events_file.exists():
        print("\n[LOG] Recent Events:")
        lines = events_file.read_text(encoding="utf-8").strip().split("\n")
        for line in lines[-5:]:
            if line:
                try:
                    evt = json.loads(line)
                    print(f"  [{evt['type']}] {evt.get('details', {}).get('message', '')[:50]}")
                except Exception:
                    pass

    return 0


def cmd_genes(args):
    """列出所有基因"""
    evolver = Evolver()
    genes = evolver._read_json(evolver.genes_file)

    print("[DNA] Stored Genes")
    print("=" * 60)

    for gene_id, gene in genes.items():
        if gene_id.startswith("_"):
            continue
        print(f"\n[{gene_id}]")
        print(f"  Desc: {gene.get('description', 'N/A')[:60]}")
        print(f"  Type: {gene.get('mutation_type', 'unknown')}")
        if "updated_at" in gene:
            import datetime

            dt = datetime.datetime.fromtimestamp(gene["updated_at"])
            print(f"  Updated: {dt.strftime('%Y-%m-%d %H:%M')}")

    if len([k for k in genes if not k.startswith("_")]) == 0:
        print("  (no genes yet)")

    return 0


def cmd_capsules(args):
    """列出所有胶囊"""
    evolver = Evolver()
    capsules = evolver._read_json(evolver.capsules_file)

    print("[DNA] Stored Capsules")
    print("=" * 60)

    for cap_id, cap in capsules.items():
        if cap_id.startswith("_"):
            continue
        print(f"\n[{cap_id}]")
        print(f"  Desc: {cap.get('description', 'N/A')[:60]}")
        genes = cap.get("genes", [])
        print(f"  Genes: {len(genes)}")

    if len([k for k in capsules if not k.startswith("_")]) == 0:
        print("  (no capsules yet)")

    return 0


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        prog="hive.evolution",
        description="Hive Evolution Engine - Self-improving swarm system",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # run 命令 (默认)
    run_parser = subparsers.add_parser("run", help="Run single evolution cycle")
    run_parser.add_argument("--review", action="store_true", help="Enable human review mode")
    run_parser.add_argument(
        "--strategy", "-s", help="Evolution strategy (balanced/innovate/harden/repair-only)"
    )
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # loop 命令
    loop_parser = subparsers.add_parser("loop", help="Run continuous evolution loop")
    loop_parser.add_argument("--review", action="store_true", help="Enable human review mode")
    loop_parser.add_argument("--strategy", "-s", help="Evolution strategy")
    loop_parser.add_argument(
        "--interval", "-i", type=int, help="Cycle interval in seconds (default 300)"
    )
    loop_parser.add_argument("--max-cycles", "-n", type=int, help="Maximum cycles")
    loop_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # status 命令
    status_parser = subparsers.add_parser("status", help="View evolution engine status")

    # genes 命令
    genes_parser = subparsers.add_parser("genes", help="List all genes")

    # capsules 命令
    capsules_parser = subparsers.add_parser("capsules", help="List all capsules")

    # 兼容旧参数 (直接在顶层)
    parser.add_argument("--review", action="store_true", help="Enable human review mode")
    parser.add_argument("--loop", action="store_true", help="Continuous run mode")
    parser.add_argument("--strategy", "-s", help="Evolution strategy")
    parser.add_argument("--interval", "-i", type=int, default=300, help="Loop interval seconds")
    parser.add_argument("--max-cycles", "-n", type=int, help="Maximum cycles")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--status", action="store_true", help="View status")

    args = parser.parse_args()

    # 设置日志
    setup_logging(getattr(args, "verbose", False))

    # 路由命令
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "loop":
        return cmd_loop(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "genes":
        return cmd_genes(args)
    elif args.command == "capsules":
        return cmd_capsules(args)
    elif getattr(args, "status", False):
        return cmd_status(args)
    elif getattr(args, "loop", False):
        return cmd_loop(args)
    else:
        # 默认执行单次运行
        return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
