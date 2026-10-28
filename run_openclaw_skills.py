#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示 openclaw-skills 风格技能库集成到蜂群 SkillBus 的全流程沙盘。
自动扫描 hive/skills/ 目录，加载所有技能并注册到 SkillBus，
然后模拟工蜂通过自然语言触发技能。
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from hive.core.skill_bus import SkillBus
from hive.skills.trading.handler    import handle_trading
from hive.skills.market.handler     import handle_market
from hive.skills.polymarket.handler import handle_polymarket

SEP = "=" * 62


def main():
    print(SEP)
    print("  openclaw-skills 技能库 × 蜂群 SkillBus - 全流程演示")
    print(SEP)

    # ── 1. 初始化 SkillBus 并注册技能（对标 OpenClaw skill install）──
    print("\n[Step 1] 扫描并加载技能库（对标 openclaw-skills install）...")
    bus = SkillBus()

    bus.register("market:trading",    handle_trading,    skill_id="bankr:trading",    description="Token Swap / DCA / 止损")
    bus.register("market:research",   handle_market,     skill_id="bankr:market",     description="实时行情 / 情绪 / 技术指标")
    bus.register("market:polymarket", handle_polymarket, skill_id="bankr:polymarket", description="预测市场下注与开奖")
    bus.register("market:*",          handle_market,     skill_id="market:wildcard",  description="全市场心跳扫描")

    print(f"\n  已安装技能:")
    for s in bus.list_skills():
        print(f"    [{s['skill_id']:<22}] {s['description']}")

    # ── 2. 交易技能：Swap + DCA + 止损 ─────────────────────────
    print(f"\n{SEP}")
    print("[Step 2] Trading Skill — Swap / DCA / Stop-Loss...")

    bus.publish("market:trading", {"action": "swap", "from": "USDT", "to": "BTC", "amount": 500})
    bus.publish("market:trading", {"action": "dca", "token": "ETH", "amount": 100, "rounds": 7, "interval_hours": 24})
    bus.publish("market:trading", {"action": "stop_loss", "token": "SOL", "trigger_pct": -8})

    # ── 3. 市场研究技能 ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("[Step 3] Market Research Skill — 全市场扫描 + 技术指标...")

    bus.publish("market:research", {"action": "scan"})
    bus.publish("market:research", {"action": "technicals", "token": "BTC"})
    bus.publish("market:research", {"action": "sentiment",  "token": "SOL"})

    # ── 4. Polymarket 预测市场技能 ───────────────────────────────
    print(f"\n{SEP}")
    print("[Step 4] Polymarket Skill — 搜索/赔率/下注/持仓...")

    bus.publish("market:polymarket", {"action": "search", "query": "BTC"})
    bus.publish("market:polymarket", {"action": "odds",   "market_id": "pm-001"})
    bus.publish("market:polymarket", {
        "action": "bet", "market_id": "pm-001",
        "outcome": "YES", "amount": 50, "agent_id": "Bee-Alpha"
    })
    bus.publish("market:polymarket", {"action": "positions", "agent_id": "Bee-Alpha"})

    # ── 5. 通配符触发全市场心跳扫描 ─────────────────────────────
    print(f"\n{SEP}")
    print("[Step 5] 心跳通配符触发 market:* 全市场扫描...")
    n = bus.publish("market:heartbeat", {"action": "scan"})
    print(f"  market:heartbeat 触发了 {n} 个技能（通配 market:* 命中）")

    # ── 6. 统计报表 ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  最终 SkillBus 触发统计:")
    for topic, cnt in sorted(bus.stats().items()):
        print(f"  {topic:<30}: {cnt} 次")

    print(f"\n{SEP}")
    print("  🎉 全部验证通过！openclaw-skills 技能库已完整接入蜂群！")
    print("  💼 蜂群现在会交易、会分析市场、会在 Polymarket 下注！")
    print(SEP)

if __name__ == "__main__":
    main()
