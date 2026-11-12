# -*- coding: utf-8 -*-
"""
Polymarket Skill Handler — 预测市场搜索/赔率/下注/持仓
对标 bankr/references/polymarket.md
"""

import logging
import random

logger = logging.getLogger("Hive.Skill.Polymarket")

# 模拟预测市场数据库
_MARKETS: list[dict] = [
    {"id": "pm-001", "title": "BTC > $100k by March 2026?", "yes": 0.62, "no": 0.38},
    {"id": "pm-002", "title": "ETH 2.0 staking yield > 5%?", "yes": 0.71, "no": 0.29},
    {"id": "pm-003", "title": "SOL hits $300 in Q1 2026?", "yes": 0.34, "no": 0.66},
    {"id": "pm-004", "title": "Fed rate cut in March 2026?", "yes": 0.55, "no": 0.45},
]
_POSITIONS: dict[str, list] = {}  # agent_id → [bet records]


def handle_polymarket(topic: str, payload: dict) -> None:
    """
    SkillBus handler — topic: market:polymarket
    payload 支持 action: search / odds / bet / positions
    """
    action = payload.get("action", "search").lower()
    agent_id = payload.get("agent_id", "anonymous")

    if action == "search":
        _search(payload.get("query", ""))
    elif action == "odds":
        _odds(payload.get("market_id", "pm-001"))
    elif action == "bet":
        _bet(payload, agent_id)
    elif action == "positions":
        _positions(agent_id)
    else:
        logger.warning(f"[Polymarket] 未知 action={action}")


def _search(query: str) -> None:
    query_lower = query.lower()
    results = [m for m in _MARKETS if not query_lower or query_lower in m["title"].lower()]
    print(f"  🔮 预测市场搜索 ({query or '全部'}):")
    for m in results:
        print(f"    [{m['id']}] {m['title']}")
        print(f"      YES={m['yes']:.0%}  NO={m['no']:.0%}")


def _odds(market_id: str) -> None:
    m = next((x for x in _MARKETS if x["id"] == market_id), None)
    if not m:
        print(f"  ❌ 市场 {market_id} 不存在")
        return
    print(f"  📈 [{market_id}] {m['title']}")
    print(f"     赔率 → YES=${1/m['yes']:.2f}  NO=${1/m['no']:.2f}")


def _bet(p: dict, agent_id: str) -> None:
    market_id = p.get("market_id", "pm-001")
    outcome = p.get("outcome", "YES").upper()
    amount = float(p.get("amount", 10))
    m = next((x for x in _MARKETS if x["id"] == market_id), None)
    if not m:
        print(f"  ❌ 市场不存在")
        return
    odds = m["yes"] if outcome == "YES" else m["no"]
    potential = round(amount / odds, 2)
    record = {"market_id": market_id, "outcome": outcome, "amount": amount, "potential": potential}
    _POSITIONS.setdefault(agent_id, []).append(record)
    print(f"  🎯 下注成功: [{market_id}] {outcome} ${amount} → 潜在收益 ${potential} ✅")
    logger.info(f"[Polymarket] {agent_id} 下注 {market_id}/{outcome} ${amount}")


def _positions(agent_id: str) -> None:
    bets = _POSITIONS.get(agent_id, [])
    if not bets:
        print(f"  📂 {agent_id} 暂无持仓")
        return
    print(f"  📂 {agent_id} 持仓 ({len(bets)} 注):")
    for b in bets:
        print(f"    [{b['market_id']}] {b['outcome']} ${b['amount']} → 潜在 ${b['potential']}")
