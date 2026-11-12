# -*- coding: utf-8 -*-
"""
Market Research Skill Handler — Price / Sentiment / Technicals
对标 bankr/references/market-research.md
"""

import logging
import random

logger = logging.getLogger("Hive.Skill.Market")

_MOCK_DATA: dict[str, dict] = {
    "BTC": {"price": 98500, "rsi": 62.4, "macd": 0.003, "sentiment": "bullish"},
    "ETH": {"price": 3210, "rsi": 58.1, "macd": 0.001, "sentiment": "neutral"},
    "SOL": {"price": 175, "rsi": 71.8, "macd": 0.008, "sentiment": "bullish"},
    "DOGE": {"price": 0.18, "rsi": 45.2, "macd": -0.001, "sentiment": "bearish"},
}


def handle_market(topic: str, payload: dict) -> None:
    """
    SkillBus handler — topic: market:research
    payload 支持 action: price / sentiment / technicals / scan
    """
    action = payload.get("action", "scan").lower()
    token = payload.get("token", "BTC").upper()

    if action == "price":
        _get_price(token)
    elif action == "sentiment":
        _get_sentiment(token)
    elif action == "technicals":
        _get_technicals(token)
    elif action == "scan":
        _full_scan()
    else:
        logger.warning(f"[Market] 未知 action={action}")


def _get_price(token: str) -> None:
    d = _MOCK_DATA.get(token, {"price": 1.0})
    noise = random.uniform(-0.005, 0.005)
    px = round(d["price"] * (1 + noise), 2)
    print(f"  💰 {token} 实时价格: ${px:,.2f}")


def _get_sentiment(token: str) -> None:
    d = _MOCK_DATA.get(token, {"sentiment": "neutral"})
    icon = {"bullish": "🟢", "neutral": "🟡", "bearish": "🔴"}.get(d["sentiment"], "⚪")
    print(f"  {icon} {token} 市场情绪: {d['sentiment'].upper()}")


def _get_technicals(token: str) -> None:
    d = _MOCK_DATA.get(token, {"rsi": 50, "macd": 0})
    rsi = round(d["rsi"] + random.uniform(-2, 2), 1)
    macd = round(d["macd"] + random.uniform(-0.001, 0.001), 4)
    sig = "超买" if rsi > 70 else ("超卖" if rsi < 30 else "中性")
    print(f"  📊 {token} 技术指标: RSI={rsi}({sig}) MACD={macd:+.4f}")


def _full_scan() -> None:
    print("  🔍 全市场扫描...")
    for token, d in _MOCK_DATA.items():
        icon = {"bullish": "🟢", "neutral": "🟡", "bearish": "🔴"}.get(d["sentiment"], "⚪")
        print(f"    {icon} {token:<6} ${d['price']:>10,.2f} | RSI={d['rsi']} | {d['sentiment']}")
