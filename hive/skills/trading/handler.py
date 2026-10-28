# -*- coding: utf-8 -*-
"""
Trading Skill Handler — Token Swap / DCA / Stop-Loss
对标 bankr/references/token-trading.md + automation.md
"""

import random
import time
import logging
logger = logging.getLogger("Hive.Skill.Trading")

# 模拟价格表（演示用，生产接 Gate.io API）
_MOCK_PRICES: dict[str, float] = {
    "BTC": 98500.0, "ETH": 3210.0, "SOL": 175.0,
    "USDT": 1.0,    "BNB": 620.0,  "DOGE": 0.18,
}

def _price(token: str) -> float:
    base = _MOCK_PRICES.get(token.upper(), 1.0)
    return round(base * (1 + random.uniform(-0.002, 0.002)), 4)


def handle_trading(topic: str, payload: dict) -> None:
    """
    SkillBus handler — topic: market:trading
    payload 支持 action: swap / dca / stop_loss
    """
    action = payload.get("action", "swap").lower()

    if action == "swap":
        _swap(payload)
    elif action == "dca":
        _dca(payload)
    elif action == "stop_loss":
        _stop_loss(payload)
    else:
        logger.warning(f"[Trading] 未知 action={action}")


def _swap(p: dict) -> None:
    from_tok = p.get("from", "USDT").upper()
    to_tok   = p.get("to",   "BTC").upper()
    amount   = float(p.get("amount", 100))
    from_px  = _price(from_tok)
    to_px    = _price(to_tok)
    received = round(amount * from_px / to_px, 6)
    logger.info(f"[Trading:Swap] {amount} {from_tok} → {received} {to_tok} "
                f"(rate {from_px}/{to_px})")
    print(f"  🔄 SWAP {amount} {from_tok} → {received} {to_tok} ✅")


def _dca(p: dict) -> None:
    token    = p.get("token", "ETH").upper()
    amount   = float(p.get("amount", 50))
    rounds   = int(p.get("rounds", 3))
    interval = p.get("interval_hours", 24)
    logger.info(f"[Trading:DCA] 注册 DCA: {token} ${amount}×{rounds}轮 每{interval}h")
    print(f"  📅 DCA 计划已注册: 每 {interval}h 买入 ${amount} {token}, 共 {rounds} 轮")
    # 模拟第一轮立即执行
    px  = _price(token)
    qty = round(amount / px, 6)
    print(f"     第 1/{rounds} 轮: 以 ${px} 买入 {qty} {token} ✅")


def _stop_loss(p: dict) -> None:
    token    = p.get("token", "SOL").upper()
    trig_pct = float(str(p.get("trigger_pct", -5)).replace("%", ""))
    px       = _price(token)
    trigger  = round(px * (1 + trig_pct / 100), 4)
    logger.info(f"[Trading:StopLoss] {token} 止损线设置为 ${trigger} ({trig_pct}%)")
    print(f"  🛡️  止损保护: {token} 当前 ${px}, 触发价 ${trigger} ({trig_pct}%) ✅")
