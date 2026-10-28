---
name: trading
description: Token swap, DCA, stop-loss, limit order automation for hive agents
topic: market:trading
---

# Trading Skill

蜂群自主交易技能。对标 bankr/trading-operations。

## 能力
- `swap(from_token, to_token, amount)` — Token 兑换（模拟执行）
- `dca(token, amount, interval_hours, rounds)` — 定投策略
- `stop_loss(token, trigger_pct)` — 止损保护监控

## 使用示例
```
swap BTC USDT 0.1
dca ETH 100 24 7
stop_loss SOL -5%
```

## SkillBus Topic
`market:trading`
