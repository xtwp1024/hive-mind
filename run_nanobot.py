#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示 HKUDS/nanobot 心跳主动唤醒 + 技能路由总线的沙盘脚本。
展示 SkillBus 通配符路由 + HiveHeartbeat 定时广播 + 热插拔技能 全闭环。
"""

import time
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from hive.core.skill_bus import SkillBus
from hive.core.heartbeat import HiveHeartbeat, HeartbeatConfig

SEP = "=" * 62
received_events: list[str] = []

# ──────────────────────────────────────── 技能处理函数 ──

def skill_market_scan(topic: str, payload: dict) -> None:
    msg = f"[市场扫描] 收到心跳 topic={topic} at={payload.get('fired_at')}"
    print(f"  🔍 {msg}")
    received_events.append(topic)

def skill_morning_brief(topic: str, payload: dict) -> None:
    msg = f"[晨报生成] 触发 at={payload.get('fired_at')}"
    print(f"  📰 {msg}")
    received_events.append(topic)

def skill_alert_wildcard(topic: str, payload: dict) -> None:
    msg = f"[通配告警] topic={topic} payload={payload}"
    print(f"  🚨 {msg}")
    received_events.append(topic)

def skill_global_logger(topic: str, payload: dict) -> None:
    print(f"  📋 [全局日志] {topic}: {list(payload.keys())}")


def main():
    print(SEP)
    print("  nanobot 心跳守护 + 技能路由总线 - 全流程演示")
    print(SEP)

    # ── 1. 创建 SkillBus 并注册技能（热插拔）─────────────────
    print("\n[Step 1] 创建 SkillBus 并注册技能（对标 nanobot skills loader）")
    bus = SkillBus()

    bus.register("system:heartbeat",   skill_morning_brief,  skill_id="morning-brief",  description="每日晨报生成器")
    bus.register("market:*",           skill_market_scan,    skill_id="market-scan",    description="市场全品种扫描")
    bus.register("alert:*",            skill_alert_wildcard, skill_id="alert-wildcard", description="全类型告警处理器")
    bus.register("*",                  skill_global_logger,  skill_id="global-logger",  description="全局事件日志", async_mode=True)

    print(f"\n  已注册技能列表:")
    for s in bus.list_skills():
        print(f"    [{s['skill_id']:<18}] topic={s['topic']:<22} async={s['async']}")

    # ── 2. 测试手动发布（验证通配符路由）───────────────────────
    print(f"\n{SEP}")
    print("[Step 2] 手动发布消息，验证通配符路由精度...")
    print(f"{'─'*62}")

    print("\n  → 发布 'market:BTC' （应匹配 market:* 和 * 共 2 个技能）")
    n = bus.publish("market:BTC", {"price": 98500, "change": "+2.1%"})
    print(f"    共触发 {n} 个技能")

    time.sleep(0.1)
    print("\n  → 发布 'alert:risk_limit' （应匹配 alert:* 和 * 共 2 个技能）")
    n = bus.publish("alert:risk_limit", {"level": "HIGH", "position": "SOL"})
    print(f"    共触发 {n} 个技能")

    time.sleep(0.1)
    print("\n  → 发布 'system:heartbeat' （应匹配 system:heartbeat 和 * 共 2 个技能）")
    n = bus.publish("system:heartbeat", {"fired_at": "08:00:00"})
    print(f"    共触发 {n} 个技能")

    # ── 3. 热插拔：运行时注销一个技能────────────────────────────
    print(f"\n{SEP}")
    print("[Step 3] 热插拔：运行时注销 global-logger 技能...")
    bus.unregister("global-logger")
    n = bus.publish("market:ETH", {"price": 3210})
    print(f"  注销后 market:ETH 触发技能数: {n}（应为 1，仅 market-scan）")
    assert n == 1, f"热插拔失败，触发了 {n} 个技能"

    # ── 4. 启动 HiveHeartbeat 主动唤醒 ───────────────────────────
    print(f"\n{SEP}")
    print("[Step 4] 启动 HiveHeartbeat（每 2s 心跳一次，运行 6s，对标 nanobot heartbeat）")

    hb = HiveHeartbeat(bus=bus)
    hb.add_beat(HeartbeatConfig(
        name         = "market_watcher",
        topic        = "market:realtime_scan",
        interval_sec = 2.0,
        payload      = {"symbols": ["BTC", "ETH", "SOL"]},
    ))
    hb.add_beat(HeartbeatConfig(
        name         = "morning_brief",
        topic        = "system:heartbeat",
        interval_sec = 3.0,
        payload      = {"briefing": "daily"},
    ))

    hb.start()
    time.sleep(6.5)  # 等 3 个周期
    hb.stop()

    # ── 5. 最终统计报表 ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("[Step 5] 最终 SkillBus 触发统计报表:")
    for topic, cnt in sorted(bus.stats().items()):
        print(f"  {topic:<30}: {cnt} 次")

    total_hb_events = sum(1 for e in received_events if "market:realtime" in e or "system:heartbeat" in e)
    print(f"\n  心跳触发事件总数: {total_hb_events}")
    assert total_hb_events >= 3, "心跳触发次数不足！"

    print(f"\n{SEP}")
    print("  🎉 全部验证通过！心跳守护 + 技能路由总线部署完成！")
    print("  💓 蜂群已升级为「主动守护者」，无需等待命令！")
    print(SEP)

if __name__ == "__main__":
    main()
