# -*- coding: utf-8 -*-
"""
HiveHeartbeat - 主动唤醒守护线程
设计灵感来源于 HKUDS/nanobot 的 heartbeat/ 模块。

核心思想：AI 不是被动等待 Request，而是按节奏自主"心跳"，
主动触发定时任务（晨报、市场扫描、定时健康检查...）。
每次心跳向 SkillBus 广播事件，由注册的技能处理函数响应。
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("Hive.Heartbeat")


@dataclass
class HeartbeatConfig:
    """单个心跳定时器配置（对标 nanobot cron/ 定时配置）"""
    name:           str             # 心跳名称 e.g. "morning_brief"
    topic:          str             # 广播的 SkillBus topic e.g. "system:heartbeat"
    interval_sec:   float           # 触发间隔（秒）
    payload:        dict = field(default_factory=dict)  # 附带数据
    enabled:        bool = True


class HiveHeartbeat:
    """
    主动唤醒守护引擎。对标 nanobot heartbeat/ 模块。
    启动后后台永续运行，按配置的间隔定期向 SkillBus 发布心跳事件。
    完全解耦：不依赖任何具体工蜂，只负责打点计时与事件广播。
    """

    def __init__(self, bus: "SkillBus"):  # type: ignore
        self._bus      = bus
        self._configs: list[HeartbeatConfig] = []
        self._threads: list[threading.Thread] = []
        self._stop_ev  = threading.Event()

    # ─────────────────────────────────────── 配置接口 ──

    def add_beat(self, config: HeartbeatConfig) -> None:
        """注册一个心跳计划（对标 nanobot cron.add）"""
        self._configs.append(config)
        logger.info(f"[Heartbeat] ➕ 注册心跳: {config.name!r} → {config.topic} / 间隔 {config.interval_sec}s")

    # ─────────────────────────────────────── 生命周期 ──

    def start(self) -> None:
        """启动全部心跳线程（对标 nanobot gateway 启动时 heartbeat 自动激活）"""
        if self._threads:
            logger.warning("[Heartbeat] 已在运行，忽略重复 start() 调用")
            return
        self._stop_ev.clear()
        for cfg in self._configs:
            if not cfg.enabled:
                continue
            t = threading.Thread(
                target  = self._beat_loop,
                args    = (cfg,),
                name    = f"Heartbeat-{cfg.name}",
                daemon  = True,
            )
            self._threads.append(t)
            t.start()
            logger.info(f"[Heartbeat] 💓 {cfg.name!r} 心跳线程已启动")

    def stop(self) -> None:
        """优雅停止全部心跳（对标 nanobot gateway shutdown）"""
        self._stop_ev.set()
        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()
        logger.info("[Heartbeat] 所有心跳已停止")

    # ─────────────────────────────────────── 内部循环 ──

    def _beat_loop(self, cfg: HeartbeatConfig) -> None:
        """单个心跳的无限循环（间隔触发）"""
        logger.info(f"[Heartbeat:{cfg.name}] 准备就绪，首次心跳将在 {cfg.interval_sec}s 后触发")
        while not self._stop_ev.wait(timeout=cfg.interval_sec):
            if not cfg.enabled:
                continue
            payload = {
                **cfg.payload,
                "heartbeat_name": cfg.name,
                "fired_at":       time.strftime("%H:%M:%S"),
            }
            logger.info(f"[Heartbeat:{cfg.name}] 💓 心跳触发 → {cfg.topic}")
            try:
                self._bus.publish(cfg.topic, payload)
            except Exception as e:
                logger.error(f"[Heartbeat:{cfg.name}] 广播异常: {e}")
