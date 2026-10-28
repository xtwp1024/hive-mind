# -*- coding: utf-8 -*-
"""
SkillBus - 技能动态注册与路由总线
设计灵感来源于 HKUDS/nanobot 的 skills/ + bus/ 模块。

核心思想：技能（skill）是一个带 topic 标签的处理函数。
任何消息到达 Bus 后，Bus 按 topic 通配符路由到匹配的处理函数。
技能可以在运行时随时热插拔注册/注销，无需重启系统。

通配符规则（对标 MQTT topic 路由）：
  - 精确匹配：  "system:heartbeat" 只匹配 "system:heartbeat"
  - 单段通配：  "market:*" 匹配   "market:scan", "market:alert"
  - 全局通配：  "*"        匹配   任意 topic
"""

import fnmatch
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("Hive.SkillBus")


@dataclass
class SkillEntry:
    """一条技能注册记录（对标 nanobot skills/xxx.py 的元信息）"""
    skill_id:   str
    topic:      str                   # 匹配的 topic（支持通配符）
    handler:    Callable[[str, dict], Any]
    description: str = ""
    async_mode: bool = False          # True=后台线程执行（对标 nanobot subagent）


class SkillBus:
    """
    技能动态路由总线。
    - register(): 热插拔注册技能（对标 nanobot skills loader）
    - publish():  广播消息，自动路由到所有匹配的技能（对标 nanobot bus dispatcher）
    - unregister(): 热摘除技能
    """

    def __init__(self):
        self._skills: list[SkillEntry] = []
        self._lock   = threading.RLock()
        self._counters: dict[str, int] = {}   # topic → 已处理计数
        logger.info("[SkillBus] 技能路由总线已上线")

    # ─────────────────────────────────── 注册/注销 ──

    def register(
        self,
        topic:       str,
        handler:     Callable[[str, dict], Any],
        skill_id:    str   = "",
        description: str   = "",
        async_mode:  bool  = False,
    ) -> str:
        """
        注册技能处理函数（对标 nanobot skills/xxx.md 自动加载）。
        支持运行时热插拔：无需重启即可加入新能力。
        返回 skill_id（用于后续注销）。
        """
        sid = skill_id or f"{topic}#{id(handler):x}"
        entry = SkillEntry(
            skill_id=sid, topic=topic, handler=handler,
            description=description, async_mode=async_mode,
        )
        with self._lock:
            self._skills.append(entry)
        logger.info(f"[SkillBus] ✅ register skill={sid!r} topic={topic!r} async={async_mode}")
        return sid

    def unregister(self, skill_id: str) -> bool:
        """热摘除技能（对标 nanobot 运行时技能卸载）"""
        with self._lock:
            before = len(self._skills)
            self._skills = [s for s in self._skills if s.skill_id != skill_id]
            removed = len(self._skills) < before
        if removed:
            logger.info(f"[SkillBus] 🗑️  unregister skill={skill_id!r}")
        return removed

    # ─────────────────────────────────── 发布/路由 ──

    def publish(self, topic: str, payload: dict) -> int:
        """
        广播消息（对标 nanobot bus.send）。
        按通配符路由到所有匹配的技能，返回触发的技能数量。
        """
        with self._lock:
            matched = [s for s in self._skills if self._match(s.topic, topic)]

        if not matched:
            logger.debug(f"[SkillBus] topic={topic!r} 无匹配技能")
            return 0

        count = 0
        for entry in matched:
            logger.info(f"[SkillBus] 📨 {topic!r} → {entry.skill_id!r}")
            count += 1
            self._counters[topic] = self._counters.get(topic, 0) + 1
            if entry.async_mode:
                # 后台线程执行（对标 nanobot subagent 异步任务）
                t = threading.Thread(
                    target  = self._safe_call,
                    args    = (entry, topic, payload),
                    daemon  = True,
                )
                t.start()
            else:
                self._safe_call(entry, topic, payload)
        return count

    # ─────────────────────────────────── 状态查询 ──

    def list_skills(self) -> list[dict]:
        """列出所有已注册技能（对标 nanobot /skills 命令）"""
        with self._lock:
            return [
                {
                    "skill_id":   s.skill_id,
                    "topic":      s.topic,
                    "description":s.description,
                    "async":      s.async_mode,
                }
                for s in self._skills
            ]

    def stats(self) -> dict[str, int]:
        """返回各 topic 的触发次数统计"""
        return dict(self._counters)

    # ─────────────────────────────────── 内部辅助 ──

    @staticmethod
    def _match(pattern: str, topic: str) -> bool:
        """通配符匹配（对标 MQTT topic filter 语义）"""
        if pattern == "*":
            return True
        return fnmatch.fnmatch(topic, pattern)

    @staticmethod
    def _safe_call(entry: SkillEntry, topic: str, payload: dict) -> None:
        try:
            entry.handler(topic, payload)
        except Exception as e:
            logger.error(f"[SkillBus] skill={entry.skill_id!r} 处理异常: {e}")
