#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TheHive SIRP (安全事件指挥与编排器)
设计灵感来源于 TheHive-Project (顶级开源安全中心 SOC)。
打破过去系统内工蜂胡乱报错的窘境。现在所有的故障、封禁、蜜罐捕获，统统折叠进这个告警面板。
当在短期内相同类型的告警（如：HTTP 403 封锁）累积超过警戒阈值，它会自动升格为立案 (Case)，
并强行召唤 WardenBee 进行裁决封锁，实现蜂群大军的自愈风控。
"""

import time
import logging
from typing import Dict, List, Any
from hive.core.service_locator import BeeServiceLocator

logger = logging.getLogger("Hive.TheHiveSIRP")

class TheHiveSIRP:
    MAX_ALERTS_POOL_SIZE = 10000  # 防止告警池无限增长

    def __init__(self):
        # alerts_pool 用来收集所有的底层告警
        self.alerts_pool: List[Dict[str, Any]] = []
        self.escalation_threshold = 3  # 同类报错 3 次即升格为严重安全事故
        logger.info("🛡️ [TheHive SIRP] 战区风控雷达网部署完毕。")

    def raise_alert(self, severity: int, alert_type: str, bee_id: str, payload: str):
        """工蜂遭遇空难或火力压制时，向军部拍发的求救告警信"""
        alert = {
            "timestamp": time.time(),
            "severity": severity,
            "type": alert_type,
            "source": bee_id,
            "payload": payload
        }
        self.alerts_pool.append(alert)

        # 防止告警池无限增长，清理超过限制的旧告警
        if len(self.alerts_pool) > self.MAX_ALERTS_POOL_SIZE:
            # 按时间排序，保留最新的 50%
            self.alerts_pool = sorted(self.alerts_pool, key=lambda x: x["timestamp"], reverse=True)[:self.MAX_ALERTS_POOL_SIZE // 2]

        logger.warning(f"⚠️ [SIRP Alert] 捕获来自工蜂 {bee_id} 的前线告警 ({alert_type}): {payload[:50]}...")

        # 随时进行聚合判决
        self._aggregate_and_escalate(alert_type)

    def _aggregate_and_escalate(self, target_type: str):
        """核心降噪机制：寻找相似告警并立案 (Cases)"""
        # 过滤近 5 分钟内的同类告警
        now = time.time()
        recent_similar_alerts = [
            a for a in self.alerts_pool 
            if a["type"] == target_type and (now - a["timestamp"]) < 300
        ]
        
        if len(recent_similar_alerts) >= self.escalation_threshold:
            self._open_case(target_type, recent_similar_alerts)
            
            # 清理已被立案的旧告警以防重复报案
            self.alerts_pool = [a for a in self.alerts_pool if a not in recent_similar_alerts]

    def _open_case(self, incident_type: str, alerts: List[Dict]):
        """立案侦查！召唤法官 WardenBee 控制事态"""
        logger.error(f"💥 [SIRP Case Opened] 事态升级！侦测到 {len(alerts)} 起相似袭击 [{incident_type}]！已自动立案交由狱长处理！")
        
        # 服务定位器发威：隔空呼叫典狱长风控组件
        wm = BeeServiceLocator.get_service("WorkerManager")
        if wm:
            from hive.core.dna import BeeDNA
            # 铸造一个针对此次风控事件专职的典狱长调查令
            case_dna = BeeDNA(
                task=f"SIRP-CASE: 系统遭遇连续同构故障 [{incident_type}]，疑似被拦截或出现断崖。请进行降维求助或环境重启预案。",
                metadata={"alerts": alerts, "priority": "CRITICAL"},
                required_tools=[] # 理论上可以分配风控工具箱
            )
            wm.spawn_worker(case_dna)
            logger.info("🚔 [TheHive SIRP] 已向指挥槽发派 WardenBee 介入案卷处理!")
        else:
            logger.error("🛑 [TheHive SIRP] 系统孤岛！无法定位下发大厅！")
