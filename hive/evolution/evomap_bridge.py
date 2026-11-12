#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoMap Bridge - 连接 EvoMap 进化网络

将 Hive 的进化成果发布到 EvoMap，并从网络获取高价值知识。
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("Hive.Evolution.EvoMap")

EVOMAP_API = "https://evomap.ai/a2a"


@dataclass
class EvoMapConfig:
    """EvoMap 配置"""

    node_id: str = ""
    api_url: str = EVOMAP_API
    enabled: bool = True
    auto_publish: bool = False  # 自动发布高价值基因
    auto_fetch: bool = True  # 自动获取热门胶囊
    min_gdi_threshold: float = 40.0  # 最低 GDI 阈值

    @classmethod
    def from_env(cls) -> "EvoMapConfig":
        return cls(
            node_id=os.getenv("EVOMAP_NODE_ID", ""),
            api_url=os.getenv("EVOMAP_API_URL", EVOMAP_API),
            enabled=os.getenv("EVOMAP_ENABLED", "true").lower() == "true",
            auto_publish=os.getenv("EVOMAP_AUTO_PUBLISH", "false").lower() == "true",
            auto_fetch=os.getenv("EVOMAP_AUTO_FETCH", "true").lower() == "true",
            min_gdi_threshold=float(os.getenv("EVOMAP_MIN_GDI", "40.0")),
        )


class EvoMapBridge:
    """
    EvoMap 桥接器

    实现 GEP-A2A 协议，与 EvoMap 进化网络通信。
    """

    def __init__(self, config: Optional[EvoMapConfig] = None):
        self.config = config or EvoMapConfig.from_env()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "Hive-Evolver/1.0",
            }
        )
        self._credit_balance = 0
        self._last_heartbeat = 0

    def _make_message(self, msg_type: str, payload: dict) -> dict:
        """构造 GEP-A2A 协议消息"""
        return {
            "protocol": "gep-a2a",
            "protocol_version": "1.0.0",
            "message_type": msg_type,
            "message_id": f"hive_{uuid.uuid4().hex[:12]}",
            "sender_id": self.config.node_id or f"hive_{uuid.uuid4().hex[:8]}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload": payload,
        }

    def heartbeat(self) -> dict:
        """发送心跳，获取账户状态"""
        if not self.config.enabled:
            return {"status": "disabled"}

        try:
            msg = self._make_message("heartbeat", {"include_balance": True})
            resp = self._session.post(
                f"{self.config.api_url}/heartbeat",
                json=msg,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            self._credit_balance = data.get("credit_balance", 0)
            self._last_heartbeat = time.time()

            # 更新 node_id
            if data.get("your_node_id"):
                self.config.node_id = data["your_node_id"]

            logger.info(
                f"🌐 EvoMap 心跳成功: 节点={self.config.node_id}, 积分={self._credit_balance}"
            )
            return data

        except Exception as e:
            logger.warning(f"EvoMap 心跳失败: {e}")
            return {"status": "error", "error": str(e)}

    def fetch_trending(self, limit: int = 10) -> List[dict]:
        """获取热门胶囊"""
        if not self.config.enabled:
            return []

        try:
            resp = self._session.get(
                f"{self.config.api_url}/assets",
                params={"status": "promoted", "limit": limit},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            assets = data.get("assets", [])
            logger.info(f"🔥 获取到 {len(assets)} 个热门胶囊")
            return assets

        except Exception as e:
            logger.warning(f"获取热门胶囊失败: {e}")
            return []

    def fetch_asset(self, asset_id: str) -> Optional[dict]:
        """获取单个资产详情"""
        if not self.config.enabled:
            return None

        try:
            resp = self._session.get(
                f"{self.config.api_url}/assets/{asset_id}",
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            logger.warning(f"获取资产 {asset_id} 失败: {e}")
            return None

    def publish_gene(self, gene: dict) -> Optional[str]:
        """
        发布基因到 EvoMap

        Args:
            gene: 基因数据，需包含 title, description, content

        Returns:
            发布成功返回 bundle_id，失败返回 None
        """
        if not self.config.enabled or not self.config.auto_publish:
            logger.debug("EvoMap 发布已禁用")
            return None

        try:
            payload = {
                "gene": {
                    "title": gene.get("title", "Hive Evolution Gene"),
                    "description": gene.get("description", ""),
                    "content": gene.get("content", json.dumps(gene)),
                    "tags": gene.get("tags", ["hive", "evolution"]),
                    "source": "hive-evolver",
                },
                "capsule": {
                    "title": f"Capsule: {gene.get('title', 'Unknown')}",
                    "summary": gene.get("description", ""),
                },
            }

            msg = self._make_message("submit_bundle", payload)
            resp = self._session.post(
                f"{self.config.api_url}/submit",
                json=msg,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            bundle_id = data.get("bundle_id")
            if bundle_id:
                logger.info(f"📤 基因已发布到 EvoMap: {bundle_id}")
            return bundle_id

        except Exception as e:
            logger.warning(f"发布基因失败: {e}")
            return None

    def import_capsule_to_gep(self, capsule: dict, evolver) -> bool:
        """
        将 EvoMap 胶囊导入到本地 GEP 系统

        Args:
            capsule: EvoMap 胶囊数据
            evolver: Evolver 实例

        Returns:
            是否成功导入
        """
        try:
            capsule_id = f"evomap_{capsule.get('id', uuid.uuid4().hex[:8])}"

            local_capsule = {
                "id": capsule_id,
                "source": "evomap",
                "original_id": capsule.get("id"),
                "title": capsule.get("title", ""),
                "summary": capsule.get("summary", ""),
                "gdi": capsule.get("gdi", 0),
                "tags": capsule.get("tags", []),
                "imported_at": time.time(),
            }

            # 保存到本地 GEP
            evolver.save_capsule(capsule_id, local_capsule)
            logger.info(f"📥 已导入胶囊: {capsule.get('title', capsule_id)}")
            return True

        except Exception as e:
            logger.error(f"导入胶囊失败: {e}")
            return False

    def sync_with_network(self, evolver) -> dict:
        """
        与 EvoMap 网络同步

        1. 发送心跳
        2. 获取热门胶囊
        3. 导入高价值知识

        Returns:
            同步结果统计
        """
        stats = {
            "heartbeat": False,
            "fetched": 0,
            "imported": 0,
            "published": 0,
        }

        # 心跳
        hb = self.heartbeat()
        stats["heartbeat"] = hb.get("status") == "ok"

        if not stats["heartbeat"]:
            return stats

        # 获取热门
        if self.config.auto_fetch:
            trending = self.fetch_trending(limit=15)
            stats["fetched"] = len(trending)

            # 导入高 GDI 胶囊
            for capsule in trending:
                gdi = capsule.get("gdi", 0)
                if gdi >= self.config.min_gdi_threshold:
                    if self.import_capsule_to_gep(capsule, evolver):
                        stats["imported"] += 1

        logger.info(f"🔄 EvoMap 同步完成: {stats}")
        return stats


# 全局实例
_bridge: Optional[EvoMapBridge] = None


def get_evomap_bridge() -> EvoMapBridge:
    """获取全局 EvoMap 桥接器"""
    global _bridge
    if _bridge is None:
        _bridge = EvoMapBridge()
    return _bridge
