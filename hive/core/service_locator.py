#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BeeServiceLocator (全局服务树/组件依赖解耦器)
设计灵感来源于 alibaba/BeeHive (iOS 顶级组件解耦路由)。
这套容器切断了神级上帝类（如 QueenBee）对其它辅助系统（进化、共识等）的硬实例依赖声明。
- 所有插拔式服务引擎在启动时向此大盘注册 (Bind)。
- 消费者想调用服务，只需按名称请求 (Resolve)，实现纯天然解耦微服务化。
"""

import logging
import threading
from typing import Any, Dict, Type

logger = logging.getLogger("Hive.ServiceLocator")


class BeeServiceLocator:
    """单一的全局微件管理大本营"""

    _services: Dict[str, Any] = {}
    _factories: Dict[str, Type] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def register_service(cls, protocol_name: str, instance: Any):
        """将一个已实例化的实体服务驻留到大本营"""
        with cls._lock:
            if protocol_name in cls._services:
                logger.warning(f"⚠️ [Locator] 服务 {protocol_name} 被覆盖注册！")
            cls._services[protocol_name] = instance
        logger.info(f"🔗 [Locator] 特种服务模块 [{protocol_name}] 注册上线。")

    @classmethod
    def register_factory(cls, protocol_name: str, factory_func: Type):
        """注册一个懒加载工厂类（当有人需要时才实例化）"""
        with cls._lock:
            cls._factories[protocol_name] = factory_func

    @classmethod
    def get_service(cls, protocol_name: str) -> Any:
        """从天网中精准抽离所需服务器件"""
        with cls._lock:
            # 1. 如果已在此前实例化则直接扔回
            if protocol_name in cls._services:
                return cls._services[protocol_name]

            # 2. 如果存在懒加载厂，则临时铸造一个
            if protocol_name in cls._factories:
                factory = cls._factories[protocol_name]
                try:
                    instance = factory()
                    cls._services[protocol_name] = instance
                    logger.info(f"🔗 [Locator] 懒加载服务 [{protocol_name}] 首次锻造上线。")
                    return instance
                except Exception as e:
                    logger.error(f"❌ [Locator] 锻造服务异常: {e}")
                    return None

        logger.error(f"🛑 [Locator] 组件游离: 未能在大世界网格中定位到服务 -> {protocol_name}")
        return None

    @classmethod
    def purge(cls):
        """清空所有路由，通常用于关机收尾"""
        with cls._lock:
            cls._services.clear()
            cls._factories.clear()
