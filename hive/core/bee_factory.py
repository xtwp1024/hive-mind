# -*- coding: utf-8 -*-
"""
BeeFactory - 蜜蜂实例工厂

负责根据 BeeDNA 的信息路由到正确的 Bee 类。
从 WorkerManager 中抽取的路由逻辑，独立成单一职责组件。

路由优先级:
1. role == "async" -> AsyncBaseBee (asyncio-based, runs in main process)
2. species -> BEE_SPECIES 查找 (case-insensitive)
3. mutation_gene -> bee_registry 查找 (case-insensitive)
4. role -> bee_registry 查找 (case-insensitive)
5. fallback -> BaseBee (multiprocessing.Process-based)
"""

import logging
from typing import Optional

from hive.core.dna import BeeDNA
from hive.bees.base_bee import BaseBee
from hive.bees.async_base_bee import AsyncBaseBee

logger = logging.getLogger("Hive.BeeFactory")


class BeeFactory:
    """
    蜜蜂实例工厂

    根据 dna.species / dna.role / dna.mutation_gene 路由到正确的 Bee 类。
    支持从 bee_registry 动态注册，也支持从 BEE_SPECIES 静态注册表查找。
    """

    def __init__(self, bee_registry: Optional[dict[str, type[BaseBee]]] = None):
        """
        Args:
            bee_registry: 可选的动态注册表，格式 {name_lower: BeeClass}
        """
        self._bee_registry: dict[str, type[BaseBee]] = bee_registry or {}

    def register(self, name: str, bee_class: type[BaseBee]) -> None:
        """动态注册一个 Bee 类"""
        self._bee_registry[name.lower()] = bee_class
        logger.debug(f"[BeeFactory] 注册蜂种: {name} -> {bee_class.__name__}")

    def register_from_species(self, species_dict: dict[str, type[BaseBee]]) -> None:
        """从 species 字典批量注册（key 会转为 lowercase）"""
        for name, cls in species_dict.items():
            self._bee_registry[name.lower()] = cls
        logger.info(f"[BeeFactory] 批量注册 {len(species_dict)} 个蜂种")

    def determine_bee_class(self, dna: BeeDNA) -> type[BaseBee]:
        """
        根据 BeeDNA 确定应该使用哪个 Bee 类

        路由优先级:
        1. role == "async" -> AsyncBaseBee
        2. species -> BEE_SPECIES (case-insensitive)
        3. mutation_gene -> _bee_registry (case-insensitive)
        4. role -> _bee_registry (case-insensitive)
        5. fallback -> BaseBee
        """
        # 1. async 角色走 AsyncBaseBee
        if dna.role == "async":
            logger.debug(f"[BeeFactory] Routing {dna.id} to AsyncBaseBee (role=async)")
            return AsyncBaseBee

        # 尝试按优先级查找注册表
        lookup_keys = [
            dna.species,           # species 优先级最高
            dna.mutation_gene,     # 然后是 mutation_gene
            dna.role,              # 最后是 role
        ]

        for key in lookup_keys:
            if not key:
                continue

            key_lower = key.lower()
            if key_lower in self._bee_registry:
                bee_class = self._bee_registry[key_lower]
                logger.debug(f"[BeeFactory] Routing {dna.id} to {bee_class.__name__} (key='{key}')")
                return bee_class

        # 2. 尝试从 BEE_SPECIES 静态注册表查找（支持 PascalCase key，大小写不敏感）
        from hive.bees import BEE_SPECIES
        for species_name, bee_class in BEE_SPECIES.items():
            if species_name.lower() == dna.species.lower():
                logger.debug(f"[BeeFactory] Routing {dna.id} to {bee_class.__name__} (BEE_SPECIES match)")
                return bee_class

        # 3. fallback
        logger.debug(f"[BeeFactory] Routing {dna.id} to BaseBee (fallback)")
        return BaseBee

    def create_bee(self, dna: BeeDNA, worker_id: str, result_queue: any) -> BaseBee:
        """
        创建 Bee 实例

        Args:
            dna: BeeDNA 实例
            worker_id: Worker ID
            result_queue: 结果队列

        Returns:
            BaseBee 或其子类的实例
        """
        bee_class = self.determine_bee_class(dna)

        # AsyncBaseBee 不继承 multiprocessing.Process
        if bee_class is AsyncBaseBee or issubclass(bee_class, AsyncBaseBee):
            return bee_class(dna, worker_id, result_queue)
        else:
            return bee_class(dna, worker_id, result_queue)
