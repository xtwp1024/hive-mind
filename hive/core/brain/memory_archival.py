"""
MemoryArchival - 长期记忆归档系统
=================================

提供记忆的层级存储和归档管理:
- StorageTier: 存储层级 (HOT/WARM/COLD/FROZEN/DEEP_FREEZE)
- MemoryArchive: 归档记忆数据结构
- MemoryArchiver: 归档管理器

Usage:
    archiver = MemoryArchiver()
    archived = await archiver.archive_if_needed(memory_item)
    retrieved = await archiver.retrieve_archived("memory_id")
    rehydrated = await archiver.rehydrate(archived)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("Hive.Brain.Archival")


class StorageTier(str, Enum):
    """存储层级"""

    HOT = "hot"  # 最近/重要，毫秒级访问
    WARM = "warm"  # 30天内未访问
    COLD = "cold"  # 归档到独立存储，秒级检索
    FROZEN = "frozen"  # 极低访问频率，分钟级检索
    DEEP_FREEZE = "deep_freeze"  # 几乎不访问，小时级检索


@dataclass
class MemoryArchive:
    """归档记忆"""

    id: str
    content: Any
    memory_type: str  # episodic/semantic/procedural
    archived_at: float
    last_accessed: float
    access_count: int = 0
    storage_tier: StorageTier = StorageTier.HOT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "archived_at": self.archived_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "storage_tier": self.storage_tier.value,
        }


class MemoryArchiver:
    """
    记忆归档管理器

    管理记忆的层级存储和归档，支持:
    - 自动判断是否需要归档
    - 层级迁移
    - 归档记忆的复活(rehydrate)
    """

    ARCHIVAL_THRESHOLDS = {
        "episodic": {"max_size": 1000, "importance_cutoff": 0.3},
        "semantic": {"max_size": 5000, "importance_cutoff": 0.5},
        "procedural": {"max_size": 2000, "importance_cutoff": 0.4},
    }

    # 层级迁移时间阈值 (秒)
    TIER_TRANSITION_TIMES = {
        StorageTier.HOT: 0,  # 始终在HOT
        StorageTier.WARM: 30 * 24 * 3600,  # 30天
        StorageTier.COLD: 90 * 24 * 3600,  # 90天
        StorageTier.FROZEN: 365 * 24 * 3600,  # 1年
        StorageTier.DEEP_FREEZE: 730 * 24 * 3600,  # 2年
    }

    # 层级优先级顺序 (数字越小优先级越高，用于比较而非字符串值)
    TIER_ORDER = {t.value: i for i, t in enumerate(StorageTier)}

    # 归档总数上限 (超过后触发基于LRU的驱逐)
    MAX_ARCHIVES_SIZE = 10000
    # 驱逐比例 (超过上限时删除多少)
    EVICTION_BATCH_SIZE = 500

    def __init__(self):
        self.archives: Dict[str, MemoryArchive] = {}
        self._lock = asyncio.Lock()

    async def archive_if_needed(self, memory) -> bool:
        """
        检查是否需要归档记忆

        Args:
            memory: MemoryItem 对象

        Returns:
            bool: 是否成功归档
        """
        async with self._lock:
            # 检查记忆类型和阈值
            memory_type = memory.memory_type
            if memory_type not in self.ARCHIVAL_THRESHOLDS:
                return False

            thresholds = self.ARCHIVAL_THRESHOLDS[memory_type]

            # 检查是否超过大小限制
            current_size = len([a for a in self.archives.values() if a.memory_type == memory_type])

            if current_size >= thresholds["max_size"]:
                # 需要归档：找出最低重要性的记忆
                await self._archive_lowest_priority(memory_type, memory.id)
                return True

            # 检查重要性是否低于阈值
            if (
                hasattr(memory, "importance")
                and memory.importance < thresholds["importance_cutoff"]
            ):
                # 检查访问频率
                days_since_access = (time.time() - memory.last_accessed) / (24 * 3600)
                if days_since_access > 7:  # 7天未访问
                    await self._archive_memory(memory)
                    return True

            return False

    async def _archive_lowest_priority(self, memory_type: str, new_memory_id: str):
        """归档最低优先级的记忆"""
        candidates = [a for a in self.archives.values() if a.memory_type == memory_type]

        if not candidates:
            return

        # 按重要性/访问时间排序
        candidates.sort(key=lambda a: (a.access_count, a.last_accessed))

        # 归档最低优先级的
        to_archive = candidates[0]
        tier = self._determine_tier_from_age(to_archive.last_accessed)
        to_archive.storage_tier = tier
        to_archive.archived_at = time.time()

        logger.debug(
            f"🧠 Archived {to_archive.id} to {tier.value} "
            f"(type={memory_type}, access_count={to_archive.access_count})"
        )

    def _determine_tier_from_age(self, last_accessed: float) -> StorageTier:
        """根据最后访问时间确定层级"""
        age = time.time() - last_accessed

        if age < self.TIER_TRANSITION_TIMES[StorageTier.WARM]:
            return StorageTier.WARM
        elif age < self.TIER_TRANSITION_TIMES[StorageTier.COLD]:
            return StorageTier.COLD
        elif age < self.TIER_TRANSITION_TIMES[StorageTier.FROZEN]:
            return StorageTier.FROZEN
        else:
            return StorageTier.DEEP_FREEZE

    async def _archive_memory(self, memory) -> MemoryArchive:
        """执行归档操作"""
        tier = self._determine_tier(memory)

        archive = MemoryArchive(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type,
            archived_at=time.time(),
            access_count=memory.access_count,
            last_accessed=memory.last_accessed,
            storage_tier=tier,
        )

        self.archives[memory.id] = archive

        # 检查是否超过大小限制，如果超过则触发驱逐
        await self._evict_by_size()

        logger.info(
            f"🧠 Archived memory {memory.id[:20]}... to {tier.value} "
            f"(type={memory.memory_type})"
        )

        return archive

    async def _evict_by_size(self):
        """
        基于LRU驱逐最旧的归档以控制内存使用

        当 archives 数量超过 MAX_ARCHIVES_SIZE 时，
        按 last_accessed 时间排序，驱逐最旧的归档
        """
        if len(self.archives) <= self.MAX_ARCHIVES_SIZE:
            return

        # 计算需要驱逐的数量
        evict_count = min(
            self.EVICTION_BATCH_SIZE,
            len(self.archives) - self.MAX_ARCHIVES_SIZE + self.EVICTION_BATCH_SIZE,
        )

        # 按最后访问时间排序（最旧的在前）
        candidates = sorted(self.archives.values(), key=lambda a: (a.last_accessed, a.access_count))

        # 驱逐最旧的
        to_evict = candidates[:evict_count]
        for archive in to_evict:
            del self.archives[archive.id]

        logger.warning(
            f"🧠 Evicted {len(to_evict)} archives due to size limit "
            f"(current={len(self.archives)}, max={self.MAX_ARCHIVES_SIZE})"
        )

    def _determine_tier(self, memory) -> StorageTier:
        """
        根据访问频率/重要性确定层级

        Args:
            memory: MemoryItem 对象

        Returns:
            StorageTier: 推荐存储层级
        """
        # 基于重要性和访问频率计算
        importance = getattr(memory, "importance", 0.5)
        access_count = memory.access_count
        days_since_access = (time.time() - memory.last_accessed) / (24 * 3600)

        # 重要性高 -> HOT
        if importance >= 0.8 and access_count > 10:
            return StorageTier.HOT

        # 重要性中高 -> WARM
        if importance >= 0.6 or access_count > 5:
            return StorageTier.WARM

        # 30天以上未访问 -> COLD
        if days_since_access > 30:
            return StorageTier.COLD

        # 1年以上未访问 -> FROZEN
        if days_since_access > 365:
            return StorageTier.FROZEN

        # 2年以上未访问 -> DEEP_FREEZE
        if days_since_access > 730:
            return StorageTier.DEEP_FREEZE

        return StorageTier.WARM

    async def retrieve_archived(self, memory_id: str) -> Optional[MemoryArchive]:
        """
        检索归档记忆

        Args:
            memory_id: 记忆ID

        Returns:
            MemoryArchive 或 None
        """
        async with self._lock:
            archive = self.archives.get(memory_id)

            if archive:
                # 更新访问统计
                archive.access_count += 1
                archive.last_accessed = time.time()

                # 如果从深层回到浅层，考虑升级
                if archive.storage_tier != StorageTier.HOT:
                    new_tier = self._determine_tier_from_access(archive)
                    if (
                        self.TIER_ORDER[new_tier.value]
                        < self.TIER_ORDER[archive.storage_tier.value]
                    ):
                        await self.migrate_tier(memory_id, new_tier)

                logger.debug(f"🧠 Retrieved archived memory: {memory_id[:20]}...")

            return archive

    def _determine_tier_from_access(self, archive: MemoryArchive) -> StorageTier:
        """根据访问历史确定是否需要升级层级"""
        # 高访问频率 -> 升级到WARM
        if archive.access_count > 5:
            return StorageTier.WARM

        # 极高访问频率 -> 升级到HOT
        if archive.access_count > 20:
            return StorageTier.HOT

        return archive.storage_tier

    async def rehydrate(self, archive: MemoryArchive):
        """
        复活归档记忆到活跃状态

        Args:
            archive: MemoryArchive 对象

        Returns:
            MemoryItem: 复活后的记忆项
        """
        from .memory_hierarchy import MemoryItem

        # 创建新的MemoryItem
        memory = MemoryItem(
            id=archive.id,
            content=archive.content,
            memory_type=archive.memory_type,
            importance=0.5,  # 重新评估
            created_at=archive.archived_at,
            last_accessed=time.time(),
            access_count=0,  # 重置访问计数
            metadata={"rehydrated": True, "original_archived_at": archive.archived_at},
        )

        # 从归档中移除
        async with self._lock:
            if archive.id in self.archives:
                del self.archives[archive.id]

        logger.info(f"🧠 Rehydrated memory {archive.id[:20]}... to active state")

        return memory

    async def migrate_tier(self, archive_id: str, new_tier: StorageTier):
        """
        迁移到新层级

        Args:
            archive_id: 归档ID
            new_tier: 新的存储层级
        """
        async with self._lock:
            archive = self.archives.get(archive_id)
            if not archive:
                logger.warning(f"⚠️ Archive {archive_id} not found for migration")
                return

            old_tier = archive.storage_tier
            archive.storage_tier = new_tier

            logger.info(
                f"🧠 Migrated archive {archive_id[:20]}... "
                f"from {old_tier.value} to {new_tier.value}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取归档统计"""
        tier_counts = {tier.value: 0 for tier in StorageTier}
        type_counts = {"episodic": 0, "semantic": 0, "procedural": 0}

        for archive in self.archives.values():
            tier_counts[archive.storage_tier.value] += 1
            if archive.memory_type in type_counts:
                type_counts[archive.memory_type] += 1

        return {
            "total_archives": len(self.archives),
            "by_tier": tier_counts,
            "by_type": type_counts,
        }

    async def cleanup_old_archives(self, max_age_days: int = 365):
        """
        清理过旧的归档

        Args:
            max_age_days: 最大保留天数
        """
        async with self._lock:
            cutoff = time.time() - (max_age_days * 24 * 3600)

            to_remove = [
                aid
                for aid, a in self.archives.items()
                if a.last_accessed < cutoff and a.storage_tier == StorageTier.DEEP_FREEZE
            ]

            for aid in to_remove:
                del self.archives[aid]

            if to_remove:
                logger.info(f"🧠 Cleaned up {len(to_remove)} old archives")
