"""
MemoryHierarchy - 记忆层次结构
===============================

统一记忆接口，整合现有组件:
- WorkingMemory → 工作记忆
- MemoryStore → 情景记忆
- VectorVault + KnowledgeBase → 语义记忆
- Cerebellum + DNA → 程序记忆

层次化检索:
1. WorkingMemory (最优先)
2. EpisodicMemory (近期经验)
3. SemanticMemory (知识概念)
4. ProceduralMemory (技能模式)

Usage:
    hierarchy = MemoryHierarchy()
    context = await hierarchy.recall(query="项目架构")
    await hierarchy.store(experience)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Hive.Brain.Memory")

# 尝试导入归档系统
try:
    from .memory_archival import MemoryArchiver, StorageTier
except ImportError:
    MemoryArchiver = None
    StorageTier = None

# 尝试导入现有组件
try:
    from hive.core.memory_store import MemoryStore
except ImportError:
    MemoryStore = None

try:
    from hive.core.vector_vault import VectorVault
except ImportError:
    VectorVault = None

try:
    from hive.core.knowledge_base import KnowledgeBase
except ImportError:
    KnowledgeBase = None

try:
    from hive.core.cerebellum import Cerebellum
except ImportError:
    Cerebellum = None


@dataclass
class MemoryItem:
    """记忆项"""

    id: str
    content: Any
    memory_type: str  # working, episodic, semantic, procedural
    importance: float = 0.5  # 0-1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def access(self):
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class Context:
    """检索返回的上下文"""

    working: Optional[MemoryItem] = None
    episodic: List[MemoryItem] = field(default_factory=list)
    semantic: List[MemoryItem] = field(default_factory=list)
    procedural: List[MemoryItem] = field(default_factory=list)
    confidence: float = 0.5  # 整体置信度

    def merge(self) -> List[MemoryItem]:
        """合并所有记忆层"""
        items = []
        if self.working:
            items.append(self.working)
        items.extend(self.episodic)
        items.extend(self.semantic)
        items.extend(self.procedural)
        return items

    def to_dict(self) -> Dict[str, Any]:
        return {
            "working": self.working.content if self.working else None,
            "episodic_count": len(self.episodic),
            "semantic_count": len(self.semantic),
            "procedural_count": len(self.procedural),
            "confidence": self.confidence,
        }


@dataclass
class Experience:
    """经验"""

    content: str
    source: str = "unknown"
    success: bool = True
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryHierarchy:
    """
    记忆层次结构

    整合 WorkingMemory, MemoryStore, VectorVault, KnowledgeBase, Cerebellum
    提供统一的存储和检索接口。
    """

    def __init__(
        self,
        working_memory_capacity: int = 50,
        episodic_limit: int = 1000,
        semantic_limit: int = 5000,
        consolidation_threshold: int = 10,
    ):
        self.working = None  # 延迟初始化
        self._episodic_store = None
        self._semantic_vectors = None
        self._knowledge_base = None
        self._procedural_store = None
        self._archiver = MemoryArchiver() if MemoryArchiver else None

        self.episodic_limit = episodic_limit
        self.semantic_limit = semantic_limit
        self.consolidation_threshold = consolidation_threshold

        # 统计
        self._stats = {
            "stores_attempted": 0,
            "stores_succeeded": 0,
            "recalls": 0,
            "consolidations": 0,
            "archives": 0,
            "rehydrations": 0,
        }

        self._lock = asyncio.Lock()
        self._initialized = False

        logger.info(
            f"🧠 MemoryHierarchy initialized "
            f"(episodic_limit={episodic_limit}, semantic_limit={semantic_limit})"
        )

    async def initialize(self):
        """初始化各层存储"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # 导入并初始化工作记忆
            from .working_memory import WorkingMemory

            self.working = WorkingMemory(max_items=50)

            # 初始化现有组件
            self._stats["stores_attempted"] = 4

            if MemoryStore:
                try:
                    self._episodic_store = MemoryStore()
                    self._stats["stores_succeeded"] += 1
                    logger.debug("🧠 Episodic store (MemoryStore) connected")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to connect MemoryStore: {e}")

            if VectorVault:
                try:
                    self._semantic_vectors = VectorVault()
                    self._stats["stores_succeeded"] += 1
                    logger.debug("🧠 Semantic vectors (VectorVault) connected")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to connect VectorVault: {e}")

            if KnowledgeBase:
                try:
                    self._knowledge_base = KnowledgeBase()
                    self._stats["stores_succeeded"] += 1
                    logger.debug("🧠 Knowledge base connected")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to connect KnowledgeBase: {e}")

            if Cerebellum:
                try:
                    self._procedural_store = Cerebellum()
                    self._stats["stores_succeeded"] += 1
                    logger.debug("🧠 Procedural store (Cerebellum) connected")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to connect Cerebellum: {e}")

            self._initialized = True
            logger.info(
                f"🧠 MemoryHierarchy initialized "
                f"({self._stats['stores_succeeded']}/{self._stats['stores_attempted']} stores)"
            )

    async def recall(
        self,
        query: str,
        layers: Optional[List[str]] = None,
        limit: int = 10,
    ) -> Context:
        """
        层次化记忆检索

        Args:
            query: 查询文本
            layers: 指定检索层，默认全部
            limit: 每层返回数量

        Returns:
            Context: 包含各层记忆的上下文
        """
        if not self._initialized:
            await self.initialize()

        layers = layers or ["working", "episodic", "semantic", "procedural"]
        context = Context()

        # 1. 工作记忆 (最高优先级)
        if "working" in layers and self.working:
            try:
                focus = self.working.attention_focus
                if focus:
                    context.working = MemoryItem(
                        id=focus.id,
                        content=focus.content,
                        memory_type="working",
                        importance=focus.attention_level,
                    )
            except Exception as e:
                logger.warning(f"⚠️ Working memory recall error: {e}")

        # 2. 情景记忆 (近期经验)
        if "episodic" in layers and self._episodic_store:
            try:
                if hasattr(self._episodic_store, "retrieve_rag"):
                    results = await self._episodic_store.retrieve_rag(query, limit=limit)
                    context.episodic = [
                        MemoryItem(
                            id=str(i),
                            content=r,
                            memory_type="episodic",
                        )
                        for i, r in enumerate(results[:limit])
                    ]
            except Exception as e:
                logger.warning(f"⚠️ Episodic recall error: {e}")

        # 3. 语义记忆 (知识概念)
        if "semantic" in layers:
            if self._semantic_vectors:
                try:
                    vector_results = await self._semantic_vectors.search(query, limit=limit)
                    context.semantic = [
                        MemoryItem(
                            id=f"vec_{i}",
                            content=r,
                            memory_type="semantic",
                        )
                        for i, r in enumerate(vector_results[:limit])
                    ]
                except Exception as e:
                    logger.warning(f"⚠️ Vector search error: {e}")

            if self._knowledge_base:
                try:
                    kb_results = self._knowledge_base.query(query)
                    context.semantic.extend(
                        [
                            MemoryItem(
                                id=f"kb_{i}",
                                content=r,
                                memory_type="semantic",
                            )
                            for i, r in enumerate(kb_results[:limit])
                        ]
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Knowledge base query error: {e}")

        # 4. 程序记忆 (技能模式)
        if "procedural" in layers and self._procedural_store:
            try:
                if hasattr(self._procedural_store, "get_reflex"):
                    reflex = self._procedural_store.get_reflex(query)
                    if reflex:
                        context.procedural = [
                            MemoryItem(
                                id="reflex_0",
                                content=reflex,
                                memory_type="procedural",
                            )
                        ]
            except Exception as e:
                logger.warning(f"⚠️ Procedural recall error: {e}")

        # 计算整体置信度
        total = len(context.episodic) + len(context.semantic) + len(context.procedural)
        context.confidence = min(1.0, total / limit) if total > 0 else 0.3

        # 5. 检查归档记忆并复活
        if self._archiver and self._archiver.archives:
            rehydrated = await self._check_and_rehydrate_archives(query)
            if rehydrated:
                context.episodic.extend(rehydrated)
                self._stats["rehydrations"] += 1
                logger.debug(f"🧠 Rehydrated {len(rehydrated)} archived memories")

        self._stats["recalls"] += 1
        return context

    async def store(self, experience: Experience) -> bool:
        """
        存储经验到合适的记忆层

        Args:
            experience: 要存储的经验

        Returns:
            bool: 是否存储成功
        """
        if not self._initialized:
            await self.initialize()

        try:
            # 创建MemoryItem用于归档检查
            memory_item = MemoryItem(
                id=f"{experience.source}_{int(time.time()*1000)}",
                content=experience.content,
                memory_type="episodic",
                importance=experience.importance,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                metadata={"source": experience.source, "tags": experience.tags},
            )

            # 检查是否需要归档
            if self._archiver:
                await self._archiver.archive_if_needed(memory_item)
                self._stats["archives"] += 1

            # 1. 存入工作记忆
            if self.working:
                item_id = await self.working.add(
                    content=experience.content,
                    tags=experience.tags,
                    attention_level=experience.importance,
                )
                logger.debug(f"🧠 Stored in working memory: {item_id}")

            # 2. 存入情景记忆
            if self._episodic_store:
                try:
                    if hasattr(self._episodic_store, "memorize"):
                        await self._episodic_store.memorize(
                            experience.content,
                            importance=experience.importance,
                        )
                        logger.debug(f"🧠 Stored in episodic memory")
                except Exception as e:
                    logger.warning(f"⚠️ Episodic store error: {e}")

            # 3. 存入语义记忆 (如果经验重要)
            if experience.importance > 0.7 and self._knowledge_base:
                try:
                    self._knowledge_base.add_knowledge(
                        content=experience.content,
                        source=experience.source,
                    )
                    logger.debug(f"🧠 Stored in semantic memory")
                except Exception as e:
                    logger.warning(f"⚠️ Knowledge base error: {e}")

            # 4. 检查是否需要整合
            await self._maybe_consolidate(experience)

            return True

        except Exception as e:
            logger.error(f"❌ Memory store error: {e}")
            return False

    async def imprint(self, pattern: str, response: str) -> bool:
        """
        印记学习 - 存储到程序记忆 (Cerebellum)

        Args:
            pattern: 触发模式
            response: 响应动作

        Returns:
            bool: 是否印记成功
        """
        if self._procedural_store:
            try:
                if hasattr(self._procedural_store, "imprint"):
                    imprint_method = self._procedural_store.imprint
                    # 检查是否为异步方法并适当处理
                    if asyncio.iscoroutinefunction(imprint_method):
                        await imprint_method(pattern, response)
                    else:
                        imprint_method(pattern, response)
                    logger.info(f"🧠 Imprinted reflex: {pattern[:50]}")
                    return True
            except Exception as e:
                logger.error(f"❌ Imprint error: {e}")
        return False

    async def _maybe_consolidate(self, experience: Experience):
        """
        检查并执行记忆整合

        当经验成功且重要时（importance >= 0.8），从情景记忆提炼到语义记忆。
        这实现了记忆的层级流动：episodic -> semantic。
        """
        if not experience.success or experience.importance < 0.8:
            return

        self._stats["consolidations"] += 1
        logger.debug(f"🧠 Triggering memory consolidation (importance={experience.importance})")

        try:
            # 1. 从情景记忆检索相关经验
            related_experiences = []
            if self._episodic_store:
                try:
                    if hasattr(self._episodic_store, "retrieve_rag"):
                        results = await self._episodic_store.retrieve_rag(
                            experience.content, limit=5
                        )
                        related_experiences = results
                except Exception as e:
                    logger.warning(f"⚠️ Failed to retrieve episodic memories: {e}")

            # 2. 提炼共性形成语义记忆
            if related_experiences and self._knowledge_base:
                # 提取关键模式
                content_parts = [experience.content]
                content_parts.extend(related_experiences[:4])

                # 生成语义摘要（简单的拼接+去重）
                consolidated = self._generate_semantic_summary(content_parts)

                if consolidated:
                    try:
                        self._knowledge_base.add_knowledge(
                            content=consolidated,
                            source="consolidation",
                            tags=["consolidated", *experience.tags],
                        )
                        logger.debug(f"🧠 Consolidated to semantic memory: {consolidated[:50]}...")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to add to knowledge base: {e}")

            # 3. 更新程序记忆（如果形成反射）
            if experience.importance >= 0.95 and self._procedural_store:
                if hasattr(self._procedural_store, "imprint"):
                    imprint_method = self._procedural_store.imprint
                    # 检查是否为异步方法并适当处理
                    if asyncio.iscoroutinefunction(imprint_method):
                        await imprint_method(
                            trigger=experience.content[:50],
                            response=f"高重要性经验: {experience.content[:100]}",
                        )
                    else:
                        imprint_method(
                            trigger=experience.content[:50],
                            response=f"高重要性经验: {experience.content[:100]}",
                        )
                    logger.debug(f"🧠 Imprinted reflex for high-importance experience")

        except Exception as e:
            logger.warning(f"⚠️ Memory consolidation error: {e}")

    def _generate_semantic_summary(self, content_parts: List[str]) -> str:
        """
        生成语义摘要

        从多个经验内容中提炼共性，形成语义记忆。
        简单的实现：取第一条并附加共性标记。
        """
        if not content_parts:
            return ""

        # 简单策略：取第一条作为主体，标记为提炼内容
        primary = content_parts[0].strip()
        if len(primary) > 200:
            primary = primary[:200] + "..."

        if len(content_parts) > 1:
            return f"[提炼] {primary}"
        return primary

    async def _check_and_rehydrate_archives(self, query: str) -> List[MemoryItem]:
        """
        检查归档记忆并复活相关的记忆

        Args:
            query: 查询文本

        Returns:
            List[MemoryItem]: 复活后的记忆列表
        """
        if not self._archiver:
            return []

        rehydrated = []
        query_lower = query.lower()

        # 遍历归档，找到与查询相关的
        archives_to_check = list(self._archiver.archives.values())

        for archive in archives_to_check:
            # 简单匹配：检查内容是否包含查询关键词
            content_str = str(archive.content).lower()
            if query_lower in content_str or self._content_similarity(query_lower, content_str):
                # 检索归档
                retrieved = await self._archiver.retrieve_archived(archive.id)
                if retrieved:
                    # 复活归档
                    memory_item = await self._archiver.rehydrate(retrieved)
                    rehydrated.append(memory_item)

        return rehydrated[:5]  # 最多返回5个

    def _content_similarity(self, query: str, content: str) -> bool:
        """简单的文本相似度判断（基于关键词重叠）"""
        query_words = set(query.split())
        content_words = set(content.split())
        overlap = query_words & content_words
        # 如果有超过2个相同的词，认为相关
        return len(overlap) >= 2

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "initialized": self._initialized,
            "working_items": len(self.working._items) if self.working else 0,
        }

    async def cleanup(self):
        """清理记忆"""
        if self.working:
            await self.working.clear()
        self._stats = {
            "stores_attempted": 4,
            "stores_succeeded": 0,
            "recalls": 0,
            "consolidations": 0,
            "archives": 0,
            "rehydrations": 0,
        }
        logger.info("🧠 MemoryHierarchy cleaned")
