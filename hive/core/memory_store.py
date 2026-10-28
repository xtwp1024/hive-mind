# -*- coding: utf-8 -*-
"""
MemoryStore - 三层层次记忆仓库
设计灵感来源于 NevaMind-AI/memU（"记忆即文件系统"哲学）。

三层架构（Hierarchical Memory Architecture）：
  Resource  → 原始内容（对话/文档），可溯源（对标 memU Resource）
  Item      → 提取的具体事实/偏好/技能（对标 memU Item）
  Category  → 自动聚合的主题目录（对标 memU Category）

双模式检索：
  RAG  → TF-IDF 余弦相似度，毫秒级（对标 memU RAG embedding 检索）
  LLM  → 关键词精确匹配推理（简化版，对标 memU LLM 深度推理接口）
"""

import atexit
import math
import time
import uuid
import threading
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("Hive.MemoryStore")

try:
    from hive.utils.memory_box import MemoryBox
    MEMORY_BOX_AVAILABLE = True
except ImportError:
    MEMORY_BOX_AVAILABLE = False

# ─────────────────────────────────────────── 数据结构 ──

@dataclass
class Resource:
    """原始内容层（对标 memU Resource / 文件挂载点）"""
    resource_id: str
    content:     str
    modality:    str    = "conversation"   # conversation | document | knowledge
    agent_id:    str    = "global"
    created_at:  float  = field(default_factory=time.time)
    metadata:    dict   = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "resource_id": self.resource_id,
            "content": self.content,
            "modality": self.modality,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Resource':
        return cls(
            resource_id=data["resource_id"],
            content=data["content"],
            modality=data.get("modality", "conversation"),
            agent_id=data.get("agent_id", "global"),
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MemoryItem:
    """事实提取层（对标 memU Item / 文件内容段落）"""
    item_id:    str
    text:       str
    tags:       list[str]   = field(default_factory=list)
    category:   str         = "general"
    agent_id:   str         = "global"
    source_id:  str         = ""         # 指向 Resource
    created_at: float       = field(default_factory=time.time)
    importance: float       = 1.0        # 0~1，越高越容易被召回

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "tags": self.tags,
            "category": self.category,
            "agent_id": self.agent_id,
            "source_id": self.source_id,
            "created_at": self.created_at,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryItem':
        return cls(
            item_id=data["item_id"],
            text=data["text"],
            tags=data.get("tags", []),
            category=data.get("category", "general"),
            agent_id=data.get("agent_id", "global"),
            source_id=data.get("source_id", ""),
            created_at=data.get("created_at", time.time()),
            importance=data.get("importance", 1.0),
        )


@dataclass
class MemoryCategory:
    """主题目录层（对标 memU Category / 文件夹）"""
    name:       str
    summary:    str                     = ""
    item_ids:   list[str]               = field(default_factory=list)
    agent_id:   str                     = "global"
    last_updated: float                 = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "summary": self.summary,
            "item_ids": self.item_ids,
            "agent_id": self.agent_id,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryCategory':
        return cls(
            name=data["name"],
            summary=data.get("summary", ""),
            item_ids=data.get("item_ids", []),
            agent_id=data.get("agent_id", "global"),
            last_updated=data.get("last_updated", time.time()),
        )


# ─────────────────────────────────────── TF-IDF 工具 ──

def _tokenize(text: str) -> list[str]:
    """简单分词（字母数字 + unicode 中文）"""
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _tfidf_vector(tokens: list[str], corpus_df: dict, n_docs: int) -> dict[str, float]:
    """计算单文档的 TF-IDF 向量"""
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    vec = {}
    for term, cnt in tf.items():
        idf = math.log((n_docs + 1) / (corpus_df.get(term, 0) + 1)) + 1
        vec[term] = (cnt / total) * idf
    return vec


def _cosine(a: dict, b: dict) -> float:
    """余弦相似度"""
    dot  = sum(a.get(t, 0) * v for t, v in b.items())
    a_n  = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    b_n  = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return dot / (a_n * b_n)


# ──────────────────────────────────────── 核心仓库 ──

class MemoryStore:
    """
    三层层次记忆仓库。
    模拟 memU 的"记忆即文件系统"哲学，无需外部数据库，完全 in-process。
    """

    # 预设分类关键词表（对标 memU 自动分类模型）
    CATEGORY_HINTS: dict[str, list[str]] = {
        "preferences":   ["prefer", "like", "dislike", "want", "喜好", "偏好", "风格"],
        "knowledge":     ["rule", "formula", "strategy", "fact", "how", "规则", "知识", "原理", "策略"],
        "market":        ["price", "btc", "eth", "signal", "buy", "sell", "market", "行情", "价格", "信号"],
        "relationships": ["contact", "team", "partner", "user", "agent", "关系", "联系人"],
        "context":       ["session", "task", "conversation", "pending", "任务", "当前", "上下文"],
    }

    def __init__(self, store_name: str = "HiveMemory", auto_save: bool = True, 
                 persistence_path: Optional[Path] = None):
        self.store_name  = store_name
        self._resources: dict[str, Resource]        = {}
        self._items:     dict[str, MemoryItem]      = {}
        self._categories:dict[str, MemoryCategory]  = {}
        self._lock = threading.RLock()
        # 构建 IDF 所需的文档频率索引
        self._corpus_df: dict[str, int] = {}   # term → 含有该词的 item 数量
        
        # 持久化配置
        self.auto_save = auto_save
        self.persistence_path = persistence_path
        self._pending_changes = 0
        self._auto_save_threshold = 10  # 每 10 次变更自动保存
        
        # MemoryBox 实例（延迟加载）
        self._memory_box: Optional[Any] = None
        
        # 尝试加载已有数据
        if auto_save:
            self._try_load()
        
        # 注册退出时自动保存
        atexit.register(self._atexit_save)
        
        logger.info(f"[MemoryStore:{self.store_name}] 三层记忆仓库已初始化 (auto_save={auto_save})")

    def _get_memory_box(self) -> Optional[Any]:
        """获取 MemoryBox 实例（延迟加载）"""
        if not MEMORY_BOX_AVAILABLE:
            return None
        if self._memory_box is None:
            try:
                box_name = f"memory_store_{self.store_name}"
                self._memory_box = MemoryBox.open_box(box_name)
            except Exception as e:
                logger.warning(f"[MemoryStore:{self.store_name}] MemoryBox 初始化失败: {e}")
                return None
        return self._memory_box

    def _try_load(self) -> bool:
        """尝试从持久化存储加载数据"""
        box = self._get_memory_box()
        if box is None:
            return False
        
        try:
            # 加载 resources
            resources_data = box.get("_ms_resources", {})
            for rid, rdata in resources_data.items():
                self._resources[rid] = Resource.from_dict(rdata)
            
            # 加载 items
            items_data = box.get("_ms_items", {})
            for iid, idata in items_data.items():
                self._items[iid] = MemoryItem.from_dict(idata)
            
            # 加载 categories
            cats_data = box.get("_ms_categories", {})
            for key, cdata in cats_data.items():
                self._categories[key] = MemoryCategory.from_dict(cdata)
            
            # 加载 corpus_df
            self._corpus_df = box.get("_ms_corpus_df", {})
            
            stats = box.get("_ms_stats", {
                "resources": 0, "items": 0, "categories": 0, "corpus_terms": 0
            })
            if stats.get("resources", 0) > 0 or stats.get("items", 0) > 0:
                logger.info(f"[MemoryStore:{self.store_name}] 从持久化加载: "
                          f"resources={stats.get('resources', 0)}, "
                          f"items={stats.get('items', 0)}, "
                          f"categories={stats.get('categories', 0)}")
                return True
            return False
        except Exception as e:
            logger.warning(f"[MemoryStore:{self.store_name}] 加载持久化数据失败: {e}")
            return False

    def save(self) -> bool:
        """保存所有数据到持久化存储"""
        box = self._get_memory_box()
        if box is None:
            return False
        
        try:
            with self._lock:
                # 转换为可序列化的 dict
                resources_dict = {rid: r.to_dict() for rid, r in self._resources.items()}
                items_dict = {iid: i.to_dict() for iid, i in self._items.items()}
                cats_dict = {key: c.to_dict() for key, c in self._categories.items()}
                
                # 保存到 MemoryBox
                box.put("_ms_resources", resources_dict)
                box.put("_ms_items", items_dict)
                box.put("_ms_categories", cats_dict)
                box.put("_ms_corpus_df", dict(self._corpus_df))
                box.put("_ms_stats", {
                    "resources": len(self._resources),
                    "items": len(self._items),
                    "categories": len(self._categories),
                    "corpus_terms": len(self._corpus_df),
                    "saved_at": time.time(),
                })
            
            self._pending_changes = 0
            logger.debug(f"[MemoryStore:{self.store_name}] 已保存: "
                       f"resources={len(self._resources)}, "
                       f"items={len(self._items)}, "
                       f"categories={len(self._categories)}")
            return True
        except Exception as e:
            logger.warning(f"[MemoryStore:{self.store_name}] 保存失败: {e}")
            return False

    def _mark_changed(self) -> None:
        """标记有变更，可能触发自动保存"""
        if not self.auto_save:
            return
        
        self._pending_changes += 1
        if self._pending_changes >= self._auto_save_threshold:
            self.save()

    def _atexit_save(self) -> None:
        """程序退出时自动保存"""
        if self.auto_save and self._pending_changes > 0:
            logger.info(f"[MemoryStore:{self.store_name}] 退出时自动保存...")
            self.save()

    def load(self) -> bool:
        """手动从持久化存储加载数据（会覆盖当前内存数据）"""
        return self._try_load()

    def clear_and_load(self) -> bool:
        """清空当前数据并重新加载"""
        with self._lock:
            self._resources.clear()
            self._items.clear()
            self._categories.clear()
            self._corpus_df.clear()
            self._pending_changes = 0
        return self._try_load()

    def get_persistence_stats(self) -> dict:
        """获取持久化状态"""
        box = self._get_memory_box()
        box_available = box is not None
        
        with self._lock:
            return {
                "auto_save": self.auto_save,
                "pending_changes": self._pending_changes,
                "auto_save_threshold": self._auto_save_threshold,
                "memory_box_available": box_available,
                "store_name": self.store_name,
            }

    # ─────────────────────────────── memorize（持续学习流水线）──

    def memorize(self, content: str, agent_id: str = "global",
                 modality: str = "conversation", metadata: dict | None = None) -> dict:
        """
        吸收新内容（对标 memU memorize()）。
        流程：存入 Resource → 提取 Items → 自动归类 → 更新索引。
        """
        # 1. 存入 Resource 层
        rid = uuid.uuid4().hex[:12]
        res = Resource(resource_id=rid, content=content, modality=modality,
                       agent_id=agent_id, metadata=metadata or {})
        with self._lock:
            self._resources[rid] = res

        # 2. 提取 Items（按句子分割，过滤过短的）
        sentences = [s.strip() for s in re.split(r"[。.!?！？\n]", content) if len(s.strip()) > 5]
        items = []
        for sent in sentences:
            category   = self._auto_categorize(sent)
            tags       = self._extract_tags(sent)
            importance = min(1.0, len(sent) / 100 + 0.3)
            iid = uuid.uuid4().hex[:12]
            item = MemoryItem(
                item_id   = iid,
                text      = sent,
                tags      = tags,
                category  = category,
                agent_id  = agent_id,
                source_id = rid,
                importance= importance,
            )
            with self._lock:
                self._items[iid] = item
                # 更新 IDF 索引
                for term in set(_tokenize(sent)):
                    self._corpus_df[term] = self._corpus_df.get(term, 0) + 1
                # 更新 Category
                self._upsert_category(category, iid, agent_id)
            items.append(item)
            logger.debug(f"[MemoryStore] 提取 Item: category={category} text={sent[:40]}")

        logger.info(f"[MemoryStore] memorize({agent_id}) → resource={rid[:8]} items={len(items)}")
        
        # 标记变更（可能触发自动保存）
        self._mark_changed()
        
        return {"resource_id": rid, "items": len(items), "categories": list({i.category for i in items})}

    # ─────────────────────────────── retrieve：双模式检索 ──

    def retrieve_rag(self, query: str, agent_id: str = "global",
                     top_k: int = 5) -> list[MemoryItem]:
        """
        RAG 快速召回（对标 memU retrieve(method='rag')）。
        TF-IDF 余弦相似度，毫秒级，低成本。
        """
        q_tokens = _tokenize(query)
        with self._lock:
            items = [i for i in self._items.values()
                     if i.agent_id in (agent_id, "global")]
            n_docs = max(len(items), 1)
            corpus_df = dict(self._corpus_df)

        q_vec = _tfidf_vector(q_tokens, corpus_df, n_docs)
        scored = []
        for item in items:
            d_vec = _tfidf_vector(_tokenize(item.text), corpus_df, n_docs)
            sim   = _cosine(q_vec, d_vec) * item.importance
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [i for _, i in scored[:top_k] if _ > 0]
        logger.info(f"[MemoryStore] retrieve_rag({query[:30]!r}) → {len(results)} 条")
        return results

    def retrieve_llm(self, query: str, agent_id: str = "global",
                     top_k: int = 3) -> list[MemoryItem]:
        """
        LLM 深度推理检索（对标 memU retrieve(method='llm')）。
        简化版：关键词精确匹配 + 分类权重提升。
        """
        keywords  = set(_tokenize(query))
        category  = self._auto_categorize(query)
        with self._lock:
            items = [i for i in self._items.values()
                     if i.agent_id in (agent_id, "global")]

        scored = []
        for item in items:
            item_words = set(_tokenize(item.text))
            overlap    = len(keywords & item_words)
            cat_bonus  = 0.3 if item.category == category else 0.0
            score      = overlap + cat_bonus
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [i for _, i in scored[:top_k] if _ > 0]
        logger.info(f"[MemoryStore] retrieve_llm({query[:30]!r}) → {len(results)} 条（预测类别={category}）")
        return results

    # ─────────────────────────────────── 文件系统式导航 ──

    def browse(self, category: str = "", agent_id: str = "global") -> dict:
        """
        文件系统式记忆导航（对标 memU 目录浏览 / ls memory/）。
        category="" 时列出所有顶层目录；否则列出该目录下的 Items。
        """
        with self._lock:
            if not category:
                cats = {n: c for n, c in self._categories.items()
                        if c.agent_id in (agent_id, "global")}
                return {
                    "type":       "directory",
                    "categories": [
                        {"name": n, "items": len(c.item_ids), "summary": c.summary}
                        for n, c in cats.items()
                    ]
                }
            cat = self._categories.get(f"{category}:{agent_id}") or \
                  self._categories.get(f"{category}:global")
            if not cat:
                return {"type": "error", "message": f"Category {category!r} 不存在"}
            items = [self._items[iid] for iid in cat.item_ids if iid in self._items]
        return {
            "type":     "category",
            "name":     category,
            "summary":  cat.summary,
            "items":    [{"text": i.text[:60], "tags": i.tags, "importance": i.importance}
                         for i in items],
        }

    def memory_stats(self) -> dict:
        with self._lock:
            return {
                "resources":  len(self._resources),
                "items":      len(self._items),
                "categories": len(self._categories),
            }

    # ──────────────────────────────────── 内部辅助 ──

    def _auto_categorize(self, text: str) -> str:
        """按关键词命中数自动选择最匹配的 Category"""
        tokens = set(_tokenize(text))
        scores = {cat: len(tokens & set(hints))
                  for cat, hints in self.CATEGORY_HINTS.items()}
        best = max(scores, key=scores.get)  # type: ignore
        return best if scores[best] > 0 else "context"

    def _extract_tags(self, text: str) -> list[str]:
        """提取大写词/数字组合作为标签"""
        tags = re.findall(r"\b[A-Z]{2,}|[A-Z][a-z]+[A-Z]\w*|\d+\.\d+%?", text)
        return list(set(tags))[:5]

    def _upsert_category(self, category: str, item_id: str, agent_id: str) -> None:
        key = f"{category}:{agent_id}"
        if key not in self._categories:
            self._categories[key] = MemoryCategory(name=category, agent_id=agent_id)
        cat = self._categories[key]
        if item_id not in cat.item_ids:
            cat.item_ids.append(item_id)
        # 更新摘要（取最新 3 条 item 的前 30 字拼接）
        recent = [self._items[i].text[:30] for i in cat.item_ids[-3:] if i in self._items]
        cat.summary    = " | ".join(recent)
        cat.last_updated = time.time()
