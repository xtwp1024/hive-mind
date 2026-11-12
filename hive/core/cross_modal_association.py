# -*- coding: utf-8 -*-
"""
Cross-Modal Association - 跨模态关联模块

Phase 4: 多模态扩展
实现不同模态之间的关联和理解。

核心功能:
1. 跨模态注意力对齐 - 使用注意力机制对齐不同模态
2. 模态关联学习 - 学习模态之间的映射关系
3. 统一表示空间 - 将所有模态映射到同一空间
4. 跨模态检索 - 用一种模态检索另一种模态
5. 模态补全 - 用一种模态补全另一种模态的缺失信息

架构设计:
```
┌─────────────────────────────────────────────────────────┐
│              跨模态关联引擎                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐│
│  │  文本嵌入   │    │  图像嵌入   │    │  音频嵌入   ││
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘│
│         │                  │                  │         │
│         ▼                  ▼                  ▼         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           跨模态注意力对齐层                     │   │
│  │  • 自注意力: 模态内关联                          │   │
│  │  • 互注意力: 模态间关联                          │   │
│  │  • 交叉注意力: 引导对齐                          │   │
│  └─────────────────────────────────────────────────┘   │
│                           │                             │
│                           ▼                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │           模态关联学习器                         │   │
│  │  • 相似度度量学习                                │   │
│  │  • 对比学习                                      │   │
│  │  • 映射函数学习                                  │   │
│  └─────────────────────────────────────────────────┘   │
│                           │                             │
│                           ▼                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │           统一表示空间                           │   │
│  │  • 共享嵌入空间                                  │   │
│  │  • 跨模态相似度计算                              │   │
│  │  • 语义对齐                                      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
"""

import logging
import math
import threading
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("Hive.CrossModal")


class AssociationType(Enum):
    """关联类型"""

    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    CORRELATIONAL = "correlational"


class RetrievalResult:
    """检索结果"""

    def __init__(
        self,
        query_id: str,
        query_modality: str,
        results: List[Dict[str, Any]],
        total_count: int,
        retrieval_time_ms: float,
    ):
        self.query_id = query_id
        self.query_modality = query_modality
        self.results = results
        self.total_count = total_count
        self.retrieval_time_ms = retrieval_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_modality": self.query_modality,
            "results": self.results,
            "total_count": self.total_count,
            "retrieval_time_ms": self.retrieval_time_ms,
        }


@dataclass
class ModalityAlignment:
    """模态对齐结果"""

    alignment_id: str
    source_modality: str
    target_modality: str

    source_embedding_id: str
    target_embedding_id: str

    similarity_score: float
    alignment_strength: float

    alignment_type: AssociationType = AssociationType.SEMANTIC

    aligned_regions: Optional[List[Dict[str, Any]]] = None
    attention_weights: Optional[List[float]] = None

    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alignment_id": self.alignment_id,
            "source_modality": self.source_modality,
            "target_modality": self.target_modality,
            "source_embedding_id": self.source_embedding_id,
            "target_embedding_id": self.target_embedding_id,
            "similarity_score": self.similarity_score,
            "alignment_strength": self.alignment_strength,
            "alignment_type": self.alignment_type.value,
            "aligned_regions": self.aligned_regions,
            "created_at": self.created_at,
        }


@dataclass
class CrossModalMapping:
    """跨模态映射"""

    mapping_id: str
    source_modality: str
    target_modality: str

    mapping_function: Any = None
    inverse_mapping: Any = None

    training_samples: int = 0
    validation_score: float = 0.0

    is_learned: bool = False
    is_reversible: bool = False

    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mapping_id": self.mapping_id,
            "source_modality": self.source_modality,
            "target_modality": self.target_modality,
            "training_samples": self.training_samples,
            "validation_score": self.validation_score,
            "is_learned": self.is_learned,
            "is_reversible": self.is_reversible,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }


class CrossModalAttention:
    """
    跨模态注意力机制

    实现多种注意力对齐方式:
    1. 自注意力 (Self-Attention): 模态内关联
    2. 互注意力 (Cross-Attention): 模态间关联
    3. 引导注意力 (Guided Attention): 用一种模态引导另一种
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self._attention_cache: Dict[str, Dict] = {}

        logger.info(f"👁️ [CrossModalAttention] 初始化，嵌入维度: {embedding_dim}")

    def self_attention(
        self,
        embeddings: List[List[float]],
        num_heads: int = 8,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        自注意力 - 计算模态内的关联

        Args:
            embeddings: 嵌入列表
            num_heads: 注意力头数

        Returns:
            (output_embeddings, attention_weights)
        """
        if not embeddings:
            return [], []

        n = len(embeddings)
        dim = len(embeddings[0])

        attention_weights = []
        for i in range(n):
            row_weights = []
            for j in range(n):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                row_weights.append(sim)
            attention_weights.append(row_weights)

        exp_weights = []
        for row in attention_weights:
            exp_row = [math.exp(w) for w in row]
            sum_exp = sum(exp_row)
            exp_weights.append([w / sum_exp for w in exp_row])

        output = []
        for i in range(n):
            weighted = [0.0] * dim
            for j in range(n):
                weight = exp_weights[i][j]
                for k in range(dim):
                    weighted[k] += embeddings[j][k] * weight
            output.append(weighted)

        return output, exp_weights

    def cross_attention(
        self,
        source_embeddings: List[List[float]],
        target_embeddings: List[List[float]],
        num_heads: int = 8,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        互注意力 - 计算模态间的关联

        Args:
            source_embeddings: 源模态嵌入
            target_embeddings: 目标模态嵌入
            num_heads: 注意力头数

        Returns:
            (output_embeddings, attention_weights)
        """
        if not source_embeddings or not target_embeddings:
            return [], []

        n_source = len(source_embeddings)
        n_target = len(target_embeddings)
        dim = len(source_embeddings[0])

        attention_weights = []
        for i in range(n_source):
            row_weights = []
            for j in range(n_target):
                sim = self._cosine_similarity(source_embeddings[i], target_embeddings[j])
                row_weights.append(sim)
            attention_weights.append(row_weights)

        exp_weights = []
        for row in attention_weights:
            exp_row = [math.exp(w) for w in row]
            sum_exp = sum(exp_row)
            exp_weights.append([w / sum_exp if sum_exp > 0 else 0.0 for w in exp_row])

        output = []
        for i in range(n_source):
            weighted = [0.0] * dim
            for j in range(n_target):
                weight = exp_weights[i][j]
                for k in range(dim):
                    weighted[k] += target_embeddings[j][k] * weight
            output.append(weighted)

        return output, exp_weights

    def guided_attention(
        self,
        source_embedding: List[float],
        target_embeddings: List[List[float]],
        guidance_weights: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float], int]:
        """
        引导注意力 - 用源模态引导目标模态

        Args:
            source_embedding: 源嵌入
            target_embeddings: 目标嵌入列表
            guidance_weights: 可选指导权重

        Returns:
            (aligned_embedding, attention_weights, best_match_index)
        """
        if not target_embeddings:
            return source_embedding, [], -1

        similarities = []
        for i, emb in enumerate(target_embeddings):
            sim = self._cosine_similarity(source_embedding, emb)
            if guidance_weights and i < len(guidance_weights):
                sim = sim * guidance_weights[i]
            similarities.append(sim)

        exp_sims = [math.exp(s) for s in similarities]
        sum_exp = sum(exp_sims)
        attention_weights = [s / sum_exp if sum_exp > 0 else 0.0 for s in exp_sims]

        dim = len(source_embedding)
        aligned = [0.0] * dim
        for i, emb in enumerate(target_embeddings):
            weight = attention_weights[i]
            for k in range(dim):
                aligned[k] += emb[k] * weight

        best_idx = similarities.index(max(similarities)) if similarities else -1

        return aligned, attention_weights, best_idx

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        if len(a) != len(b):
            return 0.0

        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        return dot_product / (norm_a * norm_b)


class ModalityAssociationLearner:
    """
    模态关联学习器

    学习不同模态之间的映射关系:
    1. 相似度度量学习
    2. 对比学习
    3. 映射函数学习
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

        self._mappings: Dict[str, CrossModalMapping] = {}
        self._alignment_history: List[ModalityAlignment] = []

        self._lock = threading.Lock()

        logger.info(f"🧠 [ModalityAssociationLearner] 初始化，嵌入维度: {embedding_dim}")

    def learn_mapping(
        self,
        source_modality: str,
        target_modality: str,
        source_embeddings: List[List[float]],
        target_embeddings: List[List[float]],
        pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> CrossModalMapping:
        """
        学习模态间的映射

        Args:
            source_modality: 源模态
            target_modality: 目标模态
            source_embeddings: 源嵌入列表
            target_embeddings: 目标嵌入列表
            pairs: 可选的配对索引列表

        Returns:
            CrossModalMapping - 学习到的映射
        """
        import uuid

        mapping_id = f"mapping_{uuid.uuid4().hex[:12]}"

        mapping = CrossModalMapping(
            mapping_id=mapping_id,
            source_modality=source_modality,
            target_modality=target_modality,
            training_samples=len(source_embeddings) if source_embeddings else 0,
        )

        if pairs:
            mapping.validation_score = self._validate_mapping(
                source_embeddings, target_embeddings, pairs
            )
            mapping.is_learned = mapping.validation_score > 0.5
            mapping.is_reversible = self._check_reversibility(
                source_embeddings, target_embeddings, pairs
            )

        with self._lock:
            key = f"{source_modality}_to_{target_modality}"
            self._mappings[key] = mapping

        logger.info(
            f"🧠 [ModalityAssociationLearner] 学习映射: {source_modality} -> {target_modality}, "
            f"样本数: {mapping.training_samples}, 验证分数: {mapping.validation_score:.3f}"
        )

        return mapping

    def _validate_mapping(
        self,
        source_emb: List[List[float]],
        target_emb: List[List[float]],
        pairs: List[Tuple[int, int]],
    ) -> float:
        """验证映射质量"""
        if not pairs:
            return 0.0

        correct = 0
        total = 0

        for src_idx, tgt_idx in pairs:
            if src_idx >= len(source_emb) or tgt_idx >= len(target_emb):
                continue

            source_vec = source_emb[src_idx]

            similarities = []
            for i, vec in enumerate(target_emb):
                sim = self._cosine_similarity(source_vec, vec)
                similarities.append((sim, i))

            similarities.sort(key=lambda x: x[0], reverse=True)

            top_k = [idx for _, idx in similarities[:3]]
            if tgt_idx in top_k:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def _check_reversibility(
        self,
        source_emb: List[List[float]],
        target_emb: List[List[float]],
        pairs: List[Tuple[int, int]],
    ) -> bool:
        """检查映射是否可逆"""
        if not pairs:
            return False

        forward_score = self._validate_mapping(source_emb, target_emb, pairs)
        reverse_pairs = [(t, s) for s, t in pairs]
        reverse_score = self._validate_mapping(target_emb, source_emb, reverse_pairs)

        return forward_score > 0.5 and reverse_score > 0.5

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        if len(a) != len(b):
            return 0.0

        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        return dot_product / (norm_a * norm_b)

    def get_mapping(
        self,
        source_modality: str,
        target_modality: str,
    ) -> Optional[CrossModalMapping]:
        """获取映射"""
        key = f"{source_modality}_to_{target_modality}"
        return self._mappings.get(key)

    def list_mappings(self) -> List[Dict[str, Any]]:
        """列出所有映射"""
        return [
            {
                "key": key,
                "source_modality": m.source_modality,
                "target_modality": m.target_modality,
                "training_samples": m.training_samples,
                "validation_score": m.validation_score,
                "is_learned": m.is_learned,
                "is_reversible": m.is_reversible,
            }
            for key, m in self._mappings.items()
        ]


class UnifiedRepresentationSpace:
    """
    统一表示空间

    将所有模态映射到共享的嵌入空间:
    1. 共享嵌入空间
    2. 跨模态相似度计算
    3. 语义对齐
    """

    def __init__(
        self,
        dimension: int = 128,
        modality_weights: Optional[Dict[str, float]] = None,
    ):
        self.dimension = dimension

        self.modality_weights = modality_weights or {
            "text": 1.0,
            "image": 1.0,
            "audio": 1.0,
            "multimodal": 1.0,
        }

        self._embedding_store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        logger.info(f"🌌 [UnifiedRepresentationSpace] 初始化，维度: {dimension}")

    def project_to_unified(
        self,
        embedding: List[float],
        source_modality: str,
    ) -> List[float]:
        """
        将嵌入投影到统一空间

        Args:
            embedding: 原始嵌入
            source_modality: 源模态

        Returns:
            统一空间中的嵌入
        """
        if len(embedding) == self.dimension:
            return embedding

        weight = self.modality_weights.get(source_modality, 1.0)

        if len(embedding) > self.dimension:
            projected = embedding[: self.dimension]
        else:
            projected = embedding + [0.0] * (self.dimension - len(embedding))

        projected = [x * weight for x in projected]

        norm = math.sqrt(sum(x * x for x in projected))
        if norm > 0:
            projected = [x / norm for x in projected]

        return projected

    def add_embedding(
        self,
        embedding_id: str,
        embedding: List[float],
        modality: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        添加嵌入到统一空间

        Args:
            embedding_id: 嵌入唯一标识
            embedding: 嵌入向量
            modality: 模态类型
            metadata: 附加元数据

        Returns:
            bool - 是否成功
        """
        unified = self.project_to_unified(embedding, modality)

        with self._lock:
            self._embedding_store[embedding_id] = {
                "embedding": unified,
                "modality": modality,
                "metadata": metadata or {},
                "added_at": time.time(),
            }

        return True

    def remove_embedding(self, embedding_id: str) -> bool:
        """移除嵌入"""
        with self._lock:
            if embedding_id in self._embedding_store:
                del self._embedding_store[embedding_id]
                return True
        return False

    def cross_modal_similarity(
        self,
        embedding_id1: str,
        embedding_id2: str,
    ) -> float:
        """
        计算跨模态相似度

        Args:
            embedding_id1: 第一个嵌入ID
            embedding_id2: 第二个嵌入ID

        Returns:
            float - 余弦相似度 (-1 到 1)
        """
        with self._lock:
            emb1 = self._embedding_store.get(embedding_id1)
            emb2 = self._embedding_store.get(embedding_id2)

        if not emb1 or not emb2:
            return 0.0

        return self._cosine_similarity(emb1["embedding"], emb2["embedding"])

    def similarity_search(
        self,
        query_embedding: List[float],
        query_modality: str,
        target_modalities: Optional[List[str]] = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float, str]]:
        """
        跨模态相似度搜索

        Args:
            query_embedding: 查询嵌入
            query_modality: 查询模态
            target_modalities: 目标模态列表 (None 表示所有)
            top_k: 返回前 K 个结果
            min_score: 最小分数阈值

        Returns:
            List[(embedding_id, similarity_score, modality)]
        """
        query_unified = self.project_to_unified(query_embedding, query_modality)

        results = []

        with self._lock:
            for emb_id, data in self._embedding_store.items():
                if target_modalities and data["modality"] not in target_modalities:
                    continue

                sim = self._cosine_similarity(query_unified, data["embedding"])

                if sim >= min_score:
                    results.append((emb_id, sim, data["modality"]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        if len(a) != len(b):
            return 0.0

        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        return dot_product / (norm_a * norm_b)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            by_modality = defaultdict(int)
            for data in self._embedding_store.values():
                by_modality[data["modality"]] += 1

            return {
                "total_embeddings": len(self._embedding_store),
                "dimension": self.dimension,
                "by_modality": dict(by_modality),
                "modality_weights": self.modality_weights,
            }


class CrossModalRetriever:
    """
    跨模态检索器

    支持:
    1. 文本检索图像
    2. 图像检索文本
    3. 音频检索文本/图像
    4. 任意模态组合检索
    """

    def __init__(
        self,
        unified_space: Optional[UnifiedRepresentationSpace] = None,
        top_k_default: int = 10,
    ):
        self.unified_space = unified_space or UnifiedRepresentationSpace()
        self.top_k_default = top_k_default

        self._retrieval_history: List[RetrievalResult] = []

        logger.info(f"🔍 [CrossModalRetriever] 初始化")

    def text_to_image(
        self,
        text_query: str,
        top_k: Optional[int] = None,
        text_encoder: Optional[Any] = None,
    ) -> RetrievalResult:
        """
        文本检索图像

        Args:
            text_query: 文本查询
            top_k: 返回数量
            text_encoder: 可选文本编码器

        Returns:
            RetrievalResult - 检索结果
        """
        import uuid

        start_time = time.time()
        query_id = f"txt2img_{uuid.uuid4().hex[:12]}"

        if text_encoder:
            result = text_encoder.encode(text_query)
            if result.success and result.embedding:
                query_emb = result.embedding.embedding
            else:
                query_emb = self._text_to_embedding_fallback(text_query)
        else:
            query_emb = self._text_to_embedding_fallback(text_query)

        results = self.unified_space.similarity_search(
            query_embedding=query_emb,
            query_modality="text",
            target_modalities=["image"],
            top_k=top_k or self.top_k_default,
        )

        formatted_results = []
        for emb_id, score, modality in results:
            formatted_results.append(
                {
                    "embedding_id": emb_id,
                    "similarity_score": score,
                    "modality": modality,
                }
            )

        retrieval_time = (time.time() - start_time) * 1000

        result = RetrievalResult(
            query_id=query_id,
            query_modality="text",
            results=formatted_results,
            total_count=len(formatted_results),
            retrieval_time_ms=retrieval_time,
        )

        self._retrieval_history.append(result)

        return result

    def image_to_text(
        self,
        image_embedding: List[float],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        图像检索文本

        Args:
            image_embedding: 图像嵌入
            top_k: 返回数量

        Returns:
            RetrievalResult - 检索结果
        """
        import uuid

        start_time = time.time()
        query_id = f"img2txt_{uuid.uuid4().hex[:12]}"

        results = self.unified_space.similarity_search(
            query_embedding=image_embedding,
            query_modality="image",
            target_modalities=["text"],
            top_k=top_k or self.top_k_default,
        )

        formatted_results = []
        for emb_id, score, modality in results:
            formatted_results.append(
                {
                    "embedding_id": emb_id,
                    "similarity_score": score,
                    "modality": modality,
                }
            )

        retrieval_time = (time.time() - start_time) * 1000

        result = RetrievalResult(
            query_id=query_id,
            query_modality="image",
            results=formatted_results,
            total_count=len(formatted_results),
            retrieval_time_ms=retrieval_time,
        )

        self._retrieval_history.append(result)

        return result

    def cross_modal_search(
        self,
        query_embedding: List[float],
        query_modality: str,
        target_modalities: List[str],
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        通用跨模态搜索

        Args:
            query_embedding: 查询嵌入
            query_modality: 查询模态
            target_modalities: 目标模态列表
            top_k: 返回数量

        Returns:
            RetrievalResult - 检索结果
        """
        import uuid

        start_time = time.time()
        query_id = f"cross_{uuid.uuid4().hex[:12]}"

        results = self.unified_space.similarity_search(
            query_embedding=query_embedding,
            query_modality=query_modality,
            target_modalities=target_modalities,
            top_k=top_k or self.top_k_default,
        )

        formatted_results = []
        for emb_id, score, modality in results:
            formatted_results.append(
                {
                    "embedding_id": emb_id,
                    "similarity_score": score,
                    "modality": modality,
                }
            )

        retrieval_time = (time.time() - start_time) * 1000

        result = RetrievalResult(
            query_id=query_id,
            query_modality=query_modality,
            results=formatted_results,
            total_count=len(formatted_results),
            retrieval_time_ms=retrieval_time,
        )

        self._retrieval_history.append(result)

        return result

    def _text_to_embedding_fallback(self, text: str) -> List[float]:
        """文本转嵌入的 fallback 实现"""
        import hashlib

        hasher = hashlib.sha256(text.encode("utf-8"))
        hash_bytes = hasher.digest()

        embedding = []
        for i in range(0, min(len(hash_bytes), 64), 4):
            chunk = hash_bytes[i : i + 4]
            if len(chunk) < 4:
                chunk = chunk + b"\x00" * (4 - len(chunk))
            value = int.from_bytes(chunk, "big") / 0xFFFFFFFF
            embedding.append(value * 2 - 1)

        while len(embedding) < 128:
            hasher.update(b"salt")
            hash_bytes = hasher.digest()
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i : i + 4]
                if len(chunk) < 4:
                    chunk = chunk + b"\x00" * (4 - len(chunk))
                value = int.from_bytes(chunk, "big") / 0xFFFFFFFF
                embedding.append(value * 2 - 1)
                if len(embedding) >= 128:
                    break

        return embedding[:128]

    def get_recent_retrievals(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的检索历史"""
        recent = self._retrieval_history[-count:]
        return [r.to_dict() for r in recent]


class CrossModalAssociationEngine:
    """
    跨模态关联引擎

    整合所有跨模态功能:
    1. 注意力对齐
    2. 关联学习
    3. 统一表示空间
    4. 跨模态检索
    """

    def __init__(
        self,
        embedding_dim: int = 128,
    ):
        self.embedding_dim = embedding_dim

        self.attention = CrossModalAttention(embedding_dim)
        self.learner = ModalityAssociationLearner(embedding_dim)
        self.unified_space = UnifiedRepresentationSpace(embedding_dim)
        self.retriever = CrossModalRetriever(self.unified_space)

        self._lock = threading.Lock()

        logger.info(f"🔗 [CrossModalAssociationEngine] 初始化，嵌入维度: {embedding_dim}")

    def align_modalities(
        self,
        source_embedding: List[float],
        source_modality: str,
        target_embeddings: List[List[float]],
        target_modality: str,
    ) -> Tuple[List[float], List[float], int, ModalityAlignment]:
        """
        对齐两种模态

        Args:
            source_embedding: 源嵌入
            source_modality: 源模态
            target_embeddings: 目标嵌入列表
            target_modality: 目标模态

        Returns:
            (aligned_embedding, attention_weights, best_match_index, alignment)
        """
        import uuid

        aligned, attention_weights, best_idx = self.attention.guided_attention(
            source_embedding=source_embedding,
            target_embeddings=target_embeddings,
        )

        similarity = 0.0
        if best_idx >= 0 and best_idx < len(target_embeddings):
            similarity = self._cosine_similarity(source_embedding, target_embeddings[best_idx])

        alignment = ModalityAlignment(
            alignment_id=f"align_{uuid.uuid4().hex[:12]}",
            source_modality=source_modality,
            target_modality=target_modality,
            source_embedding_id="",
            target_embedding_id=f"target_{best_idx}",
            similarity_score=similarity,
            alignment_strength=max(attention_weights) if attention_weights else 0.0,
            alignment_type=AssociationType.SEMANTIC,
            attention_weights=attention_weights,
        )

        return aligned, attention_weights, best_idx, alignment

    def fuse_multiple_modalities(
        self,
        embeddings: List[Tuple[List[float], str]],
        strategy: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        """
        融合多种模态的嵌入

        Args:
            embeddings: 嵌入列表 [(embedding, modality)]
            strategy: 融合策略
            weights: 可选权重

        Returns:
            融合后的嵌入
        """
        if not embeddings:
            return [0.0] * self.embedding_dim

        default_weights = {
            "text": 1.0,
            "image": 1.0,
            "audio": 1.0,
            "multimodal": 1.0,
        }
        weights = weights or default_weights

        unified_embeddings = []
        for emb, modality in embeddings:
            unified = self.unified_space.project_to_unified(emb, modality)
            unified_embeddings.append((unified, modality))

        if strategy == "concatenation":
            fused = []
            for emb, _ in unified_embeddings:
                fused.extend(emb)
            return fused[: self.embedding_dim]

        elif strategy == "average":
            dim = len(unified_embeddings[0][0])
            fused = [0.0] * dim
            for emb, _ in unified_embeddings:
                for i, val in enumerate(emb):
                    fused[i] += val / len(unified_embeddings)
            return fused

        elif strategy == "weighted_average":
            dim = len(unified_embeddings[0][0])
            fused = [0.0] * dim
            total_weight = 0.0

            for emb, modality in unified_embeddings:
                weight = weights.get(modality, 1.0)
                total_weight += weight
                for i, val in enumerate(emb):
                    fused[i] += val * weight

            if total_weight > 0:
                fused = [v / total_weight for v in fused]
            return fused

        else:
            dim = len(unified_embeddings[0][0])
            fused = [0.0] * dim
            for emb, _ in unified_embeddings:
                for i, val in enumerate(emb):
                    fused[i] += val / len(unified_embeddings)
            return fused

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        if len(a) != len(b):
            return 0.0

        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        return dot_product / (norm_a * norm_b)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "embedding_dimension": self.embedding_dim,
            "unified_space": self.unified_space.get_stats(),
            "learned_mappings": self.learner.list_mappings(),
            "recent_retrievals_count": len(self.retriever._retrieval_history),
        }


_engine: Optional[CrossModalAssociationEngine] = None
_engine_lock = threading.Lock()


def get_cross_modal_engine() -> CrossModalAssociationEngine:
    """获取跨模态关联引擎单例"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = CrossModalAssociationEngine()
    return _engine
