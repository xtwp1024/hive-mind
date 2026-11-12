#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Vault (向量金库)
Hive 2.0 的统一语义记忆系统，基于 ChromaDB。
"""

import hashlib
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb
from chromadb.config import Settings

from hive.config.settings import get_settings

logger = logging.getLogger("Hive.VectorVault")


# ============================================================================
# 轻量 Embedding Function —— 零模型下载依赖
# 用 SHA256 哈希直接生成固定维度的确定性向量
# 替代 Chroma DefaultEmbeddingFunction (ONNXMiniLM_L6_V2)，避免首次启动时
# 从 S3 下载 79MB onnx.tar.gz（速度 ~10KB/s，会卡死 Evolution 阶段）
# ============================================================================
class _HashEmbeddingFunction:
    """确定性哈希向量嵌入 —— 无外部依赖"""

    DIMENSIONS = 256  # 256 floats = 1KB/条，Chroma 默认 384 我们给 256 足够

    def __init__(self, dims: int = 256):
        self.dims = dims

    def __call__(self, input) -> List[List[float]]:
        """Chroma 要求的主接口：接收文档列表，返回向量列表"""
        if isinstance(input, str):
            input = [input]
        return [self._hash_vector(doc) for doc in input]

    def embed_query(self, input) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        return [self._hash_vector(doc) for doc in input]

    def _hash_vector(self, text: str) -> List[float]:
        """SHA256 → 足够多字节 → 归一化到 [-1,1]"""
        # 把文本扩充到 dims * 4 字节（每个 float 占 4 字节）
        target_bytes = self.dims * 4
        buf = bytearray()
        counter = 0
        # 多次 SHA256 拼接，保证足够长且确定性
        while len(buf) < target_bytes:
            h = hashlib.sha256(f"{text}::{counter}".encode("utf-8", errors="ignore")).digest()
            buf.extend(h)
            counter += 1
        # 转 float32 并归一化
        vec = []
        for i in range(self.dims):
            raw = struct.unpack_from("<f", buf, i * 4)[0]
            # inf/nan 保护 → 剪到 [-1,1]
            vec.append(
                max(-1.0, min(1.0, raw if raw == raw and abs(raw) < 1000 else (i % 20 - 10) / 10.0))
            )
        # L2 归一化
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    # ── Chroma 可选接口（新版本会警告缺失，补上更健壮）──
    @staticmethod
    def name() -> str:
        return "hash_embedding_v256"

    def get_config(self) -> Dict[str, Any]:
        return {"type": "hash_embedding", "dims": self.dims}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "_HashEmbeddingFunction":
        return _HashEmbeddingFunction(dims=config.get("dims", 256))


class VectorVault:
    """向量金库 - 基于 ChromaDB 的统一语义存储"""

    MAX_ACTIVE_DOCUMENTS = 1000000
    DEEP_STORAGE_ROTATION_SIZE_MB = 500

    def __init__(self, vault_path: str = None):
        self.settings = get_settings()
        self.persist_dir = Path(vault_path) if vault_path else self.settings.CHROMA_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.content_hashes: Set[str] = self._load_hash_index()
        self._hash_lock = __import__("threading").Lock()
        self.deep_storage_path = self.persist_dir / "deep_knowledge_vault.jsonl"
        self._init_client()

    def _init_client(self):
        try:
            logger.info(f"💾 初始化 VectorVault @ {self.persist_dir}")
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            # ⚠ 关键：显式指定 embedding_function，跳过 Chroma 默认的
            # ONNXMiniLM_L6_V2（首次启动从 S3 下载 79MB，速度 ~10KB/s 卡死）
            self._emb_fn = _HashEmbeddingFunction(dims=256)
            self.collection = self.client.get_or_create_collection(
                name="hive_knowledge",
                embedding_function=self._emb_fn,
            )
            logger.info(
                f"📚 ChromaDB 连接成功. 当前条目: {self.collection.count()} (embed={self._emb_fn.name()})"
            )
        except Exception as e:
            logger.error(f"❌ ChromaDB 初始化失败: {e}")
            raise e

    def _content_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

    def _load_hash_index(self) -> Set[str]:
        p = self.persist_dir / "content_hash_index.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception:
                pass
        return set()

    def _save_hash_index(self):
        try:
            with open(self.persist_dir / "content_hash_index.json", "w", encoding="utf-8") as f:
                json.dump(list(self.content_hashes), f)
        except Exception as e:
            logger.error(f"❌ 保存哈希索引失败: {e}")

    def add_document(
        self, doc_id: str, content: str, metadata: Dict[str, Any] = None, skip_dedup: bool = False
    ):
        if not content or not content.strip():
            return False
        try:
            if not skip_dedup and self._is_duplicate(content):
                return False

            doc_id = doc_id or hashlib.md5(content.encode()).hexdigest()
            meta = self._prepare_metadata(metadata or {}, content, skip_dedup)

            self.collection.upsert(documents=[content], metadatas=[meta], ids=[doc_id])
            self._check_and_archive()
            return True
        except Exception as e:
            logger.error(f"❌ 记忆植入失败 ({doc_id}): {e}")
            return False

    def _is_duplicate(self, content: str) -> bool:
        h = self._content_hash(content)
        # Thread-safe access to content_hashes set
        with self._hash_lock:
            if h in self.content_hashes:
                return True
            self.content_hashes.add(h)
        self._save_hash_index()
        return False

    def _prepare_metadata(self, metadata: Dict, content: str, skip_dedup: bool) -> Dict:
        m = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
        m["timestamp"] = time.time()
        if not skip_dedup:
            m["content_hash"] = self._content_hash(content)
        return m

    def add_documents_batch(self, documents: List[Dict[str, Any]], skip_dedup: bool = False) -> int:
        if not documents:
            return 0
        docs, ids, metas = [], [], []

        for doc in documents:
            c = doc.get("content")
            if not c or not c.strip():
                continue

            if not skip_dedup and self._content_hash(c) in self.content_hashes:
                continue
            if not skip_dedup:
                self.content_hashes.add(self._content_hash(c))

            docs.append(c)
            ids.append(doc.get("id") or hashlib.md5(c.encode()).hexdigest())
            metas.append(self._prepare_metadata(doc.get("metadata", {}), c, skip_dedup))

        return self._upsert_batch(docs, ids, metas, skip_dedup)

    def _upsert_batch(self, docs, ids, metas, skip_dedup) -> int:
        if not docs:
            return 0
        try:
            self.collection.upsert(documents=docs, metadatas=metas, ids=ids)
            if not skip_dedup:
                self._save_hash_index()
            logger.debug(f"📦 批量添加成功: {len(docs)} 条")
            return len(docs)
        except Exception as e:
            logger.error(f"❌ 批量添加失败: {e}")
            return 0

    def digest_folder(
        self,
        folder_path: Path,
        valid_extensions: Set[str] = None,
        max_files: int = 500,
        batch_size: int = 50,
    ) -> int:
        valid_extensions = valid_extensions or {
            ".py",
            ".md",
            ".js",
            ".ts",
            ".java",
            ".json",
            ".txt",
        }
        logger.info(f"🍽️ 消化文件夹: {folder_path}")

        state = {"added": 0, "errors": 0, "buffer": []}

        for root, _, files in os.walk(folder_path):
            if state["added"] >= max_files:
                break
            self._process_files_in_dir(root, files, valid_extensions, state, batch_size)

        self._flush_buffer(state)
        logger.info(f"✅ 消化完成: 添加 {state['added']}, 错误 {state['errors']}")
        return state["added"]

    def _process_files_in_dir(self, root, files, exts, state, batch_size):
        for file in files:
            if not self._is_valid_file(file, exts):
                continue
            try:
                doc = self._read_file_to_doc(Path(root) / file)
                if doc:
                    state["buffer"].append(doc)
                    if len(state["buffer"]) >= batch_size:
                        self._flush_buffer(state)
            except Exception:
                state["errors"] += 1

    def _is_valid_file(self, filename: str, valid_extensions: Set[str]) -> bool:
        return os.path.splitext(filename)[1].lower() in valid_extensions

    def _read_file_to_doc(self, file_path: Path) -> Optional[Dict]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            c = f.read()
        if len(c.strip()) < 50:
            return None
        return {
            "id": hashlib.md5(str(file_path).encode()).hexdigest(),
            "content": c,
            "metadata": {"path": str(file_path), "filename": file_path.name, "size": len(c)},
        }

    def _flush_buffer(self, state: Dict):
        if state["buffer"]:
            state["added"] += self.add_documents_batch(state["buffer"])
            state["buffer"] = []

    def _check_and_archive(self):
        c = self.collection.count()
        if c > self.MAX_ACTIVE_DOCUMENTS:
            logger.warning(f"⚠️ 文档数量超过限制 ({c} > {self.MAX_ACTIVE_DOCUMENTS})")

    def search(self, query: str, limit: int = 5, filter_metadata: Dict = None) -> List[Dict]:
        try:
            params = {"query_texts": [query], "n_results": limit}
            if filter_metadata:
                params["where"] = filter_metadata
            res = self.collection.query(**params)
            return self._format_search_results(res)
        except Exception as e:
            logger.error(f"❌ 检索失败: {e}")
            return []

    def _format_search_results(self, res: Dict) -> List[Dict]:
        if not res["ids"]:
            return []
        return [
            {
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i] if res["distances"] else None,
            }
            for i in range(len(res["ids"][0]))
        ]

    def get_stats(self) -> Dict[str, Any]:
        return {"total_documents": self.collection.count(), "backend": "chromadb"}

    # ===== Phase 2 P3: 异步写入支持 =====

    def async_add_document(
        self, doc_id: str, content: str, metadata: Dict[str, Any] = None, skip_dedup: bool = False
    ) -> Any:
        """
        Phase 2 P3: 异步版本 add_document
        使用线程池在后台写入，不阻塞调用线程
        返回 Future 对象

        Args:
            doc_id: 文档 ID
            content: 文档内容
            metadata: 元数据
            skip_dedup: 是否跳过去重

        Returns:
            concurrent.futures.Future
        """
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(self.add_document, doc_id, content, metadata, skip_dedup)

    async def await_add_document(
        self, doc_id: str, content: str, metadata: Dict[str, Any] = None, skip_dedup: bool = False
    ) -> bool:
        """
        Phase 2 P3: async/await 版本的文档添加
        适用于 asyncio 上下文中的非阻塞写入

        Args:
            doc_id: 文档 ID
            content: 文档内容
            metadata: 元数据
            skip_dedup: 是否跳过去重

        Returns:
            bool: 是否添加成功
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.add_document(doc_id, content, metadata, skip_dedup)  # 使用默认线程池
        )


_vault_instance = None
_vault_lock = __import__("threading").Lock()


def get_vector_vault() -> VectorVault:
    global _vault_instance
    if _vault_instance is None:
        with _vault_lock:
            if _vault_instance is None:
                _vault_instance = VectorVault()
    return _vault_instance
