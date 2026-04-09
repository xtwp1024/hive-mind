#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cerebellum (小脑) - System 1 快速反射系统
从 Hive 1.0 移植并增强
"""

import json
import re
import difflib
import logging
import hashlib
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("Hive.Cerebellum")


class Cerebellum:
    """
    小脑 - System 1 快速反射系统

    功能:
    - 静态规则匹配 (最快)
    - 模糊记忆检索 (神经路径)
    - 从大脑学习 (印记机制)
    - Phase 2 P3: 实时记忆同步到 VectorVault (毫秒级异步写入)
    """

    def __init__(self, memory_path: Optional[Path] = None):
        """
        初始化小脑

        Args:
            memory_path: 记忆文件路径,默认为 cerebellum_memory.json
        """
        self.memory_path = memory_path or Path("cerebellum_memory.json")

        # 静态规则 (正则 + 处理函数)
        self.static_rules: List[Tuple[str, callable]] = [
            (r"Generate 3 high-value.*search topics", self._instinct_search_topics),
            (r"Summarize.*file", self._instinct_summarize),
            (r"分析.*趋势", self._instinct_trend_analysis),
            (r"推荐.*关键词", self._instinct_keywords),
        ]

        # 学习记忆
        self.memory: Dict[str, str] = self._load_memory()

        # 模糊匹配阈值
        self.SIMILARITY_THRESHOLD = 0.85

        # 记忆容量限制
        self.MAX_MEMORY_SIZE = 100

        # ===== Phase 2 P3: 实时记忆同步 =====
        # 后台线程池用于异步写入 VectorVault（毫秒级同步）
        self._write_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cereb_write")
        self._write_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._sync_thread_running = False
        self._sync_thread: Optional[threading.Thread] = None

        # VectorVault 单例（懒加载）
        self._vault = None

        # 同步统计
        self._sync_stats = {"queued": 0, "written": 0, "errors": 0}
        self._stats_lock = threading.Lock()

        # 启动后台同步线程
        self._start_sync_thread()

        logger.info("🧠 小脑已初始化 (Phase 2 P3: 实时记忆同步已启用)")

    # ===== Phase 2 P3: VectorVault 集成 =====

    def _get_vault(self):
        """懒加载 VectorVault 单例"""
        if self._vault is None:
            try:
                from hive.core.vector_vault import get_vector_vault
                self._vault = get_vector_vault()
                logger.info("💾 [Cerebellum] VectorVault 连接成功")
            except ImportError:
                logger.warning("💾 [Cerebellum] VectorVault 未安装，跳过实时同步")
            except Exception as e:
                logger.warning(f"💾 [Cerebellum] VectorVault 连接失败: {e}")
        return self._vault

    def _start_sync_thread(self):
        """启动后台同步线程（低优先级，持续写入 VectorVault）"""
        if self._sync_thread_running:
            return
        self._sync_thread_running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True, name="cereb_sync")
        self._sync_thread.start()
        logger.info("💾 [Cerebellum] 后台同步线程已启动")

    def _sync_loop(self):
        """后台同步循环：持续从队列中取出记忆并写入 VectorVault"""
        while self._sync_thread_running:
            try:
                # 从队列中获取记忆条目（带超时以便检查停止标志）
                item = self._write_queue.get(timeout=1.0)
                if item is None:  # 停止信号
                    break

                prompt, response = item
                self._write_to_vault(prompt, response)
                self._write_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"💾 [Cerebellum] 同步循环错误: {e}")

        logger.info("💾 [Cerebellum] 后台同步线程已停止")

    def _write_to_vault(self, prompt: str, response: str):
        """
        将单条记忆异步写入 VectorVault
        写入策略：去重 + 元数据标注来源为 cerebellum
        """
        vault = self._get_vault()
        if vault is None:
            return

        try:
            # 构建文档内容
            content = f"Prompt: {prompt}\nResponse: {response}"
            doc_id = f"cereb_{hashlib.md5(prompt.encode('utf-8', errors='ignore')).hexdigest()}"
            metadata = {
                "source": "cerebellum",
                "type": "reflex",
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
                "response_hash": hashlib.md5(response.encode()).hexdigest(),
                "imprint_time": time.time(),
            }

            success = vault.add_document(doc_id, content, metadata, skip_dedup=False)
            if success:
                with self._stats_lock:
                    self._sync_stats["written"] += 1
                logger.debug(f"💾 [Cerebellum] 记忆已同步到 VectorVault: {prompt[:30]}...")
            # 去重不计入错误（正常情况）

        except Exception as e:
            with self._stats_lock:
                self._sync_stats["errors"] += 1
            logger.error(f"💾 [Cerebellum] VectorVault 写入失败: {e}")

    def _enqueue_sync(self, prompt: str, response: str):
        """
        Phase 2 P3 核心：enqueue 记忆用于异步同步
        这是毫秒级写入的关键——主线程绝不等待 VectorVault I/O
        """
        try:
            self._write_queue.put_nowait((prompt, response))
            with self._stats_lock:
                self._sync_stats["queued"] += 1
        except queue.Full:
            # 队列满了，跳过这次同步（不影响主流程）
            with self._stats_lock:
                self._sync_stats["errors"] += 1
            logger.warning("💾 [Cerebellum] 同步队列已满，跳过此次同步")

    def get_sync_stats(self) -> Dict[str, Any]:
        """获取同步统计"""
        with self._stats_lock:
            return {
                **self._sync_stats,
                "queue_size": self._write_queue.qsize(),
                "vault_connected": self._vault is not None,
            }

    def flush_sync(self, timeout: float = 5.0):
        """
        强制刷新同步队列（等待所有待处理记忆写入 VectorVault）
        注意：这会阻塞调用线程，仅在关闭时使用
        """
        logger.info(f"💾 [Cerebellum] 正在刷新同步队列 ({self._write_queue.qsize()} 条待写入)...")
        self._write_queue.join()
        logger.info("💾 [Cerebellum] 同步队列已全部刷新")

    def shutdown(self):
        """优雅关闭：停止同步线程，刷新队列"""
        logger.info("💾 [Cerebellum] 正在关闭...")
        self._sync_thread_running = False
        # 发送停止信号
        try:
            self._write_queue.put_nowait(None)
        except queue.Full:
            pass
        # 等待同步线程结束
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        # 关闭线程池
        self._write_executor.shutdown(wait=False)
        logger.info("💾 [Cerebellum] 已关闭")

    # ===== 原有功能 =====

    def _load_memory(self) -> Dict[str, str]:
        """加载记忆"""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"记忆损坏: {e}")
        return {}

    def _save_memory(self) -> None:
        """保存记忆到本地 JSON（快速路径）"""
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存记忆失败: {e}")

    def reflex(self, prompt: str) -> Optional[str]:
        """
        快速反射（System 1）

        Args:
            prompt: 输入提示

        Returns:
            反射响应,如果没有匹配返回 None
        """
        # 1. 静态规则匹配 (最快路径)
        for pattern, handler in self.static_rules:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.info(f"⚡ 静态规则触发: '{pattern[:30]}...'")
                return handler(prompt)

        # 2. 模糊记忆检索 (神经路径)
        best_match = None
        highest_score = 0

        prompt_clean = prompt.strip().lower()
        for mem_prompt in self.memory.keys():
            score = difflib.SequenceMatcher(
                None,
                prompt_clean,
                mem_prompt.lower()
            ).ratio()

            if score > highest_score:
                highest_score = score
                best_match = mem_prompt

        if highest_score >= self.SIMILARITY_THRESHOLD:
            logger.info(f"🧠 神经匹配成功 (相似度: {highest_score:.2f})")
            return self.memory[best_match]

        return None

    def imprint(self, prompt: str, response: str) -> None:
        """
        印记学习 - 从大脑学习新反射

        Phase 2 P3: 双重写入
        1. 同步写入本地 JSON（毫秒级，阻塞）
        2. 异步写入 VectorVault（非阻塞，不影响 reflex 性能）

        Args:
            prompt: 输入提示
            response: 大脑响应
        """
        prompt_clean = prompt.strip()

        if prompt_clean not in self.memory:
            self.memory[prompt_clean] = response

            # 记忆容量控制
            if len(self.memory) > self.MAX_MEMORY_SIZE:
                # 移除最旧的条目
                oldest_key = next(iter(self.memory))
                del self.memory[oldest_key]
                logger.debug(f"记忆已满,移除: {oldest_key[:30]}...")

            # ===== Phase 2 P3: 双重写入 =====
            # 1. 同步写入本地 JSON（立即持久化，保证不丢失）
            self._save_memory()
            logger.info(f"📝 新反射已学习: '{prompt_clean[:30]}...'")

            # 2. 异步写入 VectorVault（后台线程，不阻塞主流程）
            # 这是"毫秒级"写入的关键：主线程立即返回，I/O 由后台处理
            self._enqueue_sync(prompt_clean, response)

    def clear_memory(self) -> None:
        """清空记忆"""
        self.memory.clear()
        self._save_memory()
        logger.info("🧹 记忆已清空")

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            "total_memories": len(self.memory),
            "capacity": self.MAX_MEMORY_SIZE,
            "usage_percent": (len(self.memory) / self.MAX_MEMORY_SIZE) * 100,
            "static_rules": len(self.static_rules),
            "sync": self.get_sync_stats(),
        }

    # ========== 静态本能处理函数 ==========

    def _instinct_search_topics(self, prompt: str) -> str:
        """本能: 搜索主题生成"""
        import random

        topics = [
            "autonomous-agent-frameworks",
            "self-improving-code-patterns",
            "vector-database-efficiency",
            "transformer-memory-compression",
            "neuromorphic-computing-sim",
            "multi-agent-orchestration",
            "LLM-tool-calling",
            "agent-memory-systems"
        ]

        selected = random.sample(topics, min(3, len(topics)))

        return json.dumps({
            "strategy_name": "小脑反射 (快速路径)",
            "topics": selected,
            "analysis": "基于本能的基线演化策略"
        }, ensure_ascii=False)

    def _instinct_summarize(self, prompt: str) -> str:
        """本能: 文件摘要"""
        return json.dumps({
            "summary": "文件内容已被系统吸收 (快速摘要)",
            "keywords": ["code", "source", "asset"]
        }, ensure_ascii=False)

    def _instinct_trend_analysis(self, prompt: str) -> str:
        """本能: 趋势分析"""
        return json.dumps({
            "trend": "AI Agent 技术持续升温",
            "keywords": ["multi-agent", "autonomous", "LLM-orchestration"],
            "confidence": 0.8
        }, ensure_ascii=False)

    def _instinct_keywords(self, prompt: str) -> str:
        """本能: 关键词推荐"""
        import random

        keyword_pool = [
            "autonomous-agents", "multi-agent-systems", "LLM-orchestration",
            "agent-memory", "tool-calling", "self-improving-AI",
            "swarm-intelligence", "distributed-agents"
        ]

        selected = random.sample(keyword_pool, 3)

        return json.dumps({
            "keywords": selected,
            "source": "小脑本能"
        }, ensure_ascii=False)


# 全局单例
_cerebellum_instance: Optional[Cerebellum] = None
_cerebellum_lock = __import__('threading').Lock()


def get_cerebellum() -> Cerebellum:
    """获取小脑单例（线程安全）"""
    global _cerebellum_instance
    if _cerebellum_instance is None:
        with _cerebellum_lock:
            if _cerebellum_instance is None:
                _cerebellum_instance = Cerebellum()
    return _cerebellum_instance
