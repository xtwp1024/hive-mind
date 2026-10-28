#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-Feast (自动进食) - 积压任务自动处理系统
从 Hive 1.0 移植
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
from hive.core.dna import BeeDNA
from hive.config.settings import get_settings
from hive.utils.relevance import RelevanceConfig, RelevanceScorer
from hive.utils.targeting import load_targeting_keywords


logger = logging.getLogger("Hive.AutoFeast")


class AutoFeast:
    """
    自动进食系统

    功能:
    - 处理 Scout 积压任务
    - 自动生成 DNA
    - 持久化到历史
    """

    def __init__(self,
                 loot_file: Path = Path("scout_loot.json"),
                 history_file: Path = Path("discovery_history.json")):
        """
        初始化自动进食系统

        Args:
            loot_file: Scout 积压文件
            history_file: 发现历史文件
        """
        self.loot_file = loot_file
        self.history_file = history_file
        self.history_cache: Set[str] = set()
        self._lock = __import__('threading').Lock()
        self._load_history_cache()
    
    def _load_history_cache(self) -> None:
        """加载历史缓存"""
        if self.history_file.exists():
            try:
                with self._lock:
                    with open(self.history_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.history_cache.update(data.get("assimilated_urls", []))
                logger.info(f"📚 历史缓存已加载: {len(self.history_cache)} 条")
            except Exception as e:
                logger.warning(f"加载历史失败: {e}")
    
    def get_backlog(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取积压任务

        Args:
            limit: 最大返回数量

        Returns:
            积压任务列表
        """
        if not self.loot_file.exists():
            return []

        try:
            with open(self.loot_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 过滤已处理的
            with self._lock:
                backlog = [
                    t for t in data.get("targets", [])
                    if t.get("url") not in self.history_cache
                ]

            return backlog[:limit]
        except Exception as e:
            logger.error(f"获取积压失败: {e}")
            return []
    
    def generate_dna_from_backlog(self, backlog: List[Dict[str, Any]]) -> List[BeeDNA]:
        """
        从积压生成 DNA
        
        Args:
            backlog: 积压任务列表
            
        Returns:
            DNA 列表
        """
        dna_list = []
        
        settings = get_settings()
        languages = [lang.strip() for lang in settings.RELEVANCE_LANGUAGES.split(",") if lang.strip()]
        scorer = RelevanceScorer(
            RelevanceConfig(
                min_score=settings.GITHUB_RELEVANCE_MIN_SCORE,
                strict=settings.GITHUB_RELEVANCE_STRICT,
                preferred_languages=languages,
            )
        )
        base_keywords = load_targeting_keywords()


        for item in backlog:
            url = item.get("url", "")
            if "github.com" not in url:
                continue

            name = item.get("id", "looted").split("/")[-1]
            description = item.get("title", "") or item.get("description", "") or ""
            source_query = item.get("source_query", "")
            keywords = base_keywords + [source_query]

            score = scorer.score(
                name=name,
                description=description,
                keywords=keywords,
                extra_text=url,
            )

            if scorer.strict and not scorer.is_relevant(score):
                continue
            
            # 创建 DNA
            dna = BeeDNA(
                species="bee",
                role="worker",
                id=f"backlog_{name}",
                target_name=name,
                target_url=url,
                mutation_gene="standard",
            )
            
            dna_list.append(dna)

        
        logger.info(f"🧬 从积压生成 {len(dna_list)} 个 DNA")
        return dna_list
    
    def purge_loot(self, handled_urls: List[str]) -> None:
        """
        清除已处理的积压

        Args:
            handled_urls: 已处理的 URL 列表
        """
        if not self.loot_file.exists() or not handled_urls:
            return

        try:
            with self._lock:
                with open(self.loot_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                original_len = len(data.get("targets", []))
                handled_set = set(handled_urls)
                data["targets"] = [
                    t for t in data.get("targets", [])
                    if t.get("url") not in handled_set
                ]
                new_len = len(data["targets"])

                if original_len != new_len:
                    # 原子性写入
                    temp_file = self.loot_file.with_suffix('.json.tmp')
                    with open(temp_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    temp_file.replace(self.loot_file)
                    logger.info(f"🧹 已清理 {original_len - new_len} 个积压任务")
        except Exception as e:
            logger.error(f"清理积压失败: {e}")
    
    def persist_to_history(self, urls: List[str]) -> None:
        """
        持久化到历史

        Args:
            urls: URL 列表
        """
        if not urls:
            return

        try:
            with self._lock:
                # 读取现有历史
                if self.history_file.exists():
                    with open(self.history_file, "r", encoding="utf-8") as f:
                        hist_data = json.load(f)
                else:
                    hist_data = {"assimilated_urls": []}

                # 添加新 URL
                current_set = set(hist_data.get("assimilated_urls", []))
                new_urls = [u for u in urls if u not in current_set]

                if new_urls:
                    hist_data["assimilated_urls"].extend(new_urls)

                    # 原子性写入：先写临时文件再重命名
                    temp_file = self.history_file.with_suffix('.json.tmp')
                    with open(temp_file, "w", encoding="utf-8") as f:
                        json.dump(hist_data, f, indent=2, ensure_ascii=False)
                    temp_file.replace(self.history_file)

                    # 更新缓存
                    self.history_cache.update(new_urls)

                    logger.info(f"💾 已固化 {len(new_urls)} 条历史记录")
        except Exception as e:
            logger.error(f"持久化历史失败: {e}")
    
    def feast(self, limit: int = 10) -> List[BeeDNA]:
        """
        执行自动进食
        
        Args:
            limit: 最大处理数量
            
        Returns:
            生成的 DNA 列表
        """
        # 1. 获取积压
        backlog = self.get_backlog(limit)
        
        if not backlog:
            logger.debug("📭 无积压任务")
            return []
        
        logger.info(f"🍽️ 发现 {len(backlog)} 个积压任务,开始进食...")
        
        # 2. 生成 DNA
        dna_list = self.generate_dna_from_backlog(backlog)
        
        # 3. 清理积压
        handled_urls = [item.get("url") for item in backlog]
        self.purge_loot(handled_urls)
        
        # 4. 持久化历史
        self.persist_to_history(handled_urls)
        
        logger.info(f"✅ 进食完成,生成 {len(dna_list)} 个任务")
        return dna_list


# 全局单例
_auto_feast_instance = None
_auto_feast_lock = __import__('threading').Lock()

def get_auto_feast() -> AutoFeast:
    """获取自动进食单例"""
    global _auto_feast_instance
    if _auto_feast_instance is None:
        with _auto_feast_lock:
            if _auto_feast_instance is None:
                _auto_feast_instance = AutoFeast()
    return _auto_feast_instance
