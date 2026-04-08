# -*- coding: utf-8 -*-
"""
Async-capable GitHub Hunter

重构为真正的异步版本，支持并发搜索多个关键词。
"""

import asyncio
import os
import logging
from typing import List, Dict, Any, Optional

import httpx

from hive.config.settings import get_settings
from hive.utils.relevance import RelevanceConfig, RelevanceScorer
from hive.utils.targeting import load_targeting_keywords

logger = logging.getLogger("Hive.GitHunter")


class GithubHunter:
    """
    主动狩猎单元 - 异步版本

    支持：
    - 多个关键词并发搜索（asyncio.gather）
    - 异步 HTTP 请求（httpx.AsyncClient）
    - 异步 rate limit 延迟（asyncio.sleep）
    """

    def __init__(self):
        self.base_url = "https://api.github.com/search/repositories"
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Hive-Swarm-Hunter/2.0",
        }
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"

        settings = get_settings()
        self._preferred_languages = [
            lang.strip() for lang in settings.RELEVANCE_LANGUAGES.split(",") if lang.strip()
        ]
        self._relevance = RelevanceScorer(
            RelevanceConfig(
                min_score=settings.GITHUB_RELEVANCE_MIN_SCORE,
                strict=settings.GITHUB_RELEVANCE_STRICT,
                preferred_languages=self._preferred_languages,
            )
        )
        self._relevance_keywords = load_targeting_keywords()
        self._max_keywords = settings.GITHUB_MAX_KEYWORDS_PER_HUNT
        self._min_stars = settings.GITHUB_STARS_MIN
        self._sleep_seconds = settings.GITHUB_SEARCH_SLEEP_SECONDS
        self._rate_limit_cooldown = settings.GITHUB_RATE_LIMIT_COOLDOWN_SECONDS
        self._use_language_filter = settings.GITHUB_USE_LANGUAGE_FILTER
        self._query_in = settings.GITHUB_QUERY_IN
        # httpx 异步客户端（延迟初始化）
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建异步 HTTP 客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        return self._client

    async def close(self) -> None:
        """关闭异步客户端"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _infer_language(self, text: str) -> str | None:
        """推断语言（同步版本，供 search_one_keyword 使用）"""
        text = (text or "").lower()
        for lang in self._preferred_languages:
            if lang.lower() in text:
                return lang
        return None

    async def _search_one_keyword(
        self, keyword: str, count: int
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        搜索单个关键词，返回 (keyword, results) 元组

        所有关键词搜索结果合并到 seen_urls deduplication。
        """
        try:
            lang = self._infer_language(keyword) or (
                self._preferred_languages[0] if self._preferred_languages else "python"
            )

            query_parts = [keyword]
            if self._query_in:
                scope = self._query_in.replace(" ", "").strip(",")
                if scope:
                    query_parts.append(f"in:{scope}")
            if self._use_language_filter and lang:
                query_parts.append(f"language:{lang}")
            query_parts.append(f"stars:>{self._min_stars}")

            query = " ".join(query_parts)
            logger.info(f"🏹 狩猎开始: '{query}'")

            client = await self._get_client()
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": min(count, 50),
            }

            response = await client.get(self.base_url, params=params)
            results = []

            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                combined_keywords = [keyword] + self._relevance_keywords

                for item in items:
                    url = item.get("clone_url")
                    if not url:
                        continue

                    name = item.get("name", "")
                    description = item.get("description") or ""
                    stars = item.get("stargazers_count", 0)
                    language = item.get("language", "unknown") or "unknown"

                    score = self._relevance.score(
                        name=name,
                        description=description,
                        keywords=combined_keywords,
                        language=language,
                        stars=stars,
                    )

                    if self._relevance.strict and not self._relevance.is_relevant(score):
                        continue

                    results.append({
                        "name": name,
                        "url": url,
                        "description": description,
                        "stars": stars,
                        "language": language,
                        "relevance_score": score,
                    })

            elif response.status_code == 403:
                logger.warning("🧱 遭遇反爬虫护盾 (Rate Limit Exceeded)")
                await asyncio.sleep(self._rate_limit_cooldown)
            else:
                logger.error(f"狩猎失败: HTTP {response.status_code}")

            return keyword, results

        except Exception as e:
            logger.error(f"❌ 狩猎异常 ({keyword}): {e}")
            return keyword, []

    async def hunt(self, keywords: List[str], count: int = 10) -> List[Dict[str, Any]]:
        """
        根据关键词搜索 GitHub 仓库（真正异步并发）

        使用 asyncio.gather 并发搜索所有关键词，显著减少总等待时间。
        """
        if not keywords:
            return []

        max_keywords = min(len(keywords), self._max_keywords)
        selected_keywords = keywords[:max_keywords]

        # 并发搜索所有关键词
        tasks = [
            self._search_one_keyword(keyword, count)
            for keyword in selected_keywords
        ]
        results_tuples = await asyncio.gather(*tasks)

        # 合并结果（按 relevance_score 排序）
        all_results = []
        seen_urls = set()

        for _, results in results_tuples:
            for repo in results:
                url = repo.get("url")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                all_results.append(repo)

        # 按 stars + relevance_score 排序
        all_results.sort(
            key=lambda r: r.get("stars", 0) * 0.4 + r.get("relevance_score", 0) * 0.6,
            reverse=True
        )

        logger.info(f"🏹 捕获到 {len(all_results)} 个潜在猎物")
        return all_results[:count]

    # 保留同步版本（供不适用异步的场景）
    def hunt_sync(self, keywords: List[str], count: int = 10) -> List[Dict[str, Any]]:
        """同步版本 hunt（内部使用 asyncio.run）"""
        return asyncio.run(self.hunt(keywords, count))
