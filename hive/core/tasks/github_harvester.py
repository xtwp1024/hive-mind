# -*- coding: utf-8 -*-
"""
GitHub 采集任务（异步版本）

让蜂群从 GitHub 采集代码，积累知识。
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from hive.utils.github_hunter import GithubHunter


@dataclass
class GitHubTask:
    """GitHub 采集任务"""
    id: str
    keywords: list[str]
    target_count: int = 5
    language: str = ""
    task_type: str = "code"


class GitHubHarvester:
    """
    GitHub 采集器（异步版本）

    从 GitHub 采集代码，提取知识。
    - 搜索阶段：多个关键词并发请求（asyncio.gather）
    - 提取阶段：完全同步（CPU/内存密集，轻量）
    """

    def __init__(self):
        self.hunter = GithubHunter()
        self.harvest_count = 0

    async def execute_task(self, task: GitHubTask) -> dict[str, Any]:
        """执行采集任务（完全异步）"""
        try:
            # 搜索仓库（异步并发）
            repos = await self._search_repos(task)

            # 提取知识（同步，轻量）
            knowledge = []
            for repo in repos[:task.target_count]:
                k = self._extract_repo_knowledge(repo, task)
                knowledge.append(k)

            self.harvest_count += 1

            return {
                "status": "SUCCESS",
                "task_id": task.id,
                "repos_found": len(repos),
                "knowledge_extracted": len(knowledge),
                "knowledge": knowledge
            }

        except Exception as e:
            return {
                "status": "FAILED",
                "task_id": task.id,
                "error": str(e)
            }

    async def _search_repos(self, task: GitHubTask) -> list[dict]:
        """搜索仓库（异步并发）"""
        try:
            # GithubHunter.hunt() 是异步并发方法
            repos = await self.hunter.hunt(
                keywords=task.keywords,
                count=task.target_count * 2
            )
            return repos
        except Exception as e:
            print(f"搜索仓库失败: {e}")
            return []

    def _extract_repo_knowledge(self, repo: dict, task: GitHubTask) -> dict:
        """从仓库提取知识（同步）"""
        name = repo.get("name", "")
        description = repo.get("description", "") or ""
        language = repo.get("language", "")
        stars = repo.get("stars", 0)
        url = repo.get("url", "")
        relevance_score = repo.get("relevance_score", 0)

        # 构建知识摘要
        summary = f"{name}: {description[:150]}" if description else name

        # 计算重要性（基于stars + 相关性）
        importance = min(1.0, stars / 1000 + 0.3)
        if relevance_score:
            importance = min(1.0, (importance * 0.6) + (relevance_score * 0.4))

        return {
            "type": "repo_knowledge",
            "repo_name": name,
            "language": language,
            "stars": stars,
            "relevance_score": relevance_score,
            "summary": summary,
            "importance": importance,
            "url": url,
            "keywords": task.keywords
        }

    async def cleanup(self) -> None:
        """清理资源（关闭异步客户端）"""
        await self.hunter.close()

    def get_status(self) -> dict[str, Any]:
        """获取状态"""
        return {
            "total_harvested": self.harvest_count
        }


# 默认采集任务关键词
DEFAULT_KEYWORDS = [
    "machine learning",
    "neural network",
    "deep learning",
    "artificial intelligence",
    "python async",
    "distributed system",
    "microservice",
    "docker kubernetes",
    "react vue",
    "rust go"
]
