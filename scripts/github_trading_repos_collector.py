# -*- coding: utf-8 -*-
"""
GitHub 交易/量化/虚拟币 项目爬取脚本 (含克隆)
功能：
1. 使用多组关键词搜索 GitHub 仓库（交易、量化、虚拟币、加密货币等）
2. 去重 + 相关性评分
3. 创建 Obsidian Markdown 笔记
4. 克隆实际代码仓库到本地
5. 目标：1000个项目
"""

import os
import sys
import time
import requests
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# 搜索关键词组合
TRADING_KEYWORDS = [
    "trading bot", "algorithmic trading", "high frequency trading",
    "stock trading", "forex trading", "day trading", "swing trading",
    "trading strategy", "trading platform", "trading system",
    "quantitative trading", "quant trading", "quantitative finance",
    "algorithmic trading", "quantitative analysis", "alpha strategy",
    "cryptocurrency trading", "crypto trading bot", "bitcoin trading",
    "ethereum trading", "defi trading", "dex trading", "token trading",
    "arbitrage bot", "crypto arbitrage", "solana trading", "trading signal",
    "binance api", "coinbase api", "kraken api", "exchange api",
    "tradingview", "backtesting", "paper trading",
    "financial data", "market data", "price prediction", "stock prediction",
    "forex", "options trading", "futures trading",
    "grid trading", "martingale", "dca strategy", "dollar cost averaging",
    "portfolio rebalancing", "rebalancing bot", "trading indicator",
    "technical analysis", "price alert", "trading bot python",
]


class GitHubTradingReposCollector:
    """GitHub 交易/量化/虚拟币 仓库收集器"""

    def __init__(
        self,
        obsidian_vault_path: str = None,
        data_dir: str = None,
        github_token: str = None,
        target_count: int = 1000,
        max_workers: int = 4
    ):
        # Obsidian vault 路径
        self.obsidian_vault = Path(obsidian_vault_path) if obsidian_vault_path else Path(
            "C:/Users/Administrator/Documents/Obsidian Vault"
        )
        self.obsidian_repos_folder = self.obsidian_vault / "Trading" / "GitHub_Repos"

        # 克隆目录
        self.clone_dir = Path(data_dir) if data_dir else Path("D:/Hive/Data/trading_repos")
        self.clone_dir.mkdir(parents=True, exist_ok=True)

        # GitHub 配置
        self.github_token = github_token or os.getenv("GITHUB_TOKEN", "")
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Hive-Trading-Repos/1.0",
        }
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"

        self.base_url = "https://api.github.com"

        # 目标和并发
        self.target_count = target_count
        self.max_workers = max_workers

        # 统计
        self.lock = Lock()
        self.collected_count = 0
        self.cloned_count = 0
        self.seen_urls: Set[str] = set()
        self.seen_names: Set[str] = set()
        self.seen_clone_names: Set[str] = set()
        self.rate_limit_hits = 0

        # 加载已存在的笔记和克隆目录
        self._load_existing()

    def _load_existing(self):
        """加载已存在的笔记和克隆目录，避免重复"""
        # Obsidian 笔记
        if self.obsidian_repos_folder.exists():
            for note in self.obsidian_repos_folder.glob("*.md"):
                name = note.stem
                self.seen_names.add(name)

        # 已克隆的目录
        if self.clone_dir.exists():
            for item in self.clone_dir.iterdir():
                if item.is_dir():
                    self.seen_clone_names.add(item.name)

        logger.info(f"已扫描到 {len(self.seen_names)} 个已存在笔记, {len(self.seen_clone_names)} 个已克隆目录")

    def _search_repos(self, keyword: str, per_page: int = 50) -> List[Dict]:
        """搜索单个关键词的仓库"""
        all_items = []
        page = 1

        query = f"{keyword} stars:>5"
        if not self.github_token:
            query += " language:Python"

        while True:
            try:
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": min(per_page, 100),
                    "page": page,
                }

                response = requests.get(
                    f"{self.base_url}/search/repositories",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )

                if response.status_code == 403:
                    logger.warning(f"触发限流，等待 60 秒...")
                    self.rate_limit_hits += 1
                    time.sleep(60)
                    continue

                if response.status_code != 200:
                    logger.error(f"搜索失败 '{keyword}': HTTP {response.status_code}")
                    break

                data = response.json()
                items = data.get("items", [])

                if not items:
                    break

                all_items.extend(items)

                if len(items) < per_page or page >= 10:
                    break

                page += 1
                time.sleep(1.5)

            except Exception as e:
                logger.error(f"搜索异常 '{keyword}': {e}")
                break

        return all_items

    def _score_repo(self, repo: Dict) -> float:
        """计算仓库相关性评分"""
        name = repo.get("name", "").lower()
        desc = (repo.get("description") or "").lower()
        stars = repo.get("stargazers_count", 0)

        score = 0.0

        high_value = ["trading bot", "crypto", "bitcoin", "ethereum", "defi",
                      "quantitative", "algorithmic", "high frequency", "arbitrage",
                      "backtest", "binance", "coinbase", "exchange api"]
        for kw in high_value:
            if kw in name or kw in desc:
                score += 2.0

        medium = ["trading", "strategy", "signal", "indicator", "chart",
                  "portfolio", "investment", "finance", "market data"]
        for kw in medium:
            if kw in name or kw in desc:
                score += 1.0

        if stars > 0:
            score += min(3.0, stars ** 0.2)

        return score

    def _create_obsidian_note(self, repo: Dict) -> str:
        """为仓库创建 Obsidian Markdown 笔记"""
        name = repo.get("name", "unknown")
        full_name = repo.get("full_name", name)
        description = repo.get("description", "") or "无描述"
        clone_url = repo.get("clone_url", "")
        stars = repo.get("stargazers_count", 0)
        language = repo.get("language", "Unknown") or "Unknown"
        forks = repo.get("forks_count", 0)
        issues = repo.get("open_issues_count", 0)
        created = (repo.get("created_at") or "")[:10] if repo.get("created_at") else "Unknown"
        updated = (repo.get("updated_at") or "")[:10] if repo.get("updated_at") else "Unknown"
        pushed = (repo.get("pushed_at") or "")[:10] if repo.get("pushed_at") else "Unknown"
        homepage = repo.get("homepage", "") or ""
        topics = repo.get("topics", []) or []
        license_info = repo.get("license", {}) or {}
        license_text = license_info.get("name", "未指定") if isinstance(license_info, dict) else "未指定"

        # 本地路径
        local_path = str(self.clone_dir / name)

        content = f"""---
title: {name}
source: GitHub
url: {clone_url}
stars: {stars}
forks: {forks}
language: {language}
created: {created}
updated: {updated}
topics: {', '.join(topics) if topics else '无'}
license: {license_text}
收集时间: {datetime.now().strftime('%Y-%m-%d')}
local_path: {local_path}
tags:
  - GitHub
  - trading
  - {language.lower() if language else 'unknown'}
  - {'crypto' if any(t in ' '.join(topics).lower() for t in ['crypto', 'bitcoin', 'ethereum', 'defi']) else 'quant'}
---

# {full_name}

## 基本信息

| 属性 | 值 |
|------|-----|
| ⭐ Stars | {stars:,} |
| 🍴 Forks | {forks:,} |
| 🐛 Issues | {issues:,} |
| 💻 语言 | {language} |
| 📅 创建时间 | {created} |
| 🔄 更新时间 | {updated} |
| 📤 最后推送 | {pushed} |
| 💾 本地路径 | `{local_path}` |

## 描述

{description}

## 链接

- **GitHub**: {clone_url}
{f'- **官网**: {homepage}' if homepage else ''}
- **本地仓库**: `{local_path}`

## 主题标签

{', '.join([f'#{t}' for t in topics]) if topics else '_无标签_'}

## 许可证

{license_text}

---

> 🤖 由 Hive 自动收集 | 收集时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

        return content

    def _save_note(self, repo: Dict) -> tuple:
        """保存单个仓库笔记"""
        name = repo.get("name", "unknown")
        clone_url = repo.get("clone_url", "")

        if not clone_url:
            return (name, "skipped", "No clone URL")

        if name in self.seen_names:
            return (name, "skipped", "Note already exists")

        try:
            content = self._create_obsidian_note(repo)

            safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
            safe_name = safe_name.replace("*", "_").replace("?", "_").replace("|", "_")
            safe_name = safe_name.replace("<", "_").replace(">", "_").replace('"', "_")

            note_path = self.obsidian_repos_folder / f"{safe_name}.md"
            note_path.parent.mkdir(parents=True, exist_ok=True)

            with open(note_path, "w", encoding="utf-8") as f:
                f.write(content)

            with self.lock:
                self.seen_names.add(name)
                self.collected_count += 1

            return (name, "success", str(note_path))

        except Exception as e:
            return (name, "failed", str(e)[:100])

    def _clone_repo(self, repo: Dict) -> tuple:
        """克隆单个仓库"""
        name = repo.get("name", "unknown")
        clone_url = repo.get("clone_url", "")

        if not clone_url:
            return (name, "skipped", "No clone URL")

        if name in self.seen_clone_names:
            return (name, "skipped", "Already cloned")

        target_path = self.clone_dir / name

        if target_path.exists():
            self.seen_clone_names.add(name)
            return (name, "skipped", "Dir already exists")

        cmd = ["git", "clone", "--depth", "1", clone_url, str(target_path)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                with self.lock:
                    self.seen_clone_names.add(name)
                    self.cloned_count += 1
                return (name, "success", str(target_path))
            else:
                # 清理失败目录
                if target_path.exists():
                    shutil.rmtree(target_path, ignore_errors=True)
                return (name, "failed", result.stderr[:100])

        except subprocess.TimeoutExpired:
            if target_path.exists():
                shutil.rmtree(target_path, ignore_errors=True)
            return (name, "failed", "Clone timeout")

        except Exception as e:
            if target_path.exists():
                shutil.rmtree(target_path, ignore_errors=True)
            return (name, "failed", str(e)[:100])

    def hunt(self, keywords: List[str] = None) -> List[Dict]:
        """主动狩猎 - 使用关键词搜索 GitHub 仓库"""
        if keywords is None:
            keywords = TRADING_KEYWORDS

        all_repos = []
        seen_urls: Set[str] = set()

        logger.info(f"=" * 60)
        logger.info(f"开始狩猎，目标: {self.target_count} 个仓库")
        logger.info(f"关键词数量: {len(keywords)}")
        logger.info(f"=" * 60)

        for i, keyword in enumerate(keywords):
            if len(all_repos) >= self.target_count:
                break

            logger.info(f"[{i+1}/{len(keywords)}] 搜索: '{keyword}'")

            try:
                items = self._search_repos(keyword)

                for item in items:
                    url = item.get("clone_url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        item["_relevance_score"] = self._score_repo(item)
                        all_repos.append(item)

                logger.info(f"  → 获取 {len(items)} 个仓库，累计 {len(all_repos)} 个")
                time.sleep(1)

            except Exception as e:
                logger.error(f"  → 搜索异常: {e}")

            if (i + 1) % 20 == 0 and len(all_repos) >= self.target_count * 0.8:
                logger.info(f"已达到目标的80%，提前结束搜索")
                break

        all_repos.sort(key=lambda x: x.get("_relevance_score", 0), reverse=True)

        logger.info(f"=" * 60)
        logger.info(f"狩猎完成! 共收集 {len(all_repos)} 个仓库")
        logger.info(f"限流次数: {self.rate_limit_hits}")
        logger.info(f"=" * 60)

        return all_repos[:self.target_count]

    def collect_and_save(self, keywords: List[str] = None) -> dict:
        """完整流程：搜索 + 创建笔记 + 克隆仓库"""
        start_time = time.time()

        # Step 1: 狩猎
        repos = self.hunt(keywords)

        if not repos:
            logger.warning("没有找到任何仓库")
            return {"status": "no_repos"}

        # Step 2: 创建笔记目录
        self.obsidian_repos_folder.mkdir(parents=True, exist_ok=True)

        # Step 3: 并发保存笔记
        logger.info(f"\n开始创建 Obsidian 笔记...")
        note_success = 0
        note_skipped = 0
        note_failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_repo = {
                executor.submit(self._save_note, repo): repo
                for repo in repos
            }

            for future in as_completed(future_to_repo):
                result = future.result()
                if result[1] == "success":
                    note_success += 1
                    logger.info(f"[笔记] ✅ {result[0]}")
                elif result[1] == "skipped":
                    note_skipped += 1
                else:
                    note_failed += 1
                    logger.warning(f"[笔记] ❌ {result[0]}: {result[2]}")

        # Step 4: 并发克隆仓库
        logger.info(f"\n开始克隆仓库到: {self.clone_dir}")
        clone_success = 0
        clone_skipped = 0
        clone_failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_repo = {
                executor.submit(self._clone_repo, repo): repo
                for repo in repos
            }

            for future in as_completed(future_to_repo):
                result = future.result()
                if result[1] == "success":
                    clone_success += 1
                    logger.info(f"[克隆] ✅ {result[0]} → {result[2]}")
                elif result[1] == "skipped":
                    clone_skipped += 1
                else:
                    clone_failed += 1
                    logger.warning(f"[克隆] ❌ {result[0]}: {result[2]}")

        elapsed = time.time() - start_time

        # 汇总
        summary = {
            "status": "completed",
            "total_found": len(repos),
            "notes": {"success": note_success, "skipped": note_skipped, "failed": note_failed},
            "clones": {"success": clone_success, "skipped": clone_skipped, "failed": clone_failed},
            "elapsed_seconds": elapsed,
            "obsidian_folder": str(self.obsidian_repos_folder),
            "clone_folder": str(self.clone_dir),
            "rate_limit_hits": self.rate_limit_hits,
        }

        self._save_summary_report(summary, repos)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"收集完成!")
        logger.info(f"笔记创建: ✅ {note_success} | ⏭️ {note_skipped} | ❌ {note_failed}")
        logger.info(f"仓库克隆: ✅ {clone_success} | ⏭️ {clone_skipped} | ❌ {clone_failed}")
        logger.info(f"耗时: {elapsed:.1f} 秒")
        logger.info(f"笔记目录: {self.obsidian_repos_folder}")
        logger.info(f"克隆目录: {self.clone_dir}")
        logger.info(f"{'=' * 60}")

        return summary

    def _save_summary_report(self, summary: dict, repos: List[Dict]):
        """保存收集汇总报告"""
        top_repos = sorted(repos, key=lambda x: x.get("stargazers_count", 0), reverse=True)[:100]

        report = f"""# GitHub 交易/量化/虚拟币 项目收集报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 收集统计

| 指标 | 值 |
|------|-----|
| 找到仓库总数 | {summary['total_found']} |
| 笔记创建成功 | {summary['notes']['success']} |
| 笔记跳过 | {summary['notes']['skipped']} |
| 笔记失败 | {summary['notes']['failed']} |
| 克隆成功 | {summary['clones']['success']} |
| 克隆跳过 | {summary['clones']['skipped']} |
| 克隆失败 | {summary['clones']['failed']} |
| 耗时 | {summary['elapsed_seconds']:.1f} 秒 |
| 限流次数 | {summary['rate_limit_hits']} |
| 笔记目录 | `{summary['obsidian_folder']}` |
| 克隆目录 | `{summary['clone_folder']}` |

## Top 50 Stars 仓库

| # | 仓库名 | Stars | 语言 | 描述 |
|---|--------|-------|------|------|
"""

        for i, repo in enumerate(top_repos[:50], 1):
            name = repo.get("full_name", repo.get("name", ""))
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language", "Unknown") or "Unknown"
            desc = (repo.get("description") or "无描述")[:60]
            url = repo.get("clone_url", "")
            report += f"| {i} | [{name}]({url}) | {stars:,} | {lang} | {desc} |\n"

        report += f"""
## 本地克隆的仓库列表

"""
        for i, repo in enumerate(repos, 1):
            name = repo.get("full_name", repo.get("name", ""))
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language", "Unknown") or "Unknown"
            url = repo.get("clone_url", "")
            local = str(self.clone_dir / repo.get("name", ""))
            report += f"- [{name}]({url}) - ⭐ {stars:,} | {lang}\n"
            report += f"  - 本地: `{local}`\n"

        report += f"""
---

🤖 由 Hive 自动生成
"""

        report_path = self.obsidian_repos_folder.parent / "GitHub_Repos_Collection_Report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"汇总报告已保存: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GitHub 交易/量化/虚拟币 项目爬取工具 (含克隆)"
    )
    parser.add_argument(
        "--vault", "-v",
        default="C:/Users/Administrator/Documents/Obsidian Vault",
        help="Obsidian vault 路径"
    )
    parser.add_argument(
        "--clone-dir", "-d",
        default="D:/Hive/Data/trading_repos",
        help="仓库克隆目录"
    )
    parser.add_argument(
        "--token", "-t",
        default=os.getenv("GITHUB_TOKEN", ""),
        help="GitHub Token"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=1000,
        help="目标收集数量"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="并发线程数"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GitHub 交易/量化/虚拟币 项目爬取工具 (含克隆)")
    print("=" * 60)
    print(f"Obsidian Vault: {args.vault}")
    print(f"克隆目录: {args.clone_dir}")
    print(f"目标数量: {args.count}")
    print(f"并发线程: {args.workers}")
    print(f"GitHub Token: {'已设置' if args.token else '未设置'}")
    print("=" * 60)

    collector = GitHubTradingReposCollector(
        obsidian_vault_path=args.vault,
        data_dir=args.clone_dir,
        github_token=args.token,
        target_count=args.count,
        max_workers=args.workers
    )

    result = collector.collect_and_save()

    return result


if __name__ == "__main__":
    main()
