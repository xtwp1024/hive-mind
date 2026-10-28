# -*- coding: utf-8 -*-
"""
GitHub OpenClaw 仓库全量下载脚本

功能：
1. 搜索 GitHub 上所有包含 "openclaw" 的仓库
2. 全量克隆（不浅克隆）
3. 重名仓库按下载时间重命名，不覆盖已有仓库
"""

import os
import sys
import time
import subprocess
import requests
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GitHubOpenClawDownloader:
    """GitHub OpenClaw 仓库下载器"""

    def __init__(self, output_dir: str = None, max_workers: int = 4, github_token: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path("Data/openclaw_repos")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers
        self.github_token = github_token or os.getenv("GITHUB_TOKEN", "")

        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Hive-OpenClaw-Downloader/1.0",
        }
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"

        self.base_url = "https://api.github.com"
        self.downloaded_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.lock = Lock()
        self.seen_names = set()

        # 收集已存在的目录名
        self._scan_existing_dirs()

    def _scan_existing_dirs(self):
        """扫描已存在的目录"""
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir():
                    self.seen_names.add(item.name)
        logger.info(f"已存在 {len(self.seen_names)} 个同名目录")

    def _get_unique_dir_name(self, original_name: str) -> str:
        """
        获取唯一目录名，按下载时间重命名，不覆盖已有目录
        """
        if original_name not in self.seen_names:
            self.seen_names.add(original_name)
            return original_name

        # 格式: original_name_20260323_143052
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{original_name}_{timestamp}"

        # 如果还冲突，继续加序号
        counter = 1
        while unique_name in self.seen_names:
            unique_name = f"{original_name}_{timestamp}_{counter}"
            counter += 1

        self.seen_names.add(unique_name)
        return unique_name

    def search_all_openclaw_repos(self) -> list:
        """
        搜索 GitHub 上所有包含 openclaw 的仓库
        使用分页获取所有结果
        """
        all_repos = []
        page = 1
        per_page = 100

        logger.info("开始搜索 GitHub 上包含 'openclaw' 的仓库...")

        while True:
            try:
                params = {
                    "q": "openclaw",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": per_page,
                    "page": page,
                }

                response = requests.get(
                    f"{self.base_url}/search/repositories",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )

                if response.status_code == 403:
                    logger.warning("触发 GitHub 限流，等待 60 秒...")
                    time.sleep(60)
                    continue

                if response.status_code != 200:
                    logger.error(f"搜索失败: HTTP {response.status_code}")
                    break

                data = response.json()
                items = data.get("items", [])

                if not items:
                    break

                all_repos.extend(items)
                logger.info(f"获取第 {page} 页，完成 {len(all_repos)} 个仓库")

                # 检查是否还有更多页
                if len(items) < per_page:
                    break

                page += 1
                time.sleep(2)  # 避免触发限流

            except Exception as e:
                logger.error(f"搜索异常: {e}")
                break

        logger.info(f"搜索完成，共找到 {len(all_repos)} 个仓库")
        return all_repos

    def _clone_repo(self, repo: dict) -> tuple:
        """
        克隆单个仓库
        返回: (repo_name, clone_url, status, message)
        """
        name = repo.get("name", "unknown")
        clone_url = repo.get("clone_url", "")

        if not clone_url:
            return (name, "", "failed", "No clone URL")

        # 获取唯一目录名
        unique_dir_name = self._get_unique_dir_name(name)
        target_path = self.output_dir / unique_dir_name

        # 构建 git clone 命令（全量克隆，不浅克隆）
        cmd = [
            "git", "clone",
            "--verbose",
            clone_url,
            str(target_path)
        ]

        try:
            logger.info(f"克隆: {name} -> {unique_dir_name}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 分钟超时
            )

            if result.returncode == 0:
                with self.lock:
                    self.downloaded_count += 1
                return (name, clone_url, "success", unique_dir_name)
            else:
                # 如果克隆失败，清理可能创建的目录
                if target_path.exists():
                    import shutil
                    shutil.rmtree(target_path, ignore_errors=True)
                    self.seen_names.discard(unique_dir_name)

                with self.lock:
                    self.failed_count += 1
                return (name, clone_url, "failed", result.stderr[:200])

        except subprocess.TimeoutExpired:
            with self.lock:
                self.failed_count += 1
            return (name, clone_url, "failed", "Clone timeout")

        except Exception as e:
            with self.lock:
                self.failed_count += 1
            return (name, clone_url, "failed", str(e)[:200])

    def download_all(self, repos: list = None) -> dict:
        """
        下载所有仓库
        """
        if repos is None:
            repos = self.search_all_openclaw_repos()

        if not repos:
            logger.warning("没有仓库可下载")
            return {"status": "no_repos"}

        total = len(repos)
        logger.info(f"开始下载 {total} 个仓库，使用 {self.max_workers} 个并发线程")

        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_repo = {
                executor.submit(self._clone_repo, repo): repo
                for repo in repos
            }

            for future in as_completed(future_to_repo):
                result = future.result()
                results.append(result)

                # 进度日志
                completed = len(results)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0

                logger.info(
                    f"进度: {completed}/{total} | "
                    f"成功: {self.downloaded_count} | "
                    f"失败: {self.failed_count} | "
                    f"速率: {rate:.2f} repos/s"
                )

        elapsed = time.time() - start_time

        summary = {
            "total": total,
            "downloaded": self.downloaded_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "elapsed_seconds": elapsed,
            "rate": total / elapsed if elapsed > 0 else 0,
        }

        logger.info("=" * 60)
        logger.info("下载完成!")
        logger.info(f"总计: {total} 个仓库")
        logger.info(f"成功: {self.downloaded_count}")
        logger.info(f"失败: {self.failed_count}")
        logger.info(f"耗时: {elapsed:.1f} 秒")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)

        return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GitHub OpenClaw 仓库全量下载器")
    parser.add_argument(
        "--output", "-o",
        default="Data/openclaw_repos",
        help="输出目录 (默认: Data/openclaw_repos)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="并发下载数 (默认: 4)"
    )
    parser.add_argument(
        "--token", "-t",
        default=os.getenv("GITHUB_TOKEN", ""),
        help="GitHub Token (可从 GITHUB_TOKEN 环境变量读取)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅搜索不下载"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GitHub OpenClaw 仓库全量下载器")
    print("=" * 60)
    print(f"输出目录: {args.output}")
    print(f"并发数: {args.workers}")
    print(f"Token: {'已设置' if args.token else '未设置 (可能触发限流)'}")
    print("=" * 60)

    downloader = GitHubOpenClawDownloader(
        output_dir=args.output,
        max_workers=args.workers,
        github_token=args.token
    )

    if args.dry_run:
        repos = downloader.search_all_openclaw_repos()
        print(f"\n找到 {len(repos)} 个仓库 (dry-run 模式，未下载)")
        for repo in repos[:10]:
            print(f"  - {repo['name']} (stars: {repo.get('stargazers_count', 0)})")
        if len(repos) > 10:
            print(f"  ... 还有 {len(repos) - 10} 个")
        return

    # 搜索并下载
    summary = downloader.download_all()

    print("\n下载摘要:")
    print(f"  总计: {summary['total']}")
    print(f"  成功: {summary['downloaded']}")
    print(f"  失败: {summary['failed']}")
    print(f"  耗时: {summary['elapsed_seconds']:.1f} 秒")


if __name__ == "__main__":
    main()
