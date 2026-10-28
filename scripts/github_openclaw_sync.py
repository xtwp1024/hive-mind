# -*- coding: utf-8 -*-
"""
GitHub OpenClaw 仓库增量同步脚本

功能：
1. 增量下载 - 只下载不存在的仓库
2. 验证下载成功 - 使用 git fsck 验证克隆完整性
3. 完整去重处理 - 检测重复/损坏的仓库
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


class GitHubOpenClawSync:
    """GitHub OpenClaw 仓库增量同步器"""

    def __init__(self, output_dir: str = None, max_workers: int = 4, github_token: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path("Data/openclaw_repos")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers
        self.github_token = github_token or os.getenv("GITHUB_TOKEN", "")

        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Hive-OpenClaw-Sync/1.0",
        }
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"

        self.base_url = "https://api.github.com"

        # 统计
        self.downloaded_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        self.corrupted_count = 0
        self.duplicate_count = 0

        self.lock = Lock()
        self.seen_names = set()
        self.stats_log = []

        # 收集已存在的目录名
        self._scan_existing_dirs()

    def _scan_existing_dirs(self):
        """扫描已存在的目录"""
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir():
                    self.seen_names.add(item.name)
        logger.info(f"已扫描到 {len(self.seen_names)} 个已存在目录")

    def _get_unique_dir_name(self, original_name: str) -> str:
        """获取唯一目录名，按下载时间重命名"""
        if original_name not in self.seen_names:
            self.seen_names.add(original_name)
            return original_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{original_name}_{timestamp}"

        counter = 1
        while unique_name in self.seen_names:
            unique_name = f"{original_name}_{timestamp}_{counter}"
            counter += 1

        self.seen_names.add(unique_name)
        return unique_name

    def _is_git_repo_valid(self, repo_path: Path) -> bool:
        """验证 git 仓库完整性"""
        try:
            # 检查 .git 目录是否存在
            git_dir = repo_path / ".git"
            if not git_dir.exists():
                return False

            # 运行 git fsck 检查完整性
            result = subprocess.run(
                ["git", "fsck", "--no-progress"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            # 检查是否有错误
            if result.returncode != 0:
                return False

            # 检查是否有 commit（空仓库无效）
            commit_result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            if commit_result.returncode != 0:
                return False

            count = commit_result.stdout.strip()
            return count and int(count) > 0

        except Exception as e:
            logger.debug(f"验证失败 {repo_path}: {e}")
            return False

    def _check_repo_status(self, repo_path: Path) -> dict:
        """检查仓库状态"""
        name = repo_path.name
        git_dir = repo_path / ".git"
        has_git = git_dir.exists()
        is_valid = self._is_git_repo_valid(repo_path) if has_git else False

        # 获取仓库信息
        try:
            remote = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            remote_url = remote.stdout.strip() if remote.returncode == 0 else ""
        except:
            remote_url = ""

        return {
            "name": name,
            "path": str(repo_path),
            "has_git": has_git,
            "is_valid": is_valid,
            "remote_url": remote_url
        }

    def verify_downloads(self) -> dict:
        """验证所有已下载的仓库"""
        logger.info("=" * 60)
        logger.info("开始验证已下载仓库...")
        logger.info("=" * 60)

        verified = {"valid": [], "corrupted": [], "empty": [], "no_git": []}
        total = len(self.seen_names)

        for i, name in enumerate(sorted(self.seen_names)):
            repo_path = self.output_dir / name

            if not repo_path.exists():
                continue

            status = self._check_repo_status(repo_path)

            if not status["has_git"]:
                verified["no_git"].append(status)
                logger.warning(f"[{i+1}/{total}] 无 .git: {name}")
            elif not status["is_valid"]:
                verified["corrupted"].append(status)
                logger.warning(f"[{i+1}/{total}] 损坏: {name}")
            else:
                verified["valid"].append(status)
                logger.debug(f"[{i+1}/{total}] 正常: {name}")

        logger.info("=" * 60)
        logger.info("验证完成!")
        logger.info(f"正常: {len(verified['valid'])}")
        logger.info(f"损坏: {len(verified['corrupted'])}")
        logger.info(f"无 Git: {len(verified['no_git'])}")
        logger.info(f"总计时: {total}")
        logger.info("=" * 60)

        return verified

    def search_all_openclaw_repos(self) -> list:
        """搜索 GitHub 上所有包含 openclaw 的仓库"""
        all_repos = []
        page = 1
        per_page = 100

        logger.info("搜索 GitHub 上包含 'openclaw' 的仓库...")

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
                    logger.warning("触发限流，等待 60 秒...")
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

                if len(items) < per_page:
                    break

                page += 1
                time.sleep(2)

            except Exception as e:
                logger.error(f"搜索异常: {e}")
                break

        return all_repos

    def _clone_repo(self, repo: dict) -> tuple:
        """克隆单个仓库"""
        name = repo.get("name", "unknown")
        clone_url = repo.get("clone_url", "")

        if not clone_url:
            return (name, "", "failed", "No clone URL")

        # 检查是否已存在
        if name in self.seen_names:
            # 检查是否有效
            existing_path = self.output_dir / name
            if existing_path.exists() and self._is_git_repo_valid(existing_path):
                with self.lock:
                    self.skipped_count += 1
                return (name, clone_url, "skipped", "Already exists and valid")
            else:
                # 无效或损坏，删除后重新下载
                import shutil
                if existing_path.exists():
                    shutil.rmtree(existing_path, ignore_errors=True)
                    self.seen_names.discard(name)

        unique_dir_name = self._get_unique_dir_name(name)
        target_path = self.output_dir / unique_dir_name

        cmd = ["git", "clone", "--verbose", clone_url, str(target_path)]

        try:
            logger.info(f"克隆: {name} -> {unique_dir_name}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                # 验证克隆成功
                if self._is_git_repo_valid(target_path):
                    with self.lock:
                        self.downloaded_count += 1
                    return (name, clone_url, "success", unique_dir_name)
                else:
                    # 克隆成功但仓库无效
                    import shutil
                    shutil.rmtree(target_path, ignore_errors=True)
                    self.seen_names.discard(unique_dir_name)
                    with self.lock:
                        self.corrupted_count += 1
                    return (name, clone_url, "corrupted", "Clone succeeded but repo invalid")
            else:
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
        """增量下载所有仓库"""
        if repos is None:
            repos = self.search_all_openclaw_repos()

        if not repos:
            logger.warning("没有仓库可下载")
            return {"status": "no_repos"}

        # 过滤掉已存在的（增量下载）
        repos_to_download = []
        for repo in repos:
            name = repo.get("name", "")
            if name not in self.seen_names:
                repos_to_download.append(repo)
            else:
                existing_path = self.output_dir / name
                if existing_path.exists() and self._is_git_repo_valid(existing_path):
                    self.skipped_count += 1
                else:
                    repos_to_download.append(repo)

        total = len(repos_to_download)
        existing = len(repos) - total

        logger.info(f"总计找到 {len(repos)} 个仓库")
        logger.info(f"已存在（跳过）: {existing}")
        logger.info(f"需要下载: {total}")
        logger.info(f"使用 {self.max_workers} 个并发线程")

        if total == 0:
            logger.info("所有仓库已下载完成！")
            return {"status": "all_exist", "existing": existing}

        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_repo = {
                executor.submit(self._clone_repo, repo): repo
                for repo in repos_to_download
            }

            for future in as_completed(future_to_repo):
                result = future.result()
                results.append(result)

                completed = len(results)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0

                status = result[2]
                if status == "success":
                    logger.info(
                        f"进度: {completed}/{total} | "
                        f"成功: {self.downloaded_count} | "
                        f"跳过: {self.skipped_count} | "
                        f"损坏: {self.corrupted_count} | "
                        f"失败: {self.failed_count} | "
                        f"速率: {rate:.2f} repos/s"
                    )
                elif status == "skipped":
                    logger.debug(f"跳过: {result[0]} - {result[3]}")
                else:
                    logger.warning(f"失败: {result[0]} - {result[3]}")

        elapsed = time.time() - start_time

        summary = {
            "total_searched": len(repos),
            "existing": existing,
            "downloaded": self.downloaded_count,
            "skipped": self.skipped_count,
            "corrupted": self.corrupted_count,
            "failed": self.failed_count,
            "elapsed_seconds": elapsed,
            "rate": total / elapsed if elapsed > 0 else 0,
        }

        logger.info("=" * 60)
        logger.info("增量下载完成!")
        logger.info(f"搜索总数: {len(repos)}")
        logger.info(f"已存在: {existing}")
        logger.info(f"成功下载: {self.downloaded_count}")
        logger.info(f"跳过: {self.skipped_count}")
        logger.info(f"损坏: {self.corrupted_count}")
        logger.info(f"失败: {self.failed_count}")
        logger.info(f"耗时: {elapsed:.1f} 秒")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)

        return summary

    def full_sync(self) -> dict:
        """完整同步：验证 + 增量下载"""
        logger.info("=" * 60)
        logger.info("开始完整同步流程")
        logger.info("=" * 60)

        # Step 1: 验证已下载的仓库
        verification = self.verify_downloads()

        # Step 2: 增量下载
        download_summary = self.download_all()

        # Step 3: 最终统计
        final_summary = {
            "verification": {
                "total": len(self.seen_names),
                "valid": len(verification["valid"]),
                "corrupted": len(verification["corrupted"]),
                "no_git": len(verification["no_git"]),
            },
            "download": download_summary
        }

        logger.info("=" * 60)
        logger.info("完整同步完成!")
        logger.info(f"有效仓库: {final_summary['verification']['valid']}")
        logger.info(f"损坏仓库: {final_summary['verification']['corrupted']}")
        logger.info(f"本次下载: {final_summary['download'].get('downloaded', 0)}")
        logger.info("=" * 60)

        return final_summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GitHub OpenClaw 仓库增量同步工具")
    parser.add_argument("--output", "-o", default="Data/openclaw_repos",
                        help="输出目录")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="并发数")
    parser.add_argument("--token", "-t", default=os.getenv("GITHUB_TOKEN", ""),
                        help="GitHub Token")
    parser.add_argument("--verify-only", action="store_true",
                        help="仅验证，不下载")
    parser.add_argument("--download-only", action="store_true",
                        help="仅下载，不验证")

    args = parser.parse_args()

    print("=" * 60)
    print("GitHub OpenClaw 仓库增量同步工具")
    print("=" * 60)
    print(f"输出目录: {args.output}")
    print(f"并发数: {args.workers}")
    print(f"Token: {'已设置' if args.token else '未设置'}")
    print("=" * 60)

    sync = GitHubOpenClawSync(
        output_dir=args.output,
        max_workers=args.workers,
        github_token=args.token
    )

    if args.verify_only:
        sync.verify_downloads()
    elif args.download_only:
        sync.download_all()
    else:
        sync.full_sync()


if __name__ == "__main__":
    main()
