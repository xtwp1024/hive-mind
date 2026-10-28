# -*- coding: utf-8 -*-
"""快速验证仓库脚本"""
import os
import subprocess
from pathlib import Path

def check_repo(repo_path):
    """快速检查仓库"""
    name = repo_path.name
    git_dir = repo_path / ".git"

    if not git_dir.exists():
        return name, "no_git", None

    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            count = result.stdout.strip()
            return name, "valid", count
        return name, "error", None
    except:
        return name, "error", None

def main():
    repo_dir = Path("D:/Hive/Data/openclaw_repos")
    results = {"valid": [], "no_git": [], "error": []}

    repos = list(repo_dir.iterdir())
    total = len(repos)
    print(f"检查 {total} 个仓库...")

    for i, repo_path in enumerate(repos):
        if repo_path.is_dir():
            name, status, count = check_repo(repo_path)
            results[status].append(name)
            if (i + 1) % 100 == 0:
                print(f"进度: {i+1}/{total}")

    print(f"\n结果:")
    print(f"  正常: {len(results['valid'])}")
    print(f"  无Git: {len(results['no_git'])}")
    print(f"  错误: {len(results['error'])}")

    # 列出有问题的
    if results["no_git"]:
        print(f"\n无 .git 目录 ({len(results['no_git'])}):")
        for n in results["no_git"][:10]:
            print(f"  - {n}")
        if len(results["no_git"]) > 10:
            print(f"  ... 还有 {len(results['no_git']) - 10} 个")

if __name__ == "__main__":
    main()
