#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hive-mind 静态体检 — 长效推进轮每轮必跑。

三项口径：
  1. 语法    所有入库 .py 通过 py_compile
  2. 模块闭合 所有 from hive.* / import hive.* 目标在检出树中可解析
  3. 镜像一致 D:/repos/hive-mind HEAD 与本机 origin/master、主工作区 HEAD 对齐

只做静态检查，不 import 任何项目模块（不触发第三方依赖 / 服务启动）。
⚠ PASS 只代表「能解析、能编译」，不代表服务能跑。2026-11-10 R1 实测：
静态全 PASS 时仍存在 3 处只在运行时暴露的缺陷（缺 import os /
get_system_info 少 key / hunt_sync 嵌套 asyncio.run）。运行时验证另跑
tools/smoke_runtime.py（可选第 4 层，依赖缺失时显式 SKIP，不假报 PASS）。
退出码 0 = 全绿，1 = 有失败项。
"""
from __future__ import annotations

import ast
import py_compile
import subprocess
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore", SyntaxWarning)

ROOT = Path(__file__).resolve().parent.parent
MIRROR = Path("D:/repos/hive-mind")


def git(*args: str, cwd: Path = ROOT) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True,
            text=True, timeout=30, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return f"__ERROR__{exc}"


def tracked_py() -> list[str]:
    out = git("ls-files", "*.py")
    if out.startswith("__ERROR__"):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def check_syntax(files: list[str]) -> list[str]:
    bad = []
    for rel in files:
        path = ROOT / rel
        if not path.is_file():
            bad.append(f"{rel}: 文件不存在")
            continue
        try:
            py_compile.compile(str(path), cfile=None, doraise=True)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{rel}: {str(exc).splitlines()[-1][:120]}")
    return bad


def _is_hive(module: str) -> bool:
    # 排除 hive_brain 之类同名前缀的外部顶层模块
    return module == "hive" or module.startswith("hive.")


def _resolves(module: str) -> bool:
    rel = module.replace(".", "/")
    return (ROOT / f"{rel}.py").is_file() or (ROOT / rel).is_dir()


def check_imports(files: list[str]) -> tuple[int, list[str]]:
    targets: set[str] = set()
    for rel in files:
        path = ROOT / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=rel)
        except (OSError, SyntaxError):
            continue  # 语法错误由 check_syntax 负责报告
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and _is_hive(node.module):
                # 只验模块路径本身；from X import Y 里的 Y 多为类/函数，不是子模块
                targets.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_hive(alias.name):
                        targets.add(alias.name)

    # 逐级检查：hive.core.brain.strategy_selector 要求 hive/、hive/core/ 等都存在
    missing = []
    for mod in sorted(targets):
        parts = mod.split(".")
        if not all(_resolves(".".join(parts[:i])) for i in range(1, len(parts) + 1)):
            missing.append(mod)
    return len(targets), missing


def check_mirror() -> tuple[list[str], str]:
    """镜像口径（R1 修订）：镜像必须严格等于 origin/master；主工作区允许领先。

    旧口径要求三方严格相等 ⇒ 每轮「本地 commit 后、push 前」必然误报 FAIL，
    把正常推进态判成「基线已坏」，逼下一轮空转修基线。现拆成两条：
      1. 镜像 HEAD == origin/master（镜像作为远端参照，必须 pull --ff-only 跟上）
      2. 主工作区 HEAD == origin/master，或领先于它（origin/master 是 HEAD 的祖先）
    落后 / 分叉仍然 FAIL。
    """
    issues: list[str] = []
    if not (MIRROR / ".git").exists():
        return [f"镜像不存在: {MIRROR}"], "镜像缺失"

    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/master")
    mirror = git("rev-parse", "HEAD", cwd=MIRROR)
    for label, sha in (("主工作区 HEAD", head), ("origin/master", origin), ("镜像 HEAD", mirror)):
        if sha.startswith("__ERROR__"):
            issues.append(f"{label} 读取失败")
    if issues:
        return issues, "读取失败"

    if mirror != origin:
        issues.append(
            f"镜像与远端不一致: 镜像={mirror[:8]} origin/master={origin[:8]}（需 pull --ff-only）"
        )

    if head == origin:
        status = f"三方对齐 {head[:8]}"
    else:
        ancestor = git("merge-base", "--is-ancestor", origin, head)
        if ancestor.startswith("__ERROR__"):
            issues.append(
                f"主工作区落后或分叉: HEAD={head[:8]} origin/master={origin[:8]}（禁止直接推进）"
            )
            status = "落后/分叉"
        else:
            ahead = git("rev-list", "--count", f"{origin}..{head}")
            status = f"本地领先 {ahead} 个提交待推送（HEAD {head[:8]}）"
    return issues, status


def main() -> int:
    files = tracked_py()
    if not files:
        print("[FAIL] 取不到入库 .py 列表（是否在 git 仓库根目录？）")
        return 1

    bad_syntax = check_syntax(files)
    n_targets, bad_imports = check_imports(files)
    bad_mirror, mirror_status = check_mirror()

    print(f"[1/3] 语法       {len(files)} 个 .py，失败 {len(bad_syntax)}")
    for x in bad_syntax[:10]:
        print("      X", x)
    print(f"[2/3] 模块闭合   {n_targets} 个 hive.* 目标，缺失 {len(bad_imports)}")
    for x in bad_imports[:10]:
        print("      X", x)
    print(f"[3/3] 镜像一致   问题 {len(bad_mirror)} | {mirror_status}")
    for x in bad_mirror[:5]:
        print("      X", x)

    ok = not (bad_syntax or bad_imports or bad_mirror)
    print("SCOPE : 仅静态（语法/import 闭合/镜像一致），不含运行时；运行时另跑 tools/smoke_runtime.py")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
