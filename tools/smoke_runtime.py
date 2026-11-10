#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hive-mind 运行时冒烟验证 — 静态体检之外的第 4 层（可选）。

背景：tools/loop_check.py 只保证「能解析、能编译、import 闭合」，
实测它 PASS 的情况下仍存在三处只在运行时才暴露的缺陷（缺 import os、
get_system_info 少 key、hunt_sync 嵌套 asyncio.run）。故补本脚本。

用法（必须用装了依赖的托管 venv，不要用 base 解释器）：
    python tools/smoke_runtime.py            # 默认 1 个周期
    python tools/smoke_runtime.py --cycles 2
    python tools/smoke_runtime.py --python <解释器绝对路径>

退出码：0 = 冒烟通过；1 = 冒烟失败；2 = 依赖缺失，跳过（不算通过也不算失败）。
只做单进程有界运行，不联网、不写仓库文件（Data/ 已被 .gitignore 覆盖）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TIMEOUT = 150

# 冒烟判定项：(名称, 是否必须出现, 匹配串)
MARKERS = [
    ("主入口初始化", True, "Initializing Hive Mind Swarm System"),
    ("完成至少一个主循环周期", True, "Cycle 1/"),
    ("工蜂返回成功载荷", True, "返回载荷: SUCCESS"),
    ("资源信息为真实值非 unknown", True, "CPU: "),
    ("蜂群正常收尾", True, "Hive cleanup complete"),
]

# 出现即判失败的串（保留大小写不敏感匹配）
FATAL_MARKERS = ["fatal error", "traceback (most recent call last)"]

# 结构化日志里的 error 级别行（monitor / assign_tasks 缺陷会落在这里）
ERROR_LEVEL = '"level": "error"'


def _preflight(python: str) -> int | None:
    """检查运行时依赖是否齐备；不齐返回退出码 2。"""
    probe = subprocess.run(
        [python, "-c", "import pydantic_settings, structlog, psutil, httpx, main"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        print("[SKIP] 运行时依赖未齐备，跳过冒烟（本脚本不算通过）")
        for line in detail[-6:]:
            print("      ", line)
        print("       装依赖：pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt")
        return 2
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="hive-mind 有界运行时冒烟")
    ap.add_argument("--cycles", type=int, default=1, help="主循环周期数（默认 1）")
    ap.add_argument("--python", default=sys.executable, help="运行 main.py 的解释器")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="超时秒数")
    args = ap.parse_args()

    if args.cycles < 1:
        print("[FAIL] --cycles 必须 >= 1")
        return 1

    skip = _preflight(args.python)
    if skip is not None:
        return skip

    env = dict(os.environ)
    env["HIVE_SMOKE_CYCLES"] = str(args.cycles)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    print(f"[RUN ] {args.python} main.py  (HIVE_SMOKE_CYCLES={args.cycles}, timeout={args.timeout}s)")
    try:
        proc = subprocess.run(
            [args.python, "main.py"], cwd=str(ROOT), env=env,
            capture_output=True, text=True, timeout=args.timeout,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        print(f"[FAIL] 冒烟超时（{args.timeout}s）—— 主循环未在预算内结束")
        return 1

    out = (proc.stdout or "") + (proc.stderr or "")
    lower = out.lower()
    print(f"[INFO] 进程退出码 {proc.returncode}，输出 {len(out.splitlines())} 行")

    failures: list[str] = []
    if proc.returncode != 0:
        failures.append(f"进程退出码 {proc.returncode}（非 0）")
    for fm in FATAL_MARKERS:
        if fm in lower:
            failures.append(f"出现致命标记: {fm}")
    if ERROR_LEVEL in out:
        n = out.count(ERROR_LEVEL)
        failures.append(f"结构化日志含 {n} 条 error 级别记录")
        for line in out.splitlines():
            if ERROR_LEVEL in line:
                print("      X", line[:200])
    for name, required, marker in MARKERS:
        if required and marker not in out:
            failures.append(f"缺少标记[{name}]: {marker!r}")

    # CPU/Memory 若仍为 unknown，说明 get_system_info 的 key 又缺了
    for line in out.splitlines():
        if '"CPU: ' in line or line.startswith("CPU: "):
            if "unknown" in line:
                failures.append("CPU 仍为 unknown（get_system_info 缺 key）")
            print("      ", line.strip()[:160])
            break

    print("[CHK ] 判定项：" + ", ".join(m[0] for m in MARKERS))
    print("       + 退出码 / 无 fatal / 无 error 级别日志 / CPU 非 unknown")
    if failures:
        print(f"RESULT: FAIL ({len(failures)} 项)")
        for f in failures:
            print("      X", f)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
