#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hive-mind 静态体检 — 长效推进轮每轮必跑。

三项固定口径：
  1. 语法    所有入库 .py 通过 py_compile
  2. 模块闭合 所有 from hive.* / import hive.* 目标在检出树中可解析
  3. 镜像一致 D:/repos/hive-mind HEAD 与本机 origin/master、主工作区 HEAD 对齐

R6 元审视新增「附加守卫」（三项编号与语义不变）：
  [+] 验收链闭合 — 声明的验收工具必须存在且可编译；分诊台账静态自证必须 PASS。
      把「open 历史失败的分诊台账是否漂移」纳入每轮一条命令的硬口径，
      避免只跑静态三项就误以为全绿（沿用 R-20261104-02 收敛纪律）。
      R7 口径修订：open 数量不再写死在本文与 SCOPE 行，改为从分诊台账
      EXPECTED_OPEN 静态派生（R-20261024-12 单一真源），否则每修一个历史失败
      就会让这行输出变成假陈述。
      R8 口径修订：SCOPE 行的 P0 状态同改为从 LOOP.md 的 P0 标题派生，不再
      写死「P0-1/P0-3 未清」——完成任一项即成假陈述（同 R7 形态）。
  [+] CI 配置闭合 — R12 元审视新增。R9 让三份 workflow 从「不触发」变为
      真实运行后验收环境已变：YAML 被改坏会让 CI 变成 0 jobs 空跑（R9 实证
      ScannerError），而静态三项完全无感。故要求 workflow 在位、配置守卫
      测试在位且可编译、且守卫真的调用 yaml.safe_load（AST 防写死，沿用
      test_workflow_yaml_r9.py 自身的守卫纪律）。独立函数，不改动 R6 的
      check_acceptance_chain() 语义，其 4 项回归原样通过。
      R12 口径修订：分诊台账 open 归零后，SCOPE 不再输出「≠ 0 个 open
      历史失败已修」这类病句，改述「已全部收口」，并保留 open 数量派生串
      以兼容 test_loop_check_acceptance_chain.py 的既有断言。

只做静态检查，不 import 任何项目模块（不触发第三方依赖 / 服务启动）。
⚠ PASS 只代表「能解析、能编译 + 验收链工具自洽」，不代表服务能跑，
  也不代表 open 历史失败已修（数量以 tools/triage_check.py LEDGER 为准）。R1 实证静态三项全 PASS 时仍存在
  3 处只在运行时暴露的缺陷（缺 import os / get_system_info 少 key /
  hunt_sync 嵌套 asyncio.run）；R2/R3/R4 又三次实证静态层对路径默认值、
  根锚点、局部 import 遮蔽均无感。四层验收口径与集合范围见 D:/Hive/LOOP.md 二。
运行时验证另跑 tools/smoke_runtime.py（依赖缺失时显式 SKIP，不假报 PASS）。
退出码 0 = 全绿，1 = 有失败项。
"""
from __future__ import annotations

import ast
import py_compile
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore", SyntaxWarning)

ROOT = Path(__file__).resolve().parent.parent
MIRROR = Path("D:/repos/hive-mind")

# R6：验收链的唯一真源声明。LOOP.md 二、固定验收口径只做指针，不复制本清单。
# 台账写死的路径若漂移，体检必须报错而不是静默少跑一层。
ACCEPTANCE_TOOLS = ("tools/smoke_runtime.py", "tools/triage_check.py")
TRIAGE_TOOL = "tools/triage_check.py"
LOOP_DOC = "LOOP.md"

# R12：CI 配置闭合的唯一真源声明。R9 接通 master 后每次 push 都会触发
# 3 个 Actions run，workflow 从「静态配置」升级为「验收环境的一部分」：
# YAML 解析失败 / 分支写错 / 守卫被删，都会让云端静默少跑一层而本地
# loop_check.py 三项全 PASS。这份清单让体检能兜住「守卫是否还在、是否
# 还真的解析 YAML」，而不是只检查测试文件存在。
CI_WORKFLOWS = (
    ".github/workflows/test.yml",
    ".github/workflows/lint.yml",
    ".github/workflows/pwp-qa.yml",
)
CI_GUARD_TESTS = (
    "tests/utils/test_workflow_yaml_r9.py",
    "tests/utils/test_r12_ci_workflow.py",
)


def git(*args: str, cwd: Path = ROOT) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
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


def ledger_open_count(root: Path | None = None) -> int | None:
    """R7 口径修订：从分诊台账静态派生 open 数量（唯一真源 = EXPECTED_OPEN）。

    纯 AST，不 import、不 exec 任何项目模块；读不到就返回 None，由调用方
    在 SCOPE 行输出「数量以台账为准」而不是写死一个会过期的数字。
    """
    base = root or ROOT
    path = base / TRIAGE_TOOL
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "EXPECTED_OPEN"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)
                and not isinstance(node.value.value, bool)
            ):
                return node.value.value
    return None


_P0_HEADING = re.compile(r"^\*\*(P0-\d+)")


def p0_open_ids(root: Path | None = None) -> list[str] | None:
    """R8 口径修订：从 LOOP.md「四、待办队列 · P0」小节派生未清项。

    唯一真源是台账里 `**P0-N ...**` 标题行是否带 `✅`（本项目标记完成的
    约定）。纯文本解析，不 import、不 exec、不依赖第三方包。

    返回 None 表示台账缺失或一条 P0 标题都解析不到（格式漂移），调用方须
    输出「以 LOOP.md 为准」而不是编一个列表；返回空列表表示 P0 已清零。
    R7 已把 open 数量改为派生，本函数把同一纪律扩到 P0 状态——R8 完成
    P0-1 后，写死的「P0-1/P0-3 未清」立刻变成假陈述，正是这一形态。
    """
    base = root or ROOT
    try:
        text = (base / LOOP_DOC).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = False
    open_ids: list[str] = []
    for line in text.splitlines():
        m = _P0_HEADING.match(line.strip())
        if not m:
            continue
        found = True
        if "✅" not in line:
            open_ids.append(m.group(1))
    return open_ids if found else None


def check_acceptance_chain(root: Path | None = None) -> tuple[list[str], str]:
    """R6 附加守卫：验收工具链闭合 + 分诊台账静态自证。

    纯静态、无第三方依赖、不启动服务、不 import 项目模块。root 可注入，便于
    回归测试在临时目录构造「工具缺失 / 台账自证 FAIL」的失效态反证，全程不
    改写真实工作区（沿用 R-20261010-11 的「反证不得破坏工作区」纪律）。
    """
    base = root or ROOT
    issues: list[str] = []
    for rel in ACCEPTANCE_TOOLS:
        path = base / rel
        if not path.is_file():
            issues.append(f"验收工具缺失: {rel}")
            continue
        try:
            py_compile.compile(str(path), cfile=None, doraise=True)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"验收工具不可编译: {rel}: {str(exc).splitlines()[-1][:100]}")

    triage = base / TRIAGE_TOOL
    if triage.is_file():
        try:
            proc = subprocess.run(
                [sys.executable, str(triage)],
                cwd=str(base),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            issues.append(f"分诊台账自证无法执行: {exc}")
        else:
            if proc.returncode != 0 or "RESULT: PASS" not in proc.stdout:
                tail = proc.stdout.strip().splitlines()[-1:] or ["无输出"]
                issues.append(
                    f"分诊台账静态自证未 PASS（退出码 {proc.returncode}）: {tail[0][:100]}"
                )
    return issues, f"{len(ACCEPTANCE_TOOLS)} 个工具 / 分诊台账自证"


def check_ci_config_chain(root: Path | None = None) -> tuple[list[str], str]:
    """R12 附加守卫：CI 配置闭合（workflow 在位 + 守卫真的解析 YAML）。

    纯静态、无第三方依赖、不 import 项目模块、不连网、不启动服务。root 可注入，
    便于回归在临时目录构造「workflow 缺失 / 守卫缺失 / 守卫退化成硬编码」三种
    失效态反证，全程不改写真实工作区（沿用 R6 / R-20261010-11 纪律）。

    只检查「守卫在位且仍是真解析」，不代替 pytest 跑这些守卫、也不判断 CI
    云端结论（排队中不写成绿）——完整四层验收见 LOOP.md 二。
    """
    base = root or ROOT
    issues: list[str] = []
    for rel in CI_WORKFLOWS:
        if not (base / rel).is_file():
            issues.append(f"CI workflow 缺失: {rel}")
    present = 0
    for rel in CI_GUARD_TESTS:
        path = base / rel
        if not path.is_file():
            issues.append(f"CI 配置守卫缺失: {rel}")
            continue
        try:
            py_compile.compile(str(path), cfile=None, doraise=True)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"CI 配置守卫不可编译: {rel}: {str(exc).splitlines()[-1][:100]}")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError) as exc:
            issues.append(f"CI 配置守卫无法 AST 解析: {rel}: {exc}")
            continue
        # 防写死守卫：文件必须真的调用 yaml.safe_load，退化成字符串断言会判红
        has_yaml_parse = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "safe_load"
            for n in ast.walk(tree)
        )
        if not has_yaml_parse:
            issues.append(f"CI 配置守卫未解析 YAML（缺 yaml.safe_load 调用）: {rel}")
            continue
        present += 1
    return issues, f"{len(CI_WORKFLOWS)} 份 workflow / {present} 个 YAML 解析守卫"


def main() -> int:
    files = tracked_py()
    if not files:
        print("[FAIL] 取不到入库 .py 列表（是否在 git 仓库根目录？）")
        return 1

    bad_syntax = check_syntax(files)
    n_targets, bad_imports = check_imports(files)
    bad_mirror, mirror_status = check_mirror()
    bad_chain, chain_status = check_acceptance_chain()
    bad_ci, ci_status = check_ci_config_chain()

    print(f"[1/3] 语法       {len(files)} 个 .py，失败 {len(bad_syntax)}")
    for x in bad_syntax[:10]:
        print("      X", x)
    print(f"[2/3] 模块闭合   {n_targets} 个 hive.* 目标，缺失 {len(bad_imports)}")
    for x in bad_imports[:10]:
        print("      X", x)
    print(f"[3/3] 镜像一致   问题 {len(bad_mirror)} | {mirror_status}")
    for x in bad_mirror[:5]:
        print("      X", x)
    print(f"[+ ] 验收链闭合 问题 {len(bad_chain)} | {chain_status}（R6 附加守卫）")
    for x in bad_chain[:5]:
        print("      X", x)
    print(f"[+ ] CI 配置闭合 问题 {len(bad_ci)} | {ci_status}（R12 附加守卫）")
    for x in bad_ci[:5]:
        print("      X", x)

    ok = not (bad_syntax or bad_imports or bad_mirror or bad_chain or bad_ci)
    print(
        "SCOPE : 静态三项（语法/import 闭合/镜像一致）+ R6 附加守卫（验收链闭合）"
        "+ R12 附加守卫（CI 配置闭合）；"
        "不含运行时，运行时另跑 tools/smoke_runtime.py --cycles 1"
    )
    print(
        "SCOPE : 四层验收 = ① 本静态体检 ② 1 周期冒烟 ③ 带范围的两套 pytest"
        "（tests/ 子集 + hive/tests/ 全量；数字必须带集合范围）"
        " ④ python tools/triage_check.py --report <全量报告> 对账"
    )
    n_open = ledger_open_count()
    # R12：open 归零后「≠ 0 个 open 历史失败已修」是病句（把「已修」挂到
    # 一个为 0 的集合上）。保留派生出的数量串以兼容既有回归断言，但按
    # 0 / 非 0 / 无法派生 三态分别表述，不再制造假陈述。
    if n_open is None:
        open_desc = "open 历史失败（数量未能派生，以分诊台账为准）"
        ledger_desc = open_desc
    else:
        open_desc = f"{n_open} 个 open 历史失败"
        ledger_desc = (
            f"{open_desc}（分诊台账已全部收口）" if n_open == 0 else f"{open_desc}未全部收口"
        )
    p0 = p0_open_ids()
    if p0 is None:
        p0_desc = "P0 状态以 LOOP.md 为准"
    elif p0:
        p0_desc = "/".join(p0) + " 未清"
    else:
        p0_desc = "P0 已清零（以 LOOP.md 为准）"
    print(
        f"SCOPE : PASS ≠ 服务能跑 ≠ 服务可用；{ledger_desc}；{p0_desc}；"
        "CI 云端结论以 gh api 实测为准，排队中不写成已绿；"
        "完整运行时状态见 LOOP.md「三、当前快照」"
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
