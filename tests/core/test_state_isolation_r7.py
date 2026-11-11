#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R7 / P1-9③ 回归：两处「状态污染/顺序依赖」的 fixture 隔离契约。

背景（tools/triage_check.py LEDGER #2 / #9，R7 逐条实测复现）：
- #2 `hive/tests/test_evolution_integration.py::TestSwarmCommunicationIntegration::test_register_bee`
  `get_swarm_communication()` 是进程级单例，`registered_bees` 跨用例累积；
  按用例名字典序 `test_broadcast_discovery` 先注册同名 `"test_bee_1"`，
  轮到 `test_register_bee` 时再注册只是覆盖、计数不增 ⇒ `initial_count + 1`
  断言变 `1 != 2`（单跑 PASS、整文件 FAIL）。
- #9 `hive/tests/test_phase3_evolution.py::TestCapabilityGapAnalyzer::test_get_latest_report`
  `CapabilityGapAnalyzer.__init__` 把 `_persistence_path` 设成 CWD 相对的
  `hive/data/gap_analyzer/`，该共享目录堆着历史 `report_*.json`（`analyze()` 每跑
  一次还会再写一份），`get_latest_report()` 于是返回别人的报告，`assertIsNone` 失败。

修法：**只改测试**，两个 `setUp` 各自做 fixture 隔离（单例快照-清空-addCleanup 复原 /
独立空临时目录），不动产品行为、不删既有落盘文件、不再制造新的共享残留。

验收强度（沿用 R4/R5 与 R-20261010-11 / R-20261010-08）：
- AST 判据守卫：两个 `setUp` 必须含隔离，且隔离块在文件里唯一；
- 运行时真实调用：exec 真实源码 + unittest 真跑用例（污染态下必须 PASS）；
- 旧实现注入反证：运行时剥掉隔离块，同样污染态下必须精确变红（1 != 2 / is not None）；
- 字节级复原：全程不改写工作区源码，前后 sha256 SAME；
- 污染自造：临时目录 + 临时单例状态，结束复原 CWD 与单例，不碰真实落盘残留。
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC_SWARM = ROOT / "hive" / "tests" / "test_evolution_integration.py"
SRC_GAP = ROOT / "hive" / "tests" / "test_phase3_evolution.py"

BEGIN = "# --- R7 / P1-9③ fixture 隔离 BEGIN ---"
END = "# --- R7 / P1-9③ fixture 隔离 END ---"

# 提前在正常 CWD 下触发 hive 包初始化（其 __init__ 用 os.getcwd() 找 .env），
# 避免后面为制造污染而 chdir 时把缓存根目录建到临时目录里。
import hive.core.capability_gap_analyzer as _cga  # noqa: E402,F401
from hive.core.swarm_communication import get_swarm_communication  # noqa: E402

# get_latest_report() -> _dict_to_report() 需要的全部字段（最小合法报告）
FAKE_REPORT = {
    "report_id": "r7_injected_residue",
    "generated_at": 1.0,
    "analysis_window_hours": 24.0,
    "total_tasks_analyzed": 0,
    "total_gaps_detected": 0,
    "gaps_by_severity": {},
    "gaps_by_type": {},
    "gaps_by_capability": {},
    "remediation_plan": [],
    "evolutionary_insights": {},
}


# ------------------------------------------------------------ 源码工具

def _source_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _source_text(path: Path) -> str:
    # 两个文件都是 100% CRLF：归一化为 LF 后做 exec / AST / 锚点匹配，
    # 否则 R4 踩过的「LF 锚点静默失效」会复现。
    return _source_bytes(path).decode("utf-8").replace("\r\n", "\n")


def _strip_isolation(text: str) -> str:
    """运行时剥掉隔离块（含标记行），构造「旧实现」；只改内存字符串。"""
    out: list[str] = []
    inside = False
    hit = 0
    for line in text.split("\n"):
        if BEGIN in line:
            inside = True
            hit += 1
            continue
        if inside and END in line:
            inside = False
            continue
        if not inside:
            out.append(line)
    assert hit == 1 and not inside, f"隔离块标记异常: hit={hit}, inside={inside}"
    return "\n".join(out)


def _exec_module(source_text: str, name: str, src: Path):
    """在隔离命名空间 exec 源码，不污染 sys.modules / sys.path / 工作区。"""
    mod = types.ModuleType(name)
    mod.__file__ = str(src)
    old_path = list(sys.path)
    sys.modules[name] = mod
    try:
        exec(compile(source_text, str(src), "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
        sys.path[:] = old_path
    return mod


def _find_method(tree: ast.Module, cls_name: str, meth_name: str) -> ast.FunctionDef:
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    assert cls is not None, f"找不到类 {cls_name}"
    meth = next((n for n in cls.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == meth_name), None)
    assert meth is not None, f"找不到方法 {cls_name}::{meth_name}"
    return meth


def _run_case(cls, method: str) -> unittest.TestResult:
    suite = unittest.TestSuite([cls(method)])
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


def _digest(path: Path) -> str:
    return hashlib.sha256(_source_bytes(path)).hexdigest()


# ------------------------------------------------------------ AST 判据守卫

def test_isolation_blocks_present_and_unique():
    """两个 setUp 必须含隔离，且每份源码里隔离块恰好一处（判据唯一性）。"""
    for path, cls_name, needles in (
        (SRC_SWARM, "TestSwarmCommunicationIntegration",
         ("registered_bees.clear()", "addCleanup", "_registered_bees_snapshot")),
        (SRC_GAP, "TestCapabilityGapAnalyzer",
         ("TemporaryDirectory", "_persistence_path", "addCleanup")),
    ):
        text = _source_text(path)
        assert text.count(BEGIN) == 1, f"{path.name}: BEGIN 标记不唯一"
        assert text.count(END) == 1, f"{path.name}: END 标记不唯一"
        seg = ast.get_source_segment(text, _find_method(ast.parse(text), cls_name, "setUp"))
        assert seg is not None
        for needle in needles:
            assert needle in seg, f"{path.name} {cls_name}.setUp 缺隔离判据 {needle!r}"


# ------------------------------------------------------------ #2 单例污染

def _pollute_singleton() -> None:
    swarm = get_swarm_communication()
    swarm.register_bee("test_bee_1", object())


def _clean_singleton() -> None:
    get_swarm_communication().registered_bees.clear()


def test_swarm_new_passes_under_pollution():
    """修完必做运行时真实调用：单例已被污染，新 setUp 隔离后用例仍 PASS。"""
    text = _source_text(SRC_SWARM)
    mod = _exec_module(text, "_hive_r7_swarm_new", SRC_SWARM)
    try:
        _pollute_singleton()
        result = _run_case(mod.TestSwarmCommunicationIntegration, "test_register_bee")
        assert result.testsRun == 1
        assert result.wasSuccessful(), (
            f"污染态下仍失败: {[str(f[1])[-160:] for f in result.failures]} "
            f"{[str(e[1])[-160:] for e in result.errors]}")
    finally:
        _clean_singleton()


def test_swarm_old_isolation_removed_goes_red():
    """旧实现注入反证：剥掉隔离，同样的单例污染必须精确复现 1 != 2。"""
    before = _digest(SRC_SWARM)
    old_text = _strip_isolation(_source_text(SRC_SWARM))
    assert "registered_bees.clear()" not in old_text, "旧实现里仍残留隔离代码"
    mod = _exec_module(old_text, "_hive_r7_swarm_old", SRC_SWARM)
    try:
        _pollute_singleton()
        result = _run_case(mod.TestSwarmCommunicationIntegration, "test_register_bee")
        assert not result.wasSuccessful(), "旧实现注入后竟仍全绿，反证失效"
        assert len(result.failures) == 1, (
            f"预期 1 个断言失败，实得 failures={len(result.failures)} "
            f"errors={len(result.errors)}")
        assert "1 != 2" in result.failures[0][1], (
            f"不是预期的计数漂移: {result.failures[0][1][-240:]}")
    finally:
        _clean_singleton()
    assert _digest(SRC_SWARM) == before, "源码被改写，违反字节级复原"


# ------------------------------------------------------------ #9 磁盘残留

def _with_residue_cwd(fn) -> None:
    """在含 hive/data/gap_analyzer/report_*.json 的临时 CWD 里跑 fn，结束复原 CWD。"""
    with tempfile.TemporaryDirectory(prefix="hive_r7_residue_") as tmp:
        residue = Path(tmp) / "hive" / "data" / "gap_analyzer"
        residue.mkdir(parents=True)
        (residue / "report_r7_injected.json").write_text(
            json.dumps(FAKE_REPORT), encoding="utf-8")
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            assert list(Path("hive/data/gap_analyzer").glob("report_*.json")), (
                "污染未落地，反证会静默失效")
            fn()
        finally:
            os.chdir(old_cwd)


def test_gap_new_passes_with_residue():
    """磁盘有残留报告时，新 setUp 的独立空目录让用例确定性 PASS。"""
    text = _source_text(SRC_GAP)
    mod = _exec_module(text, "_hive_r7_gap_new", SRC_GAP)

    def run():
        result = _run_case(mod.TestCapabilityGapAnalyzer, "test_get_latest_report")
        assert result.testsRun == 1
        assert result.wasSuccessful(), (
            f"残留态下仍失败: {[str(f[1])[-160:] for f in result.failures]} "
            f"{[str(e[1])[-160:] for e in result.errors]}")

    _with_residue_cwd(run)


def test_gap_old_isolation_removed_goes_red():
    """旧实现注入反证：剥掉隔离，同样的磁盘残留必须让 assertIsNone 变红。"""
    before = _digest(SRC_GAP)
    old_text = _strip_isolation(_source_text(SRC_GAP))
    assert "_persistence_path = Path(self._report_tmp.name)" not in old_text, (
        "旧实现里仍残留隔离代码")
    mod = _exec_module(old_text, "_hive_r7_gap_old", SRC_GAP)

    def run():
        result = _run_case(mod.TestCapabilityGapAnalyzer, "test_get_latest_report")
        assert not result.wasSuccessful(), "旧实现注入后竟仍全绿，反证失效"
        assert len(result.failures) == 1, (
            f"预期 1 个断言失败，实得 failures={len(result.failures)} "
            f"errors={len(result.errors)}")
        assert "is not None" in result.failures[0][1], (
            f"不是预期的 assertIsNone 失败: {result.failures[0][1][-240:]}")

    _with_residue_cwd(run)
    assert _digest(SRC_GAP) == before, "源码被改写，违反字节级复原"


def test_sources_byte_identical_after_all_injections():
    """所有注入都在内存里：两份真实源码 sha256 与开工前一致。"""
    for path in (SRC_SWARM, SRC_GAP):
        raw = _source_bytes(path)
        assert raw.count(b"\r\n") == raw.count(b"\n"), f"{path.name}: 行尾被改写"
        assert _digest(path) == hashlib.sha256(raw).hexdigest()
