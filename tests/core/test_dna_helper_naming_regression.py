#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R5 / P1-9② 回归：TestDNAAdaptiveEngine 的 mock DNA helper 命名契约。

背景（2026-11-11 R5 实测，tools/triage_check.py LEDGER #11~#16）：
`hive/tests/test_phase3_evolution.py` 里 helper 被命名为 `test_create_mock_dna`
（以 test_ 开头 ⇒ 被 pytest 误收为一个用例，且 return MagicMock），而 6 个
调用方写的是 `self.create_mock_dna()` ⇒ 6 个用例全部抛
`AttributeError: 'TestDNAAdaptiveEngine' object has no attribute 'create_mock_dna'`。

修法：把 helper 改名为 `create_mock_dna`（去掉 test_ 前缀），6 个调用点无需改动。
预期副作用（已在台账登记）：helper 不再被收集为用例，hive/tests/ 收集数 475 → 474。

验收强度（沿用 R4 / R-20261010-11「旧实现注入反证」）：
- 运行时真实调用：exec 加载真实测试模块，用 unittest 真跑 6 个用例；
- 旧实现注入反证：运行时把名字注回 test_create_mock_dna，6 个用例必须精确抛
  AttributeError，证明新测试咬住的是命名而不是别的；
- 字节级复原：全程不改写工作区文件，前后 sha256 必须 SAME；
- AST 判据守卫：helper 名与 6 个调用点的投影必须一致。
"""
from __future__ import annotations

import ast
import hashlib
import io
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "hive" / "tests" / "test_phase3_evolution.py"
CLASS = "TestDNAAdaptiveEngine"
METHODS = [
    "test_monitor_dna",
    "test_monitor_duplicate_dna",
    "test_stop_monitoring",
    "test_record_performance",
    "test_suggest_adaptations_insufficient_data",
    "test_get_status_single",
]
_NEW_NAME = "create_mock_dna"
_OLD_NAME = "test_create_mock_dna"


def _source_bytes() -> bytes:
    return SRC.read_bytes()


def _source_text() -> str:
    # 该文件是 100% CRLF：归一化为 LF 后再做锚点匹配 / AST / exec，
    # 否则注入锚点会静默失效（R4 在 species_blueprint.py 上踩过这个坑）。
    return _source_bytes().decode("utf-8").replace("\r\n", "\n")


def _exec_module(source_text: str, name: str):
    """在隔离命名空间里 exec 源码，不污染 sys.modules / sys.path / 工作区。"""
    mod = types.ModuleType(name)
    mod.__file__ = str(SRC)
    old_path = list(sys.path)
    sys.modules[name] = mod
    try:
        exec(compile(source_text, str(SRC), "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
        sys.path[:] = old_path
    return mod


def _find_class(tree: ast.Module) -> ast.ClassDef:
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == CLASS), None)
    assert cls is not None, f"找不到类 {CLASS}，源码结构已漂移"
    return cls


def _run_six(cls) -> unittest.TestResult:
    suite = unittest.TestSuite(cls(m) for m in METHODS)
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


# ------------------------------------------------------------ AST 判据守卫

def test_helper_name_is_not_collectible():
    """helper 不得以 test_ 开头，否则会被 pytest 误收为用例。"""
    cls = _find_class(ast.parse(_source_text()))
    names = [n.name for n in cls.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert _NEW_NAME in names, f"helper {_NEW_NAME} 不存在"
    assert _OLD_NAME not in names, (
        f"helper 又被改回 {_OLD_NAME}，会被 pytest 误收为用例")


def test_six_call_sites_project_to_helper():
    """6 个调用点必须全部投影到同一个 helper（判据的每个投影都要守卫）。"""
    cls = _find_class(ast.parse(_source_text()))
    callsites = []
    for fn in cls.body:
        if (isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                and fn.name in METHODS):
            for n in ast.walk(fn):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == _NEW_NAME):
                    callsites.append(fn.name)
    assert sorted(callsites) == sorted(METHODS), (
        f"6 个调用点未全部投影到 {_NEW_NAME}: {callsites}")


# -------------------------------------------------------------- 真实调用

def test_six_cases_pass_with_real_module():
    """修完必做运行时真实调用：exec 真实源码后用 unittest 真跑 6 个用例。"""
    mod = _exec_module(_source_text(), "_hive_r5_dna_helper_real")
    result = _run_six(getattr(mod, CLASS))
    assert result.testsRun == 6
    assert result.wasSuccessful(), (
        f"6 个用例仍失败: errors={[str(e[1])[-160:] for e in result.errors]} "
        f"failures={[str(f[1])[-160:] for f in result.failures]}")


# ------------------------------------------------------- 旧实现注入反证

def test_old_helper_name_injection_red():
    """旧实现注入反证：把 helper 名注回 test_create_mock_dna，6 个用例必须变红。

    用「运行时构造旧实现」而非 `git show HEAD:` / 就地改写源码，避免
    R-20261010-19 的旧版取法腐坏与 R-20261010-11 的复原破坏性问题。
    """
    before = hashlib.sha256(_source_bytes()).hexdigest()
    text = _source_text()
    anchor = f"    def {_NEW_NAME}(self):"
    assert text.count(anchor) == 1, "注入锚点已漂移，反证失效，需重新标定"

    old_text = text.replace(anchor, f"    def {_OLD_NAME}(self):", 1)
    assert old_text != text
    old_mod = _exec_module(old_text, "_hive_r5_dna_helper_old")
    result = _run_six(getattr(old_mod, CLASS))

    assert not result.wasSuccessful(), "旧实现注入后 6 个用例竟仍全绿，反证失效"
    assert len(result.errors) == 6, (
        f"预期 6 个 AttributeError，实得 {len(result.errors)} "
        f"errors；failures={len(result.failures)}")
    for _, tb in result.errors:
        assert "AttributeError" in tb and _NEW_NAME in tb, (
            f"不是预期的 {_NEW_NAME} 缺失错误: {tb[-240:]}")

    # 字节级复原：全程未写文件，源码哈希必须 SAME
    assert hashlib.sha256(_source_bytes()).hexdigest() == before
