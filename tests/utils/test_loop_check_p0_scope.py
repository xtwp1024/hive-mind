# -*- coding: utf-8 -*-
"""R8 口径修订的回归：SCOPE 行的 P0 状态必须从 LOOP.md 派生，不得写死。

形态与 R7 的 `ledger_open_count` 同源（R-20261024-12 单一真源）：
真实仓库投影守卫 + 临时台账多状态注入反证，全程不改写真实工作区
（沿用 R-20261010-11「反证不得破坏工作区」纪律）。
R8 触发点：P0-1 完成后，loop_check.py 里写死的「P0-1/P0-3 未清」
立刻变成假陈述，且不会让 RESULT 判红 —— 是静默失真而非报错。
"""
from __future__ import annotations

import ast
from pathlib import Path

from tools import loop_check


def test_real_repo_p0_scope_matches_ledger():
    """真实台账：P0-1 已标 ✅ 故不在 open 里，P0-3 仍在。"""
    ids = loop_check.p0_open_ids()
    assert ids is not None, "LOOP.md 的 P0 标题未能解析"
    assert "P0-1" not in ids, ids
    assert "P0-3" in ids, ids


def test_main_scope_is_projected_from_ledger(capsys):
    """main() 的 SCOPE 行必须用派生结果，且不得残留写死的 P0 编号。"""
    rc = loop_check.main()
    out = capsys.readouterr().out
    ids = loop_check.p0_open_ids()
    assert ids is not None
    assert ("/".join(ids) + " 未清") in out, out
    assert "P0-1/P0-3 未清" not in out, out
    assert rc == 0, out


def test_p0_open_ids_is_derived_not_hardcoded(tmp_path: Path):
    """反证：临时台账改状态，派生结果必须跟着变；缺台账/无标题返回 None。"""
    doc = tmp_path / loop_check.LOOP_DOC
    doc.write_text(
        "**P0-1　待办甲**\n**P0-2　待办乙**\n", encoding="utf-8"
    )
    assert loop_check.p0_open_ids(root=tmp_path) == ["P0-1", "P0-2"]

    doc.write_text(
        "**P0-1　✅ R8 已完成**\n**P0-2　待办乙**\n", encoding="utf-8"
    )
    assert loop_check.p0_open_ids(root=tmp_path) == ["P0-2"]

    doc.write_text(
        "**P0-1　✅ 已完成**\n**P0-2　✅ 已完成**\n", encoding="utf-8"
    )
    assert loop_check.p0_open_ids(root=tmp_path) == []

    doc.write_text("没有 P0 标题行\n", encoding="utf-8")
    assert loop_check.p0_open_ids(root=tmp_path) is None
    doc.unlink()
    assert loop_check.p0_open_ids(root=tmp_path) is None


def test_no_hardcoded_p0_status_inside_main():
    """投影守卫：main() 内不得出现写死的 P0 编号串，且必须调用派生函数。

    直接扫 main() 函数体的字符串常量（跳过模块/函数 docstring），避免把
    「历史说明文字」误判成实现；再断言 main 体内存在 p0_open_ids 调用。
    """
    src = (Path(loop_check.ROOT) / "tools" / "loop_check.py").read_text(
        encoding="utf-8", errors="replace"
    )
    tree = ast.parse(src)
    main_fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    body = [n for stmt in main_fn.body for n in ast.walk(stmt)]
    literals = [
        n.value for n in body
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert not any("P0-1/P0-3" in s for s in literals), literals
    assert not any(
        s.strip().endswith("未清") and "P0-" in s for s in literals
    ), literals
    calls = [
        n.func.id for n in body
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "p0_open_ids" in calls, calls
