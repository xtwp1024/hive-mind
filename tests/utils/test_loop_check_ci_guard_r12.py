# -*- coding: utf-8 -*-
"""R12 元审视新增「CI 配置闭合」附加守卫的回归。

R9 让三份 workflow 从「不触发」变为每次 push 真实运行后，验收环境已变：
workflow YAML 被改坏会让 CI 变成 0 jobs 空跑（R9 实证 ScannerError），而
loop_check.py 的静态三项对此完全无感。R12 把「CI 配置守卫是否在位、是否
仍真的解析 YAML」纳入每轮一条命令的硬口径。

反证形态（沿用 R6 / R-20261010-11）：临时目录构造三种失效态，断言守卫判红；
完整态断言 0 问题，形成量化对照，证明守卫不是恒绿装饰。全程不改写真实
工作区。调用 main() 的用例按 P1-13 纪律显式隔离外部镜像状态。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools import loop_check

REAL_ROOT = Path(loop_check.ROOT)


def _make_chain(
    tmp_path: Path,
    *,
    drop_workflow: str | None = None,
    drop_guard: str | None = None,
    hardcode_guard: str | None = None,
) -> Path:
    """在临时目录造一套最小 CI 配置链；三个参数分别注入一种失效态。"""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True, exist_ok=True)
    for name in ("test.yml", "lint.yml", "pwp-qa.yml"):
        if name != drop_workflow:
            (wf / name).write_text("name: stub\n", encoding="utf-8")

    guards = tmp_path / "tests" / "utils"
    guards.mkdir(parents=True, exist_ok=True)
    for name in ("test_workflow_yaml_r9.py", "test_r12_ci_workflow.py"):
        if name == drop_guard:
            continue
        if name == hardcode_guard:
            # 退化成不含 yaml.safe_load 的恒绿断言（守卫被架空）
            body = "def test_placeholder():\n    assert True\n"
        else:
            body = "import yaml\n\ndata = yaml.safe_load('a: 1')\n"
        (guards / name).write_text(body, encoding="utf-8")
    return tmp_path


# ---------- 1. 真实仓库自洽 ----------


def test_real_repo_ci_chain_is_self_consistent():
    issues, status = loop_check.check_ci_config_chain()
    assert issues == [], issues
    assert "YAML 解析守卫" in status


def test_both_guard_files_and_workflows_exist_in_real_repo():
    """R12 点名复核①：R9 的 workflow 配置守卫必须真被纳入守卫清单。"""
    joined = " ".join(loop_check.CI_GUARD_TESTS)
    assert "test_workflow_yaml_r9.py" in joined
    assert len(loop_check.CI_WORKFLOWS) == 3
    for rel in (*loop_check.CI_WORKFLOWS, *loop_check.CI_GUARD_TESTS):
        assert (REAL_ROOT / rel).is_file(), rel


# ---------- 2. 失效态注入反证（有区分力，不是恒绿） ----------


def test_complete_temp_chain_has_no_issues(tmp_path):
    _make_chain(tmp_path)
    issues, status = loop_check.check_ci_config_chain(root=tmp_path)
    assert issues == [], issues
    assert "3 份 workflow / 2 个 YAML 解析守卫" in status


def test_missing_workflow_is_detected(tmp_path):
    _make_chain(tmp_path, drop_workflow="lint.yml")
    issues, _ = loop_check.check_ci_config_chain(root=tmp_path)
    assert any("CI workflow 缺失" in x and "lint.yml" in x for x in issues), issues


def test_missing_guard_is_detected(tmp_path):
    _make_chain(tmp_path, drop_guard="test_r12_ci_workflow.py")
    issues, _ = loop_check.check_ci_config_chain(root=tmp_path)
    assert any("CI 配置守卫缺失" in x for x in issues), issues


def test_hardcoded_guard_is_detected(tmp_path):
    """守卫退化成不解析 YAML 的硬编码断言 ⇒ 必须判红（AST 防写死）。"""
    _make_chain(tmp_path)
    ok_issues, _ = loop_check.check_ci_config_chain(root=tmp_path)
    assert ok_issues == [], ok_issues

    _make_chain(tmp_path, hardcode_guard="test_r12_ci_workflow.py")
    bad_issues, _ = loop_check.check_ci_config_chain(root=tmp_path)
    assert any("未解析 YAML" in x for x in bad_issues), bad_issues
    assert len(bad_issues) > len(ok_issues), "守卫无区分力"


def test_uncompilable_guard_is_detected(tmp_path):
    _make_chain(tmp_path)
    (tmp_path / "tests" / "utils" / "test_r12_ci_workflow.py").write_text(
        "def broken(:\n", encoding="utf-8"
    )
    issues, _ = loop_check.check_ci_config_chain(root=tmp_path)
    assert any("不可编译" in x or "无法 AST 解析" in x for x in issues), issues


# ---------- 3. structural / 接线守卫 ----------


def test_guard_is_structural_not_hardcoded():
    """守卫必须真的编译 + AST 解析，且清单驱动，不能写死路径字符串判断。"""
    tree = ast.parse(Path(loop_check.__file__).read_text(encoding="utf-8"))
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_ci_config_chain"
    )
    calls = [
        n.func.attr
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert "compile" in calls, "未真的做 py_compile"
    assert "parse" in calls, "未真的做 AST 解析"
    assert "CI_WORKFLOWS" in ast.unparse(fn)
    assert "CI_GUARD_TESTS" in ast.unparse(fn)


def test_ci_guard_is_wired_into_main(capsys, monkeypatch):
    """守卫必须接入 main() 输出与 RESULT 判定，不能只是孤立函数。

    P1-13 纪律：main() 的镜像项是外部 git 状态，与本用例职责无关，显式隔离。
    """
    monkeypatch.setattr(
        loop_check,
        "check_mirror",
        lambda: ([], "三方对齐（测试桩：隔离外部镜像状态）"),
    )
    rc = loop_check.main()
    out = capsys.readouterr().out
    assert "CI 配置闭合" in out, out
    assert "R12 附加守卫" in out, out
    assert rc == 0, out


def test_scope_is_well_formed_when_ledger_open_is_zero(capsys, monkeypatch):
    """R12 复核③：open 归零后不得再输出「≠ 0 个 open 历史失败已修」病句。

    仍保留派生出的数量串，兼容 test_loop_check_acceptance_chain.py 的断言。
    """
    if loop_check.ledger_open_count() != 0:
        pytest.skip("分诊台账 open 非 0，本断言只针对归零态")
    monkeypatch.setattr(
        loop_check,
        "check_mirror",
        lambda: ([], "三方对齐（测试桩：隔离外部镜像状态）"),
    )
    rc = loop_check.main()
    out = capsys.readouterr().out
    assert "0 个 open 历史失败" in out, out
    assert "0 个 open 历史失败已修" not in out, out
    assert "已全部收口" in out, out
    assert rc == 0, out
