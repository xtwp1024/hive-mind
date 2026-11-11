# -*- coding: utf-8 -*-
"""R6 元审视修订的回归：验收链附加守卫必须能区分「自洽 / 失效态」。

反证形态：不改写真实工作区，在临时目录构造「验收工具缺失」与「分诊台账
自证 FAIL」两种失效态，断言守卫判红；再用真实 ROOT 断言当前自洽。
这与 R1~R5 的「旧实现注入反证」同族，但注入对象是验收对象的失效态，
不是产品代码旧实现——元审视轮不碰功能代码。
"""
from __future__ import annotations

from pathlib import Path

from tools import loop_check


def _make_chain(tmp_path: Path, *, with_smoke: bool = True, triage_result: str = "PASS"):
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    if with_smoke:
        (tools / "smoke_runtime.py").write_text("# stub\n", encoding="utf-8")
    rc = 0 if triage_result == "PASS" else 1
    (tools / "triage_check.py").write_text(
        "import sys\n"
        f"print('RESULT: {triage_result}')\n"
        f"sys.exit({rc})\n",
        encoding="utf-8",
    )


def test_real_repo_chain_is_self_consistent():
    """真实工作区：两个工具在位、分诊台账静态自证 PASS。"""
    issues, status = loop_check.check_acceptance_chain()
    assert issues == [], issues
    assert "分诊台账自证" in status


def test_missing_tool_is_detected(tmp_path):
    """失效态①：声明的验收工具消失 ⇒ 守卫报错，不是静默少跑一层。"""
    _make_chain(tmp_path, with_smoke=False)
    issues, _ = loop_check.check_acceptance_chain(root=tmp_path)
    assert any("验收工具缺失" in x for x in issues), issues


def test_failing_ledger_selfcheck_is_detected(tmp_path):
    """失效态②（反证）：分诊台账自证 FAIL ⇒ 附加守卫必须判红。

    量化对照：同一函数对 PASS 的临时链返回 0 问题，对 FAIL 的临时链返回
    ≥1 问题 ⇒ 守卫不是恒绿的装饰性检查。
    """
    _make_chain(tmp_path, triage_result="PASS")
    ok_issues, _ = loop_check.check_acceptance_chain(root=tmp_path)
    assert ok_issues == [], ok_issues

    _make_chain(tmp_path, triage_result="FAIL")
    bad_issues, _ = loop_check.check_acceptance_chain(root=tmp_path)
    assert any("分诊台账静态自证未 PASS" in x for x in bad_issues), bad_issues


def test_acceptance_chain_is_wired_into_main(capsys):
    """守卫必须接入 main() 的输出与 RESULT 判定，不能只是孤立函数。"""
    rc = loop_check.main()
    out = capsys.readouterr().out
    assert "验收链闭合" in out
    assert "13 个 open" in out
    assert rc == 0, out
