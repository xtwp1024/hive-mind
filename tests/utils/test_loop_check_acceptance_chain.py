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


def test_acceptance_chain_is_wired_into_main(capsys, monkeypatch):
    """守卫必须接入 main() 的输出与 RESULT 判定，不能只是孤立函数。

    R7 口径修订：SCOPE 行里的 open 数量改为从分诊台账 EXPECTED_OPEN 静态派生，
    不再写死「13」。本断言随台账走，避免每修一个历史失败就误报或变成假陈述。

    R10 / P1-13 状态依赖隔离：main() 的 [3/3] 镜像一致要求「镜像 HEAD ==
    origin/master」，这是本用例职责之外的外部 git 状态。R9 收尾实测「已推送
    5e57aef 但镜像仍 cd19c4a」的窗口会让 main() 返回 1，本用例与 SCOPE 投影
    无关却连带判红（2 failed / 167 passed）。镜像一致性由
    tests/utils/test_loop_check_mirror.py 专门负责；此处 monkeypatch 固定为
    「无问题」态，使 rc 只反映语法 / import 闭合 / 验收链自证三个本地可控项。
    """
    monkeypatch.setattr(
        loop_check, "check_mirror",
        lambda: ([], "三方对齐（测试桩：隔离外部镜像状态）"),
    )
    rc = loop_check.main()
    out = capsys.readouterr().out
    assert "验收链闭合" in out
    n_open = loop_check.ledger_open_count()
    assert n_open is not None, "分诊台账 EXPECTED_OPEN 未能静态派生"
    assert f"{n_open} 个 open 历史失败" in out, out
    assert rc == 0, out


def test_ledger_open_count_is_derived_not_hardcoded(tmp_path):
    """反证（R7）：把临时台账的 EXPECTED_OPEN 换一个值，派生结果必须跟着变。

    用临时目录注入 EXPECTED_OPEN = 7 / 3，若函数内部写死常量就会露出原形；
    全程不改写真实工作区（沿用 R-20261010-11「反证不得破坏工作区」纪律）。
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    stub = tools / "triage_check.py"
    stub.write_text("EXPECTED_OPEN = 7\n", encoding="utf-8")
    assert loop_check.ledger_open_count(root=tmp_path) == 7
    stub.write_text("EXPECTED_OPEN = 3\n", encoding="utf-8")
    assert loop_check.ledger_open_count(root=tmp_path) == 3
    stub.write_text("# 台账缺 EXPECTED_OPEN\n", encoding="utf-8")
    assert loop_check.ledger_open_count(root=tmp_path) is None
    assert (Path(loop_check.ROOT) / loop_check.TRIAGE_TOOL).is_file()
