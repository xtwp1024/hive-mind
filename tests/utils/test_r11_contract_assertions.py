# -*- coding: utf-8 -*-
"""R11 · P1-9①：10 条「断言过期」按当前契约修复后的回归守卫。

R4 分诊把这 10 条判为「实现契约已变、测试仍按旧语义断言」，R11 逐条核对
产品代码后只改测试断言，不改产品行为。本文件提供四类守卫：

  1. 投影守卫：10 个测试方法体内必须出现新契约字面量、不得残留旧契约字面量；
  2. 台账守卫：tools/triage_check.py 的 LEDGER 必须把这 10 条标记为「已修(R11)」，
     且唯一 open 仍是 #18（ImageEncoder 行为决策，需用户拍板）；
  3. 运行时真实调用：真的 import 产品模块、真的调一遍，断言当前契约仍成立；
  4. 旧实现注入反证：把修复前的旧断言注回（内存中，不改写工作区），
     在当前产品上必须精确失败；注入前后 5 个真实源码字节级 sha256 SAME。

SCOPE：只守护这 10 条断言与分诊台账的一致性；不跑服务、不判 #18 该怎么修、
不证明项目可用。
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import multiprocessing
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = ROOT / "tools" / "triage_check.py"

# id -> (测试文件, 类, 方法, 新契约字面量, 已移除的旧契约字面量)
CONTRACTS = {
    1: ("hive/tests/test_cognitive_bee_enhanced.py", "TestReportToEvolution",
        "test_report_to_evolution_coordinator_import_error",
        ('result["reported"] is True', '"hive.core.swarm_evolution"', '"coordinator_unavailable"'),
        ('result["reported"] == False',)),
    3: ("hive/tests/test_ministers.py", "TestEngineerBee", "test_advise_method",
        ('"analysis"', '"recommendations"', '"risks"', '"priority"'),
        ('self.assertIn("requirements_analysis", advice)',
         'self.assertIn("architecture_assessment", advice)',
         'self.assertIn("feasibility", advice)')),
    4: ("hive/tests/test_ministers.py", "TestEngineerBee", "test_advise_with_technical_debt",
        ('self.assertNotIn("technical_debt", advice)', '"risks"'),
        ('self.assertGreater(len(advice["technical_debt"]), 0)', '"poor_modularity"')),
    5: ("hive/tests/test_ministers.py", "TestEngineerBee", "test_execute_strategy_method",
        ('"total_tasks"', '"completed_tasks"', '"success_rate"'),
        ('self.assertEqual(result["strategy_name"], "architecture_refactor")',
         'self.assertIn("implementation_plan", result)',
         'self.assertIn("quality_metrics", result)')),
    6: ("hive/tests/test_ministers.py", "TestEngineerBee", "test_get_bee_status_bee_interface",
        ('"mutation_pool_size"', '"active"'),
        ('status["species"]', 'status["role"]',
         'self.assertIn("engineering_db_size", status)')),
    7: ("hive/tests/test_ministers.py", "TestEngineerBee", "test_get_statistics_bee_interface",
        ('"history_cache_size"', '"minister_type"', '"decisions_made"'),
        ('self.assertIn("projects_analyzed", stats)',
         'self.assertIn("architectures_designed", stats)')),
    8: ("hive/tests/test_ministers.py", "TestEngineerBee", "test_specialized_task",
        ('"delegated"', '"Ministers delegate tasks to Workers"'),
        ('self.assertIn("project_path", analysis)',
         'self.assertIn("architecture_type", analysis)',
         'self.assertIn("quality_assessment", analysis)')),
    17: ("hive/tests/test_phase3_evolution.py", "TestParameterState", "test_record_value",
         ('[0.2, 0.3]', 'self.assertAlmostEqual', 'state.current_value'),
         ('self.assertEqual(len(state.historical_values), 3)',
          'self.assertEqual(len(state.historical_gradients), 2)')),
    19: ("hive/tests/test_phase4_multimodal.py", "TestTaskTypeClassifier", "test_classify_empty",
         ("self.TaskType.UNKNOWN", "confidence, 0.0"),
         ("self.assertEqual(task_type, self.TaskType.QUESTION_ANSWERING)",)),
    20: ("hive/tests/test_swarm_communication_enhanced.py", "TestRequestResponsePattern",
         "test_handle_response_matches_correlation",
         ("test_corr_id in comm.pending_responses", "stored_msg is response_msg"),
         ("assert test_corr_id not in comm.pending_responses",)),
}

FIXED_IDS = tuple(CONTRACTS)
BEHAVIOR_DECISION_ID = 18
SOURCE_FILES = tuple(sorted({c[0] for c in CONTRACTS.values()}))


def _method_source(rel: str, cls_name: str, meth_name: str) -> str:
    """用 AST 精确定位方法体源码，避免整文件字符串匹配带来的假阳性。"""
    path = ROOT / rel
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    assert cls is not None, f"找不到类 {cls_name} @ {rel}"
    meth = next((n for n in cls.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == meth_name), None)
    assert meth is not None, f"找不到方法 {cls_name}::{meth_name} @ {rel}"
    lines = src.splitlines()
    return "\n".join(lines[meth.lineno - 1:meth.end_lineno])


def _load_triage():
    spec = importlib.util.spec_from_file_location("r11_triage_check", LEDGER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _runtime_values() -> dict:
    """真的 import 产品模块、真的调一遍，拿到 10 条断言各自面对的当前返回值。"""
    from hive.core.dna import BeeDNA
    from hive.bees.cognitive_bee import CognitiveBee
    from hive.ministers import EngineerBee
    from hive.core.dna_adaptive import ParameterState
    from hive.core.multimodal_task_decoder import TaskTypeClassifier, TaskType
    from hive.core.swarm_communication import SwarmCommunication, SwarmMessage, MessageType

    with patch("hive.bees.base_bee.get_logger"):
        bee = CognitiveBee(BeeDNA(species="CognitiveBee", id="r11_cog"), "r11_cog", MagicMock())
        report = bee.report_to_evolution_coordinator()

    eng = EngineerBee(BeeDNA(species="engineer"), "r11_engineer", multiprocessing.Queue())
    advice = eng.advise({"system_requirements": ["r1"]})
    outcome = eng.execute_strategy({"name": "architecture_refactor", "objectives": ["a"]})
    delegated = eng.specialized_task(ROOT)
    status = eng.get_bee_status()
    stats = eng.get_statistics()

    state = ParameterState(name="learning_rate", current_value=0.1)
    state.record_value(0.2)
    state.record_value(0.3)
    state_view = SimpleNamespace(
        historical_values=list(state.historical_values),
        historical_gradients=list(state.historical_gradients),
        current_value=state.current_value,
    )

    task_type, confidence = TaskTypeClassifier().classify("")
    classify_view = SimpleNamespace(task_type=task_type, confidence=confidence,
                                    TaskType=TaskType)

    comm = SwarmCommunication()
    event = threading.Event()
    corr = "test_corr_123"
    with comm._pending_lock:
        comm.pending_responses[corr] = (event, None)
    response_msg = SwarmMessage(id="resp_r11", type=MessageType.RESPONSE,
                                from_bee="responder", to_bee="requester",
                                content={"result": "success"}, correlation_id=corr)
    comm._handle_response(response_msg)
    with comm._pending_lock:
        pending = dict(comm.pending_responses)
    comm_view = SimpleNamespace(pending=pending, event=event,
                                corr=corr, response_msg=response_msg)

    return {
        1: report, 3: advice, 4: advice, 5: outcome, 6: status, 7: stats,
        8: delegated, 17: state_view, 19: classify_view, 20: comm_view,
    }


def _new_checks(rid: int, v):
    """当前契约断言（与修复后的测试同语义）。"""
    if rid == 1:
        assert v["reported"] is True
        assert "strengths" in v and "weaknesses" in v
    elif rid == 3:
        assert v["minister"] == "EngineerBee"
        assert isinstance(v["analysis"], dict) and isinstance(v["recommendations"], list)
        assert "risks" in v and "priority" in v
    elif rid == 4:
        assert "technical_debt" not in v and v["risks"]
    elif rid == 5:
        assert v["status"] == "success" and v["total_tasks"] == 0
        assert "completed_tasks" in v and "success_rate" in v
    elif rid == 6:
        assert v["id"] == "r11_engineer" and "mutation_pool_size" in v and "active" in v
    elif rid == 7:
        assert v["minister_type"] == "EngineerBee"
        assert "history_cache_size" in v and "total_assessments" in v
    elif rid == 8:
        assert v["status"] == "delegated"
        assert v["message"] == "Ministers delegate tasks to Workers"
    elif rid == 17:
        assert v.historical_values == [0.2, 0.3]
        assert len(v.historical_gradients) == 1 and abs(v.historical_gradients[0] - 0.1) < 1e-9
        assert v.current_value == 0.3
    elif rid == 19:
        assert v.task_type == v.TaskType.UNKNOWN and v.confidence == 0.0
    elif rid == 20:
        assert v.corr in v.pending and v.event.is_set()
        assert v.pending[v.corr][1] is v.response_msg
    else:  # pragma: no cover - 守卫自身写错时立刻暴露
        raise AssertionError(f"未知 id {rid}")


def _old_checks(rid: int, v):
    """修复前的旧断言（内存注入，不写回任何源码文件）。"""
    if rid == 1:
        assert v["reported"] == False
        assert v["reason"] == "coordinator_unavailable"
    elif rid == 3:
        assert "requirements_analysis" in v and "technical_debt" in v
    elif rid == 4:
        assert len(v["technical_debt"]) > 0
    elif rid == 5:
        assert v["strategy_name"] == "architecture_refactor" and "architecture" in v
    elif rid == 6:
        assert v["species"] == "EngineerBee"
    elif rid == 7:
        assert "projects_analyzed" in v and "architectures_designed" in v
    elif rid == 8:
        assert v["project_path"] and v["architecture_type"] == "modular"
    elif rid == 17:
        assert len(v.historical_values) == 3 and len(v.historical_gradients) == 2
    elif rid == 19:
        assert v.task_type == v.TaskType.QUESTION_ANSWERING
    elif rid == 20:
        assert v.corr not in v.pending and v.event.is_set()
    else:  # pragma: no cover
        raise AssertionError(f"未知 id {rid}")


@pytest.mark.parametrize("rid", FIXED_IDS)
def test_method_body_is_projected_to_current_contract(rid):
    """10 个测试方法体必须含新契约字面量、且不得残留旧契约字面量。"""
    rel, cls, meth, new_tokens, old_tokens = CONTRACTS[rid]
    body = _method_source(rel, cls, meth)
    for token in new_tokens:
        assert token in body, f"#{rid} 缺新契约字面量 {token!r}"
    for token in old_tokens:
        assert token not in body, f"#{rid} 残留旧契约字面量 {token!r}"


def test_triage_ledger_marks_ten_fixed_and_open_cleared_in_r12():
    """分诊台账是唯一真源：R11 的 10 条仍 fixed；R12 修完 #18 后 open 归零。"""
    triage = _load_triage()
    rows = {r[0]: r for r in triage.LEDGER}
    assert "已修(R11)" in triage.DISPOSITIONS
    assert "已修(R12)" in triage.DISPOSITIONS
    for rid in FIXED_IDS:
        assert rows[rid][5] == "fixed", f"#{rid} 未标 fixed"
        assert rows[rid][4] == "已修(R11)", f"#{rid} 去向不是已修(R11)"
    # R11 时 #18 是唯一 open（需行为决策）；R12 经用户拍板后转为 已修(R12)
    assert rows[BEHAVIOR_DECISION_ID][5] == "fixed"
    assert rows[BEHAVIOR_DECISION_ID][4] == "已修(R12)"
    open_ids = sorted(r[0] for r in triage.LEDGER if r[5] == "open")
    assert open_ids == [], open_ids
    assert triage.EXPECTED_OPEN == len(open_ids) == 0
    assert triage.check_ledger() == [], "台账静态自证不应有问题"


def test_runtime_contracts_still_hold():
    """修完必做运行时真实调用：真的调产品代码，当前契约仍成立。"""
    values = _runtime_values()
    for rid in FIXED_IDS:
        _new_checks(rid, values[rid])


def test_old_assertions_fail_and_real_files_are_untouched():
    """旧实现注入反证：旧断言在当前产品上必须精确失败，且真实源码字节级不变。"""
    before = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
              for rel in SOURCE_FILES}
    values = _runtime_values()
    for rid in FIXED_IDS:
        with pytest.raises((AssertionError, KeyError)):
            _old_checks(rid, values[rid])
    after = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
             for rel in SOURCE_FILES}
    assert after == before, "反证不得改写真实源码"
    for rel in SOURCE_FILES:
        raw = (ROOT / rel).read_bytes()
        assert raw.count(b"\r\n") == raw.count(b"\n") > 0, f"{rel} 不是 100% CRLF"
