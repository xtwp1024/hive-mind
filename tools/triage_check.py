#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hive-mind 历史失败分诊台账 — R4/P1-8 建立，R5/P1-9② / R7/P1-9③ / R11/P1-9① / R12/#18 续修 — 单一真源 + 可复现校验命令。

问题背景（R3 台账原话）：「全仓库 pytest -q 有 20 个 hive/tests/ 历史失败，
台账与 CI 若只跑 tests/ 子集，这 20 个失败会被静默忽略」。本文件把「20 个
失败」从一句会过期的自然语言，变成一份**结构化、可校验、自证范围**的台账：

  python tools/triage_check.py                     # 静态校验台账自身（秒级，无依赖）
  pytest hive/tests/ -q --tb=line > /tmp/full.txt 2>&1; \
      python tools/triage_check.py --report /tmp/full.txt   # 与真实失败集合对账

⚠ R4 实测口径修正：`pyproject.toml` 的 `testpaths = ["hive/tests"]` 让裸
`pytest -q` 只收集 475 项 hive/tests，**不是「全仓库」**；维护中的 `tests/`
子集（139 项，含 R1~R4 全部回归）必须显式指定路径才会被收集。CI
（.github/workflows/test.yml）跑的是 `pytest hive/tests/`，所以它看得到这
20 个失败、却漏掉 tests/ 子集。两套一起 collect 会因 `test_dna.py` 同名报
`import file mismatch`（无 __init__.py），故不能合并成一条命令。写报告数字时
必须带集合范围（R-20261010-19 第 17 次命中）。

三项检查（任一失败即退出码 1）：
  1. 台账完整性：行数 / nodeid 唯一 / 分类·去向·状态取值合法 / 分类分布自洽
  2. 收集闭合：每个 nodeid 都能在 hive/tests/ 对应文件里用 AST 找到真实
     类 + 方法（不 import、不需要第三方包）→ 测试被改名/删除时台账会**报错**，
     而不是悄悄过期
  3. 对账（仅 --report）：真实 pytest 的 FAILED 集合必须**恰好等于**台账里
     status=open 的集合。出现「新失败」或「已修但台账没更新」都判 FAIL。

SCOPE：本工具只做分诊台账校验，**不运行 pytest**（除非调用方传入 --report）、
**不决定 xfail**、**不修任何代码**。PASS 只代表「台账与当前失败集合一致」，
不代表这些失败已被修复，更不代表服务可用。
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS_ROOT = ROOT / "hive" / "tests"

CATEGORIES = {
    "真缺陷·已修",          # 产品代码缺陷，本轮已修
    "真缺陷·待行为决策",     # 产品代码缺陷，但修法改变对外行为，需下一轮判定
    "断言过期",             # 实现契约已变，测试断言旧契约
    "测试自身缺陷",          # 测试代码写错（helper 命名、fixture 等）
    "状态污染/顺序依赖",      # 单例/磁盘残留导致单跑 PASS、全量 FAIL
    "依赖缺失",             # 缺第三方包或配置文件（本项目实测 0 条）
}
DISPOSITIONS = {"已修(R4)", "已修(R5)", "已修(R7)", "已修(R11)", "已修(R12)", "立下一轮(P1-9)", "待人工"}
FIXED_DISP_PREFIX = "已修"   # status=fixed <=> 去向以此开头；台账支持多轮陆续修复
STATUSES = {"open", "fixed"}

# 唯一真源：改这里就是改台账。每行都是 2026-11-11 R4 全仓库 pytest -q 实测证据，
# 不是推测。evidence 记的是实测断言/异常，不是「应该是什么」。
# R5 续修：#11~#16 六条「测试自身缺陷」已修，evidence 末尾追加 R5 实测记录；
# R7 续修：#2/#9 两条「状态污染/顺序依赖」已修，evidence 末尾追加 R7 实测记录。
# R11 续修：P1-9① 的 10 条「断言过期」逐条核对产品当前契约后改断言（只改测试、
# 不改产品行为），evidence 末尾追加 R11 运行时实测记录；#18 仍留 open（需行为决策）。
# R12 续修：#18 经用户拍板后按 AudioEncoder 既有语义修复产品输入校验，evidence 末尾追加
# R12 运行时实测记录；台账 20 条至此全部 fixed，open 归零（对账口径 FAILED 0 == open 0）。
# 台账支持多轮陆续修复（status=fixed <=> 去向以「已修」开头），不写死某一轮。
LEDGER = [
    # id, nodeid, category, evidence, disposition, status
    (1, "hive/tests/test_cognitive_bee_enhanced.py::TestReportToEvolution::test_report_to_evolution_coordinator_import_error",
     "断言过期", ":341 assert True == False；hive.core.swarm_evolution 现已可导入，report_to_evolution_coordinator 返回 reported=True，测试前提「协调器不可用」不再成立；【R11 已修】实测 hive.core.swarm_evolution 可导入 ⇒ reported=True + strengths/weaknesses/overall_intelligence/timestamp；旧「coordinator_unavailable」需 patch.dict(sys.modules, {...: None}) 注入 ImportError 才成立，测试现同时覆盖默认与回退两路径",
     "已修(R11)", "fixed"),
    (2, "hive/tests/test_evolution_integration.py::TestSwarmCommunicationIntegration::test_register_bee",
     "状态污染/顺序依赖", "全量 AssertionError: 1 != 2；单独跑 19 failed/53 passed 时该用例 PASS。get_swarm_communication() 单例 registered_bees 跨用例累积。【R7 已修】setUp 快照-清空 registered_bees + addCleanup 复原，每个用例从空注册表出发；整文件 21 passed（修复前 1 failed/20 passed）；旧实现注入反证精确复现 1 != 2",
     "已修(R7)", "fixed"),
    (3, "hive/tests/test_ministers.py::TestEngineerBee::test_advise_method",
     "断言过期", "advise() 实返 {minister, analysis, recommendations, priority, risks, confidence, timestamp}；断言的 requirements_analysis/technical_debt 等旧契约不存在；【R11 已修】实测 advise() 由 BaseMinister 统一实现，返回 minister/analysis/recommendations/priority/risks/confidence/timestamp，并 assertNotIn 五个旧字段",
     "已修(R11)", "fixed"),
    (4, "hive/tests/test_ministers.py::TestEngineerBee::test_advise_with_technical_debt",
     "断言过期", ":350 KeyError: 'technical_debt'；同 #3，advise() 契约已变；【R11 已修】实测技术债务评估已移除，风险统一走 risks[{type,severity,description}]，测试改断言并显式 assertNotIn technical_debt",
     "已修(R11)", "fixed"),
    (5, "hive/tests/test_ministers.py::TestEngineerBee::test_execute_strategy_method",
     "断言过期", "execute_strategy() 实返 {status: success, total_tasks, completed_tasks, success_rate}；断言的 strategy_name/architecture 等旧契约不存在；【R11 已修】实测 execute_strategy() 返回 _aggregate_results 的 status/total_tasks/completed_tasks/success_rate；无 tasks 字段时 0 任务仍 success，旧字段全部 assertNotIn",
     "已修(R11)", "fixed"),
    (6, "hive/tests/test_ministers.py::TestEngineerBee::test_get_bee_status_bee_interface",
     "断言过期", ":443 KeyError: 'species'；get_bee_status() 返回结构已变（get_statistics() 才有 species=engineer、minister_type=EngineerBee）；【R11 已修】实测 EngineerBee 重写 get_bee_status() 返回 id/name/active/mutation_pool_size，旧 species/role/engineering_db_size 已移除",
     "已修(R11)", "fixed"),
    (7, "hive/tests/test_ministers.py::TestEngineerBee::test_get_statistics_bee_interface",
     "断言过期", "实返含 total_assessments/total_strategies 等；断言的 projects_analyzed/architectures_designed 不存在；【R11 已修】实测 get_statistics() = get_minister_status() + history_cache_size/total_assessments/total_strategies，含 minister_type=EngineerBee，旧 projects_analyzed/architectures_designed 已移除",
     "已修(R11)", "fixed"),
    (8, "hive/tests/test_ministers.py::TestEngineerBee::test_specialized_task",
     "断言过期", "实返 {status: delegated, message: Ministers delegate tasks to Workers}；断言的 project_path/architecture_type 属旧实现；【R11 已修】实测 specialized_task() 返回 {status: delegated, message: Ministers delegate tasks to Workers}，测试改断言并对六个旧字段 assertNotIn",
     "已修(R11)", "fixed"),
    (9, "hive/tests/test_phase3_evolution.py::TestCapabilityGapAnalyzer::test_get_latest_report",
     "状态污染/顺序依赖", "断言 assertIsNone(report) 但返回真实 GapAnalysisReport；get_latest_report() glob hive/data/gap_analyzer/report_*.json，该目录已有 7+ 历史落盘产物（被 .gitignore 的 Data/ 覆盖）。【R7 已修】setUp 把 _persistence_path 重定向到独立空临时目录 + addCleanup 回收，既不读共享残留也不再写新残留；整类 8 passed（修复前该用例 FAIL）；旧实现注入反证在临时残留目录下精确复现 is not None",
     "已修(R7)", "fixed"),
    (10, "hive/tests/test_phase3_evolution.py::TestSpeciesBlueprint::test_crossover",
     "真缺陷·已修", "R4 前：species_blueprint.py:545 UnboundLocalError: cannot access local variable 'random'；crossover() 内 3 处函数级 import random 遮蔽模块级导入（第 24 行），双亲能力互斥时 elif 分支访问未绑定局部名",
     "已修(R4)", "fixed"),
    (11, "hive/tests/test_phase3_evolution.py::TestDNAAdaptiveEngine::test_monitor_dna",
     "测试自身缺陷", ":781 AttributeError: create_mock_dna；helper 被命名为 test_create_mock_dna（自身被 pytest 当用例收集且 return MagicMock），调用方却写 self.create_mock_dna()。【R5 已修】helper 改名为 create_mock_dna（去掉 test_ 前缀），6 个调用点无需改动；修复前 6 failed 实测复现，修复后 6 passed",
     "已修(R5)", "fixed"),
    (12, "hive/tests/test_phase3_evolution.py::TestDNAAdaptiveEngine::test_monitor_duplicate_dna",
     "测试自身缺陷", ":791 AttributeError: create_mock_dna；同 #11。【R5 已修】随 #11 一并修复，实测 PASS", "已修(R5)", "fixed"),
    (13, "hive/tests/test_phase3_evolution.py::TestDNAAdaptiveEngine::test_stop_monitoring",
     "测试自身缺陷", ":797 AttributeError: create_mock_dna；同 #11。【R5 已修】随 #11 一并修复，实测 PASS", "已修(R5)", "fixed"),
    (14, "hive/tests/test_phase3_evolution.py::TestDNAAdaptiveEngine::test_record_performance",
     "测试自身缺陷", ":808 AttributeError: create_mock_dna；同 #11。【R5 已修】随 #11 一并修复，实测 PASS", "已修(R5)", "fixed"),
    (15, "hive/tests/test_phase3_evolution.py::TestDNAAdaptiveEngine::test_suggest_adaptations_insufficient_data",
     "测试自身缺陷", ":834 AttributeError: create_mock_dna；同 #11。【R5 已修】随 #11 一并修复，实测 PASS", "已修(R5)", "fixed"),
    (16, "hive/tests/test_phase3_evolution.py::TestDNAAdaptiveEngine::test_get_status_single",
     "测试自身缺陷", ":855 AttributeError: create_mock_dna；同 #11。【R5 已修】随 #11 一并修复，实测 PASS", "已修(R5)", "fixed"),
    (17, "hive/tests/test_phase3_evolution.py::TestParameterState::test_record_value",
     "断言过期", "断言 len(historical_values)==3、len(historical_gradients)==2；record_value() 只 append 传入值、不播种 initial value，两次调用后实测 2 != 3；【R11 已修】实测 record_value() 只 append 传入值、不播种 current_value；两次调用后 historical_values=[0.2,0.3]、gradients 长度 1（0.1）、current_value=0.3",
     "已修(R11)", "fixed"),
    (18, "hive/tests/test_phase4_multimodal.py::TestImageEncoder::test_nonexistent_image",
     "真缺陷·已修", "断言 assertFalse(result.success) 但为 True；encode() 用 Path(x).exists() 判定路径，不存在的路径字符串落入 else 分支被当作 base64 原样处理 → 静默成功。修法需引入输入校验、改变对外行为，本轮不擅改；【R12 已修】经用户拍板，新增 _is_path_like_image_input() 显式区分「路径」与「base64 数据」语义，路径语义输入不存在时对齐 AudioEncoder 既有行为返回 success=False + 'Image file not found: ...'；base64/data URI/短文本输入保持原有数据语义，不误伤。新增 22 项回归（判定表 / 真调 / 旧判定注入反证 / 字节级 SAME / AST 守卫）",
     "已修(R12)", "fixed"),
    (19, "hive/tests/test_phase4_multimodal.py::TestTaskTypeClassifier::test_classify_empty",
     "断言过期", "断言 TaskType.QUESTION_ANSWERING，实测 UNKNOWN；multimodal_task_decoder.py:229 对空 prompt/空 modalities 显式返回 (UNKNOWN, 0.0)，属当前正确行为；【R11 已修】实测 classify("") 显式返回 (TaskType.UNKNOWN, 0.0)，属当前正确行为，断言与 confidence 一并更新",
     "已修(R11)", "fixed"),
    (20, "hive/tests/test_swarm_communication_enhanced.py::TestRequestResponsePattern::test_handle_response_matches_correlation",
     "断言过期", ":386 断言 test_corr_123 已不在 pending_responses；_handle_response() 只写回 (event, message) 并 set event，pop 由 send_and_wait() 在 :544 负责。清理职责已迁移，测试仍按旧语义断言；【R11 已修】实测 _handle_response() 只写回 (event, message) 并 set event，pop 由 send_and_wait() 负责；测试改为断言 pending 中存回同一 event 与同一 message",
     "已修(R11)", "fixed"),
]

EXPECTED_TOTAL = 20          # R4 实测基线；变更必须是有意的台账更新，不能静默漂移
EXPECTED_OPEN = 0            # 20 - 20（#10 R4 1 条 + #11~#16 R5 6 条 + #2/#9 R7 2 条 + P1-9① 10 条 R11 + #18 R12 1 条）；open 归零
EXPECTED_MISSING_DEPS = 0    # R4 实测：20 条里没有一条是缺依赖/缺配置


def _nodeid_parts(nodeid: str):
    parts = nodeid.replace("\\", "/").split("::")
    return parts


def check_ledger() -> list[str]:
    problems: list[str] = []
    if len(LEDGER) != EXPECTED_TOTAL:
        problems.append(f"台账行数 {len(LEDGER)} != 基线 {EXPECTED_TOTAL}（有意增删请同步改 EXPECTED_TOTAL 并写明理由）")
    ids = [r[0] for r in LEDGER]
    if ids != list(range(1, len(LEDGER) + 1)):
        problems.append(f"id 不是 1..{len(LEDGER)} 连续序列: {ids}")
    nodeids = [r[1] for r in LEDGER]
    dup = [n for n, c in Counter(nodeids).items() if c > 1]
    if dup:
        problems.append(f"nodeid 重复: {dup}")
    for rid, nodeid, cat, ev, disp, status in LEDGER:
        if cat not in CATEGORIES:
            problems.append(f"#{rid} 非法分类 {cat!r}")
        if disp not in DISPOSITIONS:
            problems.append(f"#{rid} 非法去向 {disp!r}")
        if status not in STATUSES:
            problems.append(f"#{rid} 非法状态 {status!r}")
        if not ev.strip():
            problems.append(f"#{rid} 缺实测证据")
        # 台账支持多轮陆续修复（R4 修 1 条、R5 修 6 条）：用「已修」前缀判定，
        # 不写死某一轮的 disposition 字面量，否则每修一轮都要改校验逻辑。
        is_fixed_disp = disp.startswith(FIXED_DISP_PREFIX)
        if (cat == "真缺陷·已修") and status != "fixed":
            problems.append(f"#{rid} 分类为真缺陷·已修却未标 fixed: {status}")
        if (status == "fixed") != is_fixed_disp:
            problems.append(f"#{rid} 状态与去向不自洽: {status} / {disp}")
    opens = sum(1 for r in LEDGER if r[5] == "open")
    if opens != EXPECTED_OPEN:
        problems.append(f"open 数 {opens} != {EXPECTED_OPEN}")
    missing_deps = sum(1 for r in LEDGER if r[2] == "依赖缺失")
    if missing_deps != EXPECTED_MISSING_DEPS:
        problems.append(f"依赖缺失条数 {missing_deps} != 实测 {EXPECTED_MISSING_DEPS}")
    return problems


def check_collection() -> list[str]:
    problems: list[str] = []
    cache: dict[Path, ast.Module] = {}
    for rid, nodeid, *_ in LEDGER:
        parts = _nodeid_parts(nodeid)
        if len(parts) != 3:
            problems.append(f"#{rid} nodeid 非 路径::类::方法 三段式: {nodeid}")
            continue
        rel, cls_name, meth_name = parts
        path = ROOT / rel
        if not path.is_file():
            problems.append(f"#{rid} 测试文件不存在: {rel}")
            continue
        if path not in cache:
            try:
                cache[path] = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError as exc:
                problems.append(f"#{rid} 解析失败 {rel}: {exc}")
                continue
        tree = cache[path]
        cls = next((n for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
        if cls is None:
            problems.append(f"#{rid} 找不到类 {cls_name} @ {rel}")
            continue
        meth = next((n for n in cls.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == meth_name), None)
        if meth is None:
            problems.append(f"#{rid} 找不到方法 {cls_name}::{meth_name} @ {rel}")
    return problems


def parse_failed(report: Path) -> set[str]:
    failed: set[str] = set()
    for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("FAILED "):
            nodeid = line.split()[1].replace("\\", "/")
            failed.add(nodeid)
    return failed


def main(argv: list[str]) -> int:
    report: Path | None = None
    if argv:
        if len(argv) == 2 and argv[0] == "--report":
            report = Path(argv[1])
            if not report.is_file():
                print(f"[对账] 报告文件不存在: {report}")
                return 2
        else:
            print("用法: python tools/triage_check.py [--report <pytest输出文件>]")
            return 2

    print("=" * 68)
    print("hive-mind 历史失败分诊台账校验 (R4 建立 / R5 / R7 / R11 / R12 续修)")
    print("=" * 68)

    print("[1/3] 台账完整性")
    p1 = check_ledger()
    dist = Counter(r[2] for r in LEDGER)
    for cat, n in sorted(dist.items()):
        print(f"      {cat:<18} {n:>2} 条")
    print(f"      合计 {len(LEDGER)} 条 / open {sum(1 for r in LEDGER if r[5]=='open')} / fixed {sum(1 for r in LEDGER if r[5]=='fixed')}")
    print(f"      问题 {len(p1)}")
    for x in p1:
        print(f"        ! {x}")

    print("[2/3] 收集闭合（AST，不 import、不需要第三方包）")
    p2 = check_collection()
    print(f"      已解析 {len(LEDGER)} 个 nodeid，问题 {len(p2)}")
    for x in p2:
        print(f"        ! {x}")

    print("[3/3] 与真实 pytest 失败集合对账")
    p3: list[str] = []
    if report is None:
        print("      未传 --report：跳过（仅静态校验）。须由调用方跑全仓库 pytest 后传报告")
    else:
        actual = parse_failed(report)
        expected_open = {r[1].replace("\\", "/") for r in LEDGER if r[5] == "open"}
        expected_fixed = {r[1].replace("\\", "/") for r in LEDGER if r[5] == "fixed"}
        new_failures = sorted(actual - expected_open - expected_fixed)
        unresolved = sorted(expected_open - actual)
        regressed = sorted(expected_fixed & actual)
        print(f"      报告 {report}：FAILED {len(actual)} 条；台账 open {len(expected_open)} 条")
        if new_failures:
            p3.append(f"出现台账外的新失败 {len(new_failures)} 条: {new_failures[:5]}")
        if unresolved:
            p3.append(f"台账 open 但本次未失败 {len(unresolved)} 条（可能已修或未被收集，需核）: {unresolved[:5]}")
        if regressed:
            p3.append(f"标记 fixed 却再次失败 {len(regressed)} 条: {regressed[:5]}")
        if not p3:
            print("      对账一致：无新失败、无静默修复、无回归")
        for x in p3:
            print(f"        ! {x}")

    print("-" * 68)
    print("SCOPE: 仅校验分诊台账自身 + 与 pytest FAILED 集合对账；不运行 pytest、"
          "不决定 xfail、不修代码。PASS ≠ 失败已修 ≠ 服务可用。")
    print("口径提示(R4 实测): pyproject testpaths=['hive/tests']，故裸 pytest -q 只收集 "
          "475 项 hive/tests，不是「全仓库」；CI(.github/workflows/test.yml) 亦跑 hive/tests/，"
          "能看到这 20 个失败，反而漏掉维护中的 tests/ 子集(139 项)。两集合因 "
          "test_dna.py 同名无法一起 collect(import file mismatch)。故本命令须配 --report 使用，"
          "且报告口径须写明是哪一个集合。")
    total = len(p1) + len(p2) + len(p3)
    print(f"RESULT: {'PASS' if total == 0 else 'FAIL'} (问题 {total})")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
