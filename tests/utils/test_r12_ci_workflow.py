"""R12 回归：CI 配置的三项修复必须可解析、可复现、且不能被悄悄回退。

- P1-14：``test.yml`` 的 Python 3.9 在 ubuntu-latest（Ubuntu 26.04）不可安装，
  故 3.9 固定到 ``ubuntu-22.04``，其余版本保持 latest，四版本覆盖不缩水。
- P1-15：``lint.yml`` 的 black/isort/ruff 升级为「失败即阻塞」的真门禁
  （去掉 continue-on-error），mypy/bandit/vulture 明确为报告型；
  ``pyproject.toml`` 钉死 lint 工具版本并显式声明 ruff 规则集，
  消除「本机与 CI 规则集漂移」和「step 显示绿、job 显示红」两种误导。

所有反证只写 tmp_path 影子文件，末尾校验真实文件字节级 SAME。
"""
from __future__ import annotations

import ast
import hashlib
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WF_DIR = ROOT / ".github" / "workflows"
TEST_YML = WF_DIR / "test.yml"
LINT_YML = WF_DIR / "lint.yml"
PYPROJECT = ROOT / "pyproject.toml"
DOCS_SCRIPT = ROOT / "hive" / "scripts" / "generate_api_docs.py"

EXPECTED_RUNNERS = {
    "3.9": "ubuntu-22.04",
    "3.10": "ubuntu-latest",
    "3.11": "ubuntu-latest",
    "3.12": "ubuntu-latest",
}
GATE_IDS = ("black", "isort", "ruff")
REPORT_IDS = ("mypy", "bandit", "vulture")


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _matrix_is_pinned(data) -> bool:
    job = data["jobs"]["test"]
    include = job["strategy"]["matrix"].get("include")
    if job.get("runs-on") != "${{ matrix.runner }}" or not include:
        return False
    try:
        runners = {str(item["python-version"]): item["runner"] for item in include}
    except (KeyError, TypeError):
        return False
    return runners == EXPECTED_RUNNERS


def _gates_are_real(data) -> bool:
    steps = data["jobs"]["lint"]["steps"]
    by_id = {s.get("id"): s for s in steps}
    if any(by_id[k].get("continue-on-error", False) for k in GATE_IDS):
        return False
    if not all(by_id[k].get("continue-on-error") is True for k in REPORT_IDS):
        return False
    summary = next((s for s in steps if s.get("name") == "Summary"), None)
    if not summary or summary.get("if") != "always()":
        return False
    run = summary.get("run", "")
    return all(f"steps.{k}.outcome" in run for k in GATE_IDS)


def test_p1_14_matrix_pins_only_python_39():
    data = _load(TEST_YML)
    assert _matrix_is_pinned(data)
    assert data["jobs"]["test"]["strategy"]["fail-fast"] is False
    assert "python-version" not in data["jobs"]["test"]["strategy"]["matrix"], (
        "matrix 若仍保留裸 python-version 列表，3.9 会落回 ubuntu-latest"
    )


def test_p1_14_legacy_matrix_injection_is_detected(tmp_path):
    """旧实现注入反证：把 include 换回裸列表，守卫必须判失败。"""
    before = hashlib.sha256(TEST_YML.read_bytes()).hexdigest()
    text = TEST_YML.read_text(encoding="utf-8")
    assert "runner: ubuntu-22.04" in text
    legacy = text.replace("runs-on: ${{ matrix.runner }}", "runs-on: ubuntu-latest", 1)
    # 用同结构的旧 matrix 覆盖 include 段（只改影子文件）
    start = legacy.index("        include:")
    end = legacy.index("    steps:", start)
    legacy = legacy[:start] + "        python-version: ['3.9', '3.10', '3.11', '3.12']\n\n" + legacy[end:]
    shadow = tmp_path / "test.yml"
    shadow.write_text(legacy, encoding="utf-8")
    assert _matrix_is_pinned(_load(shadow)) is False
    assert hashlib.sha256(TEST_YML.read_bytes()).hexdigest() == before


def test_p1_15_lint_gates_are_real_and_reports_are_explicit():
    data = _load(LINT_YML)
    assert _gates_are_real(data)
    by_id = {s.get("id"): s for s in data["jobs"]["lint"]["steps"]}
    assert "ruff check hive/" in by_id["ruff"]["run"]
    assert "pip install ruff" not in by_id["ruff"]["run"], "ruff 版本必须由 dev extras 钉死"


def test_p1_15_continue_on_error_injection_is_detected(tmp_path):
    """旧实现注入反证：给 black 注回 continue-on-error，守卫必须判失败。"""
    before = hashlib.sha256(LINT_YML.read_bytes()).hexdigest()
    text = LINT_YML.read_text(encoding="utf-8")
    anchor = "        black --check hive/\n"
    assert text.count(anchor) == 1
    broken = text.replace(anchor, anchor + "      continue-on-error: true\n", 1)
    shadow = tmp_path / "lint.yml"
    shadow.write_text(broken, encoding="utf-8")
    assert _gates_are_real(_load(shadow)) is False
    assert hashlib.sha256(LINT_YML.read_bytes()).hexdigest() == before


def test_p1_15_lint_tool_versions_and_rule_set_are_pinned():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    lint = data["project"]["optional-dependencies"]["lint"]
    for pin in ("black==24.10.0", "isort==9.0.2", "mypy==2.4.1", "ruff==0.17.4"):
        assert pin in lint, f"{pin} 未钉死，CI 会随最新版漂移"
    assert data["tool"]["ruff"]["lint"]["select"] == ["E9", "F63", "F7", "F82"]
    assert data["tool"]["ruff"]["target-version"] == "py39"
    assert data["tool"]["mypy"]["python_version"] == "3.10"


def test_py39_matrix_is_not_blocked_by_lint_tool_requirements():
    """isort 9 / mypy 2.4 的 Requires-Python >=3.10，不能留在 dev extras。

    test.yml 的 3.9 matrix 会 pip install -e ".[dev]"；若把这两个工具塞进
    dev，P1-14 刚修好的 3.9 job 会在 Install dependencies 阶段直接失败。
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dev = data["project"]["optional-dependencies"]["dev"]
    assert not any(x.startswith(("black", "isort", "mypy", "ruff")) for x in dev)
    assert "pytest-timeout" in " ".join(dev), "R9 守卫仍要求 dev 声明 pytest-timeout"
    assert '.[lint]' in LINT_YML.read_text(encoding="utf-8"), "lint job 必须显式安装 lint extras"


def test_graveyard_syntax_error_sample_is_excluded_from_lint_tools():
    """孵化墓园里故意留下的语法错误样本不得让 black/isort/mypy/ruff 挂掉。"""
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "incubation/graveyard" in text
    data = tomllib.loads(text)
    assert "hive/incubation/graveyard/" in data["tool"]["mypy"]["exclude"]
    assert any("graveyard" in g for g in data["tool"]["isort"]["skip_glob"])
    assert any("graveyard" in e for e in data["tool"]["ruff"]["exclude"])


def test_docs_script_parses_as_python_39():
    """PEP 701 之前的 f-string 不能复用外层引号；CI matrix 含 3.9~3.11。"""
    src = DOCS_SCRIPT.read_bytes().decode("utf-8")
    ast.parse(src, feature_version=(3, 9))


def test_guard_uses_real_parsers():
    """AST 防写死守卫：本文件必须真的解析 YAML/TOML/Python。"""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    attrs = [
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert "safe_load" in attrs
    assert "loads" in attrs
    assert "parse" in attrs


@pytest.mark.parametrize("name", ["lint.yml", "pwp-qa.yml", "test.yml"])
def test_all_workflows_still_parse_and_target_master(name):
    """R9 口径不回退：三份 workflow 必须可解析且只认 master。"""
    path = WF_DIR / name
    data = _load(path)
    on = data.get("on", data.get(True, {}))
    assert data.get("name")
    assert data.get("jobs")
    assert on["push"]["branches"] == ["master"]
    assert on["pull_request"]["branches"] == ["master"]
    assert "refs/heads/main" not in path.read_text(encoding="utf-8")
