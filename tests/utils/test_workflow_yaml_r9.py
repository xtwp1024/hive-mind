"""R9 回归：workflow 配置必须可解析、分支口径与默认分支一致、CI 依赖声明闭合。

背景（2026-11-12 实测）：``.github/workflows/test.yml`` 的 docker job 曾把
``python -c "..."`` 的多行 Python 代码顶格写在 ``run: |`` 块标量内，从第 186 行起
截断了整个块 → PyYAML ``ScannerError: could not find expected ':'`` →
GitHub 无法解析该 workflow。表现是从 2026-11-09 起每次 push 都只留一条
**0 jobs / 无日志 / name 退化为文件路径**的空失败 run（run 43092429464 实测）。

``tools/loop_check.py`` 只查 ``.py`` 语法与 ``hive.*`` import 闭合，对 ``.yml``
完全无感，故此处补针对 workflow 配置的独立守卫。

另守卫两件事：默认分支是 ``master`` 而 workflow 曾写 ``[main, develop]``；
以及 workflow 用了 ``pytest --timeout`` 但依赖里没声明 ``pytest-timeout``。
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WF_DIR = ROOT / ".github" / "workflows"
WORKFLOWS = ("lint.yml", "pwp-qa.yml", "test.yml")
EXPECTED_NAMES = {
    "lint.yml": "Code Quality",
    "pwp-qa.yml": "PWP-QA",
    "test.yml": "Tests",
}


def _load(name: str):
    text = (WF_DIR / name).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    # PyYAML 遵循 YAML 1.1，会把顶层 `on:` 解析成布尔 True；GitHub 用 1.2 当字符串。
    on = data.get("on", data.get(True, {}))
    return text, data, on


@pytest.mark.parametrize("name", WORKFLOWS)
def test_workflow_parses_and_exposes_name(name):
    """解析失败时 GitHub 会用文件路径当 run 名，且产出 0 jobs 的空失败记录。"""
    _, data, _ = _load(name)
    assert isinstance(data, dict)
    assert data.get("name") == EXPECTED_NAMES[name]
    assert data.get("jobs"), "workflow 必须含可执行 jobs"


@pytest.mark.parametrize("name", WORKFLOWS)
def test_push_and_pr_target_default_branch(name):
    """本仓库默认分支是 master；写 main/develop 会让 workflow 永远不触发。"""
    _, _, on = _load(name)
    assert on["push"]["branches"] == ["master"]
    assert on["pull_request"]["branches"] == ["master"]


def test_no_stale_main_ref_in_any_workflow():
    """分支过滤与 job 级 if 条件必须是同一口径，不能只改一半。"""
    stale = []
    for name in WORKFLOWS:
        text = (WF_DIR / name).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if "refs/heads/main" in line or "[main, develop]" in line:
                stale.append(f"{name}:{lineno}: {line.strip()}")
    assert not stale, f"残留旧分支口径: {stale}"

    test_yml = (WF_DIR / "test.yml").read_text(encoding="utf-8")
    # full_tests / integration / docs / docker 四处 job 级条件
    assert test_yml.count("github.ref == 'refs/heads/master'") == 4


def test_docker_step_python_code_is_single_line_and_intact():
    """修 YAML 不能顺手把 5 个 import 检查删掉，也不能引入缩进错误。"""
    _, data, _ = _load("test.yml")
    steps = data["jobs"]["docker"]["steps"]
    run = next(s["run"] for s in steps if s.get("name") == "Test Docker image")
    commands = [line.strip() for line in run.splitlines()
                if line.strip().startswith("docker run")]
    assert len(commands) == 1, "docker 检查必须压在单个 shell 行内，避免顶格行截断块标量"
    code = commands[0].split('python -c "', 1)[1].rsplit('"', 1)[0]
    ast.parse(code)  # 语法不合法会在本行抛 SyntaxError
    assert code.count("from hive") == 5
    for module in ("dispatcher", "memory_store", "capability_gap_analyzer",
                   "multimodal_encoder", "hive_federation"):
        assert module in code


def test_timeout_flag_has_declared_plugin():
    """workflow 里出现 --timeout，就必须在 dev extras 里声明 pytest-timeout。"""
    uses_timeout = any(
        "--timeout" in (WF_DIR / name).read_text(encoding="utf-8")
        for name in WORKFLOWS
    )
    assert uses_timeout, "test.yml 的 --timeout 参数若被移除，本守卫需同步修订"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dev_block = pyproject.split("[project.optional-dependencies]", 1)[1].split("docs =", 1)[0]
    assert "pytest-timeout" in dev_block, "缺 pytest-timeout 会让 CI 报 unrecognized arguments"


def test_toplevel_line_injection_breaks_yaml(tmp_path):
    """旧实现注入反证：把顶格 Python 代码注回 run 块，解析器必须报错。

    全程只写 tmp_path 里的影子文件，不改写真实 workflow。
    """
    target = WF_DIR / "test.yml"
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    text = target.read_text(encoding="utf-8")
    anchor = '        echo "=== Testing Docker Image ==="'
    assert anchor in text
    broken = text.replace(
        anchor,
        anchor + '\n        docker run --rm hive-mind:latest python -c "\nimport sys\n"',
        1,
    )
    shadow = tmp_path / "test.yml"
    shadow.write_text(broken, encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(shadow.read_text(encoding="utf-8"))
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before, "真实文件被改写"


def test_guard_parses_yaml_instead_of_string_matching():
    """AST 防写死守卫：本文件必须真的调用 yaml.safe_load，不能退化成硬编码断言。"""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "safe_load"
    ]
    assert calls
