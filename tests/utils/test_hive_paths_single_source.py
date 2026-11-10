# -*- coding: utf-8 -*-
"""P1-7 回归：HIVE_HOME 根锚点收敛为 hive.paths 单一真源。

R2 只修了 event_stream.py 一处，同形硬编码仍散落在 4 处。本轮把它们全部
收成一份判据（hive/paths.py），并用四类证据钉住：

  1. 唯一真源的判据：空串 / 空白 / 覆盖 / cwd 无关 / expanduser
  2. 消费点投影守卫：不得再自造默认值，必须从 hive.paths 导入
     （R-20261024-12「接口的每个投影都要派生或守卫」）
  3. 写读同路径契约 + __all__ 派生守卫（不能手工枚举漂移）
  4. 旧实现注入反证（R-20261010-11）：把旧表达式注入源码副本，守卫必须能检出

零破坏：反证只在 tmp_path 的源码副本上做变异，不改写真实工作区。
"""
import ast
import os
from pathlib import Path, PurePosixPath

import pytest

from hive import paths
from hive.utils import event_stream as es

ROOT = Path(__file__).resolve().parents[2]

# 四处纯消费点（不得自定义 resolver），以及一处刻意保留的兼容薄壳
PLAIN_CONSUMERS = (
    "hive/__init__.py",
    "hive/config/settings.py",
    "hive/bees/variants/librarian_bee.py",
    "web_dashboard.py",
)
ALL_CONSUMERS = (*PLAIN_CONSUMERS, "hive/utils/event_stream.py")

# 旧实现的形态：盘符字面量 / 自造 resolver / 从 cwd 或 getenv 拼默认值
LEGACY_LITERALS = ("D:/HIVE", "D:\\HIVE")


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _imports_resolver(rel: str) -> bool:
    """是否确实从 hive.paths 导入了 resolve_hive_home（绝对或包内相对导入）。"""
    tree = ast.parse(_src(rel))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        mod = node.module or ""
        from_paths = mod == "hive.paths" or (node.level == 1 and mod == "paths")
        if from_paths and any(a.name == "resolve_hive_home" for a in node.names):
            return True
    return False


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """收集模块 / 函数 / 类的 docstring 常量节点 id。

    守卫必须区分「说明文字」与「实际代码」：hive/paths.py 的模块说明里
    逐条引用了旧写法（``cwd / "D:/HIVE"``）作为反例，若把 docstring 里的
    文本也算成硬编码，新实现会被自己的背景说明误报成违规。
    """
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr):
                val = body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    ids.add(id(val))
    return ids


def _legacy_markers(src: str) -> list[str]:
    """扫描源码，返回仍在自造 HIVE_HOME 默认值的证据（守卫的核心判据）。"""
    tree = ast.parse(src)
    docstrings = _docstring_nodes(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings and any(lit in node.value for lit in LEGACY_LITERALS):
                found.append(f"盘符字面量: {node.value!r}")
        # 自造 resolve_hive_home 由 test_plain_consumer_defines_no_resolver 单独守卫
    return found


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv("HIVE_HOME", raising=False)
    monkeypatch.setattr(es, "_events_file", None)
    yield
    monkeypatch.setattr(es, "_events_file", None)


# ---------- 1. 唯一真源的判据 ----------

def test_project_root_is_repo_root():
    assert paths.PROJECT_ROOT == ROOT
    assert (ROOT / "hive" / "paths.py").is_file()


def test_default_home_is_project_root():
    assert paths.resolve_hive_home() == ROOT


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_hive_home_falls_back(monkeypatch, blank):
    monkeypatch.setenv("HIVE_HOME", blank)
    assert paths.resolve_hive_home() == ROOT


def test_hive_home_override(monkeypatch, tmp_path):
    target = tmp_path / "alt_home"
    monkeypatch.setenv("HIVE_HOME", str(target))
    assert paths.resolve_hive_home() == target
    assert paths.resolve_events_file() == target / "Data" / "events.jsonl"


def test_default_does_not_depend_on_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_hive_home() == ROOT


def test_explicit_root_is_injection_point(tmp_path):
    """薄壳要能注入 root 做隔离，但 HIVE_HOME 优先级不变。"""
    root = tmp_path / "injected"
    assert paths.resolve_hive_home(root) == root
    assert paths.resolve_events_file(root) == root / "Data" / "events.jsonl"


def test_resolver_has_no_cwd_or_drive_literal():
    """AST 级契约：唯一实现内部不得出现 cwd() 或盘符字面量。"""
    src = _src("hive/paths.py")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_hive_home"
    )
    segment = ast.get_source_segment(src, fn) or ""
    assert "cwd(" not in segment
    assert "getcwd(" not in segment
    for n in ast.walk(fn):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            assert not any(lit in n.value for lit in LEGACY_LITERALS)


# ---------- 2. 消费点投影守卫 ----------

@pytest.mark.parametrize("rel", ALL_CONSUMERS)
def test_consumer_imports_single_source(rel):
    assert _imports_resolver(rel), f"{rel} 没有从 hive.paths 导入 resolve_hive_home"


@pytest.mark.parametrize("rel", ALL_CONSUMERS)
def test_consumer_has_no_legacy_literals(rel):
    assert _legacy_markers(_src(rel)) == [], f"{rel} 仍残留硬编码盘符默认值"


@pytest.mark.parametrize("rel", PLAIN_CONSUMERS)
def test_plain_consumer_defines_no_resolver(rel):
    """判据只能有一份：纯消费点不得再定义同名解析函数。"""
    tree = ast.parse(_src(rel))
    defs = [
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_hive_home"
    ]
    assert defs == [], f"{rel} 又造了一份 resolve_hive_home"


def test_event_stream_is_delegating_thin_shell():
    """兼容薄壳：只转发，不含判据（无字符串常量 / 无 cwd / 单 return）。"""
    src = _src("hive/utils/event_stream.py")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_hive_home"
    )
    # 允许 docstring + 单个 return：薄壳不得含任何其它语句
    stmts = [
        n for n in fn.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str))
    ]
    assert len(stmts) == 1 and isinstance(stmts[0], ast.Return)
    segment = ast.get_source_segment(src, fn) or ""
    assert "cwd(" not in segment and "getcwd(" not in segment
    assert not [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and any(lit in n.value for lit in LEGACY_LITERALS)
    ]


def test_all_is_derived_not_hand_listed():
    """R-20261024-12：__all__ 是手工枚举的投影，必须与实现派生结果逐项相等。"""
    tree = ast.parse(_src("hive/paths.py"))
    public = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                public.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    if target.id != "__all__":
                        public.add(target.id)
    assert set(paths.__all__) == public, "__all__ 与公开符号漂移"


def test_event_stream_projection_is_guarded():
    """兼容薄壳保留的 PROJECT_ROOT 是投影，须与真源为同一对象。"""
    assert es.PROJECT_ROOT is paths.PROJECT_ROOT


# ---------- 3. 写读同路径契约 ----------

def test_writer_reader_paths_agree(monkeypatch, tmp_path):
    """写入方 event_stream 与读取方 web_dashboard 必须落在同一个文件。"""
    monkeypatch.setattr(es, "PROJECT_ROOT", tmp_path)
    written = es.init_events_file()
    assert written == tmp_path / "Data" / "events.jsonl"

    # web_dashboard 依赖 flask（本环境未装齐），不 import，改用构造式守卫
    dash = _src("web_dashboard.py")
    assert "HIVE_HOME = resolve_hive_home()" in dash
    assert 'EVENTS_FILE = HIVE_HOME / "Data" / "events.jsonl"' in dash
    assert "BASE_DIR = Path(__file__).parent" in dash

    # 两边都由同一个真源派生：真源改 root，读侧构造式等价于 resolve_events_file
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    assert paths.resolve_events_file() == written


def test_settings_default_base_dir_is_derived(monkeypatch, tmp_path):
    """settings 的兜底 BASE_DIR 必须是 <hive 根>/Data/Inbound 的派生投影。"""
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    from hive.config.settings import HiveSettings
    assert HiveSettings._default_base_dir() == tmp_path / "Data" / "Inbound"


def test_emit_writes_parseable_event(monkeypatch, tmp_path):
    """R-20261010-08：真实调用 init + emit，读回 jsonl 逐字段解析。"""
    import json
    monkeypatch.setattr(es, "PROJECT_ROOT", tmp_path)
    path = es.init_events_file()
    es.emit("cycle_start", cycle=1, total=3)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    first = json.loads(lines[0])
    assert first["type"] == "cycle_start"
    assert first["data"] == {"cycle": 1, "total": 3}
    assert first["ts"].endswith("+00:00")


# ---------- 4. 旧实现注入反证（零破坏，只变异源码副本） ----------

def test_legacy_injection_is_detected(tmp_path):
    """R-20261010-11：把旧默认值注回源码副本，守卫必须能咬住。"""
    original = _src("hive/paths.py")
    assert _legacy_markers(original) == []  # 新实现：无标记

    legacy = original.replace(
        'return root if root is not None else PROJECT_ROOT',
        'return Path.cwd() / "D:/HIVE"',
    )
    assert legacy != original
    assert _legacy_markers(legacy), "注入旧实现后守卫必须检出"

    # 字节级复原校验（本反证不写真实文件，仅证明变异可区分）
    restored = legacy.replace(
        'return Path.cwd() / "D:/HIVE"',
        'return root if root is not None else PROJECT_ROOT',
    )
    assert restored == original


def test_legacy_expression_is_cross_platform_broken():
    """显式复现跨平台歧义，证明这不是本机巧合（沿 R2 的 POSIX 证据）。"""
    posix_legacy = PurePosixPath("/srv/hive") / "D:/HIVE"
    assert str(posix_legacy) == "/srv/hive/D:/HIVE"
    assert ":" in str(posix_legacy)
    assert posix_legacy != PurePosixPath("/srv/hive")
    assert str(paths.resolve_hive_home()) != "/srv/hive/D:/HIVE"


def test_blank_env_does_not_create_empty_path(tmp_path):
    """web_dashboard 旧式 Path(os.getenv('HIVE_HOME', ...)) 空串会落 Path('')。"""
    os.environ.pop("HIVE_HOME", None)
    assert paths.resolve_hive_home() != Path("")
    assert paths.resolve_hive_home() == ROOT
