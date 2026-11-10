# -*- coding: utf-8 -*-
"""P1-1 回归：事件流根目录的跨平台默认值与「写读同路径」契约。

静态体检（py_compile / import 闭合）看不见这类缺陷：旧实现把
``cwd / "D:/HIVE"`` 当默认值，在 Windows 上因右操作数是绝对路径而被
整个覆盖成 ``D:\\HIVE``，恰好与项目根 ``D:\\Hive`` 大小写不敏感相等，
本机「碰巧能用」；但在 POSIX 语义下会拼出 ``<cwd>/D:/HIVE`` 这种带冒号
的怪目录，写入方与 web_dashboard.py 的读取方就此分家。

故本文件用三种证据钉住：默认落点契约、运行时真实写读、旧实现注入反证。
"""
import ast
import json
from pathlib import Path, PurePosixPath

import pytest

from hive.utils import event_stream as es

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """隔离环境变量与模块级 _events_file，避免测试间互相污染。"""
    monkeypatch.delenv("HIVE_HOME", raising=False)
    monkeypatch.setattr(es, "_events_file", None)
    yield
    monkeypatch.setattr(es, "_events_file", None)


def test_project_root_is_repo_root():
    """锚点必须是仓库根，且与读取方 web_dashboard.py 同目录。"""
    assert es.PROJECT_ROOT == ROOT
    assert (ROOT / "web_dashboard.py").is_file()


def test_default_home_is_project_root():
    assert es.resolve_hive_home() == ROOT


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_hive_home_falls_back(monkeypatch, blank):
    """空串/纯空白按「未设置」处理，不能落到 Path('') == cwd。"""
    monkeypatch.setenv("HIVE_HOME", blank)
    assert es.resolve_hive_home() == ROOT


def test_hive_home_override(monkeypatch, tmp_path):
    target = tmp_path / "alt_home"
    monkeypatch.setenv("HIVE_HOME", str(target))
    assert es.resolve_hive_home() == target


def test_default_does_not_depend_on_cwd(monkeypatch, tmp_path):
    """从任意目录启动进程，默认落点都不能随 cwd 漂移。"""
    monkeypatch.chdir(tmp_path)
    assert es.resolve_hive_home() == ROOT


def test_writer_reader_paths_agree(monkeypatch, tmp_path):
    """init_events_file 的落点必须等于 web_dashboard.EVENTS_FILE 的构造式。"""
    monkeypatch.setattr(es, "PROJECT_ROOT", tmp_path)
    written = es.init_events_file()
    assert written == tmp_path / "Data" / "events.jsonl"
    dash = (ROOT / "web_dashboard.py").read_text(encoding="utf-8", errors="replace")
    assert 'EVENTS_FILE = HIVE_HOME / "Data" / "events.jsonl"' in dash
    assert "BASE_DIR = Path(__file__).parent" in dash


def test_emit_writes_parseable_event(monkeypatch, tmp_path):
    """R-20261010-08：真实调用 init + emit，读回 jsonl 解析，不是只 import。"""
    monkeypatch.setattr(es, "PROJECT_ROOT", tmp_path)
    path = es.init_events_file()
    es.emit("cycle_start", cycle=1, total=3)
    es.emit("cycle_end", cycle=1, honey=7)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["type"] == "cycle_start"
    assert first["data"] == {"cycle": 1, "total": 3}
    assert first["ts"].endswith("+00:00")
    assert json.loads(lines[1])["data"]["honey"] == 7


def test_resolver_has_no_cwd_or_drive_literal():
    """AST 级契约：解析函数里不得再出现 cwd() 或带盘符的字符串常量。"""
    src = (ROOT / "hive/utils/event_stream.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_hive_home"
    )
    segment = ast.get_source_segment(src, fn) or ""
    assert "cwd(" not in segment, "不得再用 Path.cwd() 构造默认落点"
    literals = [
        n.value for n in ast.walk(fn)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert not any(
        len(v) >= 2 and v[0].isalpha() and v[1] == ":" for v in literals
    ), f"不得再硬编码盘符字面量: {literals}"


def test_legacy_expression_injection_is_red(monkeypatch, tmp_path):
    """R-20261010-11 旧实现注入反证：把旧解析逻辑注回模块，契约必须失效。

    不改写真实源码（避免破坏工作区），运行时构造旧实现并 monkeypatch 注入；
    若「默认落点 == 项目根」这条断言恒绿，说明测试没咬住修复点。
    """
    def legacy_resolve():
        return Path.cwd() / "D:/HIVE"  # 旧默认值表达式，原样复刻

    project = tmp_path / "project"
    monkeypatch.setattr(es, "PROJECT_ROOT", project)

    # 新实现：契约成立
    assert es.resolve_hive_home() == project

    # 注入旧实现：同一契约必须崩掉（POSIX 语义下是带冒号的怪目录，
    # Windows 下则被绝对路径覆盖成 D:\HIVE，两者都不等于项目根）
    monkeypatch.setattr(es, "resolve_hive_home", legacy_resolve)
    assert es.resolve_hive_home() != project

    # 另用 PurePosixPath 显式复现跨平台歧义，证明这不是本机巧合
    posix_legacy = PurePosixPath("/srv/hive") / "D:/HIVE"
    assert str(posix_legacy) == "/srv/hive/D:/HIVE"
    assert ":" in str(posix_legacy)
    assert posix_legacy != PurePosixPath("/srv/hive")
