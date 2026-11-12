# -*- coding: utf-8 -*-
"""R10 / P1-13 回归：两个「接入 main()」的单元回归不得绑到外部 git 状态。

R9 收尾实测：推送 `5e57aef` 后镜像仍停在 `cd19c4a`，`loop_check.main()`
的 [3/3] 镜像一致判 FAIL ⇒ 返回 1 ⇒
  tests/utils/test_loop_check_acceptance_chain.py::test_acceptance_chain_is_wired_into_main
  tests/utils/test_loop_check_p0_scope.py::test_main_scope_is_projected_from_ledger
因 `assert rc == 0` 连带误报（2 failed / 167 passed），而同期冒烟、全量
pytest、分诊对账全部不变。这两个用例的职责是「守卫接入 main() 输出」与
「SCOPE 由台账派生」，镜像一致性由 test_loop_check_mirror.py 专门负责，
故在用例内用 monkeypatch 把 check_mirror 固定为「无问题」态。

四类守卫（形态沿用 R7 的 tests/core/test_state_isolation_r7.py）：
  1. AST 投影守卫：两个真实用例必须含 check_mirror 隔离块、带 monkeypatch
     形参、且仍调用 main()（防隔离被删后回潮）；
  2. 隔离桩语义：桩返回空 issues（不是把 FAIL 藏起来，而是换掉外部依赖）；
  3. 运行时真实调用 + 量化对照：强制注入「镜像不一致」态后 main() rc 由
     0 翻转为 1，精确复现 R9 误报根因；
  4. 旧实现注入反证：AST 剥离隔离块后在内存 exec 出旧用例，坏镜像态下
     必须抛 AssertionError；保留隔离的新用例在同一态下 PASS；
  5. 字节级 sha256 SAME：反证全程只改内存模块属性 / exec 内存副本，
     try/finally 复原，真实源码不被改写（R-20261010-11）。
"""
from __future__ import annotations

import ast
import hashlib
import io
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

from tools import loop_check

ROOT = Path(loop_check.ROOT)
LOOP_CHECK_REL = "tools/loop_check.py"
TARGETS = [
    ("tests/utils/test_loop_check_acceptance_chain.py",
     "test_acceptance_chain_is_wired_into_main"),
    ("tests/utils/test_loop_check_p0_scope.py",
     "test_main_scope_is_projected_from_ledger"),
]


def _fn_node(rel: str, fn_name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
    return next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == fn_name)


def _is_isolation_stmt(node: ast.stmt) -> bool:
    """识别 `monkeypatch.setattr(loop_check, "check_mirror", ...)` 语句。"""
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return False
    call = node.value
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == "setattr"):
        return False
    return any(isinstance(a, ast.Constant) and a.value == "check_mirror"
               for a in call.args)


def _calls_main(fn: ast.FunctionDef) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "main" for n in ast.walk(fn))


# ---------- 1. AST 投影守卫 ----------

def test_isolation_block_present_in_both_regressions():
    """两个真实用例必须显式隔离 check_mirror，且仍需调用 main()。"""
    for rel, fn_name in TARGETS:
        fn = _fn_node(rel, fn_name)
        assert any(a.arg == "monkeypatch" for a in fn.args.args), (rel, fn_name)
        blocks = [s for s in fn.body if _is_isolation_stmt(s)]
        assert len(blocks) == 1, (rel, fn_name, len(blocks))
        assert _calls_main(fn), (rel, fn_name, "不再调用 main()")


def test_isolation_stub_returns_clean_state():
    """隔离桩必须返回空 issues（换掉外部依赖，而非把 FAIL 伪装成 PASS）。"""
    for rel, fn_name in TARGETS:
        fn = _fn_node(rel, fn_name)
        block = next(s for s in fn.body if _is_isolation_stmt(s))
        lam = next(a for a in block.value.args if isinstance(a, ast.Lambda))
        expr = ast.Expression(body=lam)
        ast.fix_missing_locations(expr)
        stub = eval(compile(expr, f"<{rel}:isolation_stub>", "eval"), {})  # noqa: S307
        issues, status = stub()
        assert issues == [], (rel, issues)
        assert "隔离" in status, (rel, status)


# ---------- 2. 反证用的内存夹具（不改写真实工作区） ----------

class _Out:
    def __init__(self, text: str):
        self.out = text
        self.err = ""


class _FakeCapsys:
    def __init__(self, buf: io.StringIO):
        self._buf = buf

    def readouterr(self) -> _Out:
        return _Out(self._buf.getvalue())


class _FakeMP:
    """最小 monkeypatch 替身，undo 时逐项复原模块属性。"""

    def __init__(self):
        self._saved: list[tuple[object, str, object]] = []

    def setattr(self, obj, name: str, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._saved):
            setattr(obj, name, old)
        self._saved.clear()


@contextmanager
def _forced_mirror(issues: list[str], status: str):
    """注入受控镜像检查结果，退出时复原真实函数。

    两态（坏 / 干净）都走注入，反证自身不读真实 git 状态 —— 否则本回归
    会重蹈 P1-13 的覆辙（把结论绑到「镜像是否 == origin/master」）。
    """
    orig = loop_check.check_mirror
    loop_check.check_mirror = lambda: (list(issues), status)
    try:
        yield
    finally:
        loop_check.check_mirror = orig


@contextmanager
def _forced_bad_mirror():
    """模拟 R9 收尾「已推送但镜像未同步」窗口。"""
    with _forced_mirror(["镜像与远端不一致（R10 反证注入）"], "落后/分叉"):
        yield


def _load_variant(rel: str, fn_name: str, *, strip_isolation: bool):
    """从真实源码抽出目标函数，可选剥离隔离块，在内存 exec 出可调用对象。"""
    fn = _fn_node(rel, fn_name)
    if strip_isolation:
        fn.body = [s for s in fn.body if not _is_isolation_stmt(s)]
        fn.args.args = [a for a in fn.args.args if a.arg != "monkeypatch"]
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"loop_check": loop_check}
    exec(compile(module, f"<{rel}:{fn_name}:variant>", "exec"), ns)  # noqa: S102
    return ns[fn_name]


def _call_variant(rel: str, fn_name: str, *, strip_isolation: bool):
    """在强制坏镜像态下真实调用变体，返回 (是否未抛断言错, 断言错对象)。"""
    fn = _load_variant(rel, fn_name, strip_isolation=strip_isolation)
    buf = io.StringIO()
    cap = _FakeCapsys(buf)
    mp = _FakeMP()
    try:
        with _forced_bad_mirror():
            try:
                with redirect_stdout(buf):
                    if strip_isolation:
                        fn(cap)          # 旧实现：无 monkeypatch 形参
                    else:
                        fn(cap, mp)      # 新实现：内部隔离覆盖坏镜像
            finally:
                mp.undo()                # 必须在坏镜像上下文内撤销，避免污染
        return True, None
    except AssertionError as exc:
        return False, exc


# ---------- 3. 运行时真实调用 + 量化对照 ----------

def test_forced_mirror_failure_flips_main_rc():
    """坏镜像态下 main() rc 必由 0 翻转为 1 —— 精确复现 R9 误报根因。

    两态都用注入桩，不读真实 git 状态（否则本回归自身又绑回外部状态）。
    """
    with _forced_mirror(["镜像与远端不一致（R10 反证注入）"], "落后/分叉"):
        with redirect_stdout(io.StringIO()):
            rc_bad = loop_check.main()
    with _forced_mirror([], "三方对齐（反证注入·干净态）"):
        with redirect_stdout(io.StringIO()):
            rc_clean = loop_check.main()
    assert rc_bad == 1, rc_bad
    assert rc_clean == 0, rc_clean
    assert rc_bad != rc_clean, "守卫无区分力：注入外部状态前后 rc 相同"


# ---------- 4. 旧实现注入反证 ----------

def test_old_implementation_reproduces_r9_false_positive():
    """剥离隔离块的旧用例在坏镜像态下必须抛 AssertionError（R9 的 2 failed）。

    精确判据：断言消息（即 main() 输出）必须同时含 `[3/3] 镜像一致 问题 1`
    的注入痕迹与 `RESULT: FAIL`，证明失败源自镜像检查项而非 SCOPE 投影断言
    —— 否则「反证变红」可能是别的原因造成的假阳性。
    """
    for rel, fn_name in TARGETS:
        passed, exc = _call_variant(rel, fn_name, strip_isolation=True)
        assert not passed, (rel, fn_name, "旧实现未复现误报，反证失效")
        assert isinstance(exc, AssertionError)
        msg = str(exc)
        assert "镜像与远端不一致（R10 反证注入）" in msg, msg
        assert "RESULT: FAIL" in msg, msg


def test_new_implementation_survives_forced_mirror_failure():
    """同一坏镜像态下，保留隔离的新用例必须 PASS —— 量化对照成立。"""
    for rel, fn_name in TARGETS:
        passed, exc = _call_variant(rel, fn_name, strip_isolation=False)
        assert passed, (rel, fn_name, exc)


# ---------- 5. 字节级复原 ----------

def test_rebuttal_does_not_rewrite_real_sources():
    """反证前后三个真实源码 sha256 SAME（全程内存注入，不写文件）。"""
    files = [ROOT / rel for rel, _ in TARGETS] + [ROOT / LOOP_CHECK_REL]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    rel, fn_name = TARGETS[0]
    _call_variant(rel, fn_name, strip_isolation=True)
    _call_variant(rel, fn_name, strip_isolation=False)
    after = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    assert before == after, "反证改写了真实源码"
    # 反证结束后 check_mirror 必须仍是真函数（未被 _FakeMP 残留覆盖）
    assert loop_check.check_mirror.__module__ == loop_check.__name__
