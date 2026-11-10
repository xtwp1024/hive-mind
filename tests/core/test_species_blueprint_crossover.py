#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R4 / P1-8 回归：SpeciesBlueprint.crossover() 的 random 单一真源守卫。

背景（2026-11-11 R4 实测，全仓库 pytest -q 20 个历史失败之一）：
`hive/core/species_blueprint.py` 模块级第 24 行已有 `import random`，但
`crossover()` 又在 `if spec1 and spec2:` 分支内（原 528 行）与亲和度循环前
（原 565 行）重复 `import random`，`mutate()` 里也有一份（原 461 行）。
Python 只要在函数体内看到 `import random`，就把 `random` 判定为该函数的
**局部变量**；当双亲能力互斥、`if spec1 and spec2` 分支未执行时，`elif spec1`
/ `elif spec2` 分支里的 `random.random()` 就会抛
`UnboundLocalError: cannot access local variable 'random' ...`。

静态体检三项（py_compile / import 闭合 / 镜像一致）对这类缺陷完全无感：
语法合法、模块可导入、镜像一致，依旧全 PASS。这是本项目
「静态全绿 ≠ 运行时可用」的第四次实证（R1 缺 import os、R2 路径默认值、
R3 根锚点、R4 函数内 import 遮蔽）。

验收强度（沿用 R-20261010-11）：
- 真实运行时调用：双亲能力互斥 / 共有 / 亲和度三条路径都真的跑一遍；
- 旧实现注入反证：**运行时构造**旧实现（不碰工作区文件），断言必然抛
  UnboundLocalError，证明新测试确实咬住了修复点；
- 字节级复原：全程不改写源码，前后 sha256 必须 SAME；
- AST 判据守卫：源码里函数体内不得再出现 `import random`。
"""
from __future__ import annotations

import ast
import hashlib
import random
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "hive" / "core" / "species_blueprint.py"

# 注入用的锚点：修复后 `if spec1 and spec2:` 下一行直接是 random.random()
_NEW_ANCHOR = "            if spec1 and spec2:\n                if random.random() < 0.5:\n"
_OLD_ANCHOR = (
    "            if spec1 and spec2:\n"
    "                import random\n\n"
    "                if random.random() < 0.5:\n"
)


def _source_bytes() -> bytes:
    return SRC.read_bytes()


def _source_text() -> str:
    # 本仓库该文件是 CRLF：统一成 LF 后再做锚点匹配 / AST / exec，避免注入
    # 反证因行尾差异静默失效（这正是「反证必须真的执行过」的一处实例）。
    return _source_bytes().decode("utf-8").replace("\r\n", "\n")


def _make_blueprints(module, *, disjoint: bool = True):
    """构造两个真实 SpeciesBlueprint，用真实 add_capability 走能力分支。"""
    bp1 = module.SpeciesBlueprint(
        blueprint_id="r4_parent_001", name="R4 Parent 1", description="crossover regression"
    )
    bp2 = module.SpeciesBlueprint(
        blueprint_id="r4_parent_002", name="R4 Parent 2", description="crossover regression"
    )
    pri = module.CapabilityPriority.PRIMARY
    if disjoint:
        # 互不重叠 => 必走 elif spec1 / elif spec2（旧实现在这里崩）
        bp1.add_capability("mining", 0.9, pri)
        bp2.add_capability("sentinel", 0.8, pri)
    else:
        # 同名能力 => 必走 if spec1 and spec2
        bp1.add_capability("mining", 0.9, pri)
        bp2.add_capability("mining", 0.7, pri)
    return bp1, bp2


def _exec_module(source_text: str, name: str):
    """在隔离命名空间里 exec 源码，返回模块对象；不污染 sys.modules 与工作区。"""
    mod = types.ModuleType(name)
    mod.__file__ = str(SRC)
    sys.modules[name] = mod
    try:
        exec(compile(source_text, str(SRC), "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


# ---------------------------------------------------------------- 真实调用

def test_module_level_random_is_single_source():
    """AST 守卫：random 只在模块级 import 一次，函数体内不得再出现。"""
    tree = ast.parse(_source_text())
    top_level = [
        n for n in tree.body
        if isinstance(n, ast.Import) and any(a.name == "random" for a in n.names)
    ]
    assert len(top_level) == 1, "模块级 import random 必须恰好一处"

    local_imports = [
        n.lineno
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for n in ast.walk(fn)
        if isinstance(n, ast.Import) and any(a.name == "random" for a in n.names)
    ]
    assert local_imports == [], f"函数体内重复 import random 会遮蔽模块级绑定: {local_imports}"


@pytest.mark.parametrize("disjoint", [True, False])
def test_crossover_real_call_no_unbound_local(disjoint):
    """真实运行时调用：互斥与共有能力两条分支都不得抛 UnboundLocalError。"""
    from hive.core import species_blueprint as real

    random.seed(20261111)
    bp1, bp2 = _make_blueprints(real, disjoint=disjoint)
    child1, child2 = bp1.crossover(bp2)

    assert child1.origin is real.BlueprintOrigin.CROSSED
    assert child2.origin is real.BlueprintOrigin.CROSSED
    assert bp1.blueprint_id in child1.parent_blueprints
    assert bp2.blueprint_id in child1.parent_blueprints
    # 子代能力必须来自双亲能力集合，不能凭空多出
    allowed = set(bp1.capabilities) | set(bp2.capabilities)
    assert set(child1.capabilities) <= allowed
    assert set(child2.capabilities) <= allowed
    # 双亲共有的能力必须被继承（这条分支不依赖随机结果）
    for shared in set(bp1.capabilities) & set(bp2.capabilities):
        assert shared in child1.capabilities
        assert shared in child2.capabilities


def test_crossover_affinity_branch_real_call():
    """真实调用亲和度循环（原第二处函数内 import random 所在路径）。"""
    from hive.core import species_blueprint as real

    random.seed(7)
    bp1, bp2 = _make_blueprints(real, disjoint=True)
    bp1.add_task_affinity("mining", 0.9)
    bp2.add_task_affinity("mining", 0.7)
    child1, child2 = bp1.crossover(bp2)
    assert child1.task_affinities or child2.task_affinities
    assert all(a.task_category == "mining" for a in child1.task_affinities)


def test_mutate_real_call_still_works():
    """mutate() 也去掉了函数内 import，真实调用必须仍可用。"""
    from hive.core import species_blueprint as real

    random.seed(11)
    bp1, _ = _make_blueprints(real, disjoint=True)
    mutated = bp1.mutate(mutation_rate=1.0)
    assert mutated.origin is real.BlueprintOrigin.MUTATED
    assert mutated.blueprint_id.startswith("r4_parent_001_mut_")
    assert set(mutated.capabilities) == set(bp1.capabilities)


# ---------------------------------------------------------- 旧实现注入反证

def test_old_implementation_injection_red():
    """旧实现注入反证：运行时把 `import random` 注回分支内，新路径必须变红。

    用「运行时构造旧实现」而非 `git show HEAD:` / 就地改写源码，避免
    R-20261010-19 的旧版取法腐坏与 R-20261010-11 的复原破坏性问题。
    """
    before = hashlib.sha256(_source_bytes()).hexdigest()
    text = _source_text()
    assert _NEW_ANCHOR in text, "注入锚点已漂移，反证失效，需重新标定"

    old_text = text.replace(_NEW_ANCHOR, _OLD_ANCHOR, 1)
    assert old_text != text
    old_mod = _exec_module(old_text, "_hive_r4_old_species_blueprint")

    random.seed(20261111)
    bp1, bp2 = _make_blueprints(old_mod, disjoint=True)
    with pytest.raises(UnboundLocalError) as exc:
        bp1.crossover(bp2)
    assert "local variable 'random'" in str(exc.value)

    # 同一份旧实现，改成共有能力（会先执行到分支内 import）则不崩 —— 精确复现
    # 「只有互斥能力才触发」的条件性，证明测试咬住的是遮蔽而非普通崩溃。
    random.seed(20261111)
    bp3, bp4 = _make_blueprints(old_mod, disjoint=False)
    c3, c4 = bp3.crossover(bp4)
    assert c3.origin is old_mod.BlueprintOrigin.CROSSED

    # 字节级复原：全程未写文件，源码哈希必须 SAME
    assert hashlib.sha256(_source_bytes()).hexdigest() == before
