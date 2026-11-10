# 历史失败分诊报告 · R4 / P1-8

> 时间：2026-11-11　轮次：R4　范围：`hive/tests/` 全仓库 20 个历史失败
> **唯一真源**：`tools/triage_check.py` 的 `LEDGER`（20 行结构化台账）。
> 本文是导读与决策记录，**不复制台账**，避免两份副本漂移（R-20261024-12）。
> 复现命令：`python tools/triage_check.py`（静态）/ `python tools/triage_check.py --report <pytest输出>`（对账）。

## 一、口径与实测基线

- 命令：托管 venv `pytest -q --tb=line`（全仓库，非 `tests/` 子集）。
- R4 修复前实测：**20 failed / 455 passed / 13 warnings，251.40s**。
- R3 台账称这 20 个失败是「`brain_config.json` 缺失导致 `BrainInterface` 加载失败、旧断言失效等」。
  **本轮逐条核对 20 条实测异常，无一条与 `brain_config.json` / `BrainInterface` / 缺依赖 / 缺配置相关（依赖缺失 0 条）**。
  这是 R-20261010-19 的又一次命中：状态文档里的举例不能当真源，必须回到实测。
  台账描述已在本文件与 LOOP.md 更正，未改动 R3 历史行（append-only）。

## 二、分类结果（证据见 LEDGER.evidence）

| 分类 | 条数 | 处置 |
| --- | --- | --- |
| 真缺陷·已修 | 1 | R4 本轮修复 + 6 项回归 + 旧实现注入反证 |
| 真缺陷·待行为决策 | 1 | `#18` 不存在的图像路径被当 base64 静默成功；修法改变对外行为，立 P1-9，不擅改 |
| 断言过期 | 10 | 立 P1-9：按当前契约改断言或标 xfail（多为 Ministers 委托语义迁移） |
| 测试自身缺陷 | 6 | 立 P1-9：`test_create_mock_dna` helper 命名错误，6 个用例调用不存在的 `create_mock_dna` |
| 状态污染/顺序依赖 | 2 | 立 P1-9：`#2` 单例 `registered_bees` 跨用例累积；`#9` 磁盘 `report_*.json` 残留 |
| 依赖缺失 | 0 | 无（推翻 R3 举例） |

## 三、关于「修还是标 xfail」的显式决定

P1-8 要求「决定修或标记 xfail」。本轮决定：**不做批量 xfail**，理由：

1. 批量 xfail 会把 19 个失败从可见信号变成静默噪声，恰恰复现 P1-8 要解决的
   「只跑 `tests/` 子集导致这 20 个失败被静默忽略」；xfail 只是把忽略挪进 pytest。
2. 19 条里混着**真缺陷**（`#18`）与**产品行为决策**，标 xfail 等于替用户把行为拍板，
   违反「不依赖用户决策 / 不擅改行为」的约束。
3. 更有效的替代：把失败集合变成**结构化台账 + 一条可复现对账命令**
   （`tools/triage_check.py`，R-20261104-02）。任何新失败、静默修复、fixed 回归都会
   让命令 FAIL，且命令自证它检查了什么。这样 19 条既不消失、也不需要逐个标 xfail。

由此 P1-8 拆成两个去向（R-20261010-15 的「修 / 立下一轮」）：
- **修**：`#10` 唯一确定的产品缺陷，本轮完成；
- **立下一轮 P1-9**：其余 19 条按分类批量处理（改断言 / 修 helper / 加 fixture 隔离 / 单独判定 `#18`）。
- **待人工**：0 条。本轮未产生新的需用户决策项。

## 四、本轮唯一产品缺陷与验收证据

`hive/core/species_blueprint.py`：模块第 24 行已有 `import random`，但 `mutate()`、
`crossover()` 内共 3 处函数级 `import random` 把它绑成函数局部变量；双亲能力互斥时
`if spec1 and spec2` 分支未执行，`elif spec1` / `elif spec2` 里的 `random.random()`
抛 `UnboundLocalError`（:545）。**静态三项全 PASS 时仍漏掉** —— 本项目
「静态全绿 ≠ 运行时可用」的**第四次实证**（R1 缺 import os、R2 路径默认值、
R3 根锚点、R4 函数内 import 遮蔽）。

修法：删除 3 处函数级 import，模块级导入成为单一真源，并加注释与 AST 守卫。

验收证据：
- `tests/core/test_species_blueprint_crossover.py` **6 passed**（真实调用：互斥/共有能力、
  亲和度分支、`mutate()`；AST 单一真源守卫；运行时构造旧实现反证）。
- **旧实现注入反证**：把 `import random` 注回真实源码 → `5 failed, 2 passed`，
  精确复现 `UnboundLocalError`；`finally` 字节级复原 `sha256 f6115dac45e01e55 SAME`
  → 复原后 `7 passed`。
- 原失败用例 `TestSpeciesBlueprint::test_crossover` 由 FAIL 转 **PASS**。
- 反证锚点必须先做 CRLF→LF 归一化：本仓库该文件是 CRLF，直接用 LF 锚点会让
  反证**静默失效**（首次实现即因此报 `注入锚点已漂移`），已修正并在测试中留注释。

## 五、附带侦察发现（立 P1-10，本轮不改）

R4 实测三条口径事实，**推翻 R3 台账「台账与 CI 若只跑 `tests/` 子集，这 20 个失败
会被静默忽略」的假设**，也再次印证 R-20261010-19：

1. `pyproject.toml:113` 的 `testpaths = ["hive/tests"]` ⇒ 裸 `pytest -q` 只收集
   **475 项 `hive/tests/`**，不是「全仓库」。R3 记的「全仓库 455 passed / 20 failed」
   其实是这个集合（475 = 455 + 20），数字对、口径名错。
2. CI（`.github/workflows/test.yml:45/80`）跑的是 `pytest hive/tests/ -v` ⇒ CI
   **看得到这 20 个失败**，反而**完全没跑**维护中的 `tests/` 子集（`--collect-only`
   实测 139 项，含 R1~R4 新增的全部回归）。漏的是新回归，不是历史失败。
3. 两套一起 collect 会失败：`pytest tests/ hive/tests/` → `ERROR collecting
   hive/tests/test_dna.py: import file mismatch`（`tests/core/test_dna.py` 同名、
   两侧都无 `__init__.py`）→ 597 collected + 1 error、退出码 2。故目前**无法**用
   一条 pytest 命令覆盖两套。

本轮**不改** `pyproject.toml` / CI / 文件命名：改验收口径属 R6 元审视的既定问题②，
且涉及收集范围与测试组织方式，提前越权修订会覆盖 R 轮的判据。按
R-20261010-15 登记为 **P1-10（立下一轮 / R6 输入）**。

## 六、边界

- 本轮**没有**把项目状态写成「可用」：P0 未清零、`--cycles 2` 仍 FAIL、依赖未装齐。
- `tools/loop_check.py` 的验收口径本轮**不改**：是否把「带范围的 pytest 子集 +
  全仓库失败对账」纳入固定口径，属 R6 元审视的既定问题②，不提前越权修订。
- 本报告与台账只做分诊与守卫，不替用户决定产品行为。
