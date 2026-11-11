# hive-mind · 长效推进台账（LOOP.md）

> 本文件是项目的**唯一待办源**，随仓库版本化。由自动化「hive-mind · 长效推进轮」（id `d1995554-83e7-498a-8a34-b85759cdb95a`，每 6 小时一轮）读取、推进、追加日志。
> 不要另开一份散落的 TODO；所有结论、验收证据、下一步都写回这里。
> 建立：2026-11-10

---

## 一、边界与原则

- 主工作区固定 `D:\Hive`（唯一可写源码树）；`D:\repos\hive-mind` 是**只读参照镜像**，只做 `pull --ff-only`，禁止在里面改代码。
- 远端 `github.com/xtwp1024/hive-mind.git`，默认分支 `master`。
- 每轮**至少完成一个实质动作并留痕**；没有可推进项时只写一行说明，不硬凑改动、不刷「一切正常」的空转汇报。
- 不删除文件、不 `git reset --hard`、不强推。破坏性操作一律转「待人工」。
- `vendor/` 只放行 `pwp_qa-0.2.3-py3-none-any.whl` 一个版本（**P0-1 / R8 已由用户授权按方案 A 入库**）。拆包核实：单文件 60KB、`pwp_qa.py` 4702 行、MIT 许可、声明公开主页、**无硬编码凭证或内部地址**；workflow 注释里「私有部署」只是「未发布到 PyPI」的意思（公共 PyPI 实测 404）。R0 台账称其「私有工具、公开仓库有泄露顾虑」是照抄注释未拆包，**已更正**。其余 8 个历史 wheel 继续排除，**新增或升级 wheel 仍需用户确认**，不要批量 `git add -f vendor/`。
- 单轮预算控制在 40 分钟内，宁可少做也不拖轮。

## 二、固定验收口径（每轮必跑，先于任何改动）

```bash
cd D:/Hive
git status --porcelain                      # 工作区基线
python tools/loop_check.py                  # 静态体检：见下
```

`loop_check.py` 给出 **三项固定口径 + R6 新增一项附加守卫**，任一失败即本轮转为修复，不做新功能：

1. **语法**：所有入库 `.py` 通过 `py_compile`
2. **模块闭合**：所有 `from hive.* / import hive.*` 目标在检出树中可解析
3. **镜像一致**（R1 修订）：`D:/repos/hive-mind` HEAD 必须 == `origin/master`；主工作区 HEAD 允许**领先**（待推送的正常推进态），但落后/分叉仍 FAIL。旧口径要求三方严格相等，会在每轮 commit 后、push 前误报基线已坏。
4. **[+] 验收链闭合**（R6 新增，不改前三项编号与语义）：声明的验收工具（`tools/smoke_runtime.py` / `tools/triage_check.py`，唯一真源是 `loop_check.py` 的 `ACCEPTANCE_TOOLS`，本表只做指针）必须在位且可编译；并以子进程跑 `triage_check.py` 静态自证，要求 `RESULT: PASS`。**这把「open 历史失败的分诊台账是否漂移」纳入每轮一条命令的硬口径**，避免只跑静态三项就误以为全绿（沿用 `R-20261104-02`）。**（R7 口径修订）**：`loop_check.py` 的 open 数量与 SCOPE 行不再写死数字，改由 `ledger_open_count()` 从 `triage_check.py` 的 `EXPECTED_OPEN`（唯一真源）静态 AST 派生——每修一个历史失败，写死的 13 就会变成假陈述；派生读不到时输出「open 历史失败」而非编一个数字（投影点 `test_loop_check_acceptance_chain.py` 已改动态断言并加 7/3/缺失三态反证）。

**四层验收口径（R6 元审视定夺；每轮按层记录，数字必须带集合范围）**：

| 层 | 命令 | 范围与判据 |
| --- | --- | --- |
| ① 静态 | `python tools/loop_check.py` | 三项 + 附加守卫；PASS ≠ 服务能跑 |
| ② 运行时 | `python tools/smoke_runtime.py --cycles 1`（须用装了依赖的托管 venv） | 1 周期 PASS；依赖缺失时退出码 2 显式 SKIP，**不算通过**；`--cycles 2` 会暴露 P0-3，目前预期 FAIL |
| ③ 测试 | `pytest tests/ -q`（子集）与 `pytest hive/tests/ -q`（全量）**分两条命令** | `pyproject.toml:113` 的 `testpaths=["hive/tests"]` ⇒ 裸 `pytest -q` 只收后者；两集合因 `test_dna.py` 同名 + 两处均无 `__init__.py` 无法一起 collect（R4 实测 `import file mismatch`、退出码 2）。**台账写 pytest 数字必须带集合范围**（R3/R4 两次口径漂移） |
| ④ 对账 | `python tools/triage_check.py --report <全量报告>` | `FAILED` 集合必须恰好等于 `LEDGER` 里 `status=open` 的集合；静态自证（不带 `--report`）已由 ① 的附加守卫每轮兜住 |

> **R6 定夺：不调整** `testpaths` / CI / 测试命名。三条实测理由：① 合并两集合会触发 `import file mismatch`（退出码 2），反而把 CI 弄坏；② 改 `.github/workflows/test.yml:45/80` 或 `pyproject.toml` 超出 R 轮「只修订本文件与 `loop_check.py`」的边界；③ 改名会动 `hive/tests/` 历史测试，属功能改动。故以「两条显式命令 + ① 的附加守卫」覆盖，不追求一条命令并集；P1-10 保留 open 但**结论已定**，后续若要动 CI/命名须另开独立轮并先验证 collect 不再冲突。

静态体检只保证「能解析、能编译 + 验收链工具自洽」，**不等于服务能跑，也不等于 open 历史失败已修**（open 数量以 `tools/triage_check.py` 的 `EXPECTED_OPEN` 为准，勿写死）。R1 实证静态三项全 PASS 时仍存在 3 处只在运行时暴露的缺陷，R2/R3/R4 又三次实证静态层对路径默认值 / 根锚点 / 局部 import 遮蔽无感（**R6 定夺：维持分层，不补通用静态契约检查**——4 次实证里 2 次是语义与运行时形态，静态层天然不可见；R4 已给 `species_blueprint.py` 配专用 AST 守卫 + 旧实现注入反证，每轮「专用回归 + 1 周期冒烟」已形成稳定覆盖，硬塞通用契约检查只会引入误报或恒绿项）。未跑通前不得把项目状态写成「可用」。

## 三、当前快照（2026-11-12 R9 刷新）

> **易变值不写死**（R1 修订，源自 R-20261010-19「HEAD 是会移动的状态」）：
> HEAD / 提交数 / 领先数一律以 git 实测为准，本表只记**不可变的开工基线**与**验收结论**。
> 开工前跑：`git rev-parse HEAD` / `git rev-list --count HEAD` / `git ls-files | wc -l`。
> R0 快照曾漏记 `12b41ef` 回填提交；R1 回填时又让「33 / 领先 1」立刻过期，故改为指针式。
> 不可变历史：**R1 功能主提交 `ec0930c`**（7 文件 +358/-33）、**R2 功能主提交 `fb757cd`**（2 文件 +153/-3）、**R3 功能主提交 `696b76e`**（7 文件 +392/-42）、**R4 功能主提交 `d92ef5f`**（4 文件 +554/-6）、**R5 功能主提交 `d2d44c2`**（3 文件 +173/-16）、**R7 功能主提交 `c40dfbc`**（6 文件 +353/-13）、**R8 功能主提交 `c6266a4`**（4 文件 +21/-13，含 1 个 60491 B wheel）、**R9 功能主提交 `5e57aef`**（4 文件，CI 配置修复 + 11 项回归）；
> 完整列表用 `git log --oneline 12b41ef..HEAD` 查，不在本表逐个枚举（枚举本身会过期）。

| 项 | 值 |
| --- | --- |
| 开工基线 HEAD | `12b41ef`（R1 轮首实测，远端 `origin/master` 同值；R2 开工时本地领先 4 个待补推提交） |
| 当前 HEAD | **以 `git rev-parse HEAD` 为准，不写死**：R2 功能主提交是 `fb757cd`，但本表自身的回填提交会让 HEAD 再前移一位（R1 正是漏记这类回填导致快照漂移）。查完整链用 `git log --oneline 12b41ef..HEAD` |
| 提交数 | 以 `git rev-list --count HEAD` 为准（**回填提交会 +1，不写死具体值**） |
| 跟踪文件 | 以 `git ls-files \| wc -l` 为准（R4 新增 3 个入库文件 `tests/core/test_species_blueprint_crossover.py`、`tools/triage_check.py`、`docs/reports/LEGACY_TEST_TRIAGE_R4.md`，.py 328 → **330**、跟踪文件 402 → **405**；R5 新增 1 个入库文件 `tests/core/test_dna_helper_naming_regression.py`，.py 330 → **331**、跟踪文件 405 → **406**；R6 新增 1 个入库文件 `tests/utils/test_loop_check_acceptance_chain.py`，.py 331 → **332**、跟踪文件 406 → **407**；R7 新增 1 个入库文件 `tests/core/test_state_isolation_r7.py`，.py 332 → **333**、跟踪文件 407 → **408**；R8 新增 2 个入库文件（`vendor/pwp_qa-0.2.3-py3-none-any.whl` 与 `tests/utils/test_loop_check_p0_scope.py`），.py 333 → **334**、跟踪文件 408 → **410**；R9 新增 1 个入库文件 `tests/utils/test_workflow_yaml_r9.py`，.py 334 → **335**、跟踪文件 410 → **411**（收尾以 `git ls-files` 实测为准）。⚠️ `docs/reports/` 被 `.gitignore` 的 `reports/` 忽略，报告须 `git add -f`） |
| 工作区 | 干净（R4 功能提交 `d92ef5f` 后、台账回填前实测；`hive/memory/cognitive_state.json` 每轮跑 pytest 会被改写，已 `git checkout --` 复原；R6 开工实测干净，并定夺：该文件**确已被 git 跟踪**（`git ls-files --error-unmatch` 命中），故 `.gitignore` 对它无效，维持「跑后复原」固定流程，不改忽略规则、不做 `git rm --cached`，详见「六、R6 元审视结论」③） |
| 推送状态 | **以 `git rev-list --count origin/master..HEAD` 实测为准，不写死**：本行曾被 R2 回填提交自己顶成过期值 ⇒ 改为纯指针式（R1 同款坑的第二次出现）。**口径**：本地领先是「已 commit、待推送」的正常推进态，体检按 R1 修订允许落后以外的领先；每轮**先补推**再同步镜像。R3 轮首先补推 R2 积压 2 提交：直连 / 代理各一次均失败（详见 R3 行），远端未分叉；功能提交后本地领先以实测为准。**R9 实证（通路同轮内二次翻转）**：轮首裸直连 fetch/push 一次成功 `b858ce3..cd19c4a` 补推 R8 积压 3 提交并同步镜像；R9 功能提交后 fetch/push 立即 `Recv failure`，改跑四层验收后轮末重新探测，裸直连又通、一次成功 `cd19c4a..5e57aef`；代理端口全程不通、`gh api` 全程可用、远端始终未分叉 |
| 静态体检 | `PASS`（**R9 实测 335 个 .py / 119 个 hive.\* 目标 0 缺失 / 镜像与 `origin/master` 一致、本地允许领先 / R6 附加守卫 0 问题 / SCOPE 动态输出 11 个 open 与派生 P0 状态**；R9 新入库 1 个 .py 使计数 334 → 335，一律以 `python tools/loop_check.py` 实测为准；HEAD 是会移动的状态，不写死，按 R1 指针式口径以 git 实测为准） |
| 运行时 | **已首次跑通且有界冒烟 PASS**：`smoke_runtime.py --cycles 1` EXIT=0、R9 实测 209 行、CPU 16 cores、正常收尾（R5~R9 复跑确认未回归）；`--cycles 2` 仍 FAIL（P0-3）。**R8 新增 CI 命令链本地实证**：临时目录用托管 venv 装 `vendor/pwp_qa-0.2.3...whl` + `tree-sitter` 后原样跑 `baseline establish / baseline check / trend --backend mock -n 3`，三条 EXIT 全 0（详见 P0-1 与 P1-11） |
| 验证层级 | 静态 PASS + 1 周期冒烟 PASS + **`tests/` 子集 169 passed**（R8 158 + R9 新增 11，退出码 0）+ **`hive/tests/` 全量 11 failed/463 passed**（474 collected；R6 为 13 failed/461 passed，恰少 2 条、无新增失败）+ `triage_check.py` 静态/对账双 PASS（静态 `20 条 / open 11 / fixed 9`、对账 `FAILED 11 == open 11`）；**完整运行时/服务仍不可用**（依赖未装齐；**P0-1 已于 R8 完成**，P0-2/P0-3 未清）。**CI 云端（R9）**：三份 workflow 已接通 `master`，`test.yml` 恢复解析（name 由路径变回 `Tests`），PWP-QA `cd19c4a` run `43092432401` = **success**（9 step 全绿）；接通后 CI 预计仍会红，见 P1-12。⚠️ **R4 口径修正仍成立**：R3 写的「全仓库 `pytest -q`」其实因 `testpaths=["hive/tests"]` 只收 `hive/tests/`，**不是全仓库**；CI 也只跑 `hive/tests/`，反而漏掉维护中的 `tests/` 子集。两集合因 `test_dna.py` 同名无法一起 collect。见「二、四层验收口径」（R6 定夺）、**P1-10**；台账写 pytest 数字必须带集合范围 |
| 资料库档案 | `https://www.workbuddy.cn/space/d/URL2Tg73gmbxGrjut4nPoj` |

2026-11-10 R0 入库 4 个提交：`d14135d` 运行时健壮性（僵死 Worker 回收、缓存落 D 盘、哈希嵌入）、`db74a7a` 可视化（事件流 + swarm.html + warmup 回退）、`ee24c6f` CI 版本 bump、`3478df9` 本台账与体检脚本，外加 `12b41ef` 回填快照。清理的是 10/31–11/01 遗留的 15 项未提交改动。

## 四、待办队列

### P0 · 阻塞「可用」

**P0-1　✅ R8 已完成 — CI 两处断点（wheel 缺失 + 分支不匹配）**

**R8 实测发现的是两个断点，不是一个**（R0 只看到第一个）：
1. `.github/workflows/pwp-qa.yml` 两处 `pip install ./vendor/pwp_qa-0.2.3-py3-none-any.whl`，而 `vendor/*.whl` 被 `.gitignore:79` 排除、`git ls-files vendor` 为空 ⇒ 云端检不到 wheel。从 10-28 建库起就存在。
2. workflow `on.push.branches` / `on.pull_request.branches` 写的是 `[main, develop]`，**本仓库默认分支是 `master`** ⇒ 推 master 根本不触发。即便修好 wheel 也白搭。R0 完全漏了这条。

**决策依据（2026-11-12 用户授权「推进本项目」，按上轮建议的方案 A 执行）**：拆包核实 wheel 非私有（MIT / 无凭证 / 无内部地址，`pwp-qa` 在公共 PyPI 实测 404），故 B「内部源」不成立；C 会丢掉这套多宇宙 prompt 分析，而仓库另有 `lint.yml`、`test.yml` 兜底，但不等于等价替代。**A 改动最小且保住 workflow 原意**。

- 落地：`.gitignore` 用 `!vendor/pwp_qa-0.2.3-py3-none-any.whl` 单版本白名单放行（不 `git add -f`、不带入其余 8 个历史 wheel）；workflow 两处分支改 `[master]`，两处「私有部署」注释改为事实描述。
- **云端判据已落地（R9 补记）**：`cd19c4a` 首次被 `master` 触发的 PWP-QA run `43092432401` = **success**，9 个 step 全绿（含 `baseline establish` / `drift check` / `trend`），两处断点修复在云端生效；本机临时目录 mock 跑通得到云端复现。
- **边界**：修完只保证「任务能触发 + 工具能装上」，**不等于 CI 绿即 QA 通过**（见 P1-11：漂移告警仍以 0 退出、且未上传报告产物）。`5e57aef` 起三份 workflow 全部接通 `master`，后续以 Actions 实际日志为准。CI 若仍失败，按新证据另立待办，不回滚本项。

**P0-2　运行时首次跑通（R1 已达成「有界冒烟」，但项目仍非可用）**

R1 实测结论（`2026-11-10`）：
- **通路**：`pypi.org` 直连、代理 7897、清华源全部 `SSLEOFError`/无匹配；**阿里源 `https://mirrors.aliyun.com/pypi/simple/` 可用**。装齐启动链轻量依赖 `pydantic-settings 2.16.2 / structlog 26.1.0 / psutil 7.2.2 / typing-inspection 0.4.5`（`httpx/pydantic/requests/numpy/python-dotenv` 本就在 venv）。
- **最小 smoke 真实跑通**：托管 venv 下 `HIVE_SMOKE_CYCLES=1 python main.py` → EXIT=0、209 行输出、10 个 `返回载荷: SUCCESS`、`Hive cleanup complete`、`Final Metrics Summary` 正常收尾。`smoke_runtime.py --cycles 1` = PASS。
- **装不上的依赖（如实登记，不假报）**：`chromadb / openai / ollama / psycopg2 / flask / scikit-learn` 本轮**未装**——启动链扫描（55 个 hive 模块）显示 `main.py` 可达路径只需 `chromadb httpx psutil pydantic pydantic_settings requests structlog`，其中 `chromadb` 仅在 `vector_vault.py` 顶层 import、未被 1 周期 warmup 路径实例化，故冒烟能过；**但完整功能（语义记忆/Web 服务/DB）依然未验证**。
- **静态 PASS 漏掉的 3 处真实运行时缺陷**（本轮实证，非推测）：
  1. `hive/utils/monitor.py` 缺 `import os` → `get_snapshot()` 抛 `name 'os' is not defined`，被 `except` 吞成全 0 默认值（日志 `CPU/Memory: unknown` 即此）。**已修**。
  2. `ResourceMonitor.get_system_info()` 不返回 `cpu_count`/`memory_total_gb`，而 `main.initialize_system` 读这两个 key → 日志恒显 unknown。**已修**（现为 `CPU: 16 cores`）。
  3. `engineer.assign_tasks` → `_perform_active_hunting` → `GithubHunter.hunt_sync` → `asyncio.run(self.hunt())`，在主循环 `asyncio.run(main())` 内嵌套调用 → 每周期 `RuntimeError: asyncio.run() cannot be called from a running event loop`，静默回退 warmup。**未修，拆为 P0-3**。
- 验收证据：`tools/smoke_runtime.py`（1 周期 PASS / 2 周期 FAIL 精确捕获第 3 条）、`tests/utils/test_r1_runtime_regressions.py` 5 项 + `test_loop_check_mirror.py` 4 项 = 9 passed、旧实现注入反证（删掉 `monitor.os` 后精确复现 `name 'os' is not defined` 与全 0 默认值）。
- 注意：base 解释器（`~/.workbuddy/binaries/python/versions/.../python.exe`）没有任何第三方包，别在它上面装。

**P0-3　主动狩猎在主循环内必然失败（R1 新增，未修）**
`hunt_sync` 用 `asyncio.run()`，而 `engineer.assign_tasks()` 是同步函数、从 `asyncio.run(main())` 的事件循环里调用 ⇒ 第 2 周期起每周期抛 `RuntimeError` 并回退 warmup，`smoke_runtime.py --cycles 2` 因此 FAIL。
修法涉及行为决策，**不擅自改**：① 把 `hunt_sync` 改为检测 running loop 并用 `loop.run_until_complete`/独立线程；② 让 `assign_tasks` 走 async 链路；③ 主循环内显式跳过狩猎（会改变「主动狩猎」语义）。另注：狩猎默认 5 关键词 × 6s 间隔、限流冷却 120s，冒烟若真触发会拖慢，需一并定超时口径。

### P1 · 质量与债务

- **P1-1　✅ R2 已完成（`fb757cd`）** 事件流根目录去掉跨平台歧义默认值。改为 `PROJECT_ROOT`（仓库根）+ 单一 `HIVE_HOME` 读取（空串/空白按未设置），与读取方 `web_dashboard.py` 的 `BASE_DIR` 对齐；新增 11 项回归，含旧实现注入反证与运行时真实落盘解析。详见「八、推进日志」R2 行。
- **P1-2** 版本 1.0.0 只写在 README / pyproject，远端没打 tag；确认版本口径后补 `git tag v1.0.0`。
- **P1-3** `hive/core/vector_vault.py` 换成 SHA256 哈希嵌入后，语义检索能力实质下降（哈希无语义）。需要评估：是保留作离线兜底，还是加回可配置的可选语义嵌入。
- **P1-4** `obsidian_notes/` 9 篇开发记录（五阶段进化路线图、代码改进总结、测试汇总）是项目史一手材料，整理进 `docs/` 后入库。
- **P1-5** `vendor/` 现有 **9 个** wheel（0.1.0 / 0.1.1 / 0.1.2 / 0.1.3 / 0.1.4 / 0.2.1 / 0.2.2 / 0.2.3 / 0.2.4；R0 台账写「8 个」、R8 说明里写「另外七个」均为**未数清的口径错误**，已按 `ls vendor/*.whl` 实测更正），CI 只用 0.2.3 且已单独入库；确认其余 8 个的保留策略，别让版本继续无序堆积。
- **P1-6** `web/swarm.html` 默认深色底 + emoji 标题，与既定「白主 / 黑 / 蓝辅、不用 emoji 改用开源图标、每个元素有出现-过渡-消失动画」规范不一致；按规范改版。
- **P1-7　✅ R3 已完成（`696b76e`）** 散落 4 处的 `HIVE_HOME` 根锚点收敛成单一真源 `hive/paths.py`（`PROJECT_ROOT` / `resolve_hive_home` / `resolve_events_file`）；`hive/__init__.py`、`settings.py`、`librarian_bee.py`、`web_dashboard.py` 四处改为调用真源，`event_stream.py` 退化为转发薄壳并保留 P1-1 回归的可注入符号。新增 32 项回归（判据唯一性、消费点投影守卫、`__all__` 派生守卫、写读同路径、旧实现注入反证）。按 `R-20261024-01` 判定：`monitor.py` 取盘符、`.env` 的 `BASE_DIR`、3 处非根锚点 `cwd()` 语义不同，**不合并**并显式登记。详见「八、推进日志」R3 行。

- **P1-8　✅ R4 已完成（`d92ef5f`）** 20 个历史失败逐条分诊 + 修唯一真缺陷。产出 `tools/triage_check.py`（**唯一真源** `LEDGER` 20 行 + 一条可复现校验命令：台账完整性 / AST 收集闭合 / 与 pytest FAILED 集合对账，自证 SCOPE）与 `docs/reports/LEGACY_TEST_TRIAGE_R4.md`（导读，不复制台账）。分类：真缺陷·已修 1 / 真缺陷·待行为决策 1 / 断言过期 10 / 测试自身缺陷 6 / 状态污染·顺序依赖 2 / **依赖缺失 0**。修了 `species_blueprint.py` 函数级 `import random` 遮蔽（静态三项全 PASS 仍漏，**第 4 次实证**），配 6 项回归 + 旧实现注入反证。全量 `hive/tests/` 从 20 failed 降为 **19 failed/456 passed**。**显式决定不批量 xfail**（会把可见失败变成静默噪声，且 `#18` 涉及对外行为决策）；19 条按去向立 P1-9。
  - **口径更正（R4 实测，推翻 R3 举例）**：R3 写的「`brain_config.json` 缺失导致 BrainInterface 加载失败」在 20 条实测异常中 **0 条命中**，依赖缺失实测为 0；R3 写的「全仓库 `pytest -q` = 455 passed/20 failed」其实是 **`hive/tests/` 单集合**（475 = 455+20），因 `testpaths=["hive/tests"]`。详见 P1-10。
  - **本轮不改** `tools/loop_check.py` 的固定验收口径：是否把分诊对账纳入固定口径属 R6 元审视既定问题②，不提前越权。

- **P1-9（R4 分诊产出；R5 已完成 ②、R7 已完成 ③，余 11 条 open）** P1-8 余下 19 条按分类批量处置：① 断言过期 10 条按当前契约改断言或标 xfail（多为 Ministers 委托语义迁移，改前需确认契约是「已变」还是「退化」）——**未修，留 open**；② **✅ R5 已完成（`d2d44c2`）** 测试自身缺陷 6 条：helper `test_create_mock_dna` 改名为 `create_mock_dna`（去掉 `test_` 前缀以免被 pytest 误收为用例），6 个调用点无需改动；`triage_check.py LEDGER` #11~#16 转 `已修(R5)`/`fixed`，并把台账升级为「已修」前缀判定以支持多轮陆续修复；新增 `tests/core/test_dna_helper_naming_regression.py` 4 项回归（AST 投影守卫 / 运行时真实调用 6 用例 / 旧实现注入反证 / 字节级 sha256 SAME）。验收：修复前 `6 failed` → 修复后 `6 passed`，全量 `hive/tests/` 19 failed → 13 failed；③ **✅ R7 已完成（`c40dfbc`）** 状态污染 2 条分别加 fixture 隔离：`get_swarm_communication()` 单例在 `setUp` 快照-清空 `registered_bees` + `addCleanup` 复原，`CapabilityGapAnalyzer` 每用例把 `_persistence_path` 重定向到独立 `TemporaryDirectory` + `addCleanup` 回收；只改测试 fixture、不改产品行为，两处字节级保持 100% CRLF。新增 `tests/core/test_state_isolation_r7.py` 6 项回归（AST 隔离块守卫 / 污染态运行时真实调用 / 剥离隔离后旧实现注入反证 / 字节级 sha256 SAME，全程不改写真实源码）；并把 `loop_check.py` 写死的「13 个 open」改为从台账派生，避免修一条就产出一个假陈述。实测整文件 21 passed（修复前 1 failed/20 passed）、整类 8 passed，全量 13 failed → 11 failed；④ `#18` 单独判定 `ImageEncoder` 对不存在路径静默成功的修法（改变对外行为，需决策）——**未修，留 open**。逐条证据见 `tools/triage_check.py LEDGER`（**唯一真源**，现 `20 条 / open 11 / fixed 9`：断言过期 10 + `#18` 行为决策 1），**不要另开散落 TODO**。

- **P1-10（R4 侦察产出，未修 / R6 输入）** 验收口径三条实测事实：① `pyproject.toml:113` `testpaths=["hive/tests"]` ⇒ 裸 `pytest -q` 只收 475 项，不是「全仓库」；② CI（`.github/workflows/test.yml:45/80`）跑 `pytest hive/tests/`，**能看到这 20 个失败、却完全没跑维护中的 `tests/` 子集（139 项，含 R1~R4 全部回归）**；③ 两集合一起 collect 会因 `test_dna.py` 同名 + 无 `__init__.py` 报 `import file mismatch`（597 collected + 1 error，退出码 2），目前无法一条命令覆盖两套。**R6 元审视已定夺（结论见「二、四层验收口径」）：不调整收集范围/CI/命名**，改以「两条显式 pytest 命令 + `loop_check.py` 附加守卫每轮兜住分诊台账」覆盖；本项保留 open 仅登记「后续若要动 CI/命名须另开独立轮并先验证 collect 不再冲突」，不再是待决策输入。附带：`docs/reports/` 被 `.gitignore:80` 的 `reports/` 忽略，但已有 12 份报告是入库的 ⇒ 新报告必须 `git add -f`，否则交付物静默丢失（R4 已用 `-f` 入库）。

- **P1-11（R8 侦察产出，未修）** `pwp-qa.yml` 是**报告型任务，不是质量门禁**。R8 在临时目录实装 0.2.3 并原样跑三条命令，三条 EXIT 全 0，但：`baseline check` 在 mock 下明确输出「⚠ 1 个 prompt 发生漂移」后**仍以 0 退出**；`baseline establish --backend mock -n 3` 的 `-n 3` 不作用于固定的 5 条 `BASELINE_PROMPTS`（工具自身行为，非本仓库缺陷）；`trend --limit 10` 只打印表格。⇒ **CI 绿 ≠ QA 通过**，且 workflow 未上传任何报告产物。候选去向（涉及对外行为，不擅改）：① 加显式阈值判定使漂移非零退出；② 把 baseline/trend 结果作为 Actions artifact 上传供人看；③ 确认无门禁价值后降级为手动 `workflow_dispatch`。**R9 已观察**：`cd19c4a` 的 PWP-QA run `43092432401` 实测 `success`（9 step 全绿），本机 mock 结论得到云端复现；故「报告型任务、CI 绿 ≠ QA 通过」判断不变，该 run 同样未上传任何报告产物。后续若加阈值门禁或 artifact 属对外行为变更，仍需用户决策。

- **P1-12（R9 侦察产出，未修）** 三份 workflow 接通 `master` 后，CI 会**真实地红**，这是既有债务暴露而非 R9 配置缺陷。实测证据：① 本地模拟 `test.yml` 的 fast tests 命令（去掉尚未声明的 `--timeout` 后）`pytest hive/tests/ -q -m "not slow and not integration" -x` → **1 failed / 51 passed**（136.96s），首个失败 `test_cognitive_bee_enhanced.py::TestReportToEvolution::test_report_to_evolution_coordinator_import_error`（`assert True == False`），属 P1-9①「断言过期 10 条」；② `requirements.txt` 含 `chromadb` / `psycopg2` 等重依赖，CI matrix 为 Python 3.9~3.12，云端 `pip install` 是否全版本可装**未实测**（R1 本机就装不上）；③ `integration` / `docs` / `docker` 三个 job `needs: test`，test 红时会被跳过，其引用的脚本与 Dockerfile 本轮不验证。**处置**：不批量改断言、不 xfail、不改 `continue-on-error` 掩盖；按 P1-9① 逐条核「契约已变 vs 行为退化」后再收口，并单独观察一次云端 Install dependencies 结果。

### P2 · 清理与归档

- **P2-1** `.temp_aden_hive` 是带独立 `.git` 的副本，确认后清理，避免和主工作区混淆。
- **P2-2** `Hive_Standalone` / `Hive_V1_Backup` 在 `D:\_D盘整理_20260930\archive\`，比对版本差异再决定是否合并。
- **P2-3** 运行期账本 json（mining_ledger / discovery_history / science_journal）已排除入库，确认是否需要单独归档策略。

### 待人工（自动化不要碰）

- ~~是否把私有 pwp-qa wheel 放进公开仓库~~ **已于 R8 决策并落地**：wheel 非私有，放行 0.2.3 单版本；**但新增/升级任何 wheel 仍需用户确认**
- 是否调整项目公开/私有状态
- 归档副本的去留
- P0-3 `hunt_sync` 修法（涉及主动狩猎语义的行为决策）
- ~~是否为 hive-mind 注册独立 deploy key~~ **R9 结论：当前不需要，保持不新增凭证**。R8 传输层三路失败只是瞬时抖动，R9 轮首裸直连即恢复并一次推成功；同轮内又经历「成功→失败→成功」二次翻转。仅当 HTTPS 长期不可用且用户明确授权时，才考虑新生成专用密钥对（原诊断备查：本机 `~/.ssh/id_ed25519` 是「星社仪表盘」仓库级 deploy key，跨仓库注册报 422）。
- **补推通路诊断（非决策，仅备查，R9 更新）**：R8 轮末三路失败 ⇒ R9 轮首裸直连恢复、一次补推 3 提交 `b858ce3..cd19c4a`；R9 功能提交后 fetch/push 又 `Recv failure`，跑完四层验收后轮末重新探测再通、一次 `cd19c4a..5e57aef`。全程代理 7897 不通、`gh api` 可用、远端始终未分叉。⇒ **传输层与 API 通路分离 + 通路不跨时刻**，按「五、每轮流程」第 4 步各试一次即止，别写长期配方（`R-20261015-05`）。

## 五、每轮流程

1. 读本文件「四、待办队列」，取优先级最高且**可独立完成、不依赖待人工决策**的一项。
2. 跑「二、固定验收口径」建立基线；基线已坏 → 本轮只修基线。
3. 实施改动 → 复跑验收 → 记录证据（命令 + 输出摘要 + 提交 sha）。
4. 本地 commit。**推送前**确认远端未分叉；推不动就记一行原因，不反复重试。
5. 更新本文件：待办状态、当前快照、追加「八、推进日志」一行。
6. 里程碑轮（P0 清零、版本 tag、运行时首次跑通）额外更新资料库档案文档。
7. 末尾追加工作区日期日志 `C:/Users/Administrator/WorkBuddy/2026-10-28-16-14-39/.workbuddy/memory/YYYY-MM-DD.md`（append-only）。
8. **（R6 新增）运行时副产物复原**：跑完 pytest / 冒烟后若 `git status --porcelain` 出现 `hive/memory/cognitive_state.json` 被改写，执行 `git checkout -- hive/memory/cognitive_state.json` 复原。该文件已入库，忽略规则无效，不删跟踪。
9. **（R6 新增）新报告入库**：`docs/reports/` 被 `.gitignore:80` 的 `reports/` 忽略，新增报告须 `git add -f` 且用 `git ls-files` 回读确认，否则交付物静默丢失。

> R2 口径修正：本节原写「七、推进日志」，但实际推进日志是**第八节**（第七节是复利回链）。已改正。
> 另：本节第 4 步「推送前确认远端未分叉；推不动就记一行原因，不反复重试」在 R2 首次**成功**兑现推送，故补一条：推送成功后须 `git -C D:/repos/hive-mind pull --ff-only` 同步镜像并复跑体检确认三方一致（R1 因推送失败未走到这一步）。

## 六、元审视（随项目演化修订本机制）

每累计 6 轮触发一次 R 轮，不推进功能，只回答三个问题并直接改本文件：

- 验收口径是否已经覆盖不住新风险？（例：引入 FastAPI 后要不要加路由 smoke）
- 待办队列有没有过期项 / 重复项 / 需要升级优先级的？
- 本文件里的路径、版本、自动假设是否随项目漂移？

触发条件变化时（换了主工作区、新增服务、依赖大改）立即触发，不等满 6 轮。
**这条是长效推进不腐化的关键：机制本身要跟着项目改，不能只让项目跟着机制转。**

> **当前轮次指针（唯一状态源，R9 收尾）**：R0 建档 + R1 + … + R9 = **10 轮**，**R6 元审视已完成、R7/R8/R9 持续功能推进**；**下一轮 R10 继续功能推进**，按每累计 6 轮一次，下一次 R 轮为 **R12**（R9 改的是三份 workflow 配置与 `pyproject.toml` 的单个测试插件声明、新增 1 个回归文件，**未变更主工作区路径 / 未新增服务 / 未大改依赖**，不触发立即 R 轮）。**但 R9 让 CI 从「不触发」变为「真实运行」，验收环境发生变化 ⇒ R12 元审视须复核「workflow 配置守卫是否纳入 `loop_check.py` 附加守卫」，本轮不提前改机制。**
> 下方「R1 元审视」段末尾的「还差 4 轮」、R3 段的「还差 2 轮」、R4 段的「还差 1 轮」、R5 行的「下一轮即触发 R6」以及本段 R5 版的「下一轮即 R6」**都是当时的历史结论**，不是当前值，勿据此判断。
> R5 未触发即时元审视：主工作区未变更、无新增服务、无依赖大改（改的是测试代码与内部台账 `tools/triage_check.py`，不是服务）；本节与 `tools/loop_check.py` 本轮均不改，两套 pytest 的集合范围（P1-10）仍留给 R6 定夺。
> **待 R 轮复核的问题（R4 补了决定性的新证据，仍不提前改机制）**：
> ① 静态体检对运行时缺陷完全无感——**R4 第四次实证**：`species_blueprint.py` 函数级 `import random` 遮蔽让 `crossover()` 在互斥能力分支抛 `UnboundLocalError`，而同期 `loop_check.py` 三项仍 `PASS`。R1 缺 import os / R2 路径默认值 / R3 根锚点 / R4 import 遮蔽，四种不同形态 ⇒ 候选 `C-20261110-01` 的「断层不止一种形态」已被反复坐实。R 轮须定夺静态层是否补契约检查，还是维持「静态只证明能解析，运行时靠回归+冒烟覆盖」的分层。
> ② **验收口径（R4 把 R3 的粗结论彻底查清，这是 R6 的主输入）**：`testpaths=["hive/tests"]` ⇒ 裸 `pytest -q` 只收 475 项，CI 也只跑 `hive/tests/`；维护中的 `tests/` 子集（139 项）从未进 CI；两集合因 `test_dna.py` 同名无法一起 collect。R6 须决定：是否把「**带范围的两套 pytest + `triage_check.py` 对账**」正式纳入固定验收口径、是否调整 `testpaths`/CI/测试命名以让两套可合并。R4 已把 20 个失败变成可校验台账（P1-8），但不擅自改收集范围。
> ③ **产物入库静默失败**：`docs/reports/` 被忽略却已有 12 份报告入库，新报告必须 `git add -f`；R4 差点把分诊报告丢在库外。R 轮复核 `.gitignore` 覆盖是否精确（复利规则 `R-20261014-01`）。
> ④ 运行时账本 `hive/memory/cognitive_state.json` 每轮跑 pytest 都被改写并需手工复原（R3 已提），R 轮复核忽略或只读策略。
> ⑤ 复利规则 `R-20261010-19` 第 17 次命中即此：连自己项目的台账**举例**也会错（R3 的 `brain_config.json` 举例实测 0 条命中），不只数字会漂。
> ⑥ **R5 新增（台账连自己回链的落点也会记错）**：R4 台账写「`d92ef5f` 在 CANDIDATES.md 7 处」，实测 **0 处**（R4 只改了主表/晋升表两处 hits 数字、未写含 sha 的证据文字；`RULES.md` 里才是 14 处）。R6 须一并复核复利回链口径：hits 两处同步 + 证据行到底落在哪个 ledger，写结论前 grep 实测。另 `triage_check.py` 本轮已升级为支持多轮修复（「已修」前缀判定），R6 复核这套口径是否够用、是否纳入固定验收口径。

### R6 元审视结论（2026-11-11，第 7 轮 · 不推进功能）

**问 1 · 验收口径是否覆盖不住新风险？→ 是，已修订。**
旧「二、固定验收口径」只列静态三项 + 可选冒烟，与项目实际已长出的 `triage_check.py`（历史失败分诊唯一真源）、两套 pytest 完全脱节：每轮只跑 `loop_check.py` 显示 PASS，却无法发现分诊台账漂移或 `tests/` 子集回归失效。R6 处置：① 把「四层验收」正式写进「二」（含集合范围纪律）；② 给 `loop_check.py` 加 `[+] 验收链闭合` 附加守卫（工具在位 + 分诊台账静态自证必须 `RESULT: PASS`），把可静态化的那层收进每轮一条命令；③ **维持分层、不补通用静态契约检查**：R1~R4 四次「静态全绿但运行时有缺陷」实证中有 2 次是语义/运行时形态，静态层天然不可见；专用 AST 守卫 + 旧实现注入反证 + 1 周期冒烟已稳定覆盖，硬塞通用检查只会带来误报或恒绿项。

**问 2 · 待办有无过期 / 重复？→ 有重复驱动，已收敛。**
「六、待 R 轮复核①~⑥」与 P1-10 实为同一批输入，R6 后全部标记定夺（见下），P1-10 改述为「结论已定、仅登记后续独立轮条件」，不再作为下一轮待决策输入；「五、每轮流程」章节号笔误 R2 已修，本轮复核无误；P0-2 描述（R1 有界冒烟已达成、项目仍非可用）与实测一致，未过期；P1-1/7/8、P1-9② 的 ✅ 状态与 `triage_check.py LEDGER`（20 条 / open 13 / fixed 7）一致。**未新增待办**：③④ 属常驻流程纪律，已写进「五、每轮流程」第 8/9 步，不立会空转的候补。

**问 3 · 路径 / 版本 / 自动假设是否漂移？→ 三处，已处置。**
① `LOOP.md` 写死的验收工具路径与 `loop_check.py` 无绑定 ⇒ 工具被删/改名时体检静默少跑一层，已由附加守卫拦住；② 「`.gitignore` 能解决运行时副产物」的假设**被实测推翻**：`hive/memory/cognitive_state.json` 确已被 git 跟踪（`git ls-files --error-unmatch` 命中），忽略规则对已跟踪文件无效；③ 「`docs/reports/` 已被忽略」只说对一半：实测已有 **12 份报告入库**，另有 1 份 `compound-engineering.local.md` 被忽略且未跟踪（用户本地文件，不动）。主工作区 / 镜像 / 远端路径与 `master` 分支均未漂移（R6 开工 `git fetch` + `gh api` + 镜像 HEAD + 静态体检四处实测同值 `1bdd4069`）。

**六条待复核逐条定夺**：
- ① **维持分层**，理由见问 1；候选 `C-20261110-01` 的「断层不止一种形态」已被 R1~R4 四次实证坐实，不再重复登记为待办。
- ② **已纳入固定验收口径**（「二、四层验收」+ `loop_check.py` 附加守卫）；**`testpaths` / CI / 命名一律不调整**，三条理由见「二、R6 定夺」引文。R6 自身按修订后的口径完整跑过四层（见「八、推进日志」R6 行）。
- ③ **不改 `.gitignore`、不 `git rm --cached`**（后者等于从仓库删除跟踪，属破坏性操作，转「待人工」）；维持跑后 `git checkout --` 复原并写进「五、每轮流程」第 8 步，让纪律不依赖每轮记忆。
- ④ **不改 `.gitignore`**：精确放宽会让任意新报告自动入库，且对已入库 12 份无影响；维持「`git add -f` + `git ls-files` 回读」并写进第 9 步。
- ⑤ **已写进硬口径**：「二、四层验收」③ 与 `loop_check.py` 的 SCOPE 行都写明「pytest 数字必须带集合范围」，不再靠台账举例自觉。
- ⑥ **回链口径复核（写结论前 grep 实测）**：`RULES.md` 证据行带 commit sha（R5 的 `d2d44c2` 实测 7 处），`CANDIDATES.md` **只改主表 / 晋升表两处 hits 数字、不写含 sha 的证据文字**（R4 台账称 7 处、实测 0 处，R5 已更正）⇒ 本轮回链按此真实格式执行，结论写进「七、复利回链」前先 grep 回读。另 `triage_check.py` 的「已修」前缀判定经 R5 实测支持多轮修复，`LEDGER` 仍是唯一真源，**够用、不另起台账**；其静态自证已由 ① 的附加守卫每轮兜住。

**触发条件复核**：R6 前主工作区未变更、无新增服务、无依赖大改，故未提前触发即时元审视；本轮只改 `tools/loop_check.py`、新增 1 个回归测试与本台账，不属「主工作区变更 / 新增服务 / 依赖大改」，**不额外触发**下一轮 R 轮，仍按每 6 轮计数（下次 R12）。

## 七、复利回链

开工前先扫 `D:/编程复利系统/rules/RULES.md`；命中既有规则时直接复用，并在对应条目上 `hits+1`（附 commit sha 或路径作证据）。没命中不硬凑。
本项目已产出的一条经验：「提交时只 add 当阶段文件 → 远端仓库变成残缺快照且不报错」，值得回写候选池。

**R1 命中 2 条（已回链、已回读校验）**：
- `R-20261015-05`「通路配方不跨对象保证，换源/换客户端必须重新探测」：pypi 直连 + 代理 7897 + 清华源全败、阿里源成功。hits `11 → 12`，证据追加至 `CANDIDATES.md` C-20261015-05、`RULES.md` 复用行。
- `R-20261010-19`「不要信状态文档，先 git log + 实测」：本台账快照 `3478df9`/31 与实测 `12b41ef`/32 漂移；另用 `gh api` 独立复核远端。hits `14 → 15`，证据追加至 C-20261010-19、`RULES.md` 复用行。
- 同时复用了 `R-20261010-11`「旧实现注入反证」与 `R-20261010-08`「修完必做运行时真实调用」的方法（未重复计数，避免硬凑）。

**R1 新增候选（R2 已回写候选池为 `C-20261110-01`，hits=1）**：静态验收口径「全绿」与运行时行为之间存在断层——`py_compile` + import 闭合都过，仍可能有 `NameError` 藏在 `except` 兜底分支里被吞成默认值。**推论：任何带宽泛 `except Exception` 返回兜底值的路径，静态检查天然不可见，必须靠运行时真实调用 + 反证覆盖**（R1 `get_snapshot()` 的 `name 'os' is not defined` 即此形态）。R2 第二次实证且补出第二种形态（路径默认值跨平台歧义），已正式登记进 `CANDIDATES.md`，hits 0 → 1。

**R2 命中 3 条（已回链、已回读校验，证据均带 `fb757cd`）**：
- `R-20261010-11`「验收必须用旧实现注入反证」：hits `5 → 6`。把旧默认值注入**真实源码** → `6 failed, 5 passed`，而同一时刻 `loop_check.py` 仍 `PASS` → `finally` 字节级复原 `sha256 SAME` → 复跑 `11 passed`。
- `R-20261010-08`「修完必做运行时真实调用」：hits `4 → 5`。真实 `init_events_file()` + `emit()` 后读回 jsonl 逐字段解析 + 1 周期 `main.py` 真启动 + 复核 `Data/events.jsonl` 落点与内容。
- `R-20261015-05`「通路配方不跨对象 / 不跨时刻，换时刻要重新探测」：hits `12 → 13`。R1 直连 Reset、代理握手失败；R2 先 fetch + `gh api` 复核未分叉，裸直连**一次成功** `12b41ef..fb757cd`。未照抄上轮结论改走代理配方。
- 同时沿用了 `R-20261010-19`「不要信状态文档，先实测」（本轮实测 pathlib 两种平台语义、实测 HEAD/提交数/跟踪文件数）与 `R-20261010-05`「状态文件落盘后 grep 校验」的方法，**未重复计数，避免硬凑**。
- 回读校验：`RULES.md` 含 `fb757cd` 3 处、`CANDIDATES.md` 4 处，三条 hits 与证据均在位；候选池表格竖线数断言通过（沿用 `C-20261109-06`）。

**R3 命中 6 条（已回链、已回读校验，证据均带 `696b76e`）**：
- `R-20261024-12`「判据收成唯一实现；接口的每个投影都要派生或守卫」：hits `6 → 7`。本轮主命题。4 处根锚点收进 `hive/paths.py`；`event_stream.py` 保留为转发薄壳（旧入口不删），配 4 类守卫：消费点不得再定义 `resolve_hive_home`、薄壳只含单 return、`__all__` 从 AST 派生逐项相等、`es.PROJECT_ROOT is paths.PROJECT_ROOT`。
- `R-20261024-01`「合并前先判语义，语义不同则改名而非合并」：hits `5 → 6`。`monitor.py` 取盘符做磁盘探测、`.env` 的 `BASE_DIR`（经 pydantic 覆盖 `_default_base_dir()`）、`darwin_bee.py` / `sentinel_bee.py` / `__init__.py` 的 3 处 `cwd()` 均与「数据落点根锚点」语义不同，**不合并**，在 `hive/paths.py` docstring 与本台账显式登记。
- `R-20261010-11`「旧实现注入反证」：hits `6 → 7`。把 `Path.cwd() / "D:/HIVE"` 注回真实源码 → `7 failed, 36 passed`，同期静态体检仍 `PASS`，字节级复原 `sha256 79d81f59…e116 SAME` → `43 passed`。
- `R-20261010-08`「修完必做运行时真实调用」：hits `5 → 6`。真实启动 `main.py` 1 周期 + 真实 `Data/events.jsonl` 6 条事件逐字段解析 + 无畸形 `D:` 目录。
- `R-20261015-05`「通路配方不跨时刻」：hits `13 → 14`。R2 裸直连一次成功；R3 直连 `Connection was reset`、代理 7897 `TLS unexpected eof`，各试一次即止，`gh api` 复核远端仍 `fb757cd` 未分叉。
- `R-20261010-19`「不要信状态文档，先实测」：hits `15 → 16`。实测推翻台账「全量 101 passed」（实为 `tests/` 子集，全仓库 455 passed / 20 failed）；实测出 `.env` 的 `HIVE_HOME=D:/HIVE` 才是本机「碰巧能用」的真源，而非代码默认值。
- 回读校验：`RULES.md` 含 `696b76e` **12 处**（6 条 × 形态行与 commit 行）、`CANDIDATES.md` **6 处**；6 条 hits 逐条回读正确，候选池各行竖线数不变（沿用 `C-20261109-06`）。回链脚本跑完即删，未留临时文件。

**R4 命中 7 条（已回链、已回读校验，证据均带 `d92ef5f`）**：
- `R-20261010-19`「不要信状态文档，先实测」：hits `16 → 17`。R3 台账称 20 个失败源于 `brain_config.json` 缺失/`BrainInterface` 加载失败，逐条实测 **0 条命中**；另实测裸 `pytest -q` 只收 475 项 `hive/tests/`，并非「全仓库」。
- `R-20261015-03`「交叉核对要核分母的来源」：hits `11 → 12`。台账「455 passed/20 failed」的分母 475 只来自 `hive/tests/` 单集合；CI 跑 `hive/tests/` 反而漏掉 `tests/` 子集（139 项）。
- `R-20261010-11`「旧实现注入反证」：hits `7 → 8`。把 `import random` 注回真实源码 → `5 failed, 2 passed` 并精确复现 `UnboundLocalError`，`finally` 字节级复原 `sha256 f6115dac45e01e55 SAME` → `7 passed`；另发现 **CRLF 行尾让 LF 注入锚点静默失效**，已在测试中做 LF 归一化并留注释（反证自身也会腐坏的新形态）。
- `R-20261010-08`「修完必做运行时真实调用」：hits `6 → 7`。真实 `crossover()` 覆盖互斥/共有能力、亲和度分支，真实 `mutate()`、1 周期 `main.py` 冒烟、全量 pytest 前后对比。
- `R-20261010-15`「审计发现 ≠ 本轮修复，三种去向」：hits `9 → 10`。P1-8 分诊 20 条 → 修 1（`#10`）/ 立下一轮 19（P1-9）/ 登记备查 1 组（P1-10）/ 待人工 0，并显式决定不批量 xfail。
- `R-20261104-02`「散装检查收敛成一条可复现命令且自证范围」：hits `3 → 4`。新增 `tools/triage_check.py`（逐项打印、任一失败非零退出、SCOPE 行写明不做什么），post-fix 报告对账 `FAILED 19 == open 19` PASS。
- `R-20261024-12`「判据收成唯一实现，每个投影派生或守卫」：hits `7 → 8`。`import random` 4 份收成模块级单一真源 + AST 守卫；分诊台账以 `LEDGER` 为唯一真源，报告只做指针不复制表格，AST 检查 20 个 nodeid 收集闭合。
- 回读校验：`RULES.md` 含 `d92ef5f` **7 处**、`CANDIDATES.md` 7 条 hits 主表/晋升表**两处同步 +1 且数值一致**，竖线数不变。⚠️ 回链脚本一度把两个 ledger 的 CRLF 写成 LF（1219/580 行全量行尾漂移），grep 回读时检出并已逐字节复原为 100% CRLF；`RULES.md` 行数 +7 恰为 7 条证据行，`CANDIDATES.md` 行数不变。

**R5 命中 7 条（已回链、已回读校验，证据均带 `d2d44c2`）**：
- `R-20261010-11`「验收必须用旧实现注入反证」：hits `8 → 9`。运行时把 helper 名注回 `test_create_mock_dna` → 精确 6 个 `AttributeError`，全程不改写工作区、`sha256 SAME`。
- `R-20261010-08`「修完必做运行时真实调用」：hits `7 → 8`。exec 真实测试模块 + unittest 真跑 6 用例、全量 pytest 前后对比、1 周期 `main.py` 冒烟。
- `R-20261010-19`「不要信状态文档，先实测」：hits `17 → 18`。开工四处实测 HEAD 而非照抄上轮结论；另实测推翻 R4 台账「`d92ef5f` 在 CANDIDATES.md 7 处」（实为 0 处）——**台账连自己回链的落点也会记错**。
- `R-20261010-15`「审计发现 ≠ 本轮修复，三种去向」：hits `10 → 11`。19 条只取「测试自身缺陷 6 条」一个可独立闭合的子集做完，余 13 条留台账 open、P1-10 归 R6，不批量 xfail、不扩面。
- `R-20261024-12`「判据收成唯一实现，每个投影派生或守卫」：hits `8 → 9`。6 个调用点全部投影到单一 helper 并加 AST 守卫；`LEDGER` 继续作唯一真源。
- `R-20261104-02`「散装检查收敛成一条可复现命令且自证范围」：hits `4 → 5`。`triage_check.py --report` 一条命令完成三检 + SCOPE 自证，并把台账升级为支持多轮修复（机制随项目演化）。
- `R-20261015-05`「通路配方不跨时刻」：hits `14 → 15`。上轮记忆写「三方已对齐」，本轮仍 `git fetch` + `gh api` + `origin/master` + 镜像四处独立复核。
- 回读校验：`RULES.md` **1227 → 1234 行**（+7 恰为 7 条证据行）、含 `d2d44c2` **14 处**（7 条 × 提交行 + commit 行）；`CANDIDATES.md` 行数与竖线数（**2709**）**均不变**，7 条主表/晋升表两处 hits 同步 +1 且数值一致；两 ledger **100% CRLF 保持**（R4 曾发生 CRLF→LF 全量漂移，本轮用二进制 `\r\n` 拼接 + 写后回读断言规避）。
- **口径实测更正**：R4 台账写「`d92ef5f` 在 CANDIDATES.md 7 处」，实测 **0 处**。本轮回链按真实格式执行（CANDIDATES.md 只改两处 hits 数字、证据行落 `RULES.md`），并把这条偏差写进 R6 待复核⑥。
- 未重复计数：`R-20261010-05`（落盘 grep 校验）、`R-20261024-01`（语义判异不合并）本轮只沿用方法，不计 hits。


**R6 命中 7 条（已回链、已回读校验，证据均带 `1148c87`）**：
- `R-20261104-02`「散装检查收敛成一条可复现命令且自证范围」：hits `5 → 6`。把分诊台账静态自证收进 `loop_check.py` 一条命令，SCOPE 升级为四层验收。
- `R-20261010-19`「不要信状态文档，先实测」：hits `18 → 19`。四处实测 HEAD；实测推翻「`.gitignore` 能解决已跟踪副产物」；实测 reports 12 份入库；回链前 grep 实测格式。
- `R-20261015-03`「交叉核对要核分母的来源」：hits `12 → 13`。把两套 pytest 的集合范围写进硬口径，本轮两个集合各自实跑。
- `R-20261024-12`「判据收成唯一实现，每个投影派生或守卫」：hits `9 → 10`。`ACCEPTANCE_TOOLS` 单一真源 + 逐投影点守卫。
- `R-20261010-15`「审计发现 ≠ 本轮修复，三种去向」：hits `11 → 12`。元审视只定夺不扩面，③④ 转常驻流程、② 纳入口径、不改 CI/命名。
- `R-20261010-11`「验收必须用反证」：hits `9 → 10`。失效态注入反证（工具缺失 / 台账自证 FAIL），不改写工作区。
- `R-20261015-05`「通路配方不跨时刻」：hits `15 → 16`。上轮直连全被 reset、本轮裸直连 fetch 一次成功，仍不写进长期配方。
- 未重复计数：`R-20261010-08`（修完真跑：本轮真跑 `loop_check.main()`、1 周期冒烟、两套 pytest）、`R-20261010-05`（落盘 grep 校验：台账与回链脚本均回读断言）本轮只沿用方法。
- 回读校验（**先 grep 实测再写结论**，待复核⑥）：`RULES.md` **1233 → 1240 行**（+7 恰为 7 条证据行）、含 `1148c87` **14 处**；`CANDIDATES.md` **580 行不变**、竖线数 **2709 不变**、7 条**主表 / 晋升表两处 hits 同步 +1 且数值一致**、含 sha **0 处**；两 ledger **100% CRLF 保持**。
- **待复核⑥定论**：`CANDIDATES.md` 每个 ID 实测恰有 **2 行**（主表 + 晋升表），主表 hits 有带 `**` 与不带两种格式；`RULES.md` 证据行才带 commit sha。R4 台账写「7 处」、R5 更正为「两处 hits 同步」——R6 实测确认「两处」指 CAND 内两行，加上 RULES 复用数共 **三处同源记录**，回链须三处一起改，写结论前必须 grep。


**R7 命中 7 条（已回链、已回读校验，证据均带 `c40dfbc`）**：
- `R-20261010-11`「验收必须用旧实现注入反证」：hits `10 → 11`。`tests/core/test_state_isolation_r7.py` 在内存 exec 剥离隔离块的旧实现，#2 精确复现 `1 != 2`、#9 精确复现 `is not None`；两处真实源码 sha256 SAME、不改写工作区。
- `R-20261010-08`「修完必做运行时真实调用」：hits `8 → 9`。exec 真实测试模块 + unittest 真跑（整文件 21 passed / 整类 8 passed）、两套 pytest、1 周期冒烟。
- `R-20261010-19`「不要信状态文档，先实测」：hits `19 → 20`。逐条实测复现 #2/#9 而非照抄分诊台账；恢复执行前重新实测 HEAD / 提交数 / 跟踪文件 / 远端 / 镜像；写盘前实测 `LOOP.md` 100% LF 而两个 ledger 100% CRLF、每 ID 恰 2 行。
- `R-20261010-15`「审计发现 ≠ 本轮修复，三种去向」：hits `12 → 13`。13 条 open 只取「状态污染 2 条」这个可独立闭合的子集，余 11 条（断言过期 10 + `#18` 行为决策 1）留 open、不批量 xfail、不改产品行为。
- `R-20261024-12`「判据收成唯一实现，每个投影派生或守卫」：hits `10 → 11`。`loop_check.py` 写死 13 与 `triage_check.py` 的 `EXPECTED_OPEN` 是同一判据的两层 ⇒ 前者改 AST 派生、后者维持唯一真源；R6 回归测试作投影改动态断言并加 7/3/缺失三态反证。
- `R-20261104-02`「一条命令自证范围」：hits `6 → 7`。R6 收敛的那条命令其 SCOPE 写死 13 在本轮变成假陈述，改为派生后自证范围才与真源一致。
- `R-20261015-05`「通路配方不跨时刻」：hits `16 → 17`。R6 轮末直连/代理均失败，R7 轮首仍四处独立复核后裸直连一次成功 `1bdd406..65fa80a`；轮末 push 前再复核，不写长期配方。
- 未重复计数：`R-20261010-05`（落盘 grep 校验）、`R-20261024-01`（语义判异不合并）、`R-20261015-03`（集合范围纪律）本轮只沿用方法；两处测试文件的字节级 CRLF 纪律沿用 R4/R5 踩坑，无对应规则条目，不计 hits。
- 回读校验：`RULES.md` **1240 → 1247 行**（+7 恰为 7 条证据行）、含 `c40dfbc` **14 处**；`CANDIDATES.md` **580 行不变**、竖线数 **2709 不变**、7 条主表/晋升表**两处 hits 同步 +1 且数值一致**、含 sha **0 处**；两 ledger **100% CRLF 保持**。回链脚本跑完即删，未留临时文件。

**R8 命中 7 条（已回链、已回读校验，证据均带 `c6266a4`）**：
- `R-20261010-19`「不要信状态文档，先实测」：hits `20 → 21`。拆包推翻 R0 照抄注释写下的「私有 wheel」前提、实测发现第二断点（分支不匹配）、实测 wheel 是 9 个而非台账「8 个」——**错误前提是自己写进台账又被自己采信**。
- `R-20261015-03`「交叉核对要核分母的来源」：hits `13 → 14`。同一批 wheel 在台账（8）、上轮说明（另外七个）、实测（9）三个口径分叉，更正后才敢写「排除其余 8 个」。
- `R-20261010-08`「修完必做运行时真实调用」：hits `9 → 10`。不只核对文件在位，而是在临时目录实装 wheel + tree-sitter，原样跑云端那三条 CI 命令（EXIT 全 0）。
- `R-20261024-12`「判据收成唯一实现，每个投影派生或守卫」：hits `11 → 12`。SCOPE 写死的 P0 状态是台账状态的第二投影，P0-1 一完成即成假陈述；改为 `p0_open_ids()` 静态派生 + 4 项回归。
- `R-20261015-05`「通路配方不跨时刻」：hits `17 → 18`。R7 轮末裸直连一次成功，R8 轮末 fetch/push 直连 reset、代理 7897 连不上，`gh api` 却通 —— 各试一次即止，不写长期配方。
- `R-20261010-15`「审计发现 ≠ 本轮修复，三种去向」：hits `13 → 14`。挖出「CI 恒绿非门禁 / `-n 3` 不生效 / 无报告产物」三项，统一立 P1-11，本轮只修两处断点、不改 QA 语义。
- `R-20261010-11`「验收必须用反证」：hits `11 → 12`。新增回归用 `tmp_path` 构造临时台账五态（全 open / 标 ✅ / 全 ✅ / 无标题 / 文件缺失），断言派生随之变化；另加 `main()` AST 投影守卫防写死串回潮，全程不改写真实工作区。
- 未重复计数：`R-20261104-02`（一条命令自证范围）、`R-20261024-01`（语义判异不合并）、`R-20261010-05`（落盘 grep 校验）本轮只沿用方法。
- 回读校验（**按 R6 待复核⑥定论，先 grep 实测格式再写结论**）：`CANDIDATES.md` **581 行（含末尾空串；等价 CRLF 580，历轮记的 580 即此）** / 竖线数 **2709 不变** / 7 个 ID 各恰 2 行且主表与晋升表 hits 同步 +1 / 含 sha **0 处**；`RULES.md` **1248 → 1255 行**（+7 恰为 7 条证据行）、含 `c6266a4` **14 处**（7 条 × 形态行与 commit 行）；两 ledger **100% CRLF 保持**。回链以 heredoc 执行、跑完即删，未留临时文件。

**R9 命中 7 条（已回链、已回读校验，证据均带 `5e57aef`）**：
- `R-20261010-19`「不要信状态文档，先实测」：hits `21 → 22`。上轮说明写「仓库另有 lint.yml、test.yml 兜底」，R9 查 Actions runs 实测发现 test.yml 从 11-09 起每个提交都是 0 jobs 空失败、Code Quality 从未触发；再用 PyYAML 解析三份文件锁定 `test.yml:186` 块标量断裂。
- `R-20261015-03`「交叉核对要核分母的来源」：hits `14 → 15`。`tests/` 子集口径 R7 记 154、R8 复跑 158、R9 实测 169；四层验收数字一律带集合范围，不把全量 474 collected 混作子集通过数。
- `R-20261010-08`「修完必做运行时真实调用」：hits `10 → 11`。不只本地 PyYAML 解析通过，还推送后用 `gh api` 查 workflows 列表确认 name 由路径恢复为 `Tests`、且 `5e57aef` 一次触发三个 workflow（云端解析实测）。
- `R-20261024-12`「判据收成唯一实现，每个投影派生或守卫」：hits `12 → 13`。分支口径同时投影在 `on.push/pull_request.branches` 与四处 job 级 `github.ref` 条件，只改前者会留半截 bug；回归对两处同时断言并禁止残留 `refs/heads/main`。
- `R-20261015-05`「通路配方不跨时刻」：hits `18 → 19`。同轮内 fetch/push 成功 → `Recv failure` → 跑完四层验收再探测又成功，两种方向都在一轮里出现，更不写长期配方。
- `R-20261010-15`「审计发现 ≠ 本轮修复，三种去向」：hits `14 → 15`。挖出「CI 接通后会因 11 条历史失败红」「重依赖在 3.9~3.12 matrix 可装性未实测」「三个 `needs: test` 的 job 未验证」，统一立 P1-12，本轮只修配置、不批量改断言。
- `R-20261010-11`「验收必须用反证」：hits `12 → 13`。`tests/utils/test_workflow_yaml_r9.py` 用 `tmp_path` 影子文件注回顶格 `import sys`，断言 PyYAML 必抛 `ScannerError`，并校验真实文件 sha256 SAME；另加 AST 防写死守卫。
- 未重复计数：`R-20261104-02`、`R-20261024-01`、`R-20261010-05` 本轮只沿用方法。
- 回读校验：`CANDIDATES.md` 行数 / 竖线数不变、7 个 ID 各恰 2 行 hits+1、含 sha 0 处；`RULES.md` +7 行、`5e57aef` 14 处；两 ledger 100% CRLF 保持。

## 八、推进日志

| 时间 | 轮次 | 做了什么 | 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-11-10 | R0 建档 | 清理 10/31–11/01 遗留 15 项未提交改动，分 3 提交入库；排除 `cache/`；建立本台账、体检脚本与长效自动化 | `d14135d` `db74a7a` `ee24c6f` `3478df9` `12b41ef`；12 文件 py_compile 全过；末次体检 `PASS` | 推进 P0-2（运行时验证），P0-1 待用户决策 |
| 2026-11-10 10:25–10:58 | R1（推进 + 即时口径修订） | 打通 P0-2：实测阿里源装齐启动链依赖、`main.py` **首次真实跑通**并正常收尾；修静态 PASS 漏掉的 2 处运行时缺陷（缺 `import os`、`get_system_info` 少 key）；新增有界周期开关、第 4 层 `smoke_runtime.py`、9 项回归测试；就地把镜像验收口径改为「允许本地领先」并令体检自证范围。遗留 `hunt_sync` 缺陷拆为 P0-3（需行为决策，未擅改） | 体检 `PASS`（322 .py / 116 import / 镜像对齐）；`smoke_runtime.py --cycles 1` = `PASS`（EXIT=0、10 SUCCESS、CPU: 16 cores），`--cycles 2` = `FAIL`（精确捕获 `asyncio.run() cannot be called from a running event loop`）；pytest 9 passed；旧实现注入反证复现 `name 'os' is not defined` 与全 0 默认值；`gh api` 复核远端 `12b41ef`、开工 ahead 0；R1 主提交 `ec0930c`（7 文件 +358/-33）；旧口径反证：此时三方 distinct=2，旧规则会误报 FAIL | **下一轮先补推**（直连/代理各试一次）；再推进 P0-3 修法决策、补装 `chromadb` 等依赖验证完整功能；P0-1 仍待用户决策 |
| 2026-11-10 17:10–17:40 | R2（推进 + 补推清零） | 取 P1-1「事件流根目录跨平台歧义」：改为 `PROJECT_ROOT` + 单一 `HIVE_HOME` 读取，与读取方 `web_dashboard.py` 锚点对齐；新增 11 项回归（含跨平台语义断言、写读同路径契约、真实 emit 落盘解析、AST 契约、旧实现注入反证）。**并补推 R1 积压的 4 个提交 + 本轮 1 个**，同步只读镜像，三方首次对齐。侦察发现同源硬编码散落 3 处，登记 P1-7 不扩面；顺手修正「五、每轮流程」章节号笔误（七→八） | 开工基线体检 `PASS`（325 .py / 116 import / 本地领先 4）；pathlib 实测 `PureWindowsPath('D:/Hive')/'D:/HIVE'` → `D:\HIVE`、`PurePosixPath('/srv/hive')/'D:/HIVE'` → `/srv/hive/D:/HIVE`（同期证明本机「碰巧能用」）；pytest 新增 11 passed / 全量 **101 passed**；旧实现注入反证 `6 failed, 5 passed` 且**静态体检同时仍 PASS** → 字节级复原 `sha256 6510078806374266 SAME` → `11 passed`；`smoke_runtime.py --cycles 1` `PASS`（EXIT=0、209 行、CPU: 16 cores）；真实运行后 `Data/events.jsonl` 593B/3 条落项目根、无畸形目录；`gh api` 复核远端未分叉 → 裸直连 `git push` 一次成功 `12b41ef..fb757cd`（60s 超时防挂起）→ 镜像 `pull --ff-only` → 复跑体检 `PASS`（326 .py / 116 import / **三方对齐 fb757cde**）、工作区干净、ahead 0。主提交 `fb757cd`（2 文件 +153/-3）；R2 台账回填另计 | 下一步：P0-3 修法仍需用户决策（不擅改）；可推进 P1-2 tag 口径确认 / P1-7 单一根锚点收敛；补装 `chromadb/openai/ollama/psycopg2/flask` 验证完整功能；P0-1 仍待用户决策。轮次 R0+R1+R2 = **3**，距 R6 元审视还差 **3** 轮 |
| 2026-11-10 23:47–00:12 | R3（推进 + 轮首补推失败） | 取 P1-7「根锚点收敛」：新增 `hive/paths.py` 作单一真源，4 处硬编码（`hive/__init__.py` / `settings.py` / `librarian_bee.py` / `web_dashboard.py`）改为调用真源，`event_stream.py` 退化为转发薄壳并保留 P1-1 回归可注入符号；新增 32 项回归覆盖判据唯一性 / 消费点投影守卫 / `__all__` 派生守卫 / 写读同路径 / 旧实现注入反证。轮首按交接先补推 R2 积压 2 提交，直连与代理各试一次均失败、只记一行不反复重试。侦察发现两处口径漂移并登记：①「全量 101 passed」实为 `tests/` 子集，全仓库另有 20 个历史失败（立为 P1-8）；② `.env` 的 `HIVE_HOME=D:/HIVE` 才是本机「碰巧能用」真源 | 开工基线体检 `PASS`（326 .py / 116 import / 本地领先 2）；补推：直连 `Recv failure: Connection was reset`、代理 7897 `TLS unexpected eof`，各一次即止；`gh api` 复核远端仍 `fb757cd` 未分叉。功能主提交 **`696b76e`**（7 文件 +392/-42）。验收：静态体检 `PASS`（**328 .py / 118 hive.\* 目标 / 镜像=origin/master、本地允许领先**）；`tests/` 子集 **133 passed**（101 原有 + 32 新增，退出码 0）；P1-1 既有 11 项回归原样通过；**旧实现注入反证**：把 `Path.cwd() / "D:/HIVE"` 注回真实源码 → `7 failed, 36 passed`，而同期 `loop_check.py` 仍 `PASS`（静态层无感第三次实证）→ 字节级复原 `sha256 79d81f59d7e4a91dc0bc81291c40d493c452e770b72512d4d486770a3a80f116 SAME` → 复原后 `43 passed`；`smoke_runtime.py --cycles 1` **PASS**（EXIT=0 / 209 行 / CPU: 16 cores / 正常收尾）；真实落盘 `D:/Hive/Data/events.jsonl` 6 条事件逐字段解析、无畸形 `D:` 目录。运行时副产物 `hive/memory/cognitive_state.json` 已 `git checkout --` 复原，工作区干净。复利回链 **6 条** hits+1，回读校验 `696b76e` 在 RULES.md 12 处 / CANDIDATES.md 6 处 | 下一步：**先补推**本地领先提交（含 `d36ed04` / `148c6f5` / `696b76e` 及本轮回填，直连 / 代理各一次，仍失败只记一行）；推送成功后镜像 `pull --ff-only` 并复跑体检。P0-3 需用户决策、P0-1 决策权归用户，均跳过；可推进 P1-2 tag 口径 / P1-8 历史失败分诊 / 补装 `chromadb` 等依赖。轮次 R0+R1+R2+R3 = **4**，距 R6 元审视还差 **2** 轮 |

| 2026-11-11 06:15–07:05 | R4（推进 + 分诊台账） | 取 **P1-8**「20 个历史失败分诊」：全仓库逐条实测分类（真缺陷·已修 1 / 待行为决策 1 / 断言过期 10 / 测试自身缺陷 6 / 状态污染 2 / **依赖缺失 0**），落地 `tools/triage_check.py` 为**唯一真源** + 一条可复现自证校验命令，报告只做指针不复制台账，**显式决定不批量 xfail**。修唯一产品真缺陷：`species_blueprint.py` 3 处函数级 `import random` 遮蔽模块级导入，双亲能力互斥时 `crossover()` 抛 `UnboundLocalError`（静态三项全 PASS 仍漏，**第 4 次实证**），配 6 项回归 + 旧实现注入反证。侦察推翻 R3 两处口径：①「`brain_config.json`/`BrainInterface` 缺依赖」举例在 20 条实测异常中 **0 条命中**；② 裸 `pytest -q` 因 `testpaths=["hive/tests"]` 只收 475 项、CI 也只跑 `hive/tests/` 而**漏掉 `tests/` 子集 139 项**，两集合因 `test_dna.py` 同名无法一起 collect ⇒ 立 P1-10。余 19 条按去向立 P1-9 | 开工基线：工作区干净、HEAD `118c2798`、42 提交、402 跟踪文件、ahead 0、`git fetch` 直连成功 + `gh api` 复核远端同值、三方对齐；静态体检 `PASS`（328 .py / 118 hive.\* 目标 / 镜像一致）。**修复前**全仓库 `pytest -q --tb=line`：**20 failed/455 passed**（251.40s）；**修复后**：**19 failed/456 passed**（255.24s），恰少 `#10` 一条、无新增失败。功能主提交 **`d92ef5f`**（4 文件 +554/-6）。验收：`tests/` 子集 **139 passed**（133+6，退出码 0）；`tests/core/test_species_blueprint_crossover.py` **6 passed**；原失败用例 `TestSpeciesBlueprint::test_crossover` FAIL→**PASS**；**旧实现注入反证**：注回 `import random` → `5 failed, 2 passed` 且精确复现 `UnboundLocalError`，`finally` 字节级复原 `sha256 f6115dac45e01e55 SAME` → `7 passed`（另发现 CRLF 行尾会让 LF 注入锚点静默失效，已修正并留注释）；`triage_check.py` 静态 `PASS`、对账 post-fix 报告 `FAILED 19 == open 19` `PASS`（对 pre-fix 报告则正确判 FAIL，证明有区分力）；`loop_check.py` `PASS`；`smoke_runtime.py --cycles 1` `PASS`（EXIT=0/209 行/CPU 16 cores/正常收尾）。运行时副产物 `hive/memory/cognitive_state.json` 已 `git checkout --` 复原。复利回链 **7 条** hits+1，`d92ef5f` 在 RULES.md 7 处、CANDIDATES.md 7 条主表/晋升表两处同步；回链脚本 CRLF→LF 全量行尾漂移已 grep 检出并逐字节复原 | 下一步：**先推送 + 同步镜像**（轮首 fetch + `gh api` 复核未分叉；直连不通试代理 7897 一次）；之后 P1-9 按分类批量处置 19 条、P1-10 交 R6。P0-3 需用户决策、P0-1 决策权归用户，均跳过。**轮次 R0+R1+R2+R3+R4 = 5，距 R6 元审视还差 1 轮**（R5 后触发，R6 须定夺两套 pytest + 分诊对账是否纳入固定验收口径） |
| 2026-11-11 13:11–13:26 | R5（推进 P1-9②） | 取 **P1-9②**「测试自身缺陷 6 条」：`test_phase3_evolution.py` 的 helper `test_create_mock_dna` 改名为 `create_mock_dna`（去掉 `test_` 前缀，6 个调用点无需改动）；`triage_check.py LEDGER` #11~#16 转 `已修(R5)`/`fixed`、`EXPECTED_OPEN` 19→13，台账改为「已修」前缀判定以支持多轮修复；新增 `tests/core/test_dna_helper_naming_regression.py` 4 项回归（AST 投影守卫 / 运行时真实调用 / 旧实现注入反证 / 字节级 sha256 SAME）。余 13 条按分类留 open，不扩面；P1-10 归 R6 | 开工基线：工作区干净、HEAD `96771d1`、**45 提交 / 405 跟踪文件**、ahead 0；`git fetch` 直连一次成功 + `gh api` + `origin/master` + 镜像 HEAD **四处实测同值、三方对齐**；静态体检 `PASS`（330 .py / 119 hive.\* / 镜像一致）。**修复前** 6 个 nodeid `6 failed`（`AttributeError: ... has no attribute 'create_mock_dna'`）→ **修复后** `6 passed`；新增回归 `4 passed`；`tests/` 子集 **143 passed**（139+4，退出码 0）；全量 `hive/tests/` **13 failed/461 passed**（474 collected，254.54s；R4 为 19 failed/456 passed，恰少 6 条、无新增失败，收集数 475→474 因 helper 不再被误收为用例）；`triage_check.py --report` 对账 **`FAILED 13 == open 13` PASS**（无新失败/静默修复/回归）、静态 `PASS`（20 条/open 13/fixed 7）；`loop_check.py` `PASS`；`smoke_runtime.py --cycles 1` `PASS`（EXIT=0 / 209 行 / CPU 16 cores / 正常收尾）。功能主提交 **`d2d44c2`**（3 文件 +173/-16）；运行时副产物 `hive/memory/cognitive_state.json` 已 `git checkout --` 复原。复利回链 **7 条** hits+1，`d2d44c2` 在 RULES.md 14 处、CANDIDATES.md 两处 hits 同步、两 ledger 100% CRLF 保持 | 下一步：**先推送 + 同步镜像**（轮末 `git fetch` + `gh api` 复核未分叉；直连不通试代理 7897 一次）；之后 **R6 元审视轮**（P1-10 两套 pytest 与分诊对账是否纳入固定验收口径 / `testpaths`·CI·命名是否调整 / 静态层对运行时缺陷无感已第 4 次实证 / `cognitive_state.json` 忽略策略 / `docs/reports/` 忽略规则 / 台账举例与回链落点也会错）。P1-9 余 13 条、P1-2 tag 口径可继续推进；P0-3 需行为决策、P0-1 决策权归用户，均跳过。**轮次 R0+R1+…+R5 = 6，下一轮即触发 R6** |

| 2026-11-11 19:31 起 | R6（元审视轮，不推进功能） | 按「每累计 6 轮触发」做元审视：三问定夺 + 六条待复核逐条落位，**直接修订 `LOOP.md` 二/三/四/五/六/七 与 `tools/loop_check.py`**。① **维持分层**，不补通用静态契约检查（4 次实证中 2 次是语义/运行时形态）；② 把「带范围的两套 pytest + `triage_check.py` 对账」正式纳入固定验收口径（四层表），并给 `loop_check.py` 加 `[+] 验收链闭合` 附加守卫（工具在位 + 分诊台账静态自证 `RESULT: PASS`），**不调整** `testpaths`/CI/测试命名（三条实测理由）；③ 实测推翻「`.gitignore` 能解决副产物」（`cognitive_state.json` 已入库、忽略规则无效），③④ 转「五、每轮流程」第 8/9 步常驻纪律；④ 六条待复核全部标记定夺、轮次指针推进为「R6 已完成 / 下一轮 R7 恢复功能推进 / 下次 R12」。新增 4 项失效态注入反证回归（临时目录、不改写工作区），未新增待办、未改功能代码 | 开工基线：工作区干净、HEAD `1bdd4069`、**47 提交 / 406 跟踪文件**、ahead/behind 均 0；`git fetch` 裸直连一次成功 + `gh api` + 镜像 HEAD 四处实测同值、三方对齐未分叉；静态体检 `PASS`（331 .py / 119 hive.\*）。P1-10 三条事实复核成立（`pyproject.toml:113` / `test.yml:45,80` / 两个同名 `test_dna.py` 且两处无 `__init__.py`）；`triage_check.py` 静态自证 `PASS`（20 条 / open 13 / fixed 7）。修订后：**`loop_check.py` PASS**（三项 + 附加守卫 0 问题 / SCOPE 四层自证）；`tests/` 子集 **147 passed**（143+4，退出码 0）；既有镜像回归 4 项原样通过；`hive/tests/` 全量 **13 failed/461 passed**（254.55s，与 R5 一致）；`triage_check.py --report` 对账 **`FAILED 13 == open 13` PASS**（无新失败 / 静默修复 / 回归）；`smoke_runtime.py --cycles 1` **PASS**（EXIT=0 / 209 行 / CPU 16 cores / 正常收尾）。**失效态注入反证**：临时目录构造「工具缺失」与「台账自证 FAIL」⇒ 守卫精确判红，同函数对 PASS 临时链 0 问题，全程不改写真实工作区。功能主提交 **`1148c87`**（2 文件 +131/-7）；台账回填提交另计（sha 不写死）。复利回链 **7 条** hits+1，回读校验 RULES `1148c87` 14 处 / 1240 行、CAND 580 行 / 竖线 2709 不变 / 两处 hits 同步 / 100% CRLF | 下一步：**先推送 + 同步镜像**（轮末 `git fetch` + `gh api` 复核未分叉；直连不通试代理 7897 一次）；**R7 恢复功能推进**：P1-9 余 13 条（断言过期 10 / 状态污染 2 / `#18` 行为决策 1）、P1-2 tag 口径、补装 `chromadb/openai/ollama/psycopg2/flask` 验证完整功能；P0-3 需用户行为决策、P0-1 决策权归用户，均跳过；「待人工」小节全部跳过。**轮次 R0+R1+…+R6 = 7，R6 元审视已完成，下次 R 轮 R12** |
| 2026-11-12 02:36 起 | R7（推进 P1-9③ + 验收口径去写死） | 取 **P1-9③**「状态污染 2 条」：`hive/tests/test_evolution_integration.py` 的 `get_swarm_communication()` 是进程级单例、`registered_bees` 跨用例累积 ⇒ `setUp` 快照-清空 + `addCleanup` 复原；`hive/tests/test_phase3_evolution.py` 的 `CapabilityGapAnalyzer` 默认 `_persistence_path` 指向共享 `hive/data/gap_analyzer`、堆着历史 `report_*.json` ⇒ `setUp` 重定向到独立 `TemporaryDirectory` + `addCleanup` 回收。只改测试 fixture、不改产品行为，字节级保持 100% CRLF。新增 `tests/core/test_state_isolation_r7.py` 6 项回归（AST 隔离块守卫 / 污染态运行时真实调用 / 剥离隔离后旧实现注入反证 / 字节级 sha256 SAME，全程不改写真实源码）。同步：`tools/triage_check.py` 的 `LEDGER` #2/#9 转 `已修(R7)`/`fixed`、`EXPECTED_OPEN` 13→11；`tools/loop_check.py` 写死的「13 个 open」改为 `ledger_open_count()` 从台账 `EXPECTED_OPEN` 静态 AST 派生（单一真源，修一条不再产出假陈述），`tests/utils/test_loop_check_acceptance_chain.py` 改动态断言并加 7/3/缺失三态派生反证。轮首补推 R6 积压 2 提交（`1148c87` + `65fa80a`）：`git fetch` + `gh api` 复核远端仍 `1bdd4069` 未分叉后裸直连一次成功 `1bdd406..65fa80a`，镜像 `pull --ff-only` 同步，三方对齐 `65fa80a1` | 开工基线：工作区干净、HEAD `65fa80a1`、49 提交 / 407 跟踪文件、ahead 0；静态体检 `PASS`（332 .py / 119 hive.\* / 三方对齐 / 附加守卫 0 问题）。**四层验收（最终树复跑）**：① `loop_check.py` `PASS`（333 .py / 119 hive.\* / SCOPE 动态输出 11 个 open）；② `smoke_runtime.py --cycles 1` `PASS`（托管 venv，EXIT=0 / 209 行 / CPU 16 cores / 正常收尾）；③ `tests/` 子集 **154 passed**（R6 147 + 7）、`hive/tests/` 全量 **11 failed/463 passed**（474 collected，251.39s；比 R6 的 13 failed/461 passed 恰少 2 条、无新增失败）；④ `triage_check.py --report` 对账 **`FAILED 11 == open 11` PASS**（无新失败 / 无静默修复 / 无回归）。`triage` 静态自证 `PASS`（20 条 / open 11 / fixed 9）。运行时副产物 `cognitive_state.json` 已 `git checkout --` 复原。功能主提交 **`c40dfbc`**（6 文件 +353/-13），台账回填提交另计（sha 不写死）。复利回链 **7 条** hits+1：`RULES` 1240→1247 行 / `c40dfbc` 14 处，`CAND` 580 行 / 竖线 2709 不变 / 两处 hits 同步 / 两 ledger 100% CRLF | 先推送 + 同步镜像（轮末 `git fetch` + `gh api` 复核未分叉；直连不通试代理 7897 一次）；**R8 继续功能推进**：P1-9 余 11 条（断言过期 10 / `#18` 行为决策 1）、P1-2 tag 口径、补装 `chromadb/openai/ollama/psycopg2/flask` 验证完整功能；P0-3 需用户行为决策、P0-1 决策权归用户、「待人工」小节一律跳过。**轮次 R0+R1+…+R7 = 8，R6 元审视已完成，下次 R 轮 R12** |
| 2026-11-12 03:05 起 | R8（用户授权「推进本项目」＝按方案 A 收口 P0-1） | **P0-1 完成**：① `.gitignore` 用单版本白名单放行 `pwp_qa-0.2.3...whl`（60KB，其余 8 个历史 wheel 仍排除）；② workflow 两处触发分支 `[main, develop]` → `[master]`（R0 漏掉的第二断点）；③ 更正 R0 台账与自动化 prompt 里「私有 wheel／不得擅自提交」的错误前提（拆包核实 MIT／无凭证／PyPI 404），并把「待人工」里的 P0-1 决策项划掉。另修 `loop_check.py` SCOPE 写死的「P0-1/P0-3 未清」⇒ 新增 `p0_open_ids()` 从本台账 P0 标题 ✅ 标记静态派生，配 4 项回归。侦察产出立 **P1-11**（该 workflow 是报告型任务、非质量门禁），不本轮扩面 | 功能主提交 **`c6266a4`**（4 文件 +21/-13，含 60491 B wheel）；`git status -uall vendor/` 实测只放行 1 个、`git check-ignore -q` 对 0.2.4 仍判忽略；workflow 引用路径与白名单 grep 一致、`gh api` 确认默认分支 `master`。**本地实证 CI 命令链**：临时目录用托管 venv 经阿里源实装 wheel + tree-sitter、设 `PWP_QA_CACHE_DIR` 后原样跑 `baseline establish` / `baseline check` / `trend --backend mock -n 3`，三条 EXIT 全 0；同时实测「漂移告警仍以 0 退出」。`四层验收`：① `loop_check.py` PASS（334 .py / 119 hive.\* / SCOPE 派生 P0）；② `smoke_runtime.py --cycles 1` PASS（EXIT=0 / 211 行 / CPU 16 cores）；③ `tests/` 子集 **154 passed**、`hive/tests/` 全量 **11 failed/463 passed**（474 collected，与 R7 一致）；④ `triage_check.py --report` 对账 `FAILED 11 == open 11` PASS。复利回链 7 条 hits+1（CAND 581 行／竖线 2709 不变／含 sha 0 处；RULES 1248→1255 行／`c6266a4` 14 处／两 ledger 100% CRLF）。**轮末推送**：fetch 直连 reset、push 直连 reset、代理 7897 连不上，各一次即止；`gh api` 复核远端仍 `b858ce3` 未分叉、本地领先（以实测为准），收尾状态一律以 `git rev-list --count origin/master..HEAD` 实测为准（见快照「推送状态」行，不写死） | **R9 继续功能推进**：P1-9 余 11 条（断言过期 10 / `#18` 行为决策 1）、P1-2 tag 口径、补装 `chromadb/openai/ollama/psycopg2/flask` 验证完整功能；**观察 P0-1 修复后首次被 master 触发的 GitHub Actions 实际日志**（本机 mock 跑通不替代云端）；P0-3 需行为决策、「待人工」小节跳过。**轮次 R0+…+R8 = 9，下次 R 轮 R12** |

| 2026-11-12 03:59 起 | R9（补推闭环 + CI 解析修复） | 轮首重新探测到裸直连恢复，一次补推 R8 积压 3 提交 `b858ce3..cd19c4a`、同步镜像、三方对齐。随后查 Actions 发现 **PWP-QA `cd19c4a` run `43092432401` = success**（P0-1 云端判据落地），但 `test.yml` 从 11-09 起每次 push 都是 **0 jobs / 无日志 / name 退化为路径**的空失败 run。PyYAML 实测锁定根因：`test.yml:186` 起 docker job 的 `python -c` 多行代码顶格写在 `run: \|` 块标量内，截断整个块 ⇒ `ScannerError: could not find expected ':'`。四处修复：① docker 的 5 个 import 检查压成单个合法 shell 行；② `test.yml`/`lint.yml` 的 push 与 PR 分支 `[main, develop]` → `[master]`，`test.yml` 四处 job 级 `refs/heads/main` → `master`；③ `pyproject.toml` dev extras 补 `pytest-timeout>=2.2.0`（workflow 用了 `--timeout` 却从未声明，本地实测 `unrecognized arguments` EXIT=4）；④ 新增 `tests/utils/test_workflow_yaml_r9.py` 11 项回归（解析/name、分支口径、docker step 语义、插件声明、tmp_path 顶格注入反证、AST 防写死守卫）。仅改配置与依赖声明，不动产品代码 | 功能主提交 **`5e57aef`**（4 文件）。**云端解析实测**：`gh api` workflows 列表里 `test.yml` 的 name 由 `.github/workflows/test.yml` 恢复为 **`Tests`**，`5e57aef` 一次触发 Tests / PWP-QA / Code Quality 三个 workflow（此前 Code Quality 从未触发）。**四层验收**：① `loop_check.py` PASS（**335 .py / 119 hive.\* / 本地领先 1 待推送 / 附加守卫 0 问题**）；② `smoke_runtime.py --cycles 1` PASS（EXIT=0 / 209 行 / CPU 16 cores）；③ `tests/` 子集 **169 passed**（158+11）、`hive/tests/` 全量 **11 failed/463 passed**（254.47s，与 R8 一致、无新增失败）；④ `triage_check.py --report` 对账 `FAILED 11 == open 11` PASS。运行时副产物已 `git checkout --` 复原、工作区干净。复利回链 7 条 hits+1。通路同轮二次翻转：功能提交后 fetch/push `Recv failure`，跑完四层验收后轮末重新探测，裸直连一次成功 `cd19c4a..5e57aef` | **R10 继续功能推进**：观察云端 Install dependencies 与 fast tests 实际日志（P1-12）、按 P1-9① 逐条核契约修 10 条断言过期、P1-2 tag 口径；P0-3 需行为决策、「待人工」小节跳过。**轮次 R0+…+R9 = 10，下次 R 轮 R12**（R12 须复核 workflow 配置守卫是否纳入 `loop_check.py`） |

**R9 里程碑判定：不算**。① P0 未清零（P0-2 完整运行时、P0-3 行为决策仍在）；② 运行时非「首次跑通」（R1 已达成，本轮只复跑）；③ 未打版本 tag；④ 完整运行时/服务仍不可用（依赖未装齐、`--cycles 2` 仍 FAIL、11 个历史失败仍在）。故**未因本轮升格项目状态**；但本轮含 **P0-1 云端结论落地 + 三份 workflow 接通 master** 两项事实变化，需同步更新资料库档案，属事实纠错而非里程碑更新。CI 从「不触发」变为「真实运行」后预计仍会红（P1-12），不得据此写成「CI 已绿」。

**R8 里程碑判定：不算**。① P0 **未清零**：P0-1 已完成，但 P0-2（完整运行时）与 P0-3（`hunt_sync` 行为决策）仍在；② 运行时非「首次跑通」（R1 已达成有界冒烟，本轮只复跑确认未回归）；③ 未打版本 tag；④ 完整运行时/服务仍不可用（依赖未装齐、`--cycles 2` 仍 FAIL、11 个历史失败仍在）。故**未因本轮升格项目状态**。但本轮含**事实纠错**，需单独核对资料库档案是否残留「私有 wheel」旧说法，属纠错而非里程碑更新。修 CI 断点、体检口径去写死与复利回链属流程/工具状态，不是项目能力里程碑。

**R7 里程碑判定：不算**。① P0 未清零（P0-1 待用户决策、P0-3 需行为决策）；② 运行时非「首次跑通」（R1 已达成有界冒烟，本轮只复跑确认未回归）；③ 未打版本 tag；④ 完整运行时/服务仍不可用（依赖未装齐、`--cycles 2` 仍 FAIL、11 个历史失败仍在）。故**未更新**资料库档案 `URL2Tg73gmbxGrjut4nPoj`。修测试 fixture、台账去写死与复利回链属流程/测试状态，不是项目能力里程碑。

**R6 里程碑判定：不算**。① P0 未清零（P0-1 待用户决策、P0-3 需行为决策）；② 运行时非「首次跑通」（R1 已达成有界冒烟，本轮只复跑确认未回归）；③ 未打版本 tag；④ 完整运行时/服务仍不可用（依赖未装齐、`--cycles 2` 仍 FAIL、13 个历史失败仍在）。故**未更新**资料库档案 `URL2Tg73gmbxGrjut4nPoj`。元审视、验收机制修订与复利回链属流程状态，不是项目能力里程碑。

**R5 里程碑判定：不算**。① P0 未清零（P0-1 待用户决策、P0-3 需行为决策）；② 运行时非「首次跑通」（R1 已达成有界冒烟，本轮只是复跑确认未回归）；③ 未打版本 tag；④ 完整运行时/服务仍不可用（依赖未装齐、`--cycles 2` 仍 FAIL、13 个历史失败仍在）。故**未更新**资料库档案 `URL2Tg73gmbxGrjut4nPoj`。修测试代码、台账升级与复利回链属流程状态，不是项目能力里程碑。


**R4 收尾补记（推送成功，append-only）**：轮末 `git fetch origin master` 直连一次成功、`gh api` 复核远端仍 `118c2798` 未分叉（behind 0 / ahead 2）→ **裸直连 `git push origin master` 一次成功 `118c279..00acf78`**（2 提交：`d92ef5f` 功能 + `00acf78` 台账回填）→ 镜像 `git -C D:/repos/hive-mind pull --ff-only origin master` 直连一次成功 `118c279..00acf78` → `gh api` 复核远端 `00acf78`。**三方对齐 `00acf78`、ahead 0、工作区干净**。末次静态体检 `PASS`（**330 个 .py / 119 个 hive.\* 目标 / 镜像一致**）；实测快照：44 提交 / 405 跟踪文件 / 330 个 .py。`docs/reports/LEGACY_TEST_TRIAGE_R4.md` 经 `git ls-files` 确认已入库（`-f` 生效）。⇒ 上方 R4 行「下一步：先推送 + 同步镜像」是**轮中时点的真值，不是收尾状态**；**下一轮无需补推**，仍按惯例开工先 `git fetch` + `gh api` 复核，别照抄本补记结论（`R-20261015-05` 通路不跨时刻）。本补记自身产生一个回填提交，其 sha 不写死于此。

**R4 里程碑判定：不算**。① P0 未清零（P0-1 待用户决策、P0-3 需行为决策）；② 运行时非「首次跑通」（R1 已达成有界冒烟，本轮只是复跑）；③ 未打版本 tag；④ 完整运行时/服务仍不可用（依赖未装齐、`--cycles 2` 仍 FAIL、19 个历史失败仍在）。故**未更新**资料库档案 `URL2Tg73gmbxGrjut4nPoj`。分诊台账与推送/三方对齐属流程状态，不是项目能力里程碑。

**R3 里程碑判定：不算**。① P0 未清零（P0-1 待用户决策、P0-3 需行为决策）；② 运行时非「首次跑通」（R1 已达成有界冒烟，本轮只是复跑确认未回归）；③ 未打版本 tag；④ 完整运行时/服务仍不可用（依赖未装齐、`--cycles 2` 仍 FAIL）。故**未更新**资料库档案 `URL2Tg73gmbxGrjut4nPoj`。

**R3 两处口径漂移登记（append-only，供 R 轮复核）**：
- **台账数字漂**：R2 行与快照里的「全量 pytest 101 passed」经实测是 `tests/` 子集口径；全仓库 `pytest -q` 实为 **455 passed / 20 failed**，20 个失败集中在 `hive/tests/`（抽样 `test_nonexistent_image` 因缺 `brain_config.json` 导致 BrainInterface 加载失败、旧断言 `success` 期望值过期）。本轮未修，立为 **P1-8**。今后台账写 pytest 数字必须带子集范围。
- **「本机碰巧能用」的真源**：`.env` 第 10 行 `HIVE_HOME=D:/HIVE`、第 13 行 `BASE_DIR=D:/HIVE/Data/Inbound`，由 `hive/__init__.py` 启动时注入 `os.environ`；pydantic 还会用 `.env` 的 `BASE_DIR` 覆盖 `_default_base_dir()`。R2 归因于「Windows 大小写不敏感」只说对一半——真正让读写两端对齐的是这份 `.env`。它与代码兜底判据语义不同（外部配置 vs 代码默认），按 `R-20261024-01` 不合并、不擅自改用户的 `.env`。

**R3 收尾补记（同轮内通路二次翻转，append-only）**：轮首补推 R2 积压 2 提交时直连 / 代理各一次均失败；**轮末做完功能与台账回填后重新探测，裸直连 `git push origin master` 一次成功 `fb757cd..e8aebfd`**（4 个提交一并推上：R2 积压 `d36ed04` / `148c6f5` + R3 功能 `696b76e` + 台账 `e8aebfd`）。`gh api` 复核远端 `e8aebfd`、未分叉。⇒ **R3 行「轮首补推失败」是轮首时点的真值，不是收尾状态**；收尾结论以本补记为准。这是 `R-20261015-05`「通路不跨时刻」在同一轮内的第二次实证（R2 是先成功后失败，R3 是先失败后成功），故两种方向都不能写进长期配方，也不能因一次失败就认定通路已死。**本补记自身会产生一个回填提交**：若其推送成功，则三方对齐；若失败，则本地领先数按 `git rev-list --count origin/master..HEAD` 实测、不写死，下一轮照例先 fetch + `gh api` 复核再补推。

**R2 里程碑判定：不算**。① P0 未清零（P0-1 待用户决策、P0-3 需行为决策）；② 运行时非「首次跑通」（R1 已达成有界冒烟，本轮只是复跑确认未回归）；③ 未打版本 tag；④ 完整运行时/服务仍不可用（依赖未装齐、`--cycles 2` 仍 FAIL）。故**未更新**资料库档案 `URL2Tg73gmbxGrjut4nPoj`。另注：推送与三方对齐属流程状态，不是项目能力里程碑，不据此升格项目状态。

**R2 收尾补记（同轮内通路二次翻转，append-only）**：功能提交 `fb757cd` 及其之前的 R1 积压 4 个提交**已成功推上远端**，镜像已 `pull --ff-only` 同步（这是本轮实质补推成果）。但随后生成的**台账回填提交 `d36ed04`** 推送失败：`git fetch` 直连 `Recv failure: Connection was reset`、裸直连 push 同错；按约定改试代理 7897（HTTP/1.1 + TLS1.2 + openssl）一次 → `TLS connect error: unexpected eof while reading`，代理侧镜像 pull 同样失败。**各试一次即止，未反复重试**。`gh api repos/xtwp1024/hive-mind/commits/master` 独立复核远端仍 `fb757cd`、**未分叉**；末次静态体检 `PASS`（326 .py / 116 import / 镜像=origin/master、本地领先），工作区干净。⇒ **R2 行里的「三方对齐、ahead 0」是功能提交时点的阶段性真值，不是收尾状态**；收尾为「本地领先、待补推」，**领先数不写死**——本补记及后续台账修正各自的提交都会让它 +1，一律以 `git rev-list --count origin/master..HEAD` 实测为准。
- **同一轮内裸直连先成功后失败**，是 `R-20261015-05`「通路不跨时刻」最紧的一次实证：间隔不到 10 分钟、同一客户端同一命令，结果反转。故不能把「直连已通」写进任何长期配方，也不能因一次成功就认定代理侧永久失效。
- **本轮第二次出现「回填提交顶掉快照」**（R1 是漏记回填提交，R2 是回填提交自己造成领先）。已把快照「推送状态」行改为纯指针式，不再写死三方对齐/领先数；下次 R 轮应复核：这类易变字段是否还有残留写死值。
- **下一轮第一件事：补推 `d36ed04` 及其后的台账修正提交**（直连 / 代理各试一次，仍失败只记一行）；推送成功后再 `git -C D:/repos/hive-mind pull --ff-only` 同步镜像并复跑体检。之后才推进 P0-3 决策 / P1-7 / 补装依赖。

**R1 元审视（提前触发，理由：本轮实证「验收口径覆盖不住新风险」，属立即触发条款）**
- 问1 验收口径是否漏新风险：**是，且是本轮最大发现**。静态三项全 PASS 仍漏 3 处运行时缺陷 ⇒ 新增第 4 层 `tools/smoke_runtime.py`（依赖缺失显式 SKIP、error 级别日志纳入判定，不恒绿）；`loop_check.py` 增加 `SCOPE` 自证行，明确「PASS ≠ 服务能跑」。
- 问2 待办是否过期/重复：P0-2 描述「从未跑通」已过期，改为「首次有界冒烟达成 + 项目仍非可用」；新暴露缺陷立为 P0-3，未与既有项重复。
- 问3 路径/假设是否漂移：**是**。① 台账快照写 `3478df9`/31，实测 `12b41ef`/32（R0 回填提交未计入）；②「三方 HEAD 严格相等」假设与「每轮先 commit 再 push」流程冲突，已改；③「依赖装不上」的假设被推翻——不是装不上，是默认 pypi 源不通，阿里源可用。
- 轮次计数：R0 建档 + R1 = **2 轮**，距 R6 元审视还差 4 轮。
- 里程碑判定：**本轮不算里程碑轮**，故未更新资料库档案 `URL2Tg73gmbxGrjut4nPoj`。理由：① 仅 1 周期有界冒烟 PASS，`--cycles 2` 仍 FAIL（P0-3），`chromadb/openai/ollama/psycopg2/flask` 未装，完整运行时未跑通，按「未跑通前不得写成可用」不能升格；② 当前资料库工具仅支持 search / export / upload，无编辑既有文档能力。待 P0-3 修完、≥2 周期冒烟 PASS 且推送成功后再补档案。
