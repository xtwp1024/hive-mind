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
- 私有二进制（`vendor/*.whl`）**不得擅自提交到公开仓库**，需要用户明确决定。
- 单轮预算控制在 40 分钟内，宁可少做也不拖轮。

## 二、固定验收口径（每轮必跑，先于任何改动）

```bash
cd D:/Hive
git status --porcelain                      # 工作区基线
python tools/loop_check.py                  # 静态体检：见下
```

`loop_check.py` 必须给出三项结论，任一失败即本轮转为修复，不做新功能：

1. **语法**：所有入库 `.py` 通过 `py_compile`
2. **模块闭合**：所有 `from hive.* / import hive.*` 目标在检出树中可解析
3. **镜像一致**（R1 修订）：`D:/repos/hive-mind` HEAD 必须 == `origin/master`；主工作区 HEAD 允许**领先**（待推送的正常推进态），但落后/分叉仍 FAIL。旧口径要求三方严格相等，会在每轮 commit 后、push 前误报基线已坏。

**第 4 层（可选，R1 新增）**：`python tools/smoke_runtime.py --cycles 1`（须用装了依赖的托管 venv）。依赖缺失时退出码 2 显式 SKIP，**不算通过**；2 周期以上会暴露 P0-3，目前预期 FAIL。

静态体检只保证「能解析、能编译」，**不等于服务能跑**。R1 实证：静态三项全 PASS 时仍存在 3 处只在运行时暴露的缺陷。未跑通前不得把项目状态写成「可用」。

## 三、当前快照（2026-11-10 R2 刷新）

> **易变值不写死**（R1 修订，源自 R-20261010-19「HEAD 是会移动的状态」）：
> HEAD / 提交数 / 领先数一律以 git 实测为准，本表只记**不可变的开工基线**与**验收结论**。
> 开工前跑：`git rev-parse HEAD` / `git rev-list --count HEAD` / `git ls-files | wc -l`。
> R0 快照曾漏记 `12b41ef` 回填提交；R1 回填时又让「33 / 领先 1」立刻过期，故改为指针式。
> 不可变历史：**R1 功能主提交 `ec0930c`**（7 文件 +358/-33）、**R2 功能主提交 `fb757cd`**（2 文件 +153/-3）；
> 完整列表用 `git log --oneline 12b41ef..HEAD` 查，不在本表逐个枚举（枚举本身会过期）。

| 项 | 值 |
| --- | --- |
| 开工基线 HEAD | `12b41ef`（R1 轮首实测，远端 `origin/master` 同值；R2 开工时本地领先 4 个待补推提交） |
| 当前 HEAD | **以 `git rev-parse HEAD` 为准，不写死**：R2 功能主提交是 `fb757cd`，但本表自身的回填提交会让 HEAD 再前移一位（R1 正是漏记这类回填导致快照漂移）。查完整链用 `git log --oneline 12b41ef..HEAD` |
| 提交数 | 以 `git rev-list --count HEAD` 为准（**回填提交会 +1，不写死具体值**） |
| 跟踪文件 | 以 `git ls-files \| wc -l` 为准（R2 新增 1 个入库文件 `tests/utils/test_event_stream_home.py`；其中 .py 数由 325 → 326） |
| 工作区 | 干净（R2 功能提交后、台账回填前实测） |
| 推送状态 | **以 `git rev-list --count origin/master..HEAD` 实测为准，不写死**：本行曾被 R2 回填提交自己顶成过期值 ⇒ 改为纯指针式（R1 同款坑的第二次出现）。**口径**：本地领先是「已 commit、待推送」的正常推进态，体检按 R1 修订允许落后以外的领先；每轮**先补推**再同步镜像。R2 实测事件见「八、推进日志」R2 行与下方收尾补记 |
| 静态体检 | `PASS`（326 个 .py 语法 0 失败 / 116 个 hive.* 目标 0 缺失；镜像与 `origin/master` 一致，主工作区允许领先） |
| 运行时 | **已首次跑通且有界冒烟 PASS**：`smoke_runtime.py --cycles 1` EXIT=0、10 个 SUCCESS 载荷、正常收尾；`--cycles 2` 仍 FAIL（P0-3） |
| 验证层级 | 静态 PASS + 1 周期冒烟 PASS + pytest 101 passed；**完整运行时/服务仍不可用**（依赖未装齐、P0-1/P0-3 未清） |
| 资料库档案 | `https://www.workbuddy.cn/space/d/URL2Tg73gmbxGrjut4nPoj` |

2026-11-10 R0 入库 4 个提交：`d14135d` 运行时健壮性（僵死 Worker 回收、缓存落 D 盘、哈希嵌入）、`db74a7a` 可视化（事件流 + swarm.html + warmup 回退）、`ee24c6f` CI 版本 bump、`3478df9` 本台账与体检脚本，外加 `12b41ef` 回填快照。清理的是 10/31–11/01 遗留的 15 项未提交改动。

## 四、待办队列

### P0 · 阻塞「可用」

**P0-1　CI 引用了仓库里没有的私有 wheel**
`.github/workflows/pwp-qa.yml` 两处 `pip install ./vendor/pwp_qa-0.2.3-py3-none-any.whl`，但 `vendor/*.whl` 被 `.gitignore` 排除，远端 CI 检不到文件，必然失败。这不是版本 bump 引入的，是从 10-28 建库起就存在的结构性缺陷。
候选解法（**需用户决策，不要替他选**）：
- A. 用 `git add -f` 把 0.2.3 纳入版本库（60KB，但属私有工具，公开仓库有泄露顾虑）
- B. workflow 改为从内部源安装，缺源时显式跳过并标注 skip 原因
- C. 把 QA 步骤改成不依赖 pwp-qa 的等价静态检查

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
- **P1-5** `vendor/` 现有 8 个 wheel（0.1.0–0.2.4），CI 只用 0.2.3；确认保留策略，别让版本继续无序堆积。
- **P1-6** `web/swarm.html` 默认深色底 + emoji 标题，与既定「白主 / 黑 / 蓝辅、不用 emoji 改用开源图标、每个元素有出现-过渡-消失动画」规范不一致；按规范改版。
- **P1-7（R2 侦察新增，未修）** P1-1 只修了 `event_stream.py`，**同源代码仍散落在 3 处**：`hive/__init__.py:35` 的 `_Path.cwd() / "D:/HIVE"`（缓存根锚点，与 P1-1 完全同形）、`hive/config/settings.py:34` 的 `Path("D:/HIVE/Data/Inbound")`（Windows 无 `HIVE_HOME` 时硬编码 D 盘）、`hive/bees/variants/librarian_bee.py:24` 的 `_os.getenv("HIVE_HOME", _os.getcwd())`（回退到 cwd）。另有 `web_dashboard.py:26` 空串 `HIVE_HOME` 会落 `Path("")`。建议收敛成**单一根锚点模块**（如 `hive/paths.py`）再逐处替换，属跨模块重构，需独立一轮；本轮按「宁可少做也不拖轮」只登记不扩面。

### P2 · 清理与归档

- **P2-1** `.temp_aden_hive` 是带独立 `.git` 的副本，确认后清理，避免和主工作区混淆。
- **P2-2** `Hive_Standalone` / `Hive_V1_Backup` 在 `D:\_D盘整理_20260930\archive\`，比对版本差异再决定是否合并。
- **P2-3** 运行期账本 json（mining_ledger / discovery_history / science_journal）已排除入库，确认是否需要单独归档策略。

### 待人工（自动化不要碰）

- 是否把私有 pwp-qa wheel 放进公开仓库（P0-1 的决策权）
- 是否调整项目公开/私有状态
- 归档副本的去留

## 五、每轮流程

1. 读本文件「四、待办队列」，取优先级最高且**可独立完成、不依赖待人工决策**的一项。
2. 跑「二、固定验收口径」建立基线；基线已坏 → 本轮只修基线。
3. 实施改动 → 复跑验收 → 记录证据（命令 + 输出摘要 + 提交 sha）。
4. 本地 commit。**推送前**确认远端未分叉；推不动就记一行原因，不反复重试。
5. 更新本文件：待办状态、当前快照、追加「八、推进日志」一行。
6. 里程碑轮（P0 清零、版本 tag、运行时首次跑通）额外更新资料库档案文档。
7. 末尾追加工作区日期日志 `C:/Users/Administrator/WorkBuddy/2026-10-28-16-14-39/.workbuddy/memory/YYYY-MM-DD.md`（append-only）。

> R2 口径修正：本节原写「七、推进日志」，但实际推进日志是**第八节**（第七节是复利回链）。已改正。
> 另：本节第 4 步「推送前确认远端未分叉；推不动就记一行原因，不反复重试」在 R2 首次**成功**兑现推送，故补一条：推送成功后须 `git -C D:/repos/hive-mind pull --ff-only` 同步镜像并复跑体检确认三方一致（R1 因推送失败未走到这一步）。

## 六、元审视（随项目演化修订本机制）

每累计 6 轮触发一次 R 轮，不推进功能，只回答三个问题并直接改本文件：

- 验收口径是否已经覆盖不住新风险？（例：引入 FastAPI 后要不要加路由 smoke）
- 待办队列有没有过期项 / 重复项 / 需要升级优先级的？
- 本文件里的路径、版本、自动假设是否随项目漂移？

触发条件变化时（换了主工作区、新增服务、依赖大改）立即触发，不等满 6 轮。
**这条是长效推进不腐化的关键：机制本身要跟着项目改，不能只让项目跟着机制转。**

> **当前轮次指针（唯一状态源，R2 收尾）**：R0 建档 + R1 + R2 = **3 轮**，距 R6 元审视还差 **3 轮**。
> 下方「R1 元审视」段末尾写的「还差 4 轮」是**当时的历史结论**，不是当前值，勿据此判断。
> R2 未触发即时元审视：主工作区未变更、无新增服务、无依赖大改；本节与 `tools/loop_check.py` 本轮均不改。
> 顺带记一条待 R 轮复核的问题：R2 实证「静态体检对路径/配置类缺陷完全无感（注入旧实现时仍 PASS）」，
> 是否该给静态层补一条**跨写读两端的路径契约检查**，还是继续只靠运行时层回归覆盖？留待 R 轮判断，不提前改机制。

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

## 八、推进日志

| 时间 | 轮次 | 做了什么 | 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-11-10 | R0 建档 | 清理 10/31–11/01 遗留 15 项未提交改动，分 3 提交入库；排除 `cache/`；建立本台账、体检脚本与长效自动化 | `d14135d` `db74a7a` `ee24c6f` `3478df9` `12b41ef`；12 文件 py_compile 全过；末次体检 `PASS` | 推进 P0-2（运行时验证），P0-1 待用户决策 |
| 2026-11-10 10:25–10:58 | R1（推进 + 即时口径修订） | 打通 P0-2：实测阿里源装齐启动链依赖、`main.py` **首次真实跑通**并正常收尾；修静态 PASS 漏掉的 2 处运行时缺陷（缺 `import os`、`get_system_info` 少 key）；新增有界周期开关、第 4 层 `smoke_runtime.py`、9 项回归测试；就地把镜像验收口径改为「允许本地领先」并令体检自证范围。遗留 `hunt_sync` 缺陷拆为 P0-3（需行为决策，未擅改） | 体检 `PASS`（322 .py / 116 import / 镜像对齐）；`smoke_runtime.py --cycles 1` = `PASS`（EXIT=0、10 SUCCESS、CPU: 16 cores），`--cycles 2` = `FAIL`（精确捕获 `asyncio.run() cannot be called from a running event loop`）；pytest 9 passed；旧实现注入反证复现 `name 'os' is not defined` 与全 0 默认值；`gh api` 复核远端 `12b41ef`、开工 ahead 0；R1 主提交 `ec0930c`（7 文件 +358/-33）；旧口径反证：此时三方 distinct=2，旧规则会误报 FAIL | **下一轮先补推**（直连/代理各试一次）；再推进 P0-3 修法决策、补装 `chromadb` 等依赖验证完整功能；P0-1 仍待用户决策 |
| 2026-11-10 17:10–17:40 | R2（推进 + 补推清零） | 取 P1-1「事件流根目录跨平台歧义」：改为 `PROJECT_ROOT` + 单一 `HIVE_HOME` 读取，与读取方 `web_dashboard.py` 锚点对齐；新增 11 项回归（含跨平台语义断言、写读同路径契约、真实 emit 落盘解析、AST 契约、旧实现注入反证）。**并补推 R1 积压的 4 个提交 + 本轮 1 个**，同步只读镜像，三方首次对齐。侦察发现同源硬编码散落 3 处，登记 P1-7 不扩面；顺手修正「五、每轮流程」章节号笔误（七→八） | 开工基线体检 `PASS`（325 .py / 116 import / 本地领先 4）；pathlib 实测 `PureWindowsPath('D:/Hive')/'D:/HIVE'` → `D:\HIVE`、`PurePosixPath('/srv/hive')/'D:/HIVE'` → `/srv/hive/D:/HIVE`（同期证明本机「碰巧能用」）；pytest 新增 11 passed / 全量 **101 passed**；旧实现注入反证 `6 failed, 5 passed` 且**静态体检同时仍 PASS** → 字节级复原 `sha256 6510078806374266 SAME` → `11 passed`；`smoke_runtime.py --cycles 1` `PASS`（EXIT=0、209 行、CPU: 16 cores）；真实运行后 `Data/events.jsonl` 593B/3 条落项目根、无畸形目录；`gh api` 复核远端未分叉 → 裸直连 `git push` 一次成功 `12b41ef..fb757cd`（60s 超时防挂起）→ 镜像 `pull --ff-only` → 复跑体检 `PASS`（326 .py / 116 import / **三方对齐 fb757cde**）、工作区干净、ahead 0。主提交 `fb757cd`（2 文件 +153/-3）；R2 台账回填另计 | 下一步：P0-3 修法仍需用户决策（不擅改）；可推进 P1-2 tag 口径确认 / P1-7 单一根锚点收敛；补装 `chromadb/openai/ollama/psycopg2/flask` 验证完整功能；P0-1 仍待用户决策。轮次 R0+R1+R2 = **3**，距 R6 元审视还差 **3** 轮 |

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
