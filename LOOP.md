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

## 三、当前快照（2026-11-10 R1 刷新）

> **易变值不写死**（R1 修订，源自 R-20261010-19「HEAD 是会移动的状态」）：
> HEAD / 提交数 / 领先数一律以 git 实测为准，本表只记**不可变的开工基线**与**验收结论**。
> 开工前跑：`git rev-parse HEAD` / `git rev-list --count HEAD` / `git ls-files | wc -l`。
> R0 快照曾漏记 `12b41ef` 回填提交；R1 回填时又让「33 / 领先 1」立刻过期，故改为指针式。
> 不可变历史：R1 主提交 `ec0930c`（7 文件 +358/-33）、R1 回填提交 `64a4089`。

| 项 | 值 |
| --- | --- |
| 开工基线 HEAD | `12b41ef`（R1 轮首实测，远端 `origin/master` 同值） |
| 当前 HEAD | 以 `git rev-parse HEAD` 为准（R1 交付共 2 个本地提交，待推送） |
| 提交数 | 以 `git rev-list --count HEAD` 为准（R1 开工基线 32） |
| 跟踪文件 | 以 `git ls-files \| wc -l` 为准（R1 交付后 399，其中 325 个 .py；新增 3 个入库文件） |
| 工作区 | 干净（R1 提交后） |
| 推送状态 | **本地领先待补推**：直连 `Recv failure: Connection was reset`、代理 7897 `schannel: failed to receive handshake`；按约定各试一次即止，未反复重试。`gh api` 复核远端仍 `12b41ef`，未分叉。领先数以 `git rev-list --count origin/master..HEAD` 为准 |
| 静态体检 | `PASS`（325 个 .py 语法 0 失败 / 116 个 hive.* 目标 0 缺失 / 镜像与远端一致、本地领先待推送） |
| 运行时 | **已首次跑通且有界冒烟 PASS**：`smoke_runtime.py --cycles 1` EXIT=0、10 个 SUCCESS 载荷、正常收尾；`--cycles 2` 仍 FAIL（P0-3） |
| 验证层级 | 静态 PASS + 1 周期冒烟 PASS；**完整运行时/服务仍不可用**（依赖未装齐、P0-1/P0-3 未清） |
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

- **P1-1** `hive/utils/event_stream.py` 里 `os.getenv("HIVE_HOME", str(Path.cwd() / "D:/HIVE"))` 回退路径跨平台歧义，应改成明确默认值 + 单一环境变量读取。
- **P1-2** 版本 1.0.0 只写在 README / pyproject，远端没打 tag；确认版本口径后补 `git tag v1.0.0`。
- **P1-3** `hive/core/vector_vault.py` 换成 SHA256 哈希嵌入后，语义检索能力实质下降（哈希无语义）。需要评估：是保留作离线兜底，还是加回可配置的可选语义嵌入。
- **P1-4** `obsidian_notes/` 9 篇开发记录（五阶段进化路线图、代码改进总结、测试汇总）是项目史一手材料，整理进 `docs/` 后入库。
- **P1-5** `vendor/` 现有 8 个 wheel（0.1.0–0.2.4），CI 只用 0.2.3；确认保留策略，别让版本继续无序堆积。
- **P1-6** `web/swarm.html` 默认深色底 + emoji 标题，与既定「白主 / 黑 / 蓝辅、不用 emoji 改用开源图标、每个元素有出现-过渡-消失动画」规范不一致；按规范改版。

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
5. 更新本文件：待办状态、当前快照、追加「七、推进日志」一行。
6. 里程碑轮（P0 清零、版本 tag、运行时首次跑通）额外更新资料库档案文档。
7. 末尾追加工作区日期日志 `C:/Users/Administrator/WorkBuddy/2026-10-28-16-14-39/.workbuddy/memory/YYYY-MM-DD.md`（append-only）。

## 六、元审视（随项目演化修订本机制）

每累计 6 轮触发一次 R 轮，不推进功能，只回答三个问题并直接改本文件：

- 验收口径是否已经覆盖不住新风险？（例：引入 FastAPI 后要不要加路由 smoke）
- 待办队列有没有过期项 / 重复项 / 需要升级优先级的？
- 本文件里的路径、版本、自动假设是否随项目漂移？

触发条件变化时（换了主工作区、新增服务、依赖大改）立即触发，不等满 6 轮。
**这条是长效推进不腐化的关键：机制本身要跟着项目改，不能只让项目跟着机制转。**

## 七、复利回链

开工前先扫 `D:/编程复利系统/rules/RULES.md`；命中既有规则时直接复用，并在对应条目上 `hits+1`（附 commit sha 或路径作证据）。没命中不硬凑。
本项目已产出的一条经验：「提交时只 add 当阶段文件 → 远端仓库变成残缺快照且不报错」，值得回写候选池。

**R1 命中 2 条（已回链、已回读校验）**：
- `R-20261015-05`「通路配方不跨对象保证，换源/换客户端必须重新探测」：pypi 直连 + 代理 7897 + 清华源全败、阿里源成功。hits `11 → 12`，证据追加至 `CANDIDATES.md` C-20261015-05、`RULES.md` 复用行。
- `R-20261010-19`「不要信状态文档，先 git log + 实测」：本台账快照 `3478df9`/31 与实测 `12b41ef`/32 漂移；另用 `gh api` 独立复核远端。hits `14 → 15`，证据追加至 C-20261010-19、`RULES.md` 复用行。
- 同时复用了 `R-20261010-11`「旧实现注入反证」与 `R-20261010-08`「修完必做运行时真实调用」的方法（未重复计数，避免硬凑）。

**本轮新增候选（值得回写候选池，hits=0）**：静态验收口径「全绿」与运行时行为之间存在断层——`py_compile` + import 闭合都过，仍可能有 `NameError` 藏在 `except` 兜底分支里被吞成默认值。**推论：任何带宽泛 `except Exception` 返回兜底值的路径，静态检查天然不可见，必须靠运行时真实调用 + 反证覆盖**（本轮 `get_snapshot()` 的 `name 'os' is not defined` 即此形态）。

## 八、推进日志

| 时间 | 轮次 | 做了什么 | 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-11-10 | R0 建档 | 清理 10/31–11/01 遗留 15 项未提交改动，分 3 提交入库；排除 `cache/`；建立本台账、体检脚本与长效自动化 | `d14135d` `db74a7a` `ee24c6f` `3478df9` `12b41ef`；12 文件 py_compile 全过；末次体检 `PASS` | 推进 P0-2（运行时验证），P0-1 待用户决策 |
| 2026-11-10 10:25–10:58 | R1（推进 + 即时口径修订） | 打通 P0-2：实测阿里源装齐启动链依赖、`main.py` **首次真实跑通**并正常收尾；修静态 PASS 漏掉的 2 处运行时缺陷（缺 `import os`、`get_system_info` 少 key）；新增有界周期开关、第 4 层 `smoke_runtime.py`、9 项回归测试；就地把镜像验收口径改为「允许本地领先」并令体检自证范围。遗留 `hunt_sync` 缺陷拆为 P0-3（需行为决策，未擅改） | 体检 `PASS`（322 .py / 116 import / 镜像对齐）；`smoke_runtime.py --cycles 1` = `PASS`（EXIT=0、10 SUCCESS、CPU: 16 cores），`--cycles 2` = `FAIL`（精确捕获 `asyncio.run() cannot be called from a running event loop`）；pytest 9 passed；旧实现注入反证复现 `name 'os' is not defined` 与全 0 默认值；`gh api` 复核远端 `12b41ef`、开工 ahead 0；R1 主提交 `ec0930c`（7 文件 +358/-33）；旧口径反证：此时三方 distinct=2，旧规则会误报 FAIL | **下一轮先补推**（直连/代理各试一次）；再推进 P0-3 修法决策、补装 `chromadb` 等依赖验证完整功能；P0-1 仍待用户决策 |

**R1 元审视（提前触发，理由：本轮实证「验收口径覆盖不住新风险」，属立即触发条款）**
- 问1 验收口径是否漏新风险：**是，且是本轮最大发现**。静态三项全 PASS 仍漏 3 处运行时缺陷 ⇒ 新增第 4 层 `tools/smoke_runtime.py`（依赖缺失显式 SKIP、error 级别日志纳入判定，不恒绿）；`loop_check.py` 增加 `SCOPE` 自证行，明确「PASS ≠ 服务能跑」。
- 问2 待办是否过期/重复：P0-2 描述「从未跑通」已过期，改为「首次有界冒烟达成 + 项目仍非可用」；新暴露缺陷立为 P0-3，未与既有项重复。
- 问3 路径/假设是否漂移：**是**。① 台账快照写 `3478df9`/31，实测 `12b41ef`/32（R0 回填提交未计入）；②「三方 HEAD 严格相等」假设与「每轮先 commit 再 push」流程冲突，已改；③「依赖装不上」的假设被推翻——不是装不上，是默认 pypi 源不通，阿里源可用。
- 轮次计数：R0 建档 + R1 = **2 轮**，距 R6 元审视还差 4 轮。
