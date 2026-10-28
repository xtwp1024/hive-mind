# 版本说明 · Hive（蜂巢思维群智系统）

> 本文档由 WorkBuddy 于 2026-09-30 盘点生成。同日已按「跟随对外宣称」口径统一版本号（见下）。

| 项 | 信息 |
|---|---|
| 项目目录 | `D:\Hive` |
| 版本标识 | **1.0.0**　（2026-09-30 统一：`pyproject.toml` 由 0.1.0 对齐至 README 的 v1.0.0） |
| 技术栈 | Python 3.8+，Flask / ChromaDB / OpenAI / Ollama / PostgreSQL，Docker |
| 代码规模 | 约 176,300 文件：.png 47,676、.py 36,754、.csv 27,289、.ckpt 5,125 |
| 版本管理 | git `master`，19 次提交，远端 github.com/xtwp1024/hive-mind.git |
| 最近改动 | 2026-04-23（目录）／最后一次提交 2026-04-09「feat: Phase 5 P3 横向扩展 — ScalabilityManager 100+并发支持」 |

## 重复与关联

- 与 `Hive_Standalone`、`Hive_V1_Backup` 构成 Hive 三件套（后两者已于 2026-09-30 移入 `D:\_D盘整理_20260930\archive\`）：
  - `Hive`　＝主开发线（有远端仓库、提交记录最多）
  - `Hive_Standalone`　＝独立/Web 版（README 自称 v2.0，已统一至此）
  - `Hive_V1_Backup`　＝V1 备份（config.yaml 写 2.0.0）
- 目录内嵌 `.temp_aden_hive`，为另一份带独立 `.git` 的副本，建议确认是否需要保留。

## 已知问题 / 待办

1. ~~版本号三处不一致（0.1.0 / v1.0.0 / R24）~~ → **已于 2026-09-30 统一为 1.0.0**（写入 `pyproject.toml`）。建议后续打 git tag 固化。
2. **工作区脏**：`git status` 显示 243 项未提交改动，存在丢失风险，建议尽快整理提交或另存分支。
3. **体积大**：`.png` 近 4.8 万、`.ckpt` 5 千余，属训练/中间产物，建议移出版本库并加 `.gitignore`。
4. **根目录杂物**：`out.txt` / `out2.txt` / `out3.txt` / `db_test.log` / `error_dump.txt` / `progress.txt` 等临时文件建议归档。
