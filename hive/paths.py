# -*- coding: utf-8 -*-
"""蜂巢系统根锚点 - 全项目唯一真源（P1-7）。

背景：HIVE_HOME 的默认值原先散落在四处，拼法互不相同：

    hive/__init__.py                      _Path.cwd() / "D:/HIVE"
    hive/config/settings.py               Path("D:/HIVE/Data/Inbound")
    hive/bees/variants/librarian_bee.py   _os.getenv("HIVE_HOME", _os.getcwd())
    web_dashboard.py                      Path(os.getenv("HIVE_HOME", str(BASE_DIR)))

R2 已实证第一类拼法在 Windows 上被右操作数（绝对路径）整个覆盖成 D:\\HIVE，
在 POSIX 语义下则拼出 <cwd>/D:/HIVE 这种带冒号的怪目录；本机「碰巧能用」
只是大小写不敏感的巧合。第二类硬编码盘符，第三类依赖 cwd，第四类遇到
空串 HIVE_HOME 会落到 Path("") 也就是 cwd。四处写入方与读取方随时可能分家。

收敛口径（只有这一份实现，其余各处只调用）：
    1. HIVE_HOME 非空（strip 后）则用它，expanduser；
    2. 空串 / 纯空白 / 未设置一律按「未设置」处理；
    3. 未设置时回退仓库根 PROJECT_ROOT，不拼 cwd、不写盘符。

语义不同的旁路不纳入：hive/utils/monitor.py 取 HIVE_HOME 首两个字符作为
磁盘使用率的探测目标（盘符语义，不是数据落点），按 R-20261024-01「语义不同
则改名而非合并」保持原样，仅在此显式登记，避免为收口而收口。

本模块只依赖标准库，import 期间零副作用（不建目录、不写环境变量），
因此可被 hive/__init__.py 在包初始化早期安全导入。
"""
from __future__ import annotations

import os
from pathlib import Path

# 本文件位于 <root>/hive/paths.py，上溯一级即仓库根。
# 与 hive/utils/event_stream.py 的 parents[2]、web_dashboard.py 的
# Path(__file__).parent 指向同一目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_hive_home(root: Path | None = None) -> Path:
    """解析 HIVE_HOME 根锚点（全项目唯一实现）。

    root 仅供调用方显式注入基准根（测试隔离 / 兼容薄壳），不改变判据：
    环境变量优先，未设置才回退 root，仍未给出则回退 PROJECT_ROOT。
    """
    home = os.environ.get("HIVE_HOME", "").strip()
    if home:
        return Path(home).expanduser()
    return root if root is not None else PROJECT_ROOT


def resolve_events_file(root: Path | None = None) -> Path:
    """事件流文件落点 - 写入方与读取方共用同一构造式。"""
    return resolve_hive_home(root) / "Data" / "events.jsonl"


__all__ = ["PROJECT_ROOT", "resolve_hive_home", "resolve_events_file"]
