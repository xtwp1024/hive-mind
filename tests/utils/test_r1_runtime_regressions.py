# -*- coding: utf-8 -*-
"""R1 运行时冒烟暴露的三处缺陷的回归测试。

静态体检（py_compile / import 闭合）全 PASS 时这三者仍会真实报错，
故用运行时真实调用钉住，避免后续改动把它们悄悄带回来：
  1. hive/utils/monitor.py 缺 `import os` → get_snapshot 抛 NameError
     被 except 吞成全 0 默认值（CPU/Memory 恒显 unknown）
  2. ResourceMonitor.get_system_info() 不返回 cpu_count / memory_total_gb
  3. main.py 无有界周期开关 → 冒烟只能跑满 10 周期或超时
"""
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_monitor_module_imports_os():
    """缺陷 1：monitor 模块必须显式 import os（源码里用到 os.environ）。"""
    from hive.utils import monitor

    src = inspect.getsource(monitor)
    assert "os.environ" in src, "monitor 已不再用 os.environ 时可删本断言"
    assert monitor.os is not None
    assert monitor.os.environ is not None


def test_get_snapshot_returns_real_values():
    """缺陷 1 的运行时反证：旧实现 NameError → 全 0；新实现应为真实硬件值。"""
    from hive.utils.monitor import ResourceMonitor

    snap = ResourceMonitor().get_snapshot()
    assert snap.cpu_count > 0, "cpu_count=0 说明又落回 except 的默认值"
    assert snap.memory_total_gb > 0, "memory_total_gb=0 说明又落回 except 的默认值"
    assert snap.disk_total_gb > 0, "disk_total_gb=0 说明又落回 except 的默认值"
    assert snap.cpu_percent >= 0
    assert snap.process_count > 0


def test_get_system_info_exposes_resource_keys():
    """缺陷 2：main.initialize_system 读这两个 key，缺失会让日志恒显 unknown。"""
    from hive.utils.monitor import ResourceMonitor

    info = ResourceMonitor().get_system_info()
    assert "cpu_count" in info and info["cpu_count"] > 0
    assert "memory_total_gb" in info and info["memory_total_gb"] > 0
    # 原有 key 不得被挤掉（hive/tests/test_monitoring.py 有断言）
    assert {"platform", "python_version", "hostname"} <= set(info)


def test_main_supports_bounded_cycles():
    """缺陷 3：冒烟必须能设周期数，否则只能跑满 10 周期。"""
    src = (ROOT / "main.py").read_text(encoding="utf-8", errors="replace")
    assert "HIVE_SMOKE_CYCLES" in src
    assert "os.environ" in src


def test_smoke_runtime_script_exists():
    """第 4 层验证资产必须存在，且声明了不假报 PASS 的 SKIP 语义。"""
    script = ROOT / "tools" / "smoke_runtime.py"
    assert script.is_file()
    src = script.read_text(encoding="utf-8", errors="replace")
    assert "HIVE_SMOKE_CYCLES" in src
    assert "ERROR_LEVEL" in src, "冒烟脚本必须检查 error 级别日志，否则恒绿"
