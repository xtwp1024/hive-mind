#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hive Mind 2.0 - Core Dispatcher
-------------------------------
这是经过“大脑精简手术”后的轻量级入口。
原本 111KB 的巨型文件已被模块化并迁移至 hive/ 目录。

Legacy 备份见: hive_mind_legacy.py
"""

import sys
import asyncio
from pathlib import Path

# 将当前目录加入路径
sys.path.append(str(Path(__file__).parent))

def main():
    """桥接到模块化的 main.py"""
    try:
        from main import main as core_main
        asyncio.run(core_main())
    except ImportError as e:
        print(f"❌ 启动失败: 核心模块缺失或导入错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 蜂群正在归巢...")
        sys.exit(0)

if __name__ == "__main__":
    main()
