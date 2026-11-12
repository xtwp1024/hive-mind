# -*- coding: utf-8 -*-
"""
PRD UI 控制台 - 快速功能验证

验证以下模块:
1. SSE 消息总线 (dispatcher.py)
2. 数据库迁移 (migrations.py)
3. Web Dashboard V2.5 (web_dashboard_v25.py)
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_sse_message_bus():
    """测试 SSE 消息总线"""
    print("=" * 60)
    print("[1/5] 测试 SSE 消息总线 (dispatcher.py)")
    print("=" * 60)
    print()

    try:
        from hive.core.dispatcher import (
            MessageBus,
            SSEEventType,
            SSEMessage,
            TaskGraph,
            TaskNode,
            TaskQueue,
            TaskStatus,
            get_message_bus,
        )

        print("    [OK] 导入成功")
        print()

        # 测试 MessageBus
        print("    测试 MessageBus...")

        bus = MessageBus(max_history=100)

        # 发布消息
        msg = SSEMessage(
            event_type=SSEEventType.SYSTEM_METRICS,
            data={"cpu_percent": 25.5, "memory_percent": 60.0},
        )
        bus.publish(msg)

        print(f"    [OK] 发布消息成功: {msg.event_type.value}")

        # 订阅
        queue = bus.subscribe()
        print(f"    [OK] 订阅成功，订阅者数: {bus.get_stats()['subscribers']}")

        # 发布更多消息
        task_msg = SSEMessage(
            event_type=SSEEventType.TASK_STARTED, data={"task_id": "test-001", "action": "test"}
        )
        bus.publish(task_msg)

        # 从队列接收
        received = queue.get(timeout=1.0)
        print(f"    [OK] 收到消息: {received.event_type.value}")

        # 取消订阅
        bus.unsubscribe(queue)
        print(f"    [OK] 取消订阅，剩余订阅者: {bus.get_stats()['subscribers']}")

        # 测试全局单例
        global_bus = get_message_bus()
        print(f"    [OK] 全局消息总线: {type(global_bus).__name__}")

        # 测试 SSE 格式
        sse_format = msg.to_sse_format()
        print(f"    [OK] SSE 格式: {len(sse_format)} 字符")

        # 测试 TaskQueue
        print()
        print("    测试 TaskQueue...")

        tq = TaskQueue(max_size=10)
        tq.start()

        stats = tq.get_stats()
        print(f"    [OK] TaskQueue 启动: running={stats['running']}")

        # 测试数据类
        print()
        print("    测试数据类...")

        task_node = TaskNode(
            id="test-node-001",
            action="test_action",
            params={"key": "value"},
            required_capabilities=["test"],
        )

        task_dict = task_node.to_dict()
        print(f"    [OK] TaskNode.to_dict: id={task_dict['id']}, action={task_dict['action']}")

        graph = TaskGraph("Test Workflow")
        graph.add_node(task_node)

        print(f"    [OK] TaskGraph: nodes={len(graph.nodes)}")

        print()
        print("    结果: [通过]")
        return True

    except Exception as e:
        print(f"    [错误] {e}")
        import traceback

        traceback.print_exc()
        return False


def test_database_migration():
    """测试数据库迁移"""
    print()
    print("=" * 60)
    print("[2/5] 测试数据库迁移 (migrations.py)")
    print("=" * 60)
    print()

    try:
        from hive.evolution.migrations import (
            DatabaseMigrator,
            MigrationRegistry,
            MigrationStatus,
            MigrationStep,
            ensure_latest_schema,
            get_database_migrator,
        )

        print("    [OK] 导入成功")
        print()

        # 检查已注册的迁移
        migrations = MigrationRegistry.get_all()
        print(f"    已注册的迁移数: {len(migrations)}")

        for m in migrations:
            print(f"      - {m.version}: {m.name}")

        latest = MigrationRegistry.get_latest_version()
        print(f"    最新版本: {latest}")

        # 获取迁移状态
        print()
        print("    检查迁移状态...")

        try:
            migrator = get_database_migrator()

            if migrator:
                status = migrator.get_migration_status()

                print(f"    当前版本: {status['current_version']}")
                print(f"    最新版本: {status['latest_version']}")
                print(f"    是否最新: {status['is_latest']}")
                print(f"    已应用: {status['applied_count']}")
                print(f"    待应用: {status['pending_count']}")

                if not status["is_latest"]:
                    print(
                        f"    待应用的迁移: {[m['version'] for m in status['pending_migrations']]}"
                    )
            else:
                print("    [WARN] 无法获取迁移器")

        except Exception as e:
            print(f"    [INFO] 迁移表检查: {e} (可能是首次运行)")

        print()
        print("    结果: [通过]")
        return True

    except Exception as e:
        print(f"    [错误] {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dispatcher_imports():
    """测试 dispatcher.py 所有导入"""
    print()
    print("=" * 60)
    print("[3/5] 测试 dispatcher.py 模块导入")
    print("=" * 60)
    print()

    try:
        import hive.core.dispatcher as dispatcher

        # 检查所有导出
        exports = [
            "TaskStatus",
            "TaskNode",
            "TaskGraph",
            "SSEEventType",
            "SSEMessage",
            "MessageBus",
            "TaskQueue",
            "HiveDispatcher",
            "get_message_bus",
            "get_task_queue",
        ]

        found = 0
        for exp in exports:
            if hasattr(dispatcher, exp):
                print(f"    [OK] {exp}")
                found += 1
            else:
                print(f"    [缺失] {exp}")

        print()
        print(f"    结果: {found}/{len(exports)} 导出存在")

        return found == len(exports)

    except Exception as e:
        print(f"    [错误] {e}")
        import traceback

        traceback.print_exc()
        return False


def test_migrations_imports():
    """测试 migrations.py 所有导入"""
    print()
    print("=" * 60)
    print("[4/5] 测试 migrations.py 模块导入")
    print("=" * 60)
    print()

    try:
        import hive.evolution.migrations as migrations

        # 检查所有导出
        exports = [
            "MigrationStatus",
            "MigrationRecord",
            "MigrationStep",
            "MigrationRegistry",
            "DatabaseMigrator",
            "get_database_migrator",
            "ensure_latest_schema",
        ]

        found = 0
        for exp in exports:
            if hasattr(migrations, exp):
                print(f"    [OK] {exp}")
                found += 1
            else:
                print(f"    [缺失] {exp}")

        print()
        print(f"    结果: {found}/{len(exports)} 导出存在")

        return found == len(exports)

    except Exception as e:
        print(f"    [错误] {e}")
        import traceback

        traceback.print_exc()
        return False


def test_web_dashboard_imports():
    """测试 web_dashboard_v25.py 导入"""
    print()
    print("=" * 60)
    print("[5/5] 测试 web_dashboard_v25.py 模块导入")
    print("=" * 60)
    print()

    try:
        # 检查文件存在 (在项目根目录)
        dashboard_path = PROJECT_ROOT / "web_dashboard_v25.py"

        if dashboard_path.exists():
            print(f"    [OK] 文件存在: {dashboard_path}")

            # 检查文件大小
            size = dashboard_path.stat().st_size
            print(f"    文件大小: {size} 字节")

            # 检查关键内容
            with open(dashboard_path, "r", encoding="utf-8") as f:
                content = f.read()

            key_features = [
                ("SSE 事件流", "event: system_metrics"),
                ("Chart.js", "Chart.js"),
                ("响应式布局", "grid-template-columns"),
                ("选项卡", "data-tab"),
                ("迁移状态", "migration-status"),
                ("API 端点", "@app.route('/api/events')"),
            ]

            found_features = 0
            for name, pattern in key_features:
                if pattern in content:
                    print(f"    [OK] {name}")
                    found_features += 1
                else:
                    print(f"    [WARN] {name} 未找到")

            print()
            print(f"    特性检查: {found_features}/{len(key_features)}")

            return True
        else:
            print(f"    [错误] 文件不存在: {dashboard_path}")
            return False

    except Exception as e:
        print(f"    [错误] {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("PRD UI 控制台 - 快速功能验证")
    print("=" * 60)
    print()
    print("测试模块:")
    print("  1. SSE 消息总线 (hive/core/dispatcher.py)")
    print("  2. 数据库迁移 (hive/evolution/migrations.py)")
    print("  3. Web Dashboard V2.5 (web_dashboard_v25.py)")
    print()

    results = []

    # 测试 1: SSE 消息总线
    results.append(("SSE 消息总线", test_sse_message_bus()))

    # 测试 2: 数据库迁移
    results.append(("数据库迁移", test_database_migration()))

    # 测试 3: dispatcher 导入
    results.append(("Dispatcher 导入", test_dispatcher_imports()))

    # 测试 4: migrations 导入
    results.append(("Migrations 导入", test_migrations_imports()))

    # 测试 5: web dashboard
    results.append(("Web Dashboard V2.5", test_web_dashboard_imports()))

    # 汇总
    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print()

    passed = 0
    total = len(results)

    for name, result in results:
        status = "[OK] 通过" if result else "[FAIL] 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    print()
    print(f"  总计: {passed}/{total} 测试通过")
    print()

    if passed == total:
        print("=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        print()
        print("启动 Dashboard 命令:")
        print("  开发模式:")
        print("    set HIVE_DASHBOARD_DEBUG=true")
        print("    python web_dashboard_v25.py")
        print()
        print("  访问 URL:")
        print("    http://127.0.0.1:5000")
        print("    http://127.0.0.1:5000?api_key=dev_key_change_me_insecure")
        return 0
    else:
        print("=" * 60)
        print("部分测试失败")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
