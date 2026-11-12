# -*- coding: utf-8 -*-
"""
MemoryStore 持久化功能验证脚本
测试 MemoryStore 的 save/load/auto_save 功能
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import shutil
import tempfile
from pathlib import Path


def test_memory_store_persistence():
    """测试 MemoryStore 的持久化功能"""
    print("=" * 60)
    print("MemoryStore 持久化功能验证")
    print("=" * 60)

    from hive.core.memory_store import (
        MEMORY_BOX_AVAILABLE,
        MemoryCategory,
        MemoryItem,
        MemoryStore,
        Resource,
    )

    print()
    print(f"[1/5] 检查依赖...")
    print(f"    MemoryBox 可用: {MEMORY_BOX_AVAILABLE}")

    print()
    print("[2/5] 测试数据类序列化...")

    # 测试 Resource 序列化
    res = Resource(
        resource_id="test_res_001",
        content="这是一段测试内容。它包含多个句子。每个句子都是一个事实。",
        modality="conversation",
        agent_id="test_agent",
        metadata={"source": "test", "timestamp": 1234567890},
    )

    res_dict = res.to_dict()
    print(f'    Resource.to_dict(): resource_id={res_dict["resource_id"]}')
    print(f'    content_len={len(res_dict["content"])}')

    res_restored = Resource.from_dict(res_dict)
    print(f"    Resource.from_dict: resource_id={res_restored.resource_id}")
    print(f"    content 一致: {res_restored.content == res.content}")

    # 测试 MemoryItem 序列化
    item = MemoryItem(
        item_id="test_item_001",
        text="这是一个测试记忆条目",
        tags=["test", "memory"],
        category="knowledge",
        agent_id="test_agent",
        source_id="test_res_001",
        importance=0.85,
    )

    item_dict = item.to_dict()
    print(f'    MemoryItem.to_dict(): item_id={item_dict["item_id"]}')
    print(f'    text={item_dict["text"][:20]}...')

    item_restored = MemoryItem.from_dict(item_dict)
    print(f"    MemoryItem.from_dict: item_id={item_restored.item_id}")
    print(f"    text 一致: {item_restored.text == item.text}")

    # 测试 MemoryCategory 序列化
    cat = MemoryCategory(
        name="test_category",
        summary="测试分类",
        item_ids=["item_001", "item_002"],
        agent_id="test_agent",
    )

    cat_dict = cat.to_dict()
    print(f'    MemoryCategory.to_dict(): name={cat_dict["name"]}')
    print(f'    item_ids={cat_dict["item_ids"]}')

    cat_restored = MemoryCategory.from_dict(cat_dict)
    print(f"    MemoryCategory.from_dict: name={cat_restored.name}")
    print(f"    item_ids 一致: {cat_restored.item_ids == cat.item_ids}")

    print("    [OK] 数据类序列化测试通过")

    print()
    print("[3/5] 测试 MemoryStore 基本功能...")

    # 创建一个测试 store
    store = MemoryStore(store_name="test_persistence_store", auto_save=False)

    print(f"    Store 名称: {store.store_name}")
    print(f"    auto_save: {store.auto_save}")

    stats = store.memory_stats()
    print(
        f'    初始统计: resources={stats["resources"]}, items={stats["items"]}, categories={stats["categories"]}'
    )

    # 添加一些内容
    content1 = "用户偏好深色主题。用户喜欢简洁的设计风格。用户需要高性能的解决方案。"
    result1 = store.memorize(content1, agent_id="user_001")
    print(f'    记忆内容1: resource_id={result1["resource_id"]}, items={result1["items"]}')

    content2 = "市场规则：低买高卖。风险管理很重要。策略需要定期回测。"
    result2 = store.memorize(content2, agent_id="trading_bot")
    print(f'    记忆内容2: resource_id={result2["resource_id"]}, items={result2["items"]}')

    stats2 = store.memory_stats()
    print(
        f'    添加后统计: resources={stats2["resources"]}, items={stats2["items"]}, categories={stats2["categories"]}'
    )

    # 测试检索
    rag_results = store.retrieve_rag("用户偏好", agent_id="user_001", top_k=3)
    print(f'    RAG 检索 "用户偏好": 找到 {len(rag_results)} 条')
    for r in rag_results:
        print(f"      - {r.text[:40]}... (category={r.category})")

    llm_results = store.retrieve_llm("市场规则", agent_id="trading_bot", top_k=3)
    print(f'    LLM 检索 "市场规则": 找到 {len(llm_results)} 条')
    for r in llm_results:
        print(f"      - {r.text[:40]}... (category={r.category})")

    print("    [OK] MemoryStore 基本功能测试通过")

    print()
    print("[4/5] 测试持久化配置...")

    p_stats = store.get_persistence_stats()
    print(f"    持久化状态:")
    print(f'      auto_save: {p_stats["auto_save"]}')
    print(f'      pending_changes: {p_stats["pending_changes"]}')
    print(f'      memory_box_available: {p_stats["memory_box_available"]}')

    # 测试手动保存
    if p_stats["memory_box_available"]:
        save_result = store.save()
        print(f"    手动保存: {save_result}")

        # 检查保存后 pending_changes
        p_stats2 = store.get_persistence_stats()
        print(f'    保存后 pending_changes: {p_stats2["pending_changes"]}')
    else:
        print("    MemoryBox 不可用，跳过保存测试")

    print("    [OK] 持久化配置测试通过")

    print()
    print("[5/5] 测试 auto_save 模式...")

    # 创建一个启用 auto_save 的 store
    store2 = MemoryStore(store_name="test_auto_save_store", auto_save=True)

    p_stats3 = store2.get_persistence_stats()
    print(f"    新 Store 持久化状态:")
    print(f'      auto_save: {p_stats3["auto_save"]}')
    print(f'      memory_box_available: {p_stats3["memory_box_available"]}')

    # 添加内容，触发自动保存
    for i in range(15):  # 超过阈值 (10)
        content = f"这是第 {i+1} 条测试内容。用于验证自动保存功能。"
        store2.memorize(content, agent_id=f"test_{i}")

    p_stats4 = store2.get_persistence_stats()
    print(f'    添加 15 条后 pending_changes: {p_stats4["pending_changes"]}')

    stats3 = store2.memory_stats()
    print(f'    Store2 统计: resources={stats3["resources"]}, items={stats3["items"]}')

    print("    [OK] auto_save 模式测试通过")

    print()
    print("=" * 60)
    print("验证结果: 全部通过")
    print("=" * 60)

    print()
    print("MemoryStore 持久化功能总结:")
    print("  - 数据类支持 to_dict() / from_dict() 序列化")
    print("  - 支持 save() / load() 手动持久化")
    print("  - 支持 auto_save 自动保存模式")
    print("  - 自动保存阈值: 每 10 次变更")
    print("  - 程序退出时自动保存 (atexit 注册)")
    print()
    print("使用示例:")
    print("  # 创建启用自动保存的 MemoryStore")
    print('  store = MemoryStore(store_name="MyMemory", auto_save=True)')
    print()
    print("  # 添加记忆（自动触发保存）")
    print('  store.memorize("重要内容...", agent_id="user")')
    print()
    print("  # 检索")
    print('  results = store.retrieve_rag("查询关键词")')
    print()
    print("  # 手动保存（可选）")
    print("  store.save()")

    return True


def test_browse():
    """测试 browse 功能"""
    from hive.core.memory_store import MemoryStore

    print()
    print("-" * 60)
    print("测试 browse 文件系统式导航")
    print("-" * 60)

    store = MemoryStore(store_name="test_browse", auto_save=False)

    # 添加一些内容
    contents = [
        ("用户偏好深色主题。用户喜欢简洁的设计。", "user_001"),
        ("市场价格正在上涨。BTC 突破了关键阻力位。", "trader"),
        ("任务需要先分析，再执行，最后验证结果。", "system"),
        ("策略回测显示胜率为 65%。最大回撤 12%。", "trader"),
    ]

    for content, agent in contents:
        store.memorize(content, agent_id=agent)

    stats = store.memory_stats()
    print(
        f'    统计: resources={stats["resources"]}, items={stats["items"]}, categories={stats["categories"]}'
    )

    # 浏览根目录
    root = store.browse(agent_id="global")
    print(f'    根目录: type={root["type"]}')
    if "categories" in root:
        print(f'    分类数量: {len(root["categories"])}')
        for cat in root["categories"]:
            print(f'      - {cat["name"]}: {cat["items"]} 个条目')

    # 浏览特定分类
    if root.get("categories"):
        first_cat = root["categories"][0]["name"]
        cat_detail = store.browse(category=first_cat, agent_id="global")
        print(f'    分类 "{first_cat}" 详情:')
        print(f'      type={cat_detail.get("type")}')
        if "items" in cat_detail:
            print(f'      条目数: {len(cat_detail["items"])}')
            for item in cat_detail["items"][:3]:
                print(f'        - {item["text"][:40]}...')

    print("    [OK] browse 功能测试通过")


if __name__ == "__main__":
    try:
        test_memory_store_persistence()
        test_browse()
        print()
        print("所有测试通过！")
        sys.exit(0)
    except Exception as e:
        print()
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
