# -*- coding: utf-8 -*-
"""验证三项进化改进"""
import sys
sys.path.insert(0, '.')

def test_tool_registry():
    print("=== 测试1: Tool Registry ===")
    from hive.core.tool_registry import BeeToolRegistry, bee_tool

    @bee_tool(name="test_double", description="测试：乘以2")
    def double(x: int) -> int:
        return x * 2

    result = BeeToolRegistry.invoke("test_double", x=5)
    assert result == 10, f"期望10, 得到{result}"
    print(f"已注册工具数: {len(BeeToolRegistry)}")
    print(f"调用 test_double(5) = {result} ✅")


def test_code_analysis():
    print("\n=== 测试2: 代码结构分析工具 ===")
    from hive.core.tool_registry import BeeToolRegistry
    import hive.bees.miner_bee  # 触发 @bee_tool 注册

    r = BeeToolRegistry.invoke("分析代码结构", file_path="hive/core/queen.py")
    print(f"  语言: {r.get('language')}")
    print(f"  行数: {r.get('lines')}")
    print(f"  类: {r.get('classes')}")
    print(f"  函数(前3): {r.get('functions', [])[:3]}")
    assert r.get("language") == "python"
    print("代码结构分析 ✅")


def test_tools_list():
    print("\n=== 测试3: 已注册工具清单 ===")
    from hive.core.tool_registry import BeeToolRegistry
    tools = BeeToolRegistry.list_tools()
    for t in tools:
        print(f"  {t['name']} - {t['description'][:40]}")
    print(f"共 {len(tools)} 个工具 ✅")


def test_adaptive_hunting():
    print("\n=== 测试4: 自适应狩猎策略更新 ===")
    import json, os
    from pathlib import Path

    # 模拟3次狩猎后的策略进化
    strategy_file = Path("intel_strategy.json")
    old = strategy_file.read_text(encoding="utf-8") if strategy_file.exists() else None

    prey = [
        {"language": "Python", "url": "https://github.com/a/b", "name": "test1"},
        {"language": "Python", "url": "https://github.com/c/d", "name": "test2"},
        {"language": "TypeScript", "url": "https://github.com/e/f", "name": "test3"},
    ]
    # 直接写入一个 runs=2 的策略，触发第3次进化
    existing = {"hunting_keywords": ["ai agent"], "runs": 2, "top_languages": {"Python": 5}}
    with open(strategy_file, "w", encoding="utf-8") as f:
        json.dump(existing, f)

    from hive.core.dna import BeeDNA
    from hive.ministers.engineer import get_engineer
    eng = get_engineer()
    eng._update_intel_strategy(prey, ["ai agent"])

    result = json.loads(strategy_file.read_text(encoding="utf-8"))
    print(f"  runs: {result['runs']}")
    print(f"  top_languages: {result['top_languages']}")
    print(f"  hunting_keywords (进化后): {result['hunting_keywords']}")
    assert result["runs"] == 3
    assert "hunting_keywords" in result
    print("自适应狩猎 ✅")


if __name__ == "__main__":
    test_tool_registry()
    test_code_analysis()
    test_tools_list()
    test_adaptive_hunting()
    print("\n🎉 所有测试通过！三项进化改进验证成功。")
