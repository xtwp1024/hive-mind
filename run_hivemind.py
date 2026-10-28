#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示 learning-at-home/hivemind 去中心化精髓的沙盘脚本。
展示 DHT 知识路由 + Gossip 去中心化结果聚合两层架构联动。
"""

import logging
import random
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from hive.core.dht_router import DHTRouter
from hive.core.gradient_averager import GradientAverager

SEP = "=" * 62

# ──────────────────────────────────────────────────────────────────────
# Part 1: DHT 去中心化知识路由演示
# ──────────────────────────────────────────────────────────────────────
def demo_dht():
    print(f"\n{SEP}")
    print("  PART 1: DHT 去中心化知识路由 (Kademlia XOR)")
    print(SEP)

    # 5 个工蜂节点同时上线，各自成为 DHT 节点
    node_names = ["Bee-Alpha", "Bee-Beta", "Bee-Gamma", "Bee-Delta", "Bee-Epsilon"]
    nodes = {n: DHTRouter(n) for n in node_names}

    # 所有节点通过 bootstrap 相互发现（对标 hivemind DHT bootstrap）
    print("\n[Step 1] 所有节点执行 bootstrap，建立去中心化路由表...")
    for n in nodes.values():
        n.bootstrap("Bee-Alpha")  # 只需要知道一个节点即可
    print("  [OK] 去中心化网络拓扑已建立（无主节点！）")

    # Bee-Alpha 发布一条知识（爬取到的数据摘要）
    print("\n[Step 2] Bee-Alpha 向 DHT 发布知识碎片...")
    nodes["Bee-Alpha"].store(
        key   = "sentiment://AAPL_2026Q1",
        value = {"score": 0.82, "source": "news_corpus", "tokens": 14200},
        ttl_seconds=120,
    )

    # Bee-Gamma（一个完全没有参与发布的节点）来查询这条知识
    print("\n[Step 3] Bee-Gamma（从未接触该数据）发起 DHT 查询...")
    result = nodes["Bee-Gamma"].get("sentiment://AAPL_2026Q1")
    print(f"  Bee-Gamma 查询结果: {result}")
    assert result is not None, "DHT 查询失败！"

    # Bee-Delta 下线（容错演示）
    print("\n[Step 4] Bee-Delta 突然下线...")
    nodes["Bee-Delta"].leave()
    # 其他节点仍然能查询（容错）
    result2 = nodes["Bee-Beta"].get("sentiment://AAPL_2026Q1")
    print(f"  Delta 掉线后 Bee-Beta 查询结果: {result2}")
    assert result2 is not None, "容错查询失败！"

    print("\n  ✅ DHT 验证通过：去中心化路由、广播、容错均正常！")

    for n in nodes.values():
        n.leave()


# ──────────────────────────────────────────────────────────────────────
# Part 2: 去中心化 Gossip 结果聚合演示
# ──────────────────────────────────────────────────────────────────────
def demo_averager():
    print(f"\n{SEP}")
    print("  PART 2: Gossip 去中心化结果聚合（无需主节点汇总）")
    print(SEP)

    averager = GradientAverager(n_rounds=5)

    # 6 个工蜂各自持有不同的局部结果
    local_values = {
        "Bee-01": 0.72,
        "Bee-02": 0.55,
        "Bee-03": 0.88,
        "Bee-04": 0.41,
        "Bee-05": 0.93,
        "Bee-06": 0.67,
    }
    true_avg = sum(local_values.values()) / len(local_values)

    print(f"\n[Step 1] 6 只工蜂提交局部结果（各自持有不同数值）")
    for bee_id, val in local_values.items():
        averager.add_local(bee_id, val)
    print(f"  理论全局平均值: {true_avg:.6f}")

    # 模拟 Bee-04 中途掉线
    print(f"\n[Step 2] Bee-04 中途掉线（容错测试）")
    averager.kill_node("Bee-04")
    alive_vals = {k: v for k, v in local_values.items() if k != "Bee-04"}
    true_avg_alive = sum(alive_vals.values()) / len(alive_vals)
    print(f"  剩余节点理论平均: {true_avg_alive:.6f}")

    # 发起 Gossip 迭代平均（无任何主节点协调）
    print(f"\n[Step 3] 发起 {averager.n_rounds} 轮 Gossip 聚合（纯 P2P，无主节点）...")
    converged = averager.run_averaging()

    print(f"\n[结果]")
    print(f"  收敛值 (Gossip):    {converged:.6f}")
    print(f"  真实全局平均:       {true_avg_alive:.6f}")
    error = abs(converged - true_avg_alive)
    print(f"  误差:               {error:.8f}")

    final_vals = averager.node_values()
    print(f"\n  各节点当前持有值:")
    for bid, v in sorted(final_vals.items()):
        print(f"    {bid}: {v:.6f}")

    assert error < 0.01, f"收敛误差过大：{error}"
    print(f"\n  ✅ Gossip 聚合验证通过：无主节点收敛，容错正常！")


if __name__ == "__main__":
    print(SEP)
    print("  learning-at-home/hivemind 去中心化知识扩散 - 全演示")
    print(SEP)

    demo_dht()
    demo_averager()

    print(f"\n{SEP}")
    print("  🎉 全部验证通过！蜂群去中心化知识层架构部署完成！")
    print(SEP)
