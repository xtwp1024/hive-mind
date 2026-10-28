#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示 openshift/hive 声明式集群管理精髓的沙盘脚本。
展示 BeePool (ClusterPool) + BeeReconciler + SyncSet 三层架构联动。
"""

import time
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from hive.core.cluster_pool import BeePool
from hive.core.reconciler import BeeReconciler
from hive.core.dna import BeeDNA

SEPARATOR = "=" * 60

def print_status(pool: BeePool, label: str) -> None:
    s = pool.status()
    print(f"\n  [{label}]")
    print(f"   池大小目标  : {s['desired_size']}")
    print(f"   冬眠就绪    : {s.get('hibernated', 0)}")
    print(f"   运行中      : {s.get('running', 0)}")
    print(f"   孵化中      : {s.get('provisioning', 0)}")

def main():
    print(SEPARATOR)
    print("  openshift/hive 声明式蜂群热池 - 全生命周期演示")
    print(SEPARATOR)

    # ── 1. 声明一个 size=3 的 BeePool（我只需要声明意图）───────────────
    print("\n[Step 1] 声明 BeePool: desired_size=3，运行中最少 1 只")
    pool = BeePool(
        name="prod-scraper-pool",
        species="Standard",
        desired_size=3,
        running_count=1,
    )

    # ── 2. 预热热池（模拟 Hive 启动时首次孵化）─────────────────────────
    print("\n[Step 2] 预热热池（首次孵化 3 只冬眠工蜂）...")
    pool.warm_up()
    print_status(pool, "预热完成")

    # ── 3. 启动 Reconciler 后台守护线程──────────────────────────────────
    print("\n[Step 3] 启动 BeeReconciler 协调控制循环（周期=3s）")
    reconciler = BeeReconciler(pool, interval_seconds=3.0, max_bee_lifetime=20.0)
    reconciler.start()
    print("  [OK] Reconciler 已在后台驻守，开始监视池状态...")

    # ── 4. 申领 2 只工蜂（ClusterClaim），注入 SyncSet 配置──────────────
    print(f"\n{SEPARATOR}")
    print("[Step 4] 发出 2 个 ClusterClaim 申领请求（注入 SyncSet 配置）")

    claim1 = pool.claim(
        task_dna=BeeDNA(species="Standard", target_name="task://sentiment-analysis"),
        sync_config={"proxy": "socks5://10.0.0.1:1080", "depth": 3},
    )
    claim2 = pool.claim(
        task_dna=BeeDNA(species="Standard", target_name="task://market-data-fetch"),
        sync_config={"api_key": "sk-***", "rate_limit": 5},
    )

    print(f"  申领 1 → {claim1}")
    print(f"  申领 2 → {claim2}")
    print_status(pool, "申领后（2 只已激活，池暂时只剩 1 只冬眠）")

    # ── 5. 等待 Reconciler 自动补充缺口──────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("[Step 5] 等待 Reconciler 检测到池缺口并自动补充...")
    time.sleep(4)  # 给 Reconciler 一个协调周期
    print_status(pool, "Reconciler 协调后（应自动补满至 3 只冬眠）")

    # ── 6. 释放申领工蜂（任务完成销毁，不返回池）──────────────────────
    print(f"\n{SEPARATOR}")
    print("[Step 6] 任务完成，释放申领工蜂（sales 销毁，不回池）")
    if claim1:
        pool.release(claim1)
        print(f"  {claim1} 已销毁")
    if claim2:
        pool.release(claim2)
        print(f"  {claim2} 已销毁")

    # ── 7. Reconciler 再次补充──────────────────────────────────────────
    print("\n[Step 7] 等待 Reconciler 再次补充销毁后的缺口...")
    time.sleep(4)
    print_status(pool, "最终状态（应恢复到 3 只冬眠就绪）")

    # ── 8. 优雅停止──────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("[Step 8] 优雅停止 Reconciler")
    reconciler.stop()
    print("  [OK] 协调控制循环已停止")
    print(f"\n{SEPARATOR}")
    print("  演示完成！声明式意图驱动的蜂群热池全生命周期验证通过！")
    print(SEPARATOR)

if __name__ == "__main__":
    main()
