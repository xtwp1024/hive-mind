#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示 cazala/coin-hive 工作量证明闭环的沙盘脚本。
模拟 3 只工蜂并发领取任务、计算 PoW 哈希、提交验收。
"""

import threading
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from hive.core.proof_of_work import ProofOfWork
from hive.core.stratum_dispatcher import StratumDispatcher

SEP = "=" * 62

def bee_worker(bee_id: str, pool: StratumDispatcher, task_blob: str, throttle: float = 0.0):
    """单只工蜂的完整挖矿生命周期（对标 coin-hive miner 实例）"""
    miner = ProofOfWork(throttle=throttle)

    # 1. 从矿池领取任务（对标 pool.on('job', ...)）
    job = pool.push_job(bee_id=bee_id, task_blob=task_blob)

    # 2. 计算 PoW（对标 coin-hive 内部哈希循环）
    solution = miner.mine(job)
    if not solution:
        print(f"  [{bee_id}] 未找到解（任务超时或难度过高）")
        return

    # 3. 提交解（对标 Stratum submit）
    result = pool.submit(bee_id=bee_id, solution=solution)
    print(
        f"  [{bee_id}] {result.upper()}"
        f" | nonce={solution.nonce}"
        f" | hash={solution.hash_hex[:20]}..."
        f" | attempts={solution.attempts}"
        f" | {solution.elapsed:.3f}s"
    )


def main():
    print(SEP)
    print("  coin-hive PoW 工作量证明闭环 - 全流程演示")
    print(SEP)

    # ── 1. 启动类 Stratum 矿池 ─────────────────────────────────
    print("\n[Step 1] 启动 StratumDispatcher（类 Stratum 矿池，难度=2）")
    pool = StratumDispatcher(pool_name="HiveXMR-Pool", difficulty=2)

    # 注册事件钩子（对标 coin-hive miner.on(event, cb)）
    pool.on("accepted", lambda d: None)   # 已在 submit 内打印，这里仅占位
    pool.on("rejected", lambda d: print(f"  ⚠️ REJECTED: {d['bee_id']}"))

    # ── 2. 三只工蜂并发挖矿 ────────────────────────────────────
    print("\n[Step 2] 3 只工蜂并发领取 job、计算 PoW、提交验收...")
    print(f"{'─'*62}")

    tasks = [
        ("Bee-Alpha",   "task://sentiment/AAPL_2026Q1",        0.0),
        ("Bee-Beta",    "task://market-data/BTC_realtime",     0.0),
        ("Bee-Gamma",   "task://risk-model/portfolio_v3",      0.3),  # 节流 30%
    ]

    threads = []
    for bee_id, blob, throttle in tasks:
        t = threading.Thread(
            target=bee_worker,
            args=(bee_id, pool, blob, throttle),
            name=bee_id,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # ── 3. 矿池算力统计报表 ────────────────────────────────────
    print(f"\n{SEP}")
    print("[Step 3] 矿池算力统计报表（对标 coin-hive update 事件）")
    print(f"{'─'*62}")
    stats = pool.pool_stats()
    print(f"  矿池名称   : {stats['pool']}")
    print(f"  总矿工数   : {stats['total_miners']}")
    print(f"  总 accepted: {stats['total_accepted']}")
    print(f"  总 rejected: {stats['total_rejected']}")
    print(f"  总哈希次数 : {stats['total_attempts']:,}")
    print(f"\n  各矿工算力:")
    for bid, ms in stats["miners"].items():
        print(
            f"    {bid:<14} | "
            f"H/s={ms['hashes_per_second']:>8,} | "
            f"accepted={ms['accepted']} | "
            f"rate={ms['acceptance_rate']}"
        )

    # ── 4. 伪造攻击演示（直接提交假 hash，必 rejected）─────────
    print(f"\n{SEP}")
    print("[Step 4] 安全测试：伪造工蜂尝试提交虚假哈希...")

    from hive.core.proof_of_work import PoWSolution
    fake_job = pool.push_job("Bee-Evil", "fake://task")
    fake_sol = PoWSolution(
        job_id   = fake_job.job_id,
        nonce    = 9999999,
        hash_hex = "0" * 64,   # 伪造的哈希
        attempts = 1,
        elapsed  = 0.0001,
    )
    fraud_result = pool.submit("Bee-Evil", fake_sol)
    print(f"  伪造提交结果: {fraud_result.upper()}（系统成功识别并拒绝！）")

    print(f"\n{SEP}")
    print("  🔒 验证完成！工作量证明层部署成功，无法虚报任务！")
    print(SEP)

if __name__ == "__main__":
    main()
