# -*- coding: utf-8 -*-
"""
GradientAverager - 去中心化结果聚合器
设计灵感来源于 hivemind 的 Decentralized Parameter Averaging 机制。

核心算法：All-Reduce Style Gossip Averaging
- 每个节点持有局部结果向量 (local_result)
- 通过多轮 peer-to-peer 两两交换，做加权平均
- 经过 O(log n) 轮迭代，所有节点的值收敛到全局平均
- 容错：某节点中途离线 → 用已有节点继续迭代，不崩溃

对标 hivemind 的 DecentralizedAverager 组件。
"""

import logging
import random
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Hive.GradientAverager")


@dataclass
class NodeResult:
    """单个节点的局部结果（对标 hivemind 中的梯度张量）"""
    node_id:  str
    value:    float          # 单维简化（真实场景为向量/张量）
    weight:   float = 1.0   # 节点权重（可按计算量加权）
    alive:    bool  = True   # 节点是否在线


class GradientAverager:
    """
    去中心化 Gossip-style 结果聚合器。
    无需女王/主节点参与，靠节点两两交换自行收敛到全局平均。
    """

    def __init__(self, n_rounds: int = 4):
        self.n_rounds = n_rounds   # 迭代轮数（log2(n_nodes) 最优）
        self._nodes: dict[str, NodeResult] = {}
        self._lock = threading.Lock()

    # ────────────────────────────────────── 节点注册 ──

    def add_local(self, node_id: str, value: float, weight: float = 1.0) -> None:
        """节点提交自己的局部结果（对标 averager.step() 前的 push_tensors）"""
        with self._lock:
            self._nodes[node_id] = NodeResult(node_id=node_id, value=value, weight=weight)
        logger.info(f"[Averager] {node_id} 提交局部值 {value:.4f} (weight={weight})")

    def kill_node(self, node_id: str) -> None:
        """模拟节点掉线（容错测试用）"""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].alive = False
        logger.warning(f"[Averager] {node_id} 已下线（容错场景）")

    # ────────────────────────────────────── 核心聚合 ──

    def _gossip_round(self) -> None:
        """执行一轮 Gossip 平均（随机配对两两交换）"""
        with self._lock:
            alive = [n for n in self._nodes.values() if n.alive]

        if len(alive) < 2:
            return

        # 随机打乱并两两配对
        random.shuffle(alive)
        pairs = [(alive[i], alive[i + 1]) for i in range(0, len(alive) - 1, 2)]

        for a, b in pairs:
            # 加权平均（对标 hivemind 的 weighted average step）
            total_w  = a.weight + b.weight
            new_val  = (a.value * a.weight + b.value * b.weight) / total_w
            with self._lock:
                self._nodes[a.node_id].value = new_val
                self._nodes[b.node_id].value = new_val
            logger.debug(
                f"[Averager] 配对 ({a.node_id} ↔ {b.node_id}) "
                f"→ [{a.value:.3f} + {b.value:.3f}] → {new_val:.4f}"
            )

    def run_averaging(self) -> float:
        """
        执行 n_rounds 轮 Gossip 平均，返回最终收敛值。
        对标 hivemind DecentralizedAverager.step() 多轮迭代。
        """
        with self._lock:
            alive_count = sum(1 for n in self._nodes.values() if n.alive)

        logger.info(f"[Averager] 开始聚合，活跃节点={alive_count}，迭代轮数={self.n_rounds}")

        for r in range(1, self.n_rounds + 1):
            self._gossip_round()
            with self._lock:
                vals = [n.value for n in self._nodes.values() if n.alive]
            spread = max(vals) - min(vals) if vals else 0
            logger.info(f"[Averager] 第 {r} 轮完成，值域spread={spread:.6f}")

        return self.converged_result()

    def converged_result(self) -> float:
        """返回收敛后的全局估计值（所有活跃节点的加权平均）"""
        with self._lock:
            alive = [n for n in self._nodes.values() if n.alive]
        if not alive:
            return 0.0
        total_w = sum(n.weight for n in alive)
        return sum(n.value * n.weight for n in alive) / total_w

    def true_average(self) -> float:
        """计算真实全局平均（用于验证收敛误差）"""
        with self._lock:
            alive = [n for n in self._nodes.values() if n.alive]
        if not alive:
            return 0.0
        return sum(n.value for n in alive) / len(alive)

    def node_values(self) -> dict[str, float]:
        """返回所有节点当前持有的值（用于展示收敛状态）"""
        with self._lock:
            return {n.node_id: n.value for n in self._nodes.values() if n.alive}
