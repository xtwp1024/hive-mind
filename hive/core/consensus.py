#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consensus Engine (多端共识状态匹配器)
灵感来自于 ethereum/hive。
用来防范大型语言模型（LLM）"单点故障"或"幻觉臆想"。
对于高危指令，中枢系统会分叉出多个独立的 Worker（模拟多客户端节点），分别得出自己的答案。
该引擎作为"仲裁官"，应用多数投票（Nakamoto-style 变种）裁决最终可用状态。

Phase 2 P2: LLM Reasoning 加持
- 纯字符串Counter无法判断语义相近的答案
- 当多数投票失败时，调用 BrainInterface 进行语义分析
- 支持多轮投票收敛
"""

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("Hive.Consensus")

# 全局 BrainInterface 引用（懒加载）
_brain = None


def _get_brain():
    """懒加载 BrainInterface 单例"""
    global _brain
    if _brain is None:
        try:
            from hive.utils.brain_interface import BrainInterface
            from hive.utils.brain_interface import get_brain as _get_brain_impl

            _brain = _get_brain_impl()
        except ImportError:
            try:
                _brain = BrainInterface()
            except Exception:
                return None
    return _brain


# ===== Phase 2: 集体决策新增枚举和数据类 =====


class DecisionType(Enum):
    """决策类型 (Phase 2 新增)"""

    UNANIMOUS = "unanimous"  # 全票通过
    MAJORITY = "majority"  # 多数通过 (>50%)
    SUPER_MAJORITY = "super_majority"  # 超多数 (>=2/3)
    PLURALITY = "plurality"  # 相对多数
    WEIGHTED = "weighted"  # 加权投票


class VoteOption(Enum):
    """投票选项 (Phase 2 新增)"""

    APPROVE = "approve"  # 批准/同意
    REJECT = "reject"  # 拒绝
    ABSTAIN = "abstain"  # 弃权
    DEFER = "defer"  # 延期决策


@dataclass
class Vote:
    """投票记录 (Phase 2 新增)"""

    voter_id: str
    option: VoteOption
    weight: float = 1.0  # 权重 (基于适应度)
    justification: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DecisionProposal:
    """决策提案 (Phase 2 新增)"""

    proposal_id: str
    proposer_id: str

    # 提案内容
    title: str
    description: str
    proposal_type: str  # e.g., "evolve_species", "allocate_resources", "change_strategy"

    # 决策规则
    decision_type: DecisionType = DecisionType.MAJORITY
    quorum: float = 0.5  # 法定人数比例
    time_limit: float = 300.0  # 决策时限 (秒)

    # 元数据
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return time.time() - self.created_at > self.time_limit
        return time.time() > self.expires_at


@dataclass
class DecisionResult:
    """决策结果 (Phase 2 新增)"""

    proposal_id: str
    approved: bool
    vote_summary: Dict[str, int]  # VoteOption.value -> count
    weighted_summary: Dict[str, float]  # VoteOption.value -> weight sum
    final_option: Optional[VoteOption]
    participating_voters: List[str]
    quorum_reached: bool
    elapsed_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsensusEngine:
    """状态裁决与幻觉过滤器"""

    # Phase 2 P2: LLM 语义判断阈值
    # 当投票分散度超过此值时，触发 LLM 语义分析
    FORK_DIVERSITY_THRESHOLD = 0.6  # 最高票占比 > 60% 则认为有共识

    @classmethod
    def match_semantics(cls, results: List[str], enable_llm: bool = True) -> Tuple[bool, str]:
        """
        基于绝对字符串或基础语义的硬分叉判定。

        Phase 2 P2 升级：
        - 第一轮：严格 Counter 多数投票（极速）
        - 第二轮（可选）：LLM 语义分析（处理语义相近的答案）
        - 第三轮：多轮投票收敛

        Args:
            results: 各节点返回的结果列表
            enable_llm: 是否启用 LLM 语义分析（当 Counter 失败时）

        Returns:
            (is_consensus, canonical_result)
        """
        if not results:
            return False, "NO_RESULTS"

        # ===== 第一轮：严格多数投票 =====
        total_nodes = len(results)
        counts = Counter(results)
        most_common_result, votes = counts.most_common(1)[0]
        vote_ratio = votes / total_nodes

        # 完全一致（所有节点都投同一结果）
        if vote_ratio == 1.0:
            logger.info(f"⚖️ [Consensus] 全票通过! {votes}/{total_nodes} 节点意见完全一致。")
            return True, most_common_result

        # 简单多数共识（>50%）
        if votes > total_nodes / 2:
            # 但分散度较高时，标记为"弱共识"
            if vote_ratio < cls.FORK_DIVERSITY_THRESHOLD:
                logger.warning(
                    f"⚖️ [Consensus] 弱共识: {votes}/{total_nodes} ({vote_ratio:.1%})，分散度较高"
                )
            else:
                logger.info(f"⚖️ [Consensus] 达成共识! {votes}/{total_nodes} 节点意见一致。")
            return True, most_common_result

        # ===== 第二轮：检测是否有语义相近的答案 =====
        unique_results = len(counts)
        diversity_ratio = unique_results / total_nodes

        if diversity_ratio > 0.5 and enable_llm:
            logger.info(
                f"🔀 [Consensus] 分散度过高 ({unique_results} 种结果/{total_nodes} 节点)，尝试 LLM 语义分析..."
            )

            # 合并语义相近的答案
            merged = cls._llm_merge_semantics(results)
            if merged:
                merged_counts = Counter(merged)
                merged_result, merged_votes = merged_counts.most_common(1)[0]
                merged_ratio = merged_votes / len(merged)
                if merged_ratio > 0.5:
                    logger.info(
                        f"⚖️ [Consensus] LLM 语义合并后达成共识: {merged_votes}/{len(merged)} ({merged_ratio:.1%})"
                    )
                    return True, merged_result

            # LLM 无法合并，使用 LLM 仲裁
            winner = cls._llm_judge(results)
            if winner:
                logger.info(f"⚖️ [Consensus] LLM 仲裁结果: {winner}")
                return True, winner

        # ===== 第三轮：返回分散状态 =====
        breakdown = ", ".join(f"{k}({v})" for k, v in counts.most_common(3))
        logger.error(
            f"🚨 [Consensus Mismatch] 发生严重硬分叉！{unique_results} 种结果，票数: {breakdown}"
        )
        return False, f"Forked State! Votes: {dict(counts)}"

    @classmethod
    def _llm_merge_semantics(cls, results: List[str]) -> Optional[List[str]]:
        """
        Phase 2 P2: 使用 LLM 合并语义相近的答案

        例如：
        - "The answer is 42"
        - "42"
        - "Result: 42"
        三个都表达同一语义，应合并
        """
        brain = _get_brain()
        if brain is None:
            return None

        unique_results = list(set(results))
        if len(unique_results) <= 2:
            return None  # 结果已经够集中，无需合并

        # 只取前10个最常见的答案进行合并
        sample = unique_results[:10]

        prompt = f"""You are a semantic clustering tool for multi-agent consensus.
Given {len(sample)} candidate answers that represent the same underlying result,
group them by semantic equivalence.

Candidates:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(sample))}

Task: Identify which candidates mean the same thing.
Return a JSON list where each element is the canonical form of the most representative answer,
preserving the order of the input list.

Example:
Input: ["42", "The answer is 42", "42.0"]
Output: ["42", "42", "42"]

Return ONLY a valid JSON list, no markdown, no explanation."""

        try:
            response = brain.consult(
                prompt,
                system_prompt="你是一个语义聚类工具，帮助多智能体系统判断答案是否等价。直接返回 JSON 列表，不要解释。",
                model="glm-4-flash",
            )
            if response:
                import re

                json_match = re.search(r"\[.*\]", response, re.DOTALL)
                if json_match:
                    import json

                    merged = json.loads(json_match.group())
                    if isinstance(merged, list) and len(merged) == len(sample):
                        logger.info(
                            f"🧠 [Consensus] LLM 语义合并: {len(sample)} → {len(set(merged))} 类"
                        )
                        # 将合并结果映射回原始顺序
                        return merged
        except Exception as e:
            logger.debug(f"🧠 [Consensus] LLM 合并失败: {e}")

        return None

    @classmethod
    def _llm_judge(cls, results: List[str]) -> Optional[str]:
        """
        Phase 2 P2: 使用 LLM 作为仲裁者，判断哪个答案最合理

        当多数投票失败时，让 LLM 判断哪个答案最可能正确
        """
        brain = _get_brain()
        if brain is None:
            return None

        unique_results = list(set(results))
        if len(unique_results) > 5:
            # 太多候选，只取票数最高的5个
            counts = Counter(results)
            unique_results = [r for r, _ in counts.most_common(5)]

        if not unique_results:
            return None

        prompt = f"""You are the final arbiter in a multi-agent consensus decision.
{len(unique_results)} agents have submitted different answers to the same task.
Your job is to pick the ONE answer that is MOST LIKELY to be correct.

Submitted answers:
{chr(10).join(f"[{i+1}] {r}" for i, r in enumerate(unique_results))}

Evaluation criteria (in order of priority):
1. Technical correctness (does the answer actually solve the problem?)
2. Completeness (does it cover all aspects of the task?)
3. Clarity (is the answer unambiguous and well-structured?)
4. Safety (does it avoid dangerous operations or hardcoded secrets?)

Return your choice as the EXACT text of the selected answer (no brackets, no numbering).
Return ONLY the answer text, no explanation."""

        try:
            response = brain.consult(
                prompt,
                system_prompt="你是一个多智能体系统的最终仲裁者。你的职责是从多个候选答案中选择最可能正确的一个。只返回选中的答案文本，不要解释。",
                model="deepseek-chat",  # 使用更聪明的模型做仲裁
            )
            if response and response.strip():
                # 验证返回的答案是否在候选列表中
                cleaned = response.strip()
                # 精确匹配或部分匹配
                for r in unique_results:
                    if r.strip() == cleaned or cleaned in r or r in cleaned:
                        logger.info(f"⚖️ [Consensus] LLM 仲裁选中: {r[:50]}...")
                        return r
                # 如果没有匹配，可能 LLM 返回了改写版本，使用最近的候选
                if unique_results:
                    logger.warning(
                        f"⚖️ [Consensus] LLM 返回未识别文本，使用第一候选: {unique_results[0][:50]}..."
                    )
                    return unique_results[0]
        except Exception as e:
            logger.debug(f"⚖️ [Consensus] LLM 仲裁失败: {e}")

        return None

    @classmethod
    def process_node_payloads(cls, payloads: List[Dict], enable_llm: bool = True) -> Dict:
        """
        从一堆工蜂传回的 payload 里提取核心 result 并发起裁决

        Args:
            payloads: 各节点返回的 payload 列表
            enable_llm: 是否启用 LLM 增强（默认开启）

        Returns:
            包含 status, canonical_result, consensus_id, nodes_participated 的字典
        """
        job_id = "unknown"
        if payloads and isinstance(payloads[0], dict):
            job_id = payloads[0].get("metadata", {}).get("consensus_id", "unknown")

        logger.debug(
            f"🔍 [Consensus] 正在核验共识块提案组 {job_id}，共 {len(payloads)} 个节点见证人。"
        )

        results_str = [str(p.get("result", p.get("status", ""))) for p in payloads]
        is_consensus_reached, canonical_result = cls.match_semantics(
            results_str, enable_llm=enable_llm
        )

        return {
            "status": "CONSENSUS_REACHED" if is_consensus_reached else "FORKED",
            "canonical_result": canonical_result,
            "consensus_id": job_id,
            "nodes_participated": len(payloads),
        }

    # ===== Phase 2: 完整投票系统实例方法 =====

    def __init__(self, communication_bus: Any = None):
        """初始化共识引擎

        Args:
            communication_bus: 通信总线 (SwarmCommunication 实例)
        """
        self.active_proposals: Dict[str, DecisionProposal] = {}
        self.votes: Dict[str, Dict[str, Vote]] = {}  # proposal_id -> {voter_id: Vote}
        self.decision_history: List[DecisionResult] = []
        self.communication_bus = communication_bus

        # 权重计算函数 - 基于蜜蜂适应度
        self.weight_calculator: Callable[[str], float] = self._default_weight_calculator

    def _default_weight_calculator(self, voter_id: str) -> float:
        """默认权重计算 - 所有投票者权重相同

        实际实现应该查询蜜蜂的适应度分数
        """
        return 1.0

    def cast_vote(
        self,
        proposal_id: str,
        voter_id: str,
        option: VoteOption,
        justification: Optional[str] = None,
    ) -> bool:
        """提交投票

        Args:
            proposal_id: 提案ID
            voter_id: 投票者ID
            option: 投票选项
            justification: 投票理由（可选）

        Returns:
            是否成功提交
        """
        if proposal_id not in self.active_proposals:
            logger.warning(f"⚖️ [Consensus] 提案不存在: {proposal_id}")
            return False

        if proposal_id not in self.votes:
            self.votes[proposal_id] = {}

        weight = self.weight_calculator(voter_id)

        self.votes[proposal_id][voter_id] = Vote(
            voter_id=voter_id,
            option=option,
            weight=weight,
            justification=justification,
        )

        logger.info(f"⚖️ [Consensus] {voter_id} 投了 {option.value} (提案: {proposal_id})")
        return True

    def get_vote_status(self, proposal_id: str) -> Dict:
        """获取提案的投票状态

        Args:
            proposal_id: 提案ID

        Returns:
            投票状态字典
        """
        if proposal_id not in self.active_proposals:
            return {"error": "proposal_not_found"}

        proposal = self.active_proposals[proposal_id]
        votes = self.votes.get(proposal_id, {})

        # 统计投票
        vote_counts: Dict[VoteOption, int] = {
            VoteOption.APPROVE: 0,
            VoteOption.REJECT: 0,
            VoteOption.ABSTAIN: 0,
            VoteOption.DEFER: 0,
        }
        weighted_counts: Dict[VoteOption, float] = {
            VoteOption.APPROVE: 0.0,
            VoteOption.REJECT: 0.0,
            VoteOption.ABSTAIN: 0.0,
            VoteOption.DEFER: 0.0,
        }

        for vote in votes.values():
            vote_counts[vote.option] += 1
            weighted_counts[vote.option] += vote.weight

        total_votes = len(votes)
        eligible_voters = len(votes)  # 暂时假设所有投票者都是合格的

        quorum_reached = (
            total_votes >= eligible_voters * proposal.quorum if eligible_voters > 0 else False
        )

        return {
            "proposal_id": proposal_id,
            "proposal": {
                "title": proposal.title,
                "type": proposal.proposal_type,
                "decision_type": proposal.decision_type.value,
            },
            "vote_counts": {k.value: v for k, v in vote_counts.items()},
            "weighted_counts": {k.value: v for k, v in weighted_counts.items()},
            "total_votes": total_votes,
            "eligible_voters": eligible_voters,
            "quorum_reached": quorum_reached,
            "quorum_required": proposal.quorum,
            "time_remaining": max(0, proposal.time_limit - (time.time() - proposal.created_at)),
            "is_expired": proposal.is_expired(),
        }

    def _try_early_decision(
        self,
        proposal: DecisionProposal,
        votes: Dict[str, Vote],
        total_eligible: int,
    ) -> Optional[DecisionResult]:
        """尝试提前决策

        根据决策类型判断是否可以提前结束投票
        """
        if not votes:
            return None

        # 统计投票
        vote_counts: Dict[VoteOption, int] = {
            VoteOption.APPROVE: 0,
            VoteOption.REJECT: 0,
            VoteOption.ABSTAIN: 0,
            VoteOption.DEFER: 0,
        }
        weighted_counts: Dict[VoteOption, float] = {
            VoteOption.APPROVE: 0.0,
            VoteOption.REJECT: 0.0,
            VoteOption.ABSTAIN: 0.0,
            VoteOption.DEFER: 0.0,
        }

        for vote in votes.values():
            vote_counts[vote.option] += 1
            weighted_counts[vote.option] += vote.weight

        # 检查法定人数
        total_votes = len(votes)
        quorum_reached = (
            total_votes >= total_eligible * proposal.quorum if total_eligible > 0 else False
        )

        if not quorum_reached:
            return None

        # 根据决策类型判断
        approved = False
        final_option = None

        if proposal.decision_type == DecisionType.UNANIMOUS:
            # 全票通过 - 所有投票者都必须 APPROVE
            all_approve = all(v.option == VoteOption.APPROVE for v in votes.values())
            if all_approve and total_votes == total_eligible:
                approved = True
                final_option = VoteOption.APPROVE

        elif proposal.decision_type == DecisionType.MAJORITY:
            # 简单多数 (>50%)
            approve_weight = weighted_counts[VoteOption.APPROVE]
            reject_weight = weighted_counts[VoteOption.REJECT]
            total_valid = approve_weight + reject_weight

            if total_valid > 0:
                if approve_weight > total_valid / 2:
                    approved = True
                    final_option = VoteOption.APPROVE
                elif reject_weight > total_valid / 2:
                    approved = False
                    final_option = VoteOption.REJECT

        elif proposal.decision_type == DecisionType.SUPER_MAJORITY:
            # 超多数 (>= 2/3)
            approve_weight = weighted_counts[VoteOption.APPROVE]
            total_valid = approve_weight + weighted_counts[VoteOption.REJECT]

            if total_valid > 0 and approve_weight >= total_valid * (2 / 3):
                approved = True
                final_option = VoteOption.APPROVE

        elif proposal.decision_type == DecisionType.PLURALITY:
            # 相对多数 - 得票最多的选项获胜
            max_weight = max(weighted_counts.values())
            leading_options = [k for k, v in weighted_counts.items() if v == max_weight]

            if len(leading_options) == 1 and max_weight > 0:
                final_option = leading_options[0]
                approved = final_option == VoteOption.APPROVE

        elif proposal.decision_type == DecisionType.WEIGHTED:
            # 加权投票 - 简单多数但考虑权重
            approve_weight = weighted_counts[VoteOption.APPROVE]
            reject_weight = weighted_counts[VoteOption.REJECT]

            if approve_weight > reject_weight and approve_weight > 0:
                approved = True
                final_option = VoteOption.APPROVE
            elif reject_weight > approve_weight:
                approved = False
                final_option = VoteOption.REJECT

        # 构建结果
        if final_option is not None:
            return self._build_result(
                proposal=proposal,
                votes=votes,
                vote_counts=vote_counts,
                weighted_counts=weighted_counts,
                final_option=final_option,
                approved=approved,
                quorum_reached=quorum_reached,
            )

        return None

    def _build_result(
        self,
        proposal: DecisionProposal,
        votes: Dict[str, Vote],
        vote_counts: Dict[VoteOption, int],
        weighted_counts: Dict[VoteOption, float],
        final_option: Optional[VoteOption],
        approved: bool,
        quorum_reached: bool,
    ) -> DecisionResult:
        """构建决策结果"""
        return DecisionResult(
            proposal_id=proposal.proposal_id,
            approved=approved,
            vote_summary={k.value: v for k, v in vote_counts.items()},
            weighted_summary={k.value: v for k, v in weighted_counts.items()},
            final_option=final_option,
            participating_voters=list(votes.keys()),
            quorum_reached=quorum_reached,
            elapsed_time=time.time() - proposal.created_at,
            metadata={
                "proposal_type": proposal.proposal_type,
                "decision_type": proposal.decision_type.value,
                "proposer_id": proposal.proposer_id,
            },
        )

    def _finalize_decision(
        self,
        proposal: DecisionProposal,
        timeout: bool = False,
    ) -> DecisionResult:
        """最终决策 - 超时或投票结束时调用"""
        votes = self.votes.get(proposal.proposal_id, {})

        # 统计投票
        vote_counts: Dict[VoteOption, int] = {
            VoteOption.APPROVE: 0,
            VoteOption.REJECT: 0,
            VoteOption.ABSTAIN: 0,
            VoteOption.DEFER: 0,
        }
        weighted_counts: Dict[VoteOption, float] = {
            VoteOption.APPROVE: 0.0,
            VoteOption.REJECT: 0.0,
            VoteOption.ABSTAIN: 0.0,
            VoteOption.DEFER: 0.0,
        }

        for vote in votes.values():
            vote_counts[vote.option] += 1
            weighted_counts[vote.option] += vote.weight

        total_votes = len(votes)
        quorum_reached = total_votes > 0

        # 找出得票最多的选项
        max_weight = max(weighted_counts.values())
        leading_options = [k for k, v in weighted_counts.items() if v == max_weight]

        final_option = leading_options[0] if len(leading_options) == 1 and max_weight > 0 else None
        approved = final_option == VoteOption.APPROVE if final_option else False

        if timeout:
            logger.warning(
                f"⚖️ [Consensus] 提案 {proposal.proposal_id} 超时，最终投票: {vote_counts}"
            )

        result = self._build_result(
            proposal=proposal,
            votes=votes,
            vote_counts=vote_counts,
            weighted_counts=weighted_counts,
            final_option=final_option,
            approved=approved,
            quorum_reached=quorum_reached,
        )

        # 记录历史
        self.decision_history.append(result)

        # 清理活跃提案
        self.active_proposals.pop(proposal.proposal_id, None)

        return result

    async def propose_decision(
        self,
        proposal: DecisionProposal,
        voter_ids: List[str],
    ) -> DecisionResult:
        """提出决策提案并等待结果

        Args:
            proposal: 决策提案
            voter_ids: 投票者ID列表

        Returns:
            决策结果
        """
        self.active_proposals[proposal.proposal_id] = proposal
        self.votes[proposal.proposal_id] = {}

        logger.info(f"⚖️ [Consensus] 提出新提案: {proposal.title} ({proposal.proposal_id})")

        # 如果有通信总线，广播提案
        if self.communication_bus:
            # 广播给所有投票者
            pass

        # 等待投票结果
        start_time = time.time()
        total_voters = len(voter_ids)

        while True:
            elapsed = time.time() - start_time

            # 检查是否超时
            if elapsed > proposal.time_limit:
                return self._finalize_decision(proposal, timeout=True)

            # 检查当前投票状态
            current_votes = self.votes.get(proposal.proposal_id, {})
            votes_received = len(current_votes)

            # 尝试提前决策
            result = self._try_early_decision(
                proposal=proposal,
                votes=current_votes,
                total_eligible=total_voters,
            )

            if result:
                # 记录历史
                self.decision_history.append(result)
                self.active_proposals.pop(proposal.proposal_id, None)

                logger.info(
                    f"⚖️ [Consensus] 提案 {proposal.proposal_id} 决策结果: {'通过' if result.approved else '拒绝'}"
                )
                return result

            # 等待一下再检查
            await asyncio.sleep(1.0)

    def get_decision_history(self) -> List[DecisionResult]:
        """获取决策历史"""
        return self.decision_history.copy()

    def get_active_proposals(self) -> List[DecisionProposal]:
        """获取活跃提案列表"""
        return list(self.active_proposals.values())

    def cancel_proposal(self, proposal_id: str) -> bool:
        """取消提案

        Args:
            proposal_id: 提案ID

        Returns:
            是否成功取消
        """
        if proposal_id not in self.active_proposals:
            return False

        logger.warning(f"⚖️ [Consensus] 取消提案: {proposal_id}")

        # 创建取消结果
        proposal = self.active_proposals[proposal_id]
        result = DecisionResult(
            proposal_id=proposal_id,
            approved=False,
            vote_summary={},
            weighted_summary={},
            final_option=None,
            participating_voters=[],
            quorum_reached=False,
            elapsed_time=time.time() - proposal.created_at,
            metadata={"cancelled": True},
        )

        self.decision_history.append(result)
        self.active_proposals.pop(proposal_id, None)
        self.votes.pop(proposal_id, None)

        return True


# ===== 便捷函数 =====

_consensus_engine_instance: Optional[ConsensusEngine] = None


def get_consensus_engine(communication_bus: Any = None) -> ConsensusEngine:
    """获取共识引擎单例

    Args:
        communication_bus: 通信总线实例（可选）

    Returns:
        ConsensusEngine 实例
    """
    global _consensus_engine_instance

    if _consensus_engine_instance is None:
        _consensus_engine_instance = ConsensusEngine(communication_bus=communication_bus)
    elif communication_bus is not None:
        _consensus_engine_instance.communication_bus = communication_bus

    return _consensus_engine_instance
