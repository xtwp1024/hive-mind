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

import logging
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("Hive.Consensus")

# 全局 BrainInterface 引用（懒加载）
_brain = None


def _get_brain():
    """懒加载 BrainInterface 单例"""
    global _brain
    if _brain is None:
        try:
            from hive.utils.brain_interface import BrainInterface, get_brain as _get_brain_impl
            _brain = _get_brain_impl()
        except ImportError:
            try:
                _brain = BrainInterface()
            except Exception:
                return None
    return _brain


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
                logger.warning(f"⚖️ [Consensus] 弱共识: {votes}/{total_nodes} ({vote_ratio:.1%})，分散度较高")
            else:
                logger.info(f"⚖️ [Consensus] 达成共识! {votes}/{total_nodes} 节点意见一致。")
            return True, most_common_result

        # ===== 第二轮：检测是否有语义相近的答案 =====
        unique_results = len(counts)
        diversity_ratio = unique_results / total_nodes

        if diversity_ratio > 0.5 and enable_llm:
            logger.info(f"🔀 [Consensus] 分散度过高 ({unique_results} 种结果/{total_nodes} 节点)，尝试 LLM 语义分析...")

            # 合并语义相近的答案
            merged = cls._llm_merge_semantics(results)
            if merged:
                merged_counts = Counter(merged)
                merged_result, merged_votes = merged_counts.most_common(1)[0]
                merged_ratio = merged_votes / len(merged)
                if merged_ratio > 0.5:
                    logger.info(f"⚖️ [Consensus] LLM 语义合并后达成共识: {merged_votes}/{len(merged)} ({merged_ratio:.1%})")
                    return True, merged_result

            # LLM 无法合并，使用 LLM 仲裁
            winner = cls._llm_judge(results)
            if winner:
                logger.info(f"⚖️ [Consensus] LLM 仲裁结果: {winner}")
                return True, winner

        # ===== 第三轮：返回分散状态 =====
        breakdown = ", ".join(f"{k}({v})" for k, v in counts.most_common(3))
        logger.error(f"🚨 [Consensus Mismatch] 发生严重硬分叉！{unique_results} 种结果，票数: {breakdown}")
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
                model="glm-4-flash"
            )
            if response:
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    import json
                    merged = json.loads(json_match.group())
                    if isinstance(merged, list) and len(merged) == len(sample):
                        logger.info(f"🧠 [Consensus] LLM 语义合并: {len(sample)} → {len(set(merged))} 类")
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
                model="deepseek-chat"  # 使用更聪明的模型做仲裁
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
                    logger.warning(f"⚖️ [Consensus] LLM 返回未识别文本，使用第一候选: {unique_results[0][:50]}...")
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

        logger.debug(f"🔍 [Consensus] 正在核验共识块提案组 {job_id}，共 {len(payloads)} 个节点见证人。")

        results_str = [str(p.get("result", p.get("status", ""))) for p in payloads]
        is_consensus_reached, canonical_result = cls.match_semantics(results_str, enable_llm=enable_llm)

        return {
            "status": "CONSENSUS_REACHED" if is_consensus_reached else "FORKED",
            "canonical_result": canonical_result,
            "consensus_id": job_id,
            "nodes_participated": len(payloads)
        }
