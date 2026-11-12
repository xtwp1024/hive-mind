# -*- coding: utf-8 -*-
"""测试 Phase 2: 集体决策机制 (Consensus Engine)

测试内容:
1. DecisionType 枚举
2. VoteOption 枚举
3. Vote 数据类
4. DecisionProposal 数据类
5. DecisionResult 数据类
6. ConsensusEngine 实例方法
7. get_consensus_engine 便捷函数
"""

import asyncio
import time
from datetime import datetime
from enum import Enum

import pytest

from hive.core.consensus import (
    ConsensusEngine,
    DecisionProposal,
    DecisionResult,
    DecisionType,
    Vote,
    VoteOption,
    get_consensus_engine,
)


class TestDecisionType:
    """测试 DecisionType 枚举"""

    def test_all_decision_types_exist(self):
        """测试所有决策类型存在"""
        assert hasattr(DecisionType, "UNANIMOUS")
        assert hasattr(DecisionType, "MAJORITY")
        assert hasattr(DecisionType, "SUPER_MAJORITY")
        assert hasattr(DecisionType, "PLURALITY")
        assert hasattr(DecisionType, "WEIGHTED")

    def test_decision_type_values(self):
        """测试决策类型值"""
        assert DecisionType.UNANIMOUS.value == "unanimous"
        assert DecisionType.MAJORITY.value == "majority"
        assert DecisionType.SUPER_MAJORITY.value == "super_majority"
        assert DecisionType.PLURALITY.value == "plurality"
        assert DecisionType.WEIGHTED.value == "weighted"

    def test_decision_type_count(self):
        """测试决策类型数量"""
        assert len(list(DecisionType)) == 5


class TestVoteOption:
    """测试 VoteOption 枚举"""

    def test_all_vote_options_exist(self):
        """测试所有投票选项存在"""
        assert hasattr(VoteOption, "APPROVE")
        assert hasattr(VoteOption, "REJECT")
        assert hasattr(VoteOption, "ABSTAIN")
        assert hasattr(VoteOption, "DEFER")

    def test_vote_option_values(self):
        """测试投票选项值"""
        assert VoteOption.APPROVE.value == "approve"
        assert VoteOption.REJECT.value == "reject"
        assert VoteOption.ABSTAIN.value == "abstain"
        assert VoteOption.DEFER.value == "defer"

    def test_vote_option_count(self):
        """测试投票选项数量"""
        assert len(list(VoteOption)) == 4


class TestVoteDataClass:
    """测试 Vote 数据类"""

    def test_vote_creation_defaults(self):
        """测试创建投票（默认值）"""
        vote = Vote(
            voter_id="bee_001",
            option=VoteOption.APPROVE,
        )

        assert vote.voter_id == "bee_001"
        assert vote.option == VoteOption.APPROVE
        assert vote.weight == 1.0
        assert vote.justification is None
        assert vote.timestamp <= time.time()

    def test_vote_creation_with_params(self):
        """测试创建投票（带参数）"""
        vote = Vote(
            voter_id="bee_002",
            option=VoteOption.REJECT,
            weight=1.5,
            justification="这个提案有安全风险",
        )

        assert vote.voter_id == "bee_002"
        assert vote.option == VoteOption.REJECT
        assert vote.weight == 1.5
        assert vote.justification == "这个提案有安全风险"


class TestDecisionProposal:
    """测试 DecisionProposal 数据类"""

    def test_proposal_creation_defaults(self):
        """测试创建提案（默认值）"""
        proposal = DecisionProposal(
            proposal_id="proposal_001",
            proposer_id="queen_bee",
            title="测试提案",
            description="这是一个测试提案",
            proposal_type="test_proposal",
        )

        assert proposal.proposal_id == "proposal_001"
        assert proposal.proposer_id == "queen_bee"
        assert proposal.title == "测试提案"
        assert proposal.description == "这是一个测试提案"
        assert proposal.proposal_type == "test_proposal"

        # 默认值
        assert proposal.decision_type == DecisionType.MAJORITY
        assert proposal.quorum == 0.5
        assert proposal.time_limit == 300.0
        assert proposal.expires_at is None
        assert proposal.metadata == {}

    def test_proposal_creation_with_params(self):
        """测试创建提案（带参数）"""
        proposal = DecisionProposal(
            proposal_id="proposal_002",
            proposer_id="darwin_bee",
            title="进化新蜂种",
            description="创建一个新的 specialized 蜂种",
            proposal_type="evolve_species",
            decision_type=DecisionType.SUPER_MAJORITY,
            quorum=0.7,
            time_limit=600.0,
            expires_at=time.time() + 3600,
            metadata={"priority": "high"},
        )

        assert proposal.proposal_type == "evolve_species"
        assert proposal.decision_type == DecisionType.SUPER_MAJORITY
        assert proposal.quorum == 0.7
        assert proposal.time_limit == 600.0
        assert proposal.metadata == {"priority": "high"}

    def test_proposal_is_expired(self):
        """测试提案过期检查"""
        # 未过期
        proposal = DecisionProposal(
            proposal_id="p1",
            proposer_id="test",
            title="未过期",
            description="测试",
            proposal_type="test",
            time_limit=300.0,  # 5 分钟
        )
        assert proposal.is_expired() == False

        # 已过期（通过 expires_at）
        proposal2 = DecisionProposal(
            proposal_id="p2",
            proposer_id="test",
            title="已过期",
            description="测试",
            proposal_type="test",
            expires_at=time.time() - 100,  # 100 秒前过期
        )
        assert proposal2.is_expired() == True

        # 已过期（通过 time_limit）
        proposal3 = DecisionProposal(
            proposal_id="p3",
            proposer_id="test",
            title="已过期",
            description="测试",
            proposal_type="test",
            time_limit=0.01,  # 非常短
        )
        time.sleep(0.02)
        assert proposal3.is_expired() == True


class TestDecisionResult:
    """测试 DecisionResult 数据类"""

    def test_result_creation(self):
        """测试创建决策结果"""
        result = DecisionResult(
            proposal_id="proposal_001",
            approved=True,
            vote_summary={"approve": 5, "reject": 2, "abstain": 1, "defer": 0},
            weighted_summary={"approve": 5.5, "reject": 2.0, "abstain": 1.0, "defer": 0.0},
            final_option=VoteOption.APPROVE,
            participating_voters=["bee_1", "bee_2", "bee_3", "bee_4", "bee_5"],
            quorum_reached=True,
            elapsed_time=45.5,
            metadata={"proposal_type": "test"},
        )

        assert result.proposal_id == "proposal_001"
        assert result.approved == True
        assert result.vote_summary["approve"] == 5
        assert result.weighted_summary["approve"] == 5.5
        assert result.final_option == VoteOption.APPROVE
        assert len(result.participating_voters) == 5
        assert result.quorum_reached == True
        assert result.elapsed_time == 45.5
        assert result.metadata["proposal_type"] == "test"

    def test_result_creation_rejected(self):
        """测试创建被拒绝的决策结果"""
        result = DecisionResult(
            proposal_id="proposal_002",
            approved=False,
            vote_summary={"approve": 2, "reject": 5, "abstain": 1, "defer": 0},
            weighted_summary={"approve": 2.0, "reject": 5.5, "abstain": 1.0, "defer": 0.0},
            final_option=VoteOption.REJECT,
            participating_voters=["bee_1", "bee_2", "bee_3"],
            quorum_reached=True,
            elapsed_time=120.0,
        )

        assert result.approved == False
        assert result.final_option == VoteOption.REJECT


class TestConsensusEngine:
    """测试 ConsensusEngine"""

    def test_engine_initialization(self):
        """测试引擎初始化"""
        engine = ConsensusEngine()

        assert isinstance(engine.active_proposals, dict)
        assert isinstance(engine.votes, dict)
        assert isinstance(engine.decision_history, list)
        assert engine.communication_bus is None
        assert callable(engine.weight_calculator)

    def test_cast_vote_nonexistent_proposal(self):
        """测试给不存在的提案投票"""
        engine = ConsensusEngine()

        result = engine.cast_vote(
            proposal_id="nonexistent",
            voter_id="bee_1",
            option=VoteOption.APPROVE,
        )

        assert result == False

    def test_cast_vote(self):
        """测试投票"""
        engine = ConsensusEngine()

        # 先创建一个提案
        proposal = DecisionProposal(
            proposal_id="test_proposal",
            proposer_id="queen",
            title="测试提案",
            description="测试",
            proposal_type="test",
        )

        engine.active_proposals["test_proposal"] = proposal

        # 投票
        result = engine.cast_vote(
            proposal_id="test_proposal",
            voter_id="bee_1",
            option=VoteOption.APPROVE,
            justification="我支持这个提案",
        )

        assert result == True
        assert "test_proposal" in engine.votes
        assert "bee_1" in engine.votes["test_proposal"]

        vote = engine.votes["test_proposal"]["bee_1"]
        assert vote.voter_id == "bee_1"
        assert vote.option == VoteOption.APPROVE
        assert vote.justification == "我支持这个提案"
        assert vote.weight == 1.0

    def test_cast_vote_multiple(self):
        """测试多个投票"""
        engine = ConsensusEngine()

        proposal = DecisionProposal(
            proposal_id="multi_vote_test",
            proposer_id="queen",
            title="多投票测试",
            description="测试",
            proposal_type="test",
        )

        engine.active_proposals["multi_vote_test"] = proposal

        # 多个蜜蜂投票
        votes = [
            ("bee_1", VoteOption.APPROVE, "支持"),
            ("bee_2", VoteOption.APPROVE, "同意"),
            ("bee_3", VoteOption.REJECT, "反对"),
            ("bee_4", VoteOption.ABSTAIN, "弃权"),
        ]

        for voter_id, option, just in votes:
            engine.cast_vote(
                proposal_id="multi_vote_test",
                voter_id=voter_id,
                option=option,
                justification=just,
            )

        assert len(engine.votes["multi_vote_test"]) == 4

    def test_get_vote_status_nonexistent(self):
        """测试获取不存在提案的投票状态"""
        engine = ConsensusEngine()

        status = engine.get_vote_status("nonexistent")
        assert status == {"error": "proposal_not_found"}

    def test_get_vote_status(self):
        """测试获取投票状态"""
        engine = ConsensusEngine()

        proposal = DecisionProposal(
            proposal_id="status_test",
            proposer_id="queen",
            title="状态测试",
            description="测试",
            proposal_type="test",
            quorum=0.5,
        )

        engine.active_proposals["status_test"] = proposal

        # 投 2 票
        engine.cast_vote("status_test", "bee_1", VoteOption.APPROVE)
        engine.cast_vote("status_test", "bee_2", VoteOption.REJECT)

        status = engine.get_vote_status("status_test")

        assert status["proposal_id"] == "status_test"
        assert status["proposal"]["title"] == "状态测试"
        assert status["vote_counts"]["approve"] == 1
        assert status["vote_counts"]["reject"] == 1
        assert status["vote_counts"]["abstain"] == 0
        assert status["total_votes"] == 2
        assert status["quorum_reached"] == True  # 2 票 >= 0 合格选民的 50%
        assert status["is_expired"] == False

    def test_default_weight_calculator(self):
        """测试默认权重计算器"""
        engine = ConsensusEngine()

        assert engine._default_weight_calculator("any_bee") == 1.0
        assert engine._default_weight_calculator("another_bee") == 1.0

    def test_custom_weight_calculator(self):
        """测试自定义权重计算器"""
        engine = ConsensusEngine()

        # 自定义：bee_1 权重 2.0，bee_2 权重 0.5，其他 1.0
        def custom_weight(voter_id: str) -> float:
            if voter_id == "bee_1":
                return 2.0
            elif voter_id == "bee_2":
                return 0.5
            return 1.0

        engine.weight_calculator = custom_weight

        proposal = DecisionProposal(
            proposal_id="weight_test",
            proposer_id="queen",
            title="权重测试",
            description="测试",
            proposal_type="test",
        )

        engine.active_proposals["weight_test"] = proposal

        # 投票
        engine.cast_vote("weight_test", "bee_1", VoteOption.APPROVE)
        engine.cast_vote("weight_test", "bee_2", VoteOption.REJECT)
        engine.cast_vote("weight_test", "bee_3", VoteOption.REJECT)

        # 检查权重
        votes = engine.votes["weight_test"]
        assert votes["bee_1"].weight == 2.0
        assert votes["bee_2"].weight == 0.5
        assert votes["bee_3"].weight == 1.0

    def test_get_active_proposals(self):
        """测试获取活跃提案"""
        engine = ConsensusEngine()

        # 初始为空
        assert engine.get_active_proposals() == []

        # 添加提案
        proposal = DecisionProposal(
            proposal_id="active_test",
            proposer_id="queen",
            title="活跃测试",
            description="测试",
            proposal_type="test",
        )

        engine.active_proposals["active_test"] = proposal

        active = engine.get_active_proposals()
        assert len(active) == 1
        assert active[0].proposal_id == "active_test"

    def test_get_decision_history(self):
        """测试获取决策历史"""
        engine = ConsensusEngine()

        # 初始为空
        assert engine.get_decision_history() == []

        # 添加历史记录
        result = DecisionResult(
            proposal_id="history_test",
            approved=True,
            vote_summary={"approve": 3, "reject": 1},
            weighted_summary={"approve": 3.0, "reject": 1.0},
            final_option=VoteOption.APPROVE,
            participating_voters=["bee_1", "bee_2", "bee_3", "bee_4"],
            quorum_reached=True,
            elapsed_time=30.0,
        )

        engine.decision_history.append(result)

        history = engine.get_decision_history()
        assert len(history) == 1
        assert history[0].proposal_id == "history_test"

    def test_cancel_proposal_nonexistent(self):
        """测试取消不存在的提案"""
        engine = ConsensusEngine()

        result = engine.cancel_proposal("nonexistent")
        assert result == False

    def test_cancel_proposal(self):
        """测试取消提案"""
        engine = ConsensusEngine()

        proposal = DecisionProposal(
            proposal_id="cancel_test",
            proposer_id="queen",
            title="取消测试",
            description="测试",
            proposal_type="test",
        )

        engine.active_proposals["cancel_test"] = proposal
        engine.votes["cancel_test"] = {}  # 空投票

        # 取消
        result = engine.cancel_proposal("cancel_test")

        assert result == True
        assert "cancel_test" not in engine.active_proposals
        assert "cancel_test" not in engine.votes

        # 检查历史记录
        history = engine.get_decision_history()
        assert len(history) == 1
        assert history[0].proposal_id == "cancel_test"
        assert history[0].approved == False
        assert history[0].metadata.get("cancelled") == True


class TestConsensusEngineVotingRules:
    """测试 ConsensusEngine 投票规则"""

    def test_majority_approval(self):
        """测试简单多数批准"""
        engine = ConsensusEngine()

        proposal = DecisionProposal(
            proposal_id="majority_test",
            proposer_id="queen",
            title="多数测试",
            description="测试",
            proposal_type="test",
            decision_type=DecisionType.MAJORITY,
        )

        engine.active_proposals["majority_test"] = proposal

        # 5 个投票者，3 个批准（多数）
        engine.cast_vote("majority_test", "bee_1", VoteOption.APPROVE)
        engine.cast_vote("majority_test", "bee_2", VoteOption.APPROVE)
        engine.cast_vote("majority_test", "bee_3", VoteOption.APPROVE)
        engine.cast_vote("majority_test", "bee_4", VoteOption.REJECT)
        engine.cast_vote("majority_test", "bee_5", VoteOption.REJECT)

        votes = engine.votes["majority_test"]

        # 测试内部方法
        result = engine._try_early_decision(
            proposal=proposal,
            votes=votes,
            total_eligible=5,
        )

        # 3 批准 > 2 拒绝，应该通过
        assert result is not None
        assert result.approved == True
        assert result.final_option == VoteOption.APPROVE

    def test_majority_rejection(self):
        """测试简单多数拒绝"""
        engine = ConsensusEngine()

        proposal = DecisionProposal(
            proposal_id="majority_reject_test",
            proposer_id="queen",
            title="多数拒绝测试",
            description="测试",
            proposal_type="test",
            decision_type=DecisionType.MAJORITY,
        )

        engine.active_proposals["majority_reject_test"] = proposal

        # 5 个投票者，2 个批准，3 个拒绝
        engine.cast_vote("majority_reject_test", "bee_1", VoteOption.APPROVE)
        engine.cast_vote("majority_reject_test", "bee_2", VoteOption.APPROVE)
        engine.cast_vote("majority_reject_test", "bee_3", VoteOption.REJECT)
        engine.cast_vote("majority_reject_test", "bee_4", VoteOption.REJECT)
        engine.cast_vote("majority_reject_test", "bee_5", VoteOption.REJECT)

        votes = engine.votes["majority_reject_test"]

        result = engine._try_early_decision(
            proposal=proposal,
            votes=votes,
            total_eligible=5,
        )

        assert result is not None
        assert result.approved == False
        assert result.final_option == VoteOption.REJECT

    def test_unanimous_approval(self):
        """测试全票通过"""
        engine = ConsensusEngine()

        proposal = DecisionProposal(
            proposal_id="unanimous_test",
            proposer_id="queen",
            title="全票测试",
            description="测试",
            proposal_type="test",
            decision_type=DecisionType.UNANIMOUS,
        )

        engine.active_proposals["unanimous_test"] = proposal

        # 3 个投票者，全部批准
        engine.cast_vote("unanimous_test", "bee_1", VoteOption.APPROVE)
        engine.cast_vote("unanimous_test", "bee_2", VoteOption.APPROVE)
        engine.cast_vote("unanimous_test", "bee_3", VoteOption.APPROVE)

        votes = engine.votes["unanimous_test"]

        result = engine._try_early_decision(
            proposal=proposal,
            votes=votes,
            total_eligible=3,
        )

        assert result is not None
        assert result.approved == True

    def test_unanimous_fails_with_dissent(self):
        """测试有异议时全票不通过"""
        engine = ConsensusEngine()

        proposal = DecisionProposal(
            proposal_id="unanimous_fail_test",
            proposer_id="queen",
            title="全票失败测试",
            description="测试",
            proposal_type="test",
            decision_type=DecisionType.UNANIMOUS,
        )

        engine.active_proposals["unanimous_fail_test"] = proposal

        # 3 个投票者，1 个拒绝
        engine.cast_vote("unanimous_fail_test", "bee_1", VoteOption.APPROVE)
        engine.cast_vote("unanimous_fail_test", "bee_2", VoteOption.REJECT)
        engine.cast_vote("unanimous_fail_test", "bee_3", VoteOption.APPROVE)

        votes = engine.votes["unanimous_fail_test"]

        result = engine._try_early_decision(
            proposal=proposal,
            votes=votes,
            total_eligible=3,
        )

        # 全票不通过时，_try_early_decision 应该返回 None（无法提前决策）
        assert result is None

    def test_super_majority_passes(self):
        """测试超多数通过 (>= 2/3)"""
        engine = ConsensusEngine()

        proposal = DecisionProposal(
            proposal_id="super_majority_test",
            proposer_id="queen",
            title="超多数测试",
            description="测试",
            proposal_type="test",
            decision_type=DecisionType.SUPER_MAJORITY,
        )

        engine.active_proposals["super_majority_test"] = proposal

        # 6 个投票者，4 个批准 (4/6 = 2/3，刚好通过)
        engine.cast_vote("super_majority_test", "bee_1", VoteOption.APPROVE)
        engine.cast_vote("super_majority_test", "bee_2", VoteOption.APPROVE)
        engine.cast_vote("super_majority_test", "bee_3", VoteOption.APPROVE)
        engine.cast_vote("super_majority_test", "bee_4", VoteOption.APPROVE)
        engine.cast_vote("super_majority_test", "bee_5", VoteOption.REJECT)
        engine.cast_vote("super_majority_test", "bee_6", VoteOption.REJECT)

        votes = engine.votes["super_majority_test"]

        result = engine._try_early_decision(
            proposal=proposal,
            votes=votes,
            total_eligible=6,
        )

        # 4 批准 / (4 批准 + 2 拒绝) = 4/6 = 2/3，应该通过
        assert result is not None
        assert result.approved == True

    def test_super_majority_fails(self):
        """测试超多数不通过"""
        engine = ConsensusEngine()

        proposal = DecisionProposal(
            proposal_id="super_majority_fail_test",
            proposer_id="queen",
            title="超多数失败测试",
            description="测试",
            proposal_type="test",
            decision_type=DecisionType.SUPER_MAJORITY,
        )

        engine.active_proposals["super_majority_fail_test"] = proposal

        # 6 个投票者，3 个批准 (3/6 = 0.5 < 2/3)
        engine.cast_vote("super_majority_fail_test", "bee_1", VoteOption.APPROVE)
        engine.cast_vote("super_majority_fail_test", "bee_2", VoteOption.APPROVE)
        engine.cast_vote("super_majority_fail_test", "bee_3", VoteOption.APPROVE)
        engine.cast_vote("super_majority_fail_test", "bee_4", VoteOption.REJECT)
        engine.cast_vote("super_majority_fail_test", "bee_5", VoteOption.REJECT)
        engine.cast_vote("super_majority_fail_test", "bee_6", VoteOption.REJECT)

        votes = engine.votes["super_majority_fail_test"]

        result = engine._try_early_decision(
            proposal=proposal,
            votes=votes,
            total_eligible=6,
        )

        # 3 批准 / (3 批准 + 3 拒绝) = 0.5 < 2/3，不应该通过
        # 注意：_try_early_decision 可能返回 None，因为无法确定
        if result:
            assert result.approved == False

    def test_weighted_voting(self):
        """测试加权投票"""
        engine = ConsensusEngine()

        # 自定义权重
        def custom_weight(voter_id: str) -> float:
            weights = {
                "bee_senior": 3.0,  # 资深蜜蜂权重高
                "bee_mid": 2.0,  # 中级蜜蜂
                "bee_junior": 1.0,  # 初级蜜蜂
            }
            return weights.get(voter_id, 1.0)

        engine.weight_calculator = custom_weight

        proposal = DecisionProposal(
            proposal_id="weighted_test",
            proposer_id="queen",
            title="加权投票测试",
            description="测试",
            proposal_type="test",
            decision_type=DecisionType.WEIGHTED,
        )

        engine.active_proposals["weighted_test"] = proposal

        # 场景：2 个初级批准，1 个资深拒绝
        # 批准权重: 1 + 1 = 2
        # 拒绝权重: 3
        # 拒绝权重更高，应该失败
        engine.cast_vote("weighted_test", "bee_junior_1", VoteOption.APPROVE)
        engine.cast_vote("weighted_test", "bee_junior_2", VoteOption.APPROVE)
        engine.cast_vote("weighted_test", "bee_senior", VoteOption.REJECT)

        votes = engine.votes["weighted_test"]

        result = engine._try_early_decision(
            proposal=proposal,
            votes=votes,
            total_eligible=3,
        )

        # 拒绝权重 3 > 批准权重 2，应该拒绝
        assert result is not None
        assert result.approved == False


class TestGetConsensusEngine:
    """测试 get_consensus_engine 便捷函数"""

    def test_get_consensus_engine_returns_instance(self):
        """测试返回实例"""
        engine = get_consensus_engine()

        assert isinstance(engine, ConsensusEngine)

    def test_get_consensus_engine_singleton(self):
        """测试单例模式"""
        engine1 = get_consensus_engine()
        engine2 = get_consensus_engine()

        assert engine1 is engine2

    def test_get_consensus_engine_with_bus(self):
        """测试带通信总线"""
        mock_bus = "mock_bus"  # 模拟总线

        engine = get_consensus_engine(communication_bus=mock_bus)

        assert engine.communication_bus == mock_bus


class TestConsensusIntegration:
    """测试 ConsensusEngine 集成场景"""

    def test_full_voting_flow(self):
        """测试完整的投票流程"""
        engine = ConsensusEngine()

        # 1. 创建提案
        proposal = DecisionProposal(
            proposal_id="full_flow_test",
            proposer_id="darwin_bee",
            title="创建新蜂种: SecurityExpertBee",
            description="创建一个专门处理安全审计的蜂种",
            proposal_type="evolve_species",
            decision_type=DecisionType.MAJORITY,
            quorum=0.5,
            time_limit=120.0,
        )

        engine.active_proposals["full_flow_test"] = proposal

        # 2. 获取初始状态
        initial_status = engine.get_vote_status("full_flow_test")
        assert initial_status["total_votes"] == 0
        assert initial_status["quorum_reached"] == False

        # 3. 多个蜜蜂投票
        votes = [
            ("engineer_bee", VoteOption.APPROVE, "安全审计很重要，支持"),
            ("security_bee", VoteOption.APPROVE, "我就是做安全的，支持创建"),
            ("architect_bee", VoteOption.APPROVE, "架构上合理"),
            ("coder_bee", VoteOption.REJECT, "现有蜜蜂已经能处理安全问题"),
        ]

        for voter_id, option, just in votes:
            engine.cast_vote(
                proposal_id="full_flow_test",
                voter_id=voter_id,
                option=option,
                justification=just,
            )

        # 4. 获取投票状态
        status = engine.get_vote_status("full_flow_test")

        assert status["total_votes"] == 4
        assert status["vote_counts"]["approve"] == 3
        assert status["vote_counts"]["reject"] == 1
        assert status["quorum_reached"] == True

        # 5. 尝试提前决策
        votes_dict = engine.votes["full_flow_test"]
        result = engine._try_early_decision(
            proposal=proposal,
            votes=votes_dict,
            total_eligible=4,
        )

        # 3 批准 > 1 拒绝，应该通过
        assert result is not None
        assert result.approved == True
        assert result.final_option == VoteOption.APPROVE
        assert len(result.participating_voters) == 4
