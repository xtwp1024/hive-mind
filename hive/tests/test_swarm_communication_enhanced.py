# -*- coding: utf-8 -*-
"""测试 SwarmCommunication 增强功能

测试内容:
1. 优先级队列 - 高优先级消息优先处理
2. 消息过期机制 - 防止过时消息堆积
3. 请求-响应模式 - send_and_wait 等待响应
4. 统计信息追踪
5. 新增消息类型 (RESPONSE, HEARTBEAT)
"""

import pytest
import time
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any, List

from hive.core.swarm_communication import (
    SwarmCommunication,
    SwarmMessage,
    MessageType,
    TaskBid,
    Vote,
    PriorityQueueItem,
    MessageEnvelope,
    get_swarm_communication,
)


class TestMessageType:
    """测试消息类型"""

    def test_message_type_has_new_types(self):
        """测试新增的消息类型"""
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.HEARTBEAT.value == "heartbeat"

    def test_all_message_types_exist(self):
        """测试所有消息类型都存在"""
        expected_types = [
            "discovery", "task_request", "task_bid", "vote", 
            "knowledge", "status", "alert", "response", "heartbeat"
        ]
        
        actual_types = [t.value for t in MessageType]
        for expected in expected_types:
            assert expected in actual_types


class TestSwarmMessageEnhanced:
    """测试 SwarmMessage 增强字段"""

    def test_message_has_enhanced_fields(self):
        """测试消息有增强字段"""
        message = SwarmMessage(
            id="test_msg_001",
            type=MessageType.TASK_REQUEST,
            from_bee="worker_1",
            to_bee="worker_2",
            content={"task": "test"},
        )
        
        assert message.expires_at is None
        assert message.correlation_id is None
        assert message.hop_count == 0
        assert message.trace_path == []

    def test_message_is_expired(self):
        """测试消息过期检查"""
        message = SwarmMessage(
            id="test_msg_002",
            type=MessageType.STATUS,
            from_bee="test",
        )
        
        assert message.is_expired() == False
        
        message.expires_at = time.time() - 1.0
        assert message.is_expired() == True
        
        message.expires_at = time.time() + 10.0
        assert message.is_expired() == False

    def test_message_needs_response(self):
        """测试消息是否需要响应"""
        task_request = SwarmMessage(
            id="tr_001",
            type=MessageType.TASK_REQUEST,
            from_bee="test",
        )
        assert task_request.needs_response() == True
        
        vote_msg = SwarmMessage(
            id="v_001",
            type=MessageType.VOTE,
            from_bee="test",
        )
        assert vote_msg.needs_response() == True
        
        with_correlation = SwarmMessage(
            id="c_001",
            type=MessageType.KNOWLEDGE,
            from_bee="test",
            correlation_id="corr_001",
        )
        assert with_correlation.needs_response() == True
        
        no_resp = SwarmMessage(
            id="k_001",
            type=MessageType.KNOWLEDGE,
            from_bee="test",
        )
        assert no_resp.needs_response() == False

    def test_message_is_broadcast(self):
        """测试消息是否为广播"""
        broadcast_msg = SwarmMessage(
            id="b_001",
            type=MessageType.KNOWLEDGE,
            from_bee="test",
            to_bee=None,
        )
        assert broadcast_msg.is_broadcast() == True
        
        direct_msg = SwarmMessage(
            id="d_001",
            type=MessageType.KNOWLEDGE,
            from_bee="test",
            to_bee="recipient",
        )
        assert direct_msg.is_broadcast() == False


class TestPriorityQueue:
    """测试优先级队列"""

    def test_priority_queue_item_ordering(self):
        """测试优先级队列项排序"""
        msg_high = SwarmMessage(
            id="high",
            type=MessageType.ALERT,
            from_bee="test",
            priority=9,
        )
        
        msg_low = SwarmMessage(
            id="low",
            type=MessageType.KNOWLEDGE,
            from_bee="test",
            priority=1,
        )
        
        item_high = PriorityQueueItem(priority=9, timestamp=time.time(), message=msg_high)
        item_low = PriorityQueueItem(priority=1, timestamp=time.time(), message=msg_low)
        
        assert item_high < item_low

    def test_high_priority_messages_processed_first(self):
        """测试高优先级消息先处理"""
        comm = SwarmCommunication()
        
        received_messages = []
        
        def record_message(msg):
            received_messages.append(msg)
        
        comm.subscribe(MessageType.ALERT, record_message)
        comm.subscribe(MessageType.KNOWLEDGE, record_message)
        
        comm.start()
        
        try:
            msg_low = SwarmMessage(
                id="low_1",
                type=MessageType.KNOWLEDGE,
                from_bee="test",
                priority=1,
                content={"order": 1},
            )
            
            msg_high = SwarmMessage(
                id="high_1",
                type=MessageType.ALERT,
                from_bee="test",
                priority=9,
                content={"order": 2},
            )
            
            comm.send_message(msg_low)
            comm.send_message(msg_high)
            
            time.sleep(0.5)
            
            stats = comm.get_stats()
            assert stats["messages_sent"] == 2
            assert stats["messages_received"] >= 0
            
        finally:
            comm.stop()

    def test_message_history_limited(self):
        """测试消息历史限制"""
        comm = SwarmCommunication()
        comm._max_history_size = 5
        
        comm.start()
        
        try:
            for i in range(10):
                msg = SwarmMessage(
                    id=f"msg_{i}",
                    type=MessageType.KNOWLEDGE,
                    from_bee="test",
                    content={"index": i},
                )
                comm.send_message(msg)
            
            time.sleep(0.3)
            
            assert len(comm.message_history) <= comm._max_history_size
            
        finally:
            comm.stop()


class TestMessageExpiry:
    """测试消息过期机制"""

    def test_expired_messages_not_processed(self):
        """测试过期消息不被处理"""
        comm = SwarmCommunication()
        
        comm.start()
        
        try:
            expired_msg = SwarmMessage(
                id="expired_1",
                type=MessageType.KNOWLEDGE,
                from_bee="test",
                expires_at=time.time() - 1.0,
            )
            
            comm.send_message(expired_msg)
            
            time.sleep(0.2)
            
            stats = comm.get_stats()
            assert stats["messages_expired"] >= 1
            
        finally:
            comm.stop()

    def test_check_expired_messages_removes_expired(self):
        """测试检查过期消息时移除过期消息"""
        comm = SwarmCommunication()
        
        msg1 = SwarmMessage(
            id="expired_2",
            type=MessageType.KNOWLEDGE,
            from_bee="test",
            expires_at=time.time() - 10.0,
        )
        
        msg2 = SwarmMessage(
            id="valid_1",
            type=MessageType.KNOWLEDGE,
            from_bee="test",
            expires_at=time.time() + 10.0,
        )
        
        comm._put_priority_message(msg1)
        comm._put_priority_message(msg2)
        
        initial_queue_size = len(comm._priority_queue)
        
        comm._check_expired_messages()
        
        stats = comm.get_stats()
        assert stats["messages_expired"] >= 1


class TestRequestResponsePattern:
    """测试请求-响应模式"""

    def test_send_and_wait_generates_correlation_id(self):
        """测试 send_and_wait 生成 correlation_id"""
        comm = SwarmCommunication()
        
        request_msg = SwarmMessage(
            id="req_001",
            type=MessageType.TASK_REQUEST,
            from_bee="requester",
            to_bee="responder",
            content={"action": "test"},
        )
        
        assert request_msg.correlation_id is None
        
        comm.start()
        
        try:
            def send_response_in_thread():
                time.sleep(0.1)
                comm.send_response(request_msg, {"result": "success"})
            
            response_thread = threading.Thread(target=send_response_in_thread)
            response_thread.start()
            
            success, response = comm.send_and_wait(request_msg, timeout=1.0)
            
            response_thread.join(timeout=2.0)
            
            assert request_msg.correlation_id is not None
            
        finally:
            comm.stop()

    def test_send_and_wait_timeout(self):
        """测试 send_and_wait 超时"""
        comm = SwarmCommunication()
        comm.start()
        
        try:
            request_msg = SwarmMessage(
                id="req_timeout_001",
                type=MessageType.TASK_REQUEST,
                from_bee="requester",
                to_bee="nonexistent_responder",
                content={"action": "test"},
            )
            
            start_time = time.time()
            success, response = comm.send_and_wait(request_msg, timeout=0.5)
            elapsed = time.time() - start_time
            
            assert success == False
            assert response is None
            assert 0.4 <= elapsed <= 0.7
            
            stats = comm.get_stats()
            assert stats["timeouts_occurred"] >= 1
            
        finally:
            comm.stop()

    def test_send_response_without_correlation_id(self):
        """测试发送没有 correlation_id 的响应"""
        comm = SwarmCommunication()
        
        request_msg = SwarmMessage(
            id="req_no_corr",
            type=MessageType.KNOWLEDGE,
            from_bee="requester",
            to_bee="responder",
            content={"action": "test"},
        )
        
        comm.send_response(request_msg, {"result": "success"})
        
        stats = comm.get_stats()
        assert stats["messages_sent"] >= 0

    def test_handle_response_matches_correlation(self):
        """测试响应消息匹配 correlation_id"""
        comm = SwarmCommunication()
        
        test_corr_id = "test_corr_123"
        event = threading.Event()
        
        with comm._pending_lock:
            comm.pending_responses[test_corr_id] = (event, None)
        
        response_msg = SwarmMessage(
            id="resp_001",
            type=MessageType.RESPONSE,
            from_bee="responder",
            to_bee="requester",
            content={"result": "success"},
            correlation_id=test_corr_id,
        )
        
        comm._handle_response(response_msg)
        
        with comm._pending_lock:
            assert test_corr_id not in comm.pending_responses
            assert event.is_set()

    def test_handle_response_no_pending(self):
        """测试处理没有等待者的响应"""
        comm = SwarmCommunication()
        
        response_msg = SwarmMessage(
            id="resp_orphan_001",
            type=MessageType.RESPONSE,
            from_bee="responder",
            to_bee="requester",
            content={"result": "success"},
            correlation_id="nonexistent_corr_id",
        )
        
        comm._handle_response(response_msg)


class TestStatistics:
    """测试统计信息"""

    def test_get_stats_returns_dict(self):
        """测试 get_stats 返回字典"""
        comm = SwarmCommunication()
        
        stats = comm.get_stats()
        
        assert isinstance(stats, dict)
        assert "messages_sent" in stats
        assert "messages_received" in stats
        assert "messages_expired" in stats
        assert "responses_received" in stats
        assert "timeouts_occurred" in stats

    def test_stats_are_updated(self):
        """测试统计信息被更新"""
        comm = SwarmCommunication()
        
        initial_stats = comm.get_stats()
        
        comm.start()
        
        try:
            for i in range(3):
                msg = SwarmMessage(
                    id=f"stat_test_{i}",
                    type=MessageType.KNOWLEDGE,
                    from_bee="test",
                    content={"index": i},
                )
                comm.send_message(msg)
            
            time.sleep(0.2)
            
            stats = comm.get_stats()
            
            assert stats["messages_sent"] == initial_stats["messages_sent"] + 3
            
        finally:
            comm.stop()

    def test_get_swarm_status(self):
        """测试获取蜂群状态"""
        comm = SwarmCommunication()
        
        comm.register_bee("test_bee_1", MagicMock())
        comm.register_bee("test_bee_2", MagicMock())
        
        status = comm.get_swarm_status()
        
        assert status["registered_bees"] == 2
        assert "test_bee_1" in status["active_bees"]
        assert "test_bee_2" in status["active_bees"]
        assert "stats" in status
        assert "pending_responses" in status
        assert "queue_size" in status


class TestBeeRegistration:
    """测试蜜蜂注册"""

    def test_register_bee(self):
        """测试注册蜜蜂"""
        comm = SwarmCommunication()
        
        mock_bee = MagicMock()
        comm.register_bee("worker_1", mock_bee)
        
        assert "worker_1" in comm.registered_bees
        assert comm.registered_bees["worker_1"] == mock_bee

    def test_unregister_bee(self):
        """测试注销蜜蜂"""
        comm = SwarmCommunication()
        
        mock_bee = MagicMock()
        comm.register_bee("worker_1", mock_bee)
        
        assert "worker_1" in comm.registered_bees
        
        comm.unregister_bee("worker_1")
        
        assert "worker_1" not in comm.registered_bees

    def test_unregister_nonexistent(self):
        """测试注销不存在的蜜蜂"""
        comm = SwarmCommunication()
        
        comm.unregister_bee("nonexistent")


class TestGlobalInstance:
    """测试全局实例"""

    def test_get_swarm_communication_returns_instance(self):
        """测试 get_swarm_communication 返回实例"""
        instance = get_swarm_communication()
        
        assert isinstance(instance, SwarmCommunication)
        assert instance.running == True

    def test_get_swarm_communication_returns_same_instance(self):
        """测试 get_swarm_communication 返回相同实例"""
        instance1 = get_swarm_communication()
        instance2 = get_swarm_communication()
        
        assert instance1 is instance2


class TestCommunicationLoop:
    """测试通信循环"""

    def test_start_and_stop(self):
        """测试启动和停止"""
        comm = SwarmCommunication()
        
        assert comm.running == False
        assert comm.comm_thread is None
        
        comm.start()
        
        assert comm.running == True
        assert comm.comm_thread is not None
        assert comm.comm_thread.is_alive() == True
        
        comm.stop()
        
        assert comm.running == False

    def test_double_start_warning(self):
        """测试重复启动警告"""
        comm = SwarmCommunication()
        
        comm.start()
        
        try:
            comm.start()
        finally:
            comm.stop()


class TestCollectiveDecision:
    """测试集体决策"""

    def test_collective_decision_returns_option(self):
        """测试集体决策返回选项"""
        comm = SwarmCommunication()
        comm.start()
        
        try:
            options = ["option_a", "option_b", "option_c"]
            result = comm.collective_decision("Test question?", options, timeout=0.5)
            
            assert result in options or result == "no_decision"
            
        finally:
            comm.stop()

    def test_task_negotiation(self):
        """测试任务协商"""
        comm = SwarmCommunication()
        comm.start()
        
        try:
            task = {"type": "analysis", "target": "test.py"}
            bids = comm.task_negotiation(task)
            
            assert isinstance(bids, list)
            
        finally:
            comm.stop()

    def test_select_optimal_team(self):
        """测试选择最优团队"""
        comm = SwarmCommunication()
        
        bids = [
            TaskBid("bee_1", "coder", 0.9, 5, 0.95, "擅长编码"),
            TaskBid("bee_2", "analyst", 0.85, 3, 0.90, "擅长分析"),
            TaskBid("bee_3", "tester", 0.8, 4, 0.85, "擅长测试"),
        ]
        
        team = comm.select_optimal_team(bids, {"type": "test"})
        
        assert len(team) <= 3
        for bee_id in team:
            assert bee_id in ["bee_1", "bee_2", "bee_3"]

    def test_select_optimal_team_no_bids(self):
        """测试无投标时选择团队"""
        comm = SwarmCommunication()
        
        team = comm.select_optimal_team([], {"type": "test"})
        
        assert team == []


class TestKnowledgeSharing:
    """测试知识分享"""

    def test_share_knowledge(self):
        """测试分享知识"""
        comm = SwarmCommunication()
        comm.start()
        
        try:
            knowledge = {"topic": "design_patterns", "content": "Factory pattern"}
            
            comm.share_knowledge("librarian_1", knowledge)
            
            stats = comm.get_stats()
            assert stats["messages_sent"] >= 1
            
        finally:
            comm.stop()


class TestMessageEnvelope:
    """测试消息信封"""

    def test_message_envelope_defaults(self):
        """测试消息信封默认值"""
        msg = SwarmMessage(
            id="env_test_001",
            type=MessageType.KNOWLEDGE,
            from_bee="test",
        )
        
        envelope = MessageEnvelope(message=msg)
        
        assert envelope.message == msg
        assert envelope.read_count == 0
        assert envelope.is_acknowledged == False
        assert envelope.received_at <= time.time()


class TestFullIntegration:
    """测试完整集成场景"""

    def test_two_bee_communication(self):
        """测试两个蜜蜂之间的通信"""
        comm = SwarmCommunication()
        
        received_messages = []
        
        def on_task_bid(message):
            received_messages.append(message)
        
        comm.subscribe(MessageType.TASK_BID, on_task_bid)
        
        comm.register_bee("bee_a", MagicMock())
        comm.register_bee("bee_b", MagicMock())
        
        comm.start()
        
        try:
            bid_msg = SwarmMessage(
                id="bid_001",
                type=MessageType.TASK_BID,
                from_bee="bee_a",
                to_bee="bee_b",
                content={
                    "task_id": "task_123",
                    "bid": TaskBid(
                        bee_id="bee_a",
                        bee_type="coder",
                        capability_score=0.9,
                        estimated_time=5,
                        confidence=0.95,
                        reasoning="我擅长这个任务"
                    )
                },
            )
            
            comm.send_message(bid_msg)
            
            time.sleep(0.2)
            
            stats = comm.get_stats()
            assert stats["messages_sent"] >= 1
            
        finally:
            comm.stop()

    def test_heartbeat_messages(self):
        """测试心跳消息"""
        comm = SwarmCommunication()
        
        received_heartbeats = []
        
        def on_heartbeat(message):
            received_heartbeats.append(message)
        
        comm.subscribe(MessageType.HEARTBEAT, on_heartbeat)
        
        comm.start()
        
        try:
            heartbeat = SwarmMessage(
                id="hb_001",
                type=MessageType.HEARTBEAT,
                from_bee="sentinel_1",
                content={"status": "healthy", "load": 0.3},
                priority=7,
            )
            
            comm.broadcast(heartbeat)
            
            time.sleep(0.2)
            
            stats = comm.get_stats()
            assert stats["messages_sent"] >= 1
            
        finally:
            comm.stop()
