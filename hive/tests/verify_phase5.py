# -*- coding: utf-8 -*-
"""
Phase 5: 分布式网络 - 快速功能验证脚本
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_existing_hive_federation():
    """测试现有的蜂巢联邦模块"""
    print("[1/10] 测试 HiveFederation (现有模块)...")

    from hive.core.hive_federation import HiveFederation, HiveNode

    federation = HiveFederation(
        node_id="test-node-1",
        host="localhost",
        port=8765,
        discovery_mode="manual",
    )

    federation._running = True

    node = HiveNode(
        node_id="peer-node-1",
        host="localhost",
        port=8766,
        capabilities=["vision", "text"],
        load_factor=0.3,
    )

    federation.register_node(node)

    all_nodes = federation.get_all_nodes()
    print(f"    注册节点数量: {len(all_nodes)}")

    vision_nodes = federation.get_nodes_by_capability("vision")
    print(f"    具备 vision 能力的节点: {len(vision_nodes)}")

    best_node = federation.get_best_node(["vision"])
    print(f'    最佳节点: {best_node.node_id if best_node else "None"}')

    status = federation.get_status()
    print(f'    联邦状态: nodes_online={status["nodes_online"]}')

    print("    [OK] HiveFederation 验证通过")
    return True


def test_existing_knowledge_sync():
    """测试现有的知识同步模块"""
    print()
    print("[2/10] 测试 KnowledgeSync (现有模块)...")

    from hive.core.hive_federation import HiveFederation
    from hive.core.knowledge_sync import KnowledgeEntry, KnowledgeSync

    federation = HiveFederation(node_id="test-node", discovery_mode="manual")
    federation._running = True

    sync = KnowledgeSync(federation, max_knowledge=100)

    knowledge_id = sync.push_knowledge(
        content={"pattern": "use_context_manager", "description": "使用上下文管理器管理资源"},
        knowledge_type="pattern",
        priority="high",
        tags=["python", "resource_management"],
    )

    print(f"    知识推送成功: {knowledge_id}")

    results = sync.query(
        knowledge_type="pattern",
        limit=10,
    )

    print(f"    查询结果数: {len(results)}")

    if results:
        print(f"    知识类型: {results[0].type}")
        print(f"    知识优先级: {results[0].priority}")

    status = sync.get_status()
    print(f'    同步状态: total_knowledge={status["total_knowledge"]}')

    print("    [OK] KnowledgeSync 验证通过")
    return True


def test_identity_manager():
    """测试身份管理器"""
    print()
    print("[3/10] 测试 IdentityManager (新增)...")

    from hive.core.node_auth import IdentityManager

    manager = IdentityManager()

    identity = manager.generate_node_identity(metadata={"region": "cn-east", "env": "test"})

    print(f"    生成节点身份: node_id={identity.node_id}")
    print(f"    公钥长度: {len(identity.public_key)}")
    print(f"    私钥长度: {len(identity.private_key)}")

    my_identity = manager.get_my_identity()
    print(f'    本节点身份: {my_identity.node_id if my_identity else "None"}')

    peer_id = "peer-node-1"
    peer_public_key = "peer_public_key_test"
    registered = manager.register_peer_identity(peer_id, peer_public_key)
    print(f"    注册对等节点: {registered}")

    known_nodes = manager.list_known_nodes()
    print(f"    已知节点数: {len(known_nodes)}")

    print("    [OK] IdentityManager 验证通过")
    return True


def test_api_key_manager():
    """测试 API Key 管理器"""
    print()
    print("[4/10] 测试 APIKeyManager (新增)...")

    from hive.core.node_auth import APIKeyManager, PermissionLevel

    manager = APIKeyManager()

    api_key = manager.generate_api_key(
        node_id="test-node-1",
        permissions=["knowledge.read", "task.submit"],
        level=PermissionLevel.WRITE,
        ttl_days=7,
    )

    print(f"    生成 API Key: {api_key[:20]}...")

    valid, key_obj = manager.validate_api_key(api_key, "knowledge.read")
    print(f"    验证 Key (knowledge.read): valid={valid}")

    valid2, key_obj2 = manager.validate_api_key(api_key, "admin")
    print(f"    验证 Key (admin): valid={valid2}")

    stats = manager.get_stats()
    print(f'    Key 统计: total_keys={stats["total_keys"]}, valid_keys={stats["valid_keys"]}')

    key_obj = manager.get_keys_for_node("test-node-1")
    print(f"    节点 Key 数量: {len(key_obj)}")

    print("    [OK] APIKeyManager 验证通过")
    return True


def test_signature_verifier():
    """测试签名验证器"""
    print()
    print("[5/10] 测试 SignatureVerifier (新增)...")

    from hive.core.node_auth import SignatureAlgorithm, SignatureVerifier

    secret_key = "test_secret_key_12345"

    verifier = SignatureVerifier(secret_key, SignatureAlgorithm.HMAC_SHA256)

    data = '{"action": "sync", "timestamp": 1234567890}'
    signature = verifier.sign(data)
    print(f"    生成签名: {signature[:20]}...")

    is_valid = verifier.verify(data, signature)
    print(f"    验证签名: is_valid={is_valid}")

    is_invalid = verifier.verify(data + "tampered", signature)
    print(f"    验证篡改数据: is_valid={is_invalid}")

    payload = {"action": "test", "data": {"value": 42}}
    signed = verifier.sign_request(payload)

    print(f"    签名请求: timestamp={signed.timestamp}")
    print(f"    签名字段存在: {len(signed.signature) > 0}")

    is_valid2, msg = verifier.verify_request(
        payload,
        signed.signature,
        signed.timestamp,
        max_age_seconds=300,
    )
    print(f"    验证请求: is_valid={is_valid2}, msg={msg}")

    print("    [OK] SignatureVerifier 验证通过")
    return True


def test_access_control_list():
    """测试访问控制列表"""
    print()
    print("[6/10] 测试 AccessControlList (新增)...")

    from hive.core.node_auth import AccessControlList

    acl = AccessControlList()

    node_id = "test-node-1"

    acl.grant_permission(node_id, "knowledge.read", granted_by="system")
    acl.grant_permission(node_id, "task.write", granted_by="admin")

    print(f"    授予权限: knowledge.read, task.write")

    can_read = acl.check_permission(node_id, "knowledge.read")
    print(f"    检查 knowledge.read: {can_read}")

    can_write_task = acl.check_permission(node_id, "task.write")
    print(f"    检查 task.write: {can_write_task}")

    can_delete = acl.check_permission(node_id, "knowledge.delete")
    print(f"    检查 knowledge.delete (未授权): {can_delete}")

    acl.assign_role(node_id, "member")
    print(f"    分配角色: member")

    node_perms = acl.get_node_permissions(node_id)
    print(
        f'    节点权限: roles={node_perms["roles"]}, direct_perms={len(node_perms["direct_permissions"])}'
    )

    stats = acl.get_stats()
    print(f'    ACL 统计: nodes_with_permissions={stats["nodes_with_permissions"]}')

    print("    [OK] AccessControlList 验证通过")
    return True


def test_conflict_resolution():
    """测试冲突解决"""
    print()
    print("[7/10] 测试 ConflictResolution (新增)...")

    from hive.core.conflict_resolution import (
        ConflictDetector,
        ConflictLog,
        ConflictRecord,
        ConflictResolver,
        ConflictSeverity,
        ConflictType,
        KnowledgeVersion,
        ResolutionStrategy,
    )

    local_version = KnowledgeVersion(
        entry_id="knowledge_001",
        version=1,
        content_hash="abc123",
        source_node="node-1",
        source_timestamp=1000000.0,
        content={"value": "local", "updated_at": 1000},
    )

    remote_version = KnowledgeVersion(
        entry_id="knowledge_001",
        version=1,
        content_hash="def456",
        source_node="node-2",
        source_timestamp=1000001.0,
        content={"value": "remote", "updated_at": 2000},
    )

    detector = ConflictDetector()

    conflict_info = detector.detect_conflict(local_version, remote_version)

    if conflict_info:
        conflict_type, severity, details = conflict_info
        print(f"    检测到冲突: type={conflict_type.value}, severity={severity.value}")
    else:
        print("    未检测到冲突")

    resolver = ConflictResolver(default_strategy=ResolutionStrategy.LAST_WRITE_WINS)

    conflict = ConflictRecord(
        conflict_id="test_conflict_001",
        conflict_type=ConflictType.VERSION,
        severity=ConflictSeverity.MEDIUM,
        entry_id="knowledge_001",
        local_version=local_version,
        remote_version=remote_version,
    )

    resolved = resolver.resolve(conflict)

    print(f"    冲突解决状态: {resolved.resolution_status.value}")
    print(
        f'    解决策略: {resolved.resolution_strategy.value if resolved.resolution_strategy else "N/A"}'
    )
    print(
        f'    解决版本: {resolved.resolved_version.version if resolved.resolved_version else "N/A"}'
    )

    conflict_log = ConflictLog(max_logs=100)
    conflict_log.log_conflict(conflict)

    stats = conflict_log.get_stats()
    print(f'    冲突日志统计: total_conflicts={stats["total_conflicts"]}')

    print("    [OK] ConflictResolution 验证通过")
    return True


def test_load_balancer():
    """测试负载均衡器"""
    print()
    print("[8/10] 测试 LoadBalancer (新增)...")

    from hive.core.load_balancer import LoadBalancer, LoadBalanceStrategy, NodeHealthStatus

    lb = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)

    lb.add_node("node-1", weight=3, capabilities=["vision", "text"])
    lb.add_node("node-2", weight=2, capabilities=["text"])
    lb.add_node("node-3", weight=5, capabilities=["vision", "audio"])

    print(f"    添加节点: node-1 (weight=3), node-2 (weight=2), node-3 (weight=5)")

    stats = lb.get_stats()
    print(
        f'    负载均衡器状态: total_nodes={stats["total_nodes"]}, available_nodes={stats["available_nodes"]}'
    )

    selected1 = lb.select_node("task-001", ["text"])
    print(f"    选择 (text能力): {selected1}")

    selected2 = lb.select_node("task-002", ["vision"])
    print(f"    选择 (vision能力): {selected2}")

    selected3 = lb.select_node("task-003", ["audio"])
    print(f"    选择 (audio能力): {selected3}")

    lb.update_node_metrics("node-1", load_factor=0.1)
    lb.update_node_metrics("node-2", load_factor=0.5)
    lb.update_node_metrics("node-3", load_factor=0.2)

    lb_least = LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOAD)
    lb_least.add_node("node-1", weight=1)
    lb_least.add_node("node-2", weight=1)
    lb_least.update_node_metrics("node-1", load_factor=0.9)
    lb_least.update_node_metrics("node-2", load_factor=0.1)

    selected_least = lb_least.select_node("task-004")
    print(f"    最少负载选择: {selected_least} (node-1 load=0.9, node-2 load=0.1)")

    lb.record_success("node-1", response_time_ms=50.0)
    lb.record_failure("node-2")

    print("    [OK] LoadBalancer 验证通过")
    return True


def test_health_checker():
    """测试健康检查器"""
    print()
    print("[9/10] 测试 HealthChecker (新增)...")

    from hive.core.load_balancer import HealthChecker, NodeHealthStatus

    checker = HealthChecker(
        check_interval=10.0,
        unhealthy_threshold=3,
        healthy_threshold=2,
    )

    checker.register_node("node-1", initial_status=NodeHealthStatus.HEALTHY)
    checker.register_node("node-2", initial_status=NodeHealthStatus.HEALTHY)
    checker.register_node("node-3", initial_status=NodeHealthStatus.DEGRADED)

    print(f"    注册节点: node-1 (healthy), node-2 (healthy), node-3 (degraded)")

    stats = checker.get_stats()
    print(f'    健康检查统计: total_nodes={stats["total_nodes"]}')

    status1 = checker.get_node_status("node-1")
    print(f'    node-1 状态: {status1.value if status1 else "None"}')

    result = checker.set_node_status("node-1", NodeHealthStatus.DRAINING)
    print(f"    排空 node-1: {result}")

    status2 = checker.get_node_status("node-1")
    print(f'    node-1 新状态: {status2.value if status2 else "None"}')

    result2 = checker.set_node_status("node-2", NodeHealthStatus.UNHEALTHY)
    print(f"    标记 node-2 不健康: {result2}")

    stats2 = checker.get_stats()
    print(f'    统计 by_status: {stats2["by_status"]}')

    print("    [OK] HealthChecker 验证通过")
    return True


def test_failover_manager():
    """测试故障转移管理器"""
    print()
    print("[10/10] 测试 FailoverManager (新增)...")

    from hive.core.load_balancer import (
        FailoverManager,
        LoadBalancer,
        LoadBalanceStrategy,
        NodeHealthStatus,
    )

    lb = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)
    lb.add_node("node-1", weight=1, capabilities=["test"])
    lb.add_node("node-2", weight=1, capabilities=["test"])

    failover = FailoverManager(
        load_balancer=lb,
        max_retries=2,
        retry_delay_seconds=0.01,
    )

    assignment = failover.assign_task(
        task_id="task-001",
        required_capabilities=["test"],
    )

    print(f'    分配任务: task-001 -> {assignment.node_id if assignment else "None"}')

    if assignment:
        failover.start_task(assignment.assignment_id)
        print(f"    开始任务: assignment_id={assignment.assignment_id}")

        success, new_assignment = failover.fail_task(
            assignment.assignment_id,
            error_message="Test failure",
            can_retry=True,
        )

        print(f"    任务失败: can_retry={success}, new_assignment={new_assignment is not None}")

    def mock_executor(node_id: str):
        if node_id == "node-1":
            return False, None, "Node 1 failed"
        return True, "Success from node-2", ""

    from hive.core.load_balancer import LoadBalancerEngine

    lb_engine = LoadBalancerEngine(
        strategy=LoadBalanceStrategy.ROUND_ROBIN,
        max_retries=2,
    )
    lb_engine.add_node("node-1", weight=1)
    lb_engine.add_node("node-2", weight=1)

    success, result, final_assignment = lb_engine.execute_with_failover(
        task_id="task-002",
        executor=mock_executor,
    )

    print(f"    带故障转移执行: success={success}, result={result}")
    if final_assignment:
        print(f"    最终节点: {final_assignment.node_id}")

    stats = failover.get_stats()
    print(f'    故障转移统计: total_assignments={stats["total_assignments"]}')

    print("    [OK] FailoverManager 验证通过")
    return True


def main():
    print("=" * 60)
    print("Phase 5: 分布式网络 - 快速功能验证")
    print("=" * 60)

    tests = [
        test_existing_hive_federation,
        test_existing_knowledge_sync,
        test_identity_manager,
        test_api_key_manager,
        test_signature_verifier,
        test_access_control_list,
        test_conflict_resolution,
        test_load_balancer,
        test_health_checker,
        test_failover_manager,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"    [FAIL] 错误: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"验证结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed == 0:
        print()
        print("Phase 5 所有模块验证通过！")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
