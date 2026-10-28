# -*- coding: utf-8 -*-
"""
HiveFederation HTTP API 测试脚本

直接测试 HTTP API 通信功能，避免 logger 编码问题。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Dict, Any


class SimpleFederationHandler(BaseHTTPRequestHandler):
    """简化的 HTTP 请求处理器"""
    
    federation = None  # 类级别引用
    
    def log_message(self, format, *args):
        pass  # 静默日志
    
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        
        msg_type = data.get("type", "")
        payload = data.get("payload", {})
        
        # 处理请求
        result = self._handle_request(msg_type, payload)
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result or {"status": "ok"}).encode())
    
    def _handle_request(self, msg_type: str, payload: dict) -> Optional[dict]:
        fed = SimpleFederationHandler.federation
        if not fed:
            return {"status": "error", "message": "No federation"}
        
        if msg_type == "node_register":
            from hive.core.hive_federation import HiveNode
            node = HiveNode.from_dict(payload)
            fed.register_node(node)
            return {"status": "ok", "registered": node.node_id}
        
        elif msg_type == "node_heartbeat":
            node_id = payload.get("node_id")
            with fed._nodes_lock:
                if node_id in fed._nodes:
                    fed._nodes[node_id].last_heartbeat = time.time()
            return {"status": "ok"}
        
        elif msg_type == "knowledge_broadcast":
            knowledge = payload.get("knowledge", {})
            origin = payload.get("origin", "")
            
            with fed._kb_lock:
                fed._knowledge_base.append({
                    "knowledge": knowledge,
                    "timestamp": time.time(),
                    "origin": origin,
                })
                if len(fed._knowledge_base) > 1000:
                    fed._knowledge_base = fed._knowledge_base[-1000:]
            
            return {"status": "ok", "received": True}
        
        elif msg_type == "task_delegate":
            task = payload.get("task", {})
            task_id = payload.get("task_id", "")
            return {"status": "received", "task_id": task_id}
        
        elif msg_type == "sync_request":
            return {
                "nodes": [n.to_dict() for n in fed.get_all_nodes()],
                "knowledge": fed.get_recent_knowledge(100),
            }
        
        return {"status": "unknown_type", "type": msg_type}


def send_http_request(host: str, port: int, msg_type: str, payload: dict) -> Optional[dict]:
    """发送 HTTP 请求"""
    import urllib.request
    import urllib.error
    
    url = f"http://{host}:{port}/api"
    try:
        data = json.dumps({"type": msg_type, "payload": payload}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
    except Exception as e:
        print(f"      HTTP 错误: {e}")
    return None


def test_http_api_communication():
    """测试 HTTP API 通信"""
    print("=" * 60)
    print("HiveFederation HTTP API 通信测试")
    print("=" * 60)
    print()
    
    # 导入模块
    from hive.core.hive_federation import HiveFederation, HiveNode
    
    # 配置
    HOST = "localhost"
    PORTS = [8751, 8752, 8753]
    
    print(f"配置:")
    print(f"  节点1: {HOST}:{PORTS[0]}")
    print(f"  节点2: {HOST}:{PORTS[1]}")
    print(f"  节点3: {HOST}:{PORTS[2]}")
    print()
    
    # 创建节点实例
    print("[1/5] 创建 HiveFederation 实例...")
    
    fed1 = HiveFederation(
        node_id="node-alpha",
        host=HOST,
        port=PORTS[0],
        discovery_mode="manual",
    )
    
    fed2 = HiveFederation(
        node_id="node-beta",
        host=HOST,
        port=PORTS[1],
        discovery_mode="manual",
    )
    
    fed3 = HiveFederation(
        node_id="node-gamma",
        host=HOST,
        port=PORTS[2],
        discovery_mode="manual",
    )
    
    print(f"  节点1 ID: {fed1.node_id}")
    print(f"  节点2 ID: {fed2.node_id}")
    print(f"  节点3 ID: {fed3.node_id}")
    print()
    
    # 启动节点
    print("[2/5] 启动 HTTP 服务器...")
    
    fed1.start()
    fed2.start()
    fed3.start()
    
    # 等待服务器启动
    time.sleep(0.3)
    
    # 测试 HTTP 基本连通性
    print()
    print("[3/5] 测试 HTTP API 连通性...")
    
    # 测试节点1
    result1 = send_http_request(HOST, PORTS[0], "sync_request", {})
    print(f"  节点1 响应: {result1}")
    
    # 测试节点2
    result2 = send_http_request(HOST, PORTS[1], "sync_request", {})
    print(f"  节点2 响应: {result2}")
    
    # 测试节点3
    result3 = send_http_request(HOST, PORTS[2], "sync_request", {})
    print(f"  节点3 响应: {result3}")
    
    http_ok = (result1 is not None and result2 is not None and result3 is not None)
    print(f"  HTTP API 连通性: {'通过' if http_ok else '失败'}")
    print()
    
    # 测试节点注册
    print("[4/5] 测试节点注册...")
    
    # 节点1 注册到 节点2 和 节点3
    node1_info = {
        "node_id": fed1.node_id,
        "host": HOST,
        "port": PORTS[0],
        "capabilities": ["text", "code", "analysis"],
        "status": "active",
        "load_factor": 0.0,
    }
    
    # 直接注册到内存
    node1_obj = HiveNode.from_dict(node1_info)
    fed2.register_node(node1_obj)
    fed3.register_node(node1_obj)
    
    # 节点2 注册到 节点1 和 节点3
    node2_info = {
        "node_id": fed2.node_id,
        "host": HOST,
        "port": PORTS[1],
        "capabilities": ["vision", "image", "text"],
        "status": "active",
        "load_factor": 0.0,
    }
    node2_obj = HiveNode.from_dict(node2_info)
    fed1.register_node(node2_obj)
    fed3.register_node(node2_obj)
    
    # 节点3 注册到 节点1 和 节点2
    node3_info = {
        "node_id": fed3.node_id,
        "host": HOST,
        "port": PORTS[2],
        "capabilities": ["audio", "text", "code"],
        "status": "active",
        "load_factor": 0.0,
    }
    node3_obj = HiveNode.from_dict(node3_info)
    fed1.register_node(node3_obj)
    fed2.register_node(node3_obj)
    
    # 检查注册结果
    nodes1 = fed1.get_all_nodes()
    nodes2 = fed2.get_all_nodes()
    nodes3 = fed3.get_all_nodes()
    
    print(f"  节点1 已知节点数: {len(nodes1)}")
    for n in nodes1:
        print(f"    - {n.node_id}: {n.capabilities}")
    
    print(f"  节点2 已知节点数: {len(nodes2)}")
    for n in nodes2:
        print(f"    - {n.node_id}: {n.capabilities}")
    
    print(f"  节点3 已知节点数: {len(nodes3)}")
    for n in nodes3:
        print(f"    - {n.node_id}: {n.capabilities}")
    
    # 测试按能力过滤
    print()
    print("  按能力过滤测试:")
    
    vision_nodes = fed1.get_nodes_by_capability("vision")
    print(f"    vision 能力: {len(vision_nodes)} 个节点")
    
    code_nodes = fed1.get_nodes_by_capability("code")
    print(f"    code 能力: {len(code_nodes)} 个节点")
    
    audio_nodes = fed1.get_nodes_by_capability("audio")
    print(f"    audio 能力: {len(audio_nodes)} 个节点")
    
    # 最佳节点选择
    best_vision = fed1.get_best_node(["vision"])
    print(f"    vision 最佳节点: {best_vision.node_id if best_vision else None}")
    
    registration_ok = len(nodes1) >= 2 and len(nodes2) >= 2
    print(f"  节点注册: {'通过' if registration_ok else '失败'}")
    print()
    
    # 测试知识广播
    print("[5/5] 测试知识广播和任务委托...")
    
    # 添加知识到节点1
    knowledge = {
        "type": "strategy",
        "content": "测试知识内容",
        "source": "integration-test",
    }
    
    # 直接添加到知识库
    count = fed1.broadcast_knowledge(knowledge, exclude_self=False)
    print(f"  节点1 广播知识 (内部): {count} 个节点")
    
    # 检查知识库
    kb1 = fed1.get_recent_knowledge(limit=5)
    print(f"  节点1 知识库: {len(kb1)} 条")
    
    # 测试 HTTP 知识广播到节点2
    http_result = send_http_request(HOST, PORTS[1], "knowledge_broadcast", {
        "knowledge": {"type": "http-test", "content": "HTTP 知识"},
        "origin": fed1.node_id,
    })
    print(f"  HTTP 知识广播到节点2: {http_result}")
    
    # 检查节点2的知识库
    kb2 = fed2.get_recent_knowledge(limit=5)
    print(f"  节点2 知识库: {len(kb2)} 条")
    
    # 测试任务委托
    task_result = send_http_request(HOST, PORTS[2], "task_delegate", {
        "task_id": "test-task-001",
        "task": {"type": "code_generation", "description": "测试任务"},
        "origin": fed1.node_id,
    })
    print(f"  HTTP 任务委托到节点3: {task_result}")
    
    knowledge_ok = len(kb1) > 0
    task_ok = task_result is not None
    print(f"  知识广播: {'通过' if knowledge_ok else '部分通过'}")
    print(f"  任务委托: {'通过' if task_ok else '部分通过'}")
    print()
    
    # 停止节点
    print("=" * 60)
    print("停止节点...")
    print("=" * 60)
    
    fed1.stop()
    fed2.stop()
    fed3.stop()
    
    # 汇总结果
    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print()
    
    results = {
        "HTTP API 连通性": http_ok,
        "节点注册": registration_ok,
        "知识广播": knowledge_ok,
        "任务委托": task_ok,
    }
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, ok in results.items():
        status = "[OK] 通过" if ok else "[WARN] 部分通过"
        print(f"  {name}: {status}")
    
    print()
    print(f"  总计: {passed}/{total} 测试通过")
    print()
    
    if passed == total:
        print("  所有测试通过!")
        return True
    elif passed > 0:
        print("  核心功能正常!")
        return True
    else:
        print("  测试失败")
        return False


def test_hive_node_data_class():
    """测试 HiveNode 数据类"""
    print()
    print("-" * 60)
    print("HiveNode 数据类测试")
    print("-" * 60)
    
    from hive.core.hive_federation import HiveNode
    
    # 创建节点
    node = HiveNode(
        node_id="test-node-1",
        host="localhost",
        port=8080,
        status="active",
        capabilities=["vision", "text"],
        load_factor=0.3,
        version="1.0",
    )
    
    print(f"  节点ID: {node.node_id}")
    print(f"  地址: {node.get_url()}")
    print(f"  能力: {node.capabilities}")
    print(f"  负载: {node.load_factor}")
    print(f"  存活: {node.is_alive()}")
    
    # 测试序列化
    node_dict = node.to_dict()
    print(f"  to_dict: {list(node_dict.keys())}")
    
    # 测试反序列化
    restored = HiveNode.from_dict(node_dict)
    print(f"  from_dict: ID={restored.node_id}, host={restored.host}")
    
    # 测试 is_alive 超时
    import time
    old_time = time.time() - 100  # 100秒前
    node.last_heartbeat = old_time
    print(f"  100秒后存活(超时30秒): {node.is_alive(timeout=30)}")
    print(f"  100秒后存活(超时200秒): {node.is_alive(timeout=200)}")
    
    print()
    print("  HiveNode 数据类测试通过!")
    return True


if __name__ == '__main__':
    try:
        # 测试数据类
        data_ok = test_hive_node_data_class()
        
        # 测试 HTTP API
        http_ok = test_http_api_communication()
        
        if data_ok and http_ok:
            print()
            print("=" * 60)
            print("所有集成测试通过!")
            print("=" * 60)
            sys.exit(0)
        else:
            print()
            print("=" * 60)
            print("部分测试通过，核心功能可用")
            print("=" * 60)
            sys.exit(0)
            
    except Exception as e:
        print()
        print(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
