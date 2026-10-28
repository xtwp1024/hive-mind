# -*- coding: utf-8 -*-
"""
Autopoietic Mesh Network - Self-Organization System
自创生网状网络 - 自组织系统

基于蜂巢2.0框架，实现一个去中心化、自组织的网状网络系统。

核心概念：
1. 去中心化：无中央控制，每个节点自主决策
2. 自组织：网络拓扑动态调整，自动优化
3. 演化性：基于DNA的节点特性演化
4. 韧性：节点失效自动恢复，网络持续运行
"""

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutopoieticMeshNetwork")


# ==================== 数据结构 ====================

class NodeState(Enum):
    """节点状态枚举"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    OVERLOADED = "overloaded"
    DISCONNECTED = "disconnected"
    RECOVERING = "recovering"


class MessageType(Enum):
    """消息类型枚举"""
    DISCOVERY = "discovery"           # 节点发现
    HEARTBEAT = "heartbeat"           # 心跳检测
    DATA_EXCHANGE = "data_exchange"   # 数据交换
    TOPOLOGY_UPDATE = "topology_update"  # 拓扑更新
    LOAD_BALANCE = "load_balance"     # 负载均衡
    DNA_SYNC = "dna_sync"             # DNA同步
    FAILURE_DETECTED = "failure_detected"  # 故障检测


@dataclass
class NetworkAddress:
    """网络地址"""
    node_id: str
    host: str
    port: int
    
    def __hash__(self):
        return hash((self.node_id, self.host, self.port))
    
    def __eq__(self, other):
        if not isinstance(other, NetworkAddress):
            return False
        return self.node_id == other.node_id


@dataclass
class NetworkMessage:
    """网络消息"""
    type: MessageType
    source: NetworkAddress
    destination: NetworkAddress
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default="")
    
    def __post_init__(self):
        """生成消息ID"""
        if not self.message_id:
            content = f"{self.source.node_id}_{self.type.value}_{self.timestamp}"
            self.message_id = hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class NodeDNA:
    """节点DNA - 节点特性编码"""
    node_id: str
    
    # 基础属性
    generation: int = 1
    fitness: float = 0.5
    
    # 能力权重（0.0-1.0）
    processing_power: float = 0.5      # 处理能力
    storage_capacity: float = 0.5       # 存储容量
    network_bandwidth: float = 0.5      # 网络带宽
    reliability: float = 0.5            # 可靠性
    social_skill: float = 0.5            # 社交能力（连接倾向）
    innovation: float = 0.5             # 创新能力（新连接探索）
    
    # 演化参数
    mutation_rate: float = 0.1
    learning_rate: float = 0.1
    
    def mutate(self) -> 'NodeDNA':
        """DNA突变"""
        new_dna = NodeDNA(
            node_id=self.node_id,
            generation=self.generation + 1,
            fitness=self.fitness
        )
        
        # 对每个能力进行突变
        capabilities = [
            'processing_power', 'storage_capacity', 'network_bandwidth',
            'reliability', 'social_skill', 'innovation'
        ]
        
        for cap in capabilities:
            current_value = getattr(self, cap)
            if random.random() < self.mutation_rate:
                # 高斯突变
                delta = random.gauss(0, 0.1)
                new_value = max(0.0, min(1.0, current_value + delta))
                setattr(new_dna, cap, new_value)
            else:
                setattr(new_dna, cap, current_value)
        
        return new_dna
    
    def crossover(self, other: 'NodeDNA') -> 'NodeDNA':
        """DNA交叉"""
        child_dna = NodeDNA(
            node_id=self.node_id,
            generation=max(self.generation, other.generation) + 1,
            fitness=(self.fitness + other.fitness) / 2
        )
        
        # 平均交叉
        capabilities = [
            'processing_power', 'storage_capacity', 'network_bandwidth',
            'reliability', 'social_skill', 'innovation'
        ]
        
        for cap in capabilities:
            self_value = getattr(self, cap)
            other_value = getattr(other, cap)
            child_value = (self_value + other_value) / 2
            setattr(child_dna, cap, child_value)
        
        return child_dna
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'node_id': self.node_id,
            'generation': self.generation,
            'fitness': self.fitness,
            'processing_power': self.processing_power,
            'storage_capacity': self.storage_capacity,
            'network_bandwidth': self.network_bandwidth,
            'reliability': self.reliability,
            'social_skill': self.social_skill,
            'innovation': self.innovation
        }


@dataclass
class NodeLoad:
    """节点负载信息"""
    cpu_usage: float = 0.0      # CPU使用率（0.0-1.0）
    memory_usage: float = 0.0   # 内存使用率（0.0-1.0）
    network_usage: float = 0.0  # 网络使用率（0.0-1.0）
    active_connections: int = 0 # 活跃连接数
    
    def get_overall_load(self) -> float:
        """获取总体负载（0.0-1.0）"""
        # 加权平均
        weights = {'cpu_usage': 0.4, 'memory_usage': 0.3, 'network_usage': 0.3}
        total = 0.0
        for metric, weight in weights.items():
            total += getattr(self, metric) * weight
        return total
    
    def is_overloaded(self, threshold: float = 0.8) -> bool:
        """检查是否过载"""
        return self.get_overall_load() > threshold


# ==================== 网络节点 ====================

class NetworkNode(ABC):
    """网络节点抽象基类"""
    
    def __init__(self, dna: NodeDNA, address: NetworkAddress):
        """
        初始化节点
        
        Args:
            dna: 节点DNA
            address: 网络地址
        """
        self.dna = dna
        self.address = address
        self.state = NodeState.INITIALIZING
        
        # 网络
        self.neighbors: Set[NetworkAddress] = set()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # 负载
        self.load = NodeLoad()
        
        # 统计
        self.messages_sent = 0
        self.messages_received = 0
        self.uptime = time.time()
        
        # 运行标志
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """启动节点"""
        logger.info(f"Node {self.address.node_id} starting...")
        self.state = NodeState.ACTIVE
        self._running = True
        
        # 启动后台任务
        self._tasks.append(asyncio.create_task(self._message_loop()))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.append(asyncio.create_task(self._self_organize_loop()))
        
        logger.info(f"Node {self.address.node_id} started")
    
    async def stop(self) -> None:
        """停止节点"""
        logger.info(f"Node {self.address.node_id} stopping...")
        self._running = False
        self.state = NodeState.DISCONNECTED
        
        # 取消所有任务
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info(f"Node {self.address.node_id} stopped")
    
    async def _message_loop(self) -> None:
        """消息处理循环"""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                await self._process_message(message)
                self.messages_received += 1
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while self._running:
            try:
                await asyncio.sleep(5.0)
                await self._send_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _self_organize_loop(self) -> None:
        """自组织循环"""
        while self._running:
            try:
                await asyncio.sleep(10.0)
                await self._self_organize()
            except Exception as e:
                logger.error(f"Self-organization error: {e}")
    
    @abstractmethod
    async def _process_message(self, message: NetworkMessage) -> None:
        """处理消息（子类实现）"""
        pass
    
    @abstractmethod
    async def _send_heartbeat(self) -> None:
        """发送心跳（子类实现）"""
        pass
    
    @abstractmethod
    async def _self_organize(self) -> None:
        """自组织逻辑（子类实现）"""
        pass
    
    async def send_message(self, message: NetworkMessage) -> bool:
        """发送消息"""
        # 基类实现：放入队列
        self.messages_sent += 1
        return True
    
    def update_load(self, cpu: float, memory: float, network: float) -> None:
        """更新负载"""
        self.load.cpu_usage = cpu
        self.load.memory_usage = memory
        self.load.network_usage = network
        self.load.active_connections = len(self.neighbors)
        
        # 检查过载
        if self.load.is_overloaded():
            if self.state != NodeState.OVERLOADED:
                self.state = NodeState.OVERLOADED
                logger.warning(f"Node {self.address.node_id} is overloaded")
        else:
            if self.state == NodeState.OVERLOADED:
                self.state = NodeState.ACTIVE
    
    def get_status(self) -> Dict[str, Any]:
        """获取节点状态"""
        return {
            'node_id': self.address.node_id,
            'state': self.state.value,
            'dna': self.dna.to_dict(),
            'load': {
                'cpu': self.load.cpu_usage,
                'memory': self.load.memory_usage,
                'network': self.load.network_usage,
                'active_connections': self.load.active_connections,
                'overall_load': self.load.get_overall_load()
            },
            'neighbors': [n.node_id for n in self.neighbors],
            'stats': {
                'messages_sent': self.messages_sent,
                'messages_received': self.messages_received,
                'uptime': time.time() - self.uptime
            }
        }


class MeshNode(NetworkNode):
    """网状网络节点 - 具体实现"""
    
    def __init__(self, dna: NodeDNA, address: NetworkAddress, mesh_network: 'MeshNetwork'):
        """
        初始化网状节点
        
        Args:
            dna: 节点DNA
            address: 网络地址
            mesh_network: 所属网状网络
        """
        super().__init__(dna, address)
        self.mesh_network = mesh_network
        self.discovery_interval = 5.0
    
    async def _process_message(self, message: NetworkMessage) -> None:
        """处理消息"""
        try:
            if message.type == MessageType.DISCOVERY:
                await self._handle_discovery(message)
            elif message.type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(message)
            elif message.type == MessageType.TOPOLOGY_UPDATE:
                await self._handle_topology_update(message)
            elif message.type == MessageType.LOAD_BALANCE:
                await self._handle_load_balance(message)
            elif message.type == MessageType.DNA_SYNC:
                await self._handle_dna_sync(message)
            elif message.type == MessageType.FAILURE_DETECTED:
                await self._handle_failure_detected(message)
            else:
                logger.warning(f"Unknown message type: {message.type}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _send_heartbeat(self) -> None:
        """发送心跳"""
        for neighbor in self.neighbors:
            message = NetworkMessage(
                type=MessageType.HEARTBEAT,
                source=self.address,
                destination=neighbor,
                payload={'status': self.get_status()}
            )
            await self.mesh_network.route_message(message)
    
    async def _self_organize(self) -> None:
        """自组织逻辑"""
        # 1. 发现新节点
        await self._discover_neighbors()
        
        # 2. 负载均衡
        if self.load.is_overloaded():
            await self._initiate_load_balance()
        
        # 3. 拓扑优化
        await self._optimize_topology()
        
        # 4. DNA演化
        await self._evolve_dna()
    
    async def _discover_neighbors(self) -> None:
        """发现邻居节点"""
        # 发送发现消息
        discovery_msg = NetworkMessage(
            type=MessageType.DISCOVERY,
            source=self.address,
            destination=NetworkAddress(node_id="broadcast", host="", port=0),
            payload={'dna': self.dna.to_dict()}
        )
        await self.mesh_network.route_message(discovery_msg)
    
    async def _handle_discovery(self, message: NetworkMessage) -> None:
        """处理发现消息"""
        # 添加到邻居列表
        if message.source != self.address:
            self.neighbors.add(message.source)
            
            # 响应发现消息
            response = NetworkMessage(
                type=MessageType.DISCOVERY,
                source=self.address,
                destination=message.source,
                payload={'dna': self.dna.to_dict(), 'response': True}
            )
            await self.mesh_network.route_message(response)
    
    async def _handle_heartbeat(self, message: NetworkMessage) -> None:
        """处理心跳消息"""
        # 更新邻居状态
        if message.source in self.neighbors:
            # 记录心跳时间
            pass
    
    async def _handle_topology_update(self, message: NetworkMessage) -> None:
        """处理拓扑更新"""
        # 更新邻居列表
        new_neighbors = message.payload.get('neighbors', [])
        for neighbor_addr in new_neighbors:
            self.neighbors.add(neighbor_addr)
    
    async def _handle_load_balance(self, message: NetworkMessage) -> None:
        """处理负载均衡请求"""
        # 检查是否可以接受任务
        if not self.load.is_overloaded():
            # 可以接受任务
            response = NetworkMessage(
                type=MessageType.LOAD_BALANCE,
                source=self.address,
                destination=message.source,
                payload={'can_accept': True, 'load': self.load.get_overall_load()}
            )
            await self.mesh_network.route_message(response)
    
    async def _handle_dna_sync(self, message: NetworkMessage) -> None:
        """处理DNA同步"""
        # 学习邻居DNA
        neighbor_dna_data = message.payload.get('dna')
        if neighbor_dna_data:
            # 可以选择交叉或学习
            pass
    
    async def _handle_failure_detected(self, message: NetworkMessage) -> None:
        """处理故障检测"""
        failed_node = message.payload.get('failed_node')
        if failed_node and failed_node in self.neighbors:
            self.neighbors.remove(failed_node)
            logger.warning(f"Neighbor {failed_node} failed, removing from neighbors")
    
    async def _initiate_load_balance(self) -> None:
        """发起负载均衡"""
        # 向邻居发送负载均衡请求
        for neighbor in self.neighbors:
            message = NetworkMessage(
                type=MessageType.LOAD_BALANCE,
                source=self.address,
                destination=neighbor,
                payload={'request_balance': True}
            )
            await self.mesh_network.route_message(message)
    
    async def _optimize_topology(self) -> None:
        """优化拓扑结构"""
        # 基于DNA特性决定是否建立新连接
        target_neighbors = int(self.dna.social_skill * 10)  # 最多10个邻居
        
        if len(self.neighbors) < target_neighbors and random.random() < self.dna.innovation:
            # 随机连接新节点
            pass
        
        elif len(self.neighbors) > target_neighbors:
            # 断开最少使用的连接
            pass
    
    async def _evolve_dna(self) -> None:
        """演化DNA"""
        # 基于性能更新fitness
        performance_score = 1.0 - self.load.get_overall_load()
        
        # 指数移动平均
        alpha = self.dna.learning_rate
        self.dna.fitness = alpha * performance_score + (1 - alpha) * self.dna.fitness
        
        # 定期突变
        if random.random() < self.dna.mutation_rate:
            self.dna = self.dna.mutate()
            logger.info(f"Node {self.address.node_id} DNA mutated to generation {self.dna.generation}")


# ==================== 网状网络 ====================

class MeshNetwork:
    """自组织网状网络"""
    
    def __init__(self, name: str = "MeshNetwork"):
        """
        初始化网状网络
        
        Args:
            name: 网络名称
        """
        self.name = name
        self.nodes: Dict[str, MeshNode] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._routing_table: Dict[str, List[NetworkAddress]] = {}
        
        # 统计
        self.total_messages = 0
        self.failed_messages = 0
    
    async def start(self) -> None:
        """启动网络"""
        logger.info(f"MeshNetwork {self.name} starting...")
        self._running = True
        
        # 启动路由任务
        asyncio.create_task(self._routing_loop())
        
        # 启动所有节点
        for node in self.nodes.values():
            await node.start()
        
        logger.info(f"MeshNetwork {self.name} started with {len(self.nodes)} nodes")
    
    async def stop(self) -> None:
        """停止网络"""
        logger.info(f"MeshNetwork {self.name} stopping...")
        self._running = False
        
        # 停止所有节点
        for node in self.nodes.values():
            await node.stop()
        
        logger.info(f"MeshNetwork {self.name} stopped")
    
    def add_node(self, node: MeshNode) -> None:
        """添加节点"""
        self.nodes[node.address.node_id] = node
        logger.info(f"Added node {node.address.node_id} to network")
    
    def remove_node(self, node_id: str) -> None:
        """移除节点"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            asyncio.create_task(node.stop())
            del self.nodes[node_id]
            
            # 通知其他节点
            asyncio.create_task(self._broadcast_failure(node_id))
            
            logger.info(f"Removed node {node_id} from network")
    
    async def _routing_loop(self) -> None:
        """路由循环"""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                await self._route_message_internal(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Routing error: {e}")
    
    async def route_message(self, message: NetworkMessage) -> bool:
        """路由消息"""
        try:
            self.total_messages += 1
            await self.message_queue.put(message)
            return True
        except Exception as e:
            self.failed_messages += 1
            logger.error(f"Failed to route message: {e}")
            return False
    
    async def _route_message_internal(self, message: NetworkMessage) -> None:
        """内部路由逻辑"""
        # 广播消息
        if message.destination.node_id == "broadcast":
            for node in self.nodes.values():
                if node.address != message.source:
                    await node.send_message(message)
        # 单播消息
        elif message.destination.node_id in self.nodes:
            target_node = self.nodes[message.destination.node_id]
            await target_node.send_message(message)
        # 多跳路由（简化实现）
        else:
            logger.warning(f"Unknown destination: {message.destination.node_id}")
    
    async def _broadcast_failure(self, failed_node_id: str) -> None:
        """广播节点故障"""
        for node in self.nodes.values():
            message = NetworkMessage(
                type=MessageType.FAILURE_DETECTED,
                source=NetworkAddress(node_id="network", host="", port=0),
                destination=node.address,
                payload={'failed_node': failed_node_id}
            )
            await node.send_message(message)
    
    def get_network_status(self) -> Dict[str, Any]:
        """获取网络状态"""
        nodes_status = {}
        total_load = 0.0
        total_connections = 0
        
        for node_id, node in self.nodes.items():
            nodes_status[node_id] = node.get_status()
            total_load += node.load.get_overall_load()
            total_connections += len(node.neighbors)
        
        avg_load = total_load / len(self.nodes) if self.nodes else 0.0
        avg_connections = total_connections / len(self.nodes) if self.nodes else 0.0
        
        return {
            'name': self.name,
            'node_count': len(self.nodes),
            'total_messages': self.total_messages,
            'failed_messages': self.failed_messages,
            'average_load': avg_load,
            'average_connections': avg_connections,
            'nodes': nodes_status
        }
    
    def visualize_network(self) -> str:
        """可视化网络拓扑"""
        lines = []
        lines.append(f"Network: {self.name}")
        lines.append(f"Nodes: {len(self.nodes)}")
        lines.append("Topology:")
        
        for node_id, node in self.nodes.items():
            neighbors = [n.node_id for n in node.neighbors]
            lines.append(f"  {node_id} -> {neighbors}")
        
        return "\n".join(lines)


# ==================== 演示和测试 ====================

async def demo_autopoietic_network():
    """演示自组织网络"""
    print("=" * 60)
    print("Autopoietic Mesh Network - Self-Organization Demo")
    print("=" * 60)
    
    # 创建网络
    network = MeshNetwork("DemoNetwork")
    
    # 创建节点（使用不同的DNA）
    nodes = []
    for i in range(5):
        dna = NodeDNA(
            node_id=f"node_{i}",
            generation=1,
            fitness=0.5,
            processing_power=random.random(),
            storage_capacity=random.random(),
            network_bandwidth=random.random(),
            reliability=random.random(),
            social_skill=random.random(),
            innovation=random.random()
        )
        
        address = NetworkAddress(
            node_id=f"node_{i}",
            host="localhost",
            port=8000 + i
        )
        
        node = MeshNode(dna, address, network)
        network.add_node(node)
        nodes.append(node)
    
    # 启动网络
    await network.start()
    
    # 运行一段时间
    print("\nNetwork running for 30 seconds...")
    print("-" * 60)
    
    for i in range(6):
        await asyncio.sleep(5)
        status = network.get_network_status()
        print(f"\n--- Time: {i * 5}s ---")
        print(f"Active nodes: {status['node_count']}")
        print(f"Total messages: {status['total_messages']}")
        print(f"Average load: {status['average_load']:.2f}")
        print(f"Average connections: {status['average_connections']:.2f}")
        
        # 显示节点状态
        for node_id, node_status in status['nodes'].items():
            load_info = node_status['load']
            print(f"  {node_id}: load={load_info['overall_load']:.2f}, "
                  f"neighbors={len(node_status['neighbors'])}, "
                  f"fitness={node_status['dna']['fitness']:.2f}")
    
    # 显示网络拓扑
    print("\n" + network.visualize_network())
    
    # 测试节点故障恢复
    print("\n--- Testing Failure Recovery ---")
    print("Removing node_0...")
    network.remove_node("node_0")
    await asyncio.sleep(10)
    
    status = network.get_network_status()
    print(f"Active nodes after removal: {status['node_count']}")
    print("\nUpdated topology:")
    print(network.visualize_network())
    
    # 停止网络
    await network.stop()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    """主程序入口"""
    print("Starting Autopoietic Mesh Network Demo...")
    asyncio.run(demo_autopoietic_network())
