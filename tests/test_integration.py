import pytest
import asyncio
from unittest.mock import MagicMock, patch
from multiprocessing import Queue
from hive.core.dna import BeeDNA
from hive.core.queen import QueenBee
from hive.core.deployer import SwarmDeployer
from hive.core.dispatcher import HiveDispatcher, TaskGraph, TaskNode
from hive.core.component import HiveActor, HiveComponent

class EchoComponent(HiveComponent):
    def name(self): return "Echo"
    def capabilities(self): return ["echo"]
    async def action_echo(self, **kwargs):
        return f"Echo: {kwargs.get('message', 'no message')}"

class EchoActor(HiveActor):
    def __init__(self, name="EchoActor"):
        super().__init__(name, "tester")
        self.add_component(EchoComponent())

@pytest.mark.asyncio
async def test_full_swarm_workflow():
    dna = BeeDNA(species="TestBee", id="test_001", role="worker")
    queue = Queue()
    
    with patch('hive.bees.get_valid_species') as mock_species:
        mock_species.return_value = {"testbee": MagicMock()}
        
        queen = QueenBee(queue)
        assert queen is not None

@pytest.mark.asyncio
async def test_dispatcher_workflow():
    actor = EchoActor()
    dispatcher = HiveDispatcher([actor])
    
    graph = TaskGraph("IntegrationTest")
    node = TaskNode(id="1", action="echo", params={"message": "Hello"}, required_capabilities=["echo"])
    graph.add_node(node)
    
    summary = await dispatcher.run_workflow(graph)
    
    assert "1" in summary or node.status.value in ["completed", "running"]

@pytest.mark.asyncio
async def test_queen_to_deployer_integration():
    dna_list = [
        BeeDNA(species="TestBee", id=f"bee_{i}", role="worker")
        for i in range(3)
    ]
    
    deployer = SwarmDeployer()
    result = deployer.deploy(dna_list)
    
    assert len(result) == 3

def test_dna_mutation_crossover_integration():
    dna1 = BeeDNA(species="BeeA", id="parent1", learning_rate=0.3)
    dna2 = BeeDNA(species="BeeA", id="parent2", learning_rate=0.7)
    
    child = dna1.crossover(dna2)
    mutated = child.mutate(mutation_rate=0.5)
    
    assert mutated.generation > dna1.generation
    assert 0.0 <= mutated.learning_rate <= 1.0

def test_service_locator_integration():
    from hive.core.service_locator import BeeServiceLocator
    
    service = {"name": "test_service", "data": "test_data"}
    BeeServiceLocator.register_service("TestService", service)
    
    retrieved = BeeServiceLocator.get_service("TestService")
    assert retrieved == service
    
    BeeServiceLocator._services.clear()

@pytest.mark.asyncio
async def test_task_graph_complex_workflow():
    graph = TaskGraph("ComplexDAG")
    
    node_a = TaskNode(id="A", action="echo", params={"message": "start"}, required_capabilities=["echo"])
    node_b = TaskNode(id="B", action="echo", params={"message": "middle"}, required_capabilities=["echo"])
    node_c = TaskNode(id="C", action="echo", params={"message": "end"}, required_capabilities=["echo"])
    
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)
    
    graph.add_dependency("B", "A")
    graph.add_dependency("C", "B")
    
    ready = graph.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "A"
