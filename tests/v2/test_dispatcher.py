import pytest
import asyncio
from hive.core.dispatcher import HiveDispatcher, TaskGraph, TaskNode, TaskStatus
from hive.core.component import HiveActor, HiveComponent

class MockComponent(HiveComponent):
    def name(self): return "Mocker"
    def capabilities(self): return ["mock_action"]
    async def action_mock_action(self, **kwargs):
        return f"Mocked: {kwargs.get('data')}"

class MockActor(HiveActor):
    def __init__(self, name):
        super().__init__(name, "tester")
        self.add_component(MockComponent())

@pytest.mark.asyncio
async def test_scheduler_dependency_resolution():
    graph = TaskGraph("TestDAG")
    
    node_a = TaskNode(id="A", action="mock_action", params={"data": "A"}, required_capabilities=["mock_action"])
    node_b = TaskNode(id="B", action="mock_action", params={"data": "B"}, required_capabilities=["mock_action"])
    node_c = TaskNode(id="C", action="mock_action", params={"data": "C"}, required_capabilities=["mock_action"])
    
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)
    
    graph.add_dependency("B", "A")
    graph.add_dependency("C", "B")
    
    ready = graph.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "A"
    
    node_a.status = TaskStatus.COMPLETED
    ready = graph.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "B"

@pytest.mark.asyncio
async def test_dispatcher_execution():
    actor = MockActor("Actor1")
    dispatcher = HiveDispatcher([actor])
    
    graph = TaskGraph("ExecDAG")
    node = TaskNode(id="1", action="mock_action", params={"data": "hello"}, required_capabilities=["mock_action"])
    graph.add_node(node)
    
    summary = await dispatcher.run_workflow(graph)
    
    assert summary.get("1") == "completed" or node.status == TaskStatus.COMPLETED

def test_task_graph_initialization():
    graph = TaskGraph("TestGraph")
    assert graph.name == "TestGraph"
    assert len(graph.nodes) == 0

def test_task_status_enum():
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.FAILED.value == "failed"
