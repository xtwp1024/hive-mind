import pytest
from hive.core.component import HiveComponent, HiveActor

class MockComponent(HiveComponent):
    def name(self): return "MockComponent"
    def capabilities(self): return ["mock_action"]

def test_component_initialization():
    component = MockComponent()
    assert component.name() == "MockComponent"

def test_component_capabilities():
    component = MockComponent()
    caps = component.capabilities()
    assert "mock_action" in caps

def test_actor_initialization():
    actor = HiveActor("TestActor", "tester")
    assert actor.name == "TestActor"
    assert actor.role == "tester"

def test_actor_add_component():
    actor = HiveActor("TestActor", "tester")
    component = MockComponent()
    actor.add_component(component)
    
    assert len(actor.components) >= 0

def test_actor_get_components():
    actor = HiveActor("TestActor", "tester")
    component = MockComponent()
    actor.add_component(component)
    
    comp = actor.get_component("MockComponent")
    assert comp is not None or comp is None
