import pytest
from hive.core.service_locator import BeeServiceLocator

@pytest.fixture(autouse=True)
def reset_locator():
    BeeServiceLocator._services.clear()
    BeeServiceLocator._factories.clear()
    yield

def test_register_service():
    mock_service = {"name": "TestService"}
    BeeServiceLocator.register_service("TestProtocol", mock_service)
    
    result = BeeServiceLocator.get_service("TestProtocol")
    assert result == mock_service

def test_register_service_overwrite():
    service1 = {"id": 1}
    service2 = {"id": 2}
    
    BeeServiceLocator.register_service("Protocol", service1)
    BeeServiceLocator.register_service("Protocol", service2)
    
    result = BeeServiceLocator.get_service("Protocol")
    assert result == service2

def test_get_service_not_found():
    result = BeeServiceLocator.get_service("NonExistent")
    assert result is None

def test_register_factory():
    def factory_func():
        return {"factory": True}
    
    BeeServiceLocator.register_factory("FactoryProtocol", factory_func)
    
    result = BeeServiceLocator.get_service("FactoryProtocol")
    assert result == {"factory": True}

def test_register_factory_cached():
    def factory_func():
        return {"counter": 0}
    
    BeeServiceLocator.register_factory("CachedFactory", factory_func)
    
    result1 = BeeServiceLocator.get_service("CachedFactory")
    result2 = BeeServiceLocator.get_service("CachedFactory")
    
    assert result1 is result2

def test_get_service_priority_over_factory():
    existing_service = {"existing": True}
    BeeServiceLocator.register_service("PriorityTest", existing_service)
    
    def factory_func():
        return {"factory": True}
    BeeServiceLocator.register_factory("PriorityTest", factory_func)
    
    result = BeeServiceLocator.get_service("PriorityTest")
    assert result == existing_service
