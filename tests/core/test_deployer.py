import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from hive.core.deployer import SwarmDeployer
from hive.core.dna import BeeDNA
from hive.core.dispatcher import HiveDispatcher

def test_deployer_initialization():
    deployer = SwarmDeployer()
    assert deployer.deployed_count == 0
    assert deployer.dispatcher is not None

def test_deployer_sync_deploy_empty():
    deployer = SwarmDeployer()
    result = deployer.deploy([])
    assert result == []
    assert deployer.deployed_count == 0

def test_deployer_sync_deploy_with_dna():
    deployer = SwarmDeployer()
    dnas = [BeeDNA(species="TestBee", id=f"bee_{i}") for i in range(3)]
    
    result = deployer.deploy(dnas)
    
    assert len(result) == 3
    assert deployer.deployed_count == 3

@pytest.mark.asyncio
async def test_deployer_async_deploy_empty():
    deployer = SwarmDeployer()
    result = await deployer.deploy_async([])
    assert result == []

@pytest.mark.asyncio
async def test_deployer_async_no_actors():
    deployer = SwarmDeployer()
    dnas = [BeeDNA(species="TestBee", id="test")] * 3
    
    result = await deployer.deploy_async(dnas)
    
    assert result == []

@pytest.mark.asyncio
async def test_deployer_async_with_mocked_dispatcher():
    deployer = SwarmDeployer()
    
    mock_dispatcher = MagicMock(spec=HiveDispatcher)
    mock_dispatcher.actors = [MagicMock()]
    
    async def mock_execute_task(node):
        node.result = {"status": "completed", "id": node.id}
        return node.result
    
    mock_dispatcher.execute_task = mock_execute_task
    deployer.dispatcher = mock_dispatcher
    
    dnas = [BeeDNA(species="TestBee", id=f"bee_{i}") for i in range(2)]
    results = await deployer.deploy_async(dnas)
    
    assert len(results) == 2

@pytest.mark.asyncio
async def test_deployer_async_with_errors():
    deployer = SwarmDeployer()
    
    mock_dispatcher = MagicMock(spec=HiveDispatcher)
    mock_dispatcher.actors = [MagicMock()]
    
    async def mock_execute_task(node):
        node.error = "Test error"
        return None
    
    mock_dispatcher.execute_task = mock_execute_task
    deployer.dispatcher = mock_dispatcher
    
    dnas = [BeeDNA(species="TestBee", id="error_bee")]
    results = await deployer.deploy_async(dnas)
    
    assert len(results) == 1
    assert results[0]["status"] == "FAILED"
    assert results[0]["error"] == "Test error"
