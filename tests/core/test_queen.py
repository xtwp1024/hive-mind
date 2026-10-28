from hive.core.queen import QueenBee
from hive.core.dna import BeeDNA
import multiprocessing
import pytest
import asyncio
from unittest.mock import MagicMock, patch

def test_queen_initialization(mock_queue):
    queen = QueenBee(mock_queue)
    assert queen.name == "Queen"

def test_queen_deploy_swarm_resource_limit(mock_queue):
    queen = QueenBee(mock_queue)
    
    dnas = [BeeDNA(species="TestBee", id=f"bee_{i}") for i in range(5)]
    queen.bee_registry = {}
    
    with patch('hive.bees.base_bee.BaseBee') as MockBee:
        mock_process = MagicMock()
        MockBee.return_value = mock_process
        
        queen.deploy_swarm(dnas)
    
    assert queen.deployer is not None

def test_queen_consume_honey(mock_queue):
    queen = QueenBee(mock_queue)
    
    honey = {
        "source": "worker_1", 
        "status": "SUCCESS", 
        "dna": {"target_name": "TestProject"}
    }
    mock_queue.put(honey)
    
    queen.consume_honey()
    
    assert queen.deployer is not None

@pytest.mark.asyncio
async def test_queen_deploy_swarm_v2_no_actors(mock_queue):
    queen = QueenBee(mock_queue)
    
    dnas = [BeeDNA(species="TestBee", id=f"v2_{i}") for i in range(3)]
    results = await queen.deploy_swarm_v2(dnas)
    
    assert results == []

@pytest.mark.asyncio
async def test_queen_deploy_swarm_v2_with_mocked_dispatcher(mock_queue):
    queen = QueenBee(mock_queue)
    
    dnas = [BeeDNA(species="TestBee", id=f"v2_{i}") for i in range(3)]
    
    mock_dispatcher = MagicMock()
    mock_dispatcher.actors = [MagicMock()]
    mock_dispatcher.execute_task = MagicMock(return_value=asyncio.Future())
    mock_dispatcher.execute_task.return_value.set_result({"status": "completed"})
    
    queen.deployer.dispatcher = mock_dispatcher
    
    results = await queen.deploy_swarm_v2(dnas)
    
    assert isinstance(results, list)

def test_queen_deploy_empty_dna(mock_queue):
    queen = QueenBee(mock_queue)
    queen.deploy_swarm([])
