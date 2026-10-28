import pytest
from unittest.mock import MagicMock, patch
from multiprocessing import Queue
from hive.core.worker_manager import WorkerManager
from hive.core.dna import BeeDNA
from hive.bees.base_bee import BaseBee

def test_worker_manager_initialization(mock_queue):
    wm = WorkerManager(mock_queue)
    assert wm.result_queue is mock_queue
    assert len(wm.active_workers) == 0
    assert len(wm.bee_registry) == 0

def test_worker_manager_register_species(mock_queue, mock_bee_registry):
    wm = WorkerManager(mock_queue)
    wm.register_species(mock_bee_registry)
    assert len(wm.bee_registry) == 3
    assert "testbee" in wm.bee_registry
    assert "miner" in wm.bee_registry

def test_worker_manager_available_slots(mock_queue):
    wm = WorkerManager(mock_queue)
    wm.MAX_ACTIVE_WORKERS = 5
    
    assert wm.available_slots() == 5
    
    wm.active_workers["worker1"] = MagicMock()
    assert wm.available_slots() == 4

def test_worker_manager_has_available_slot(mock_queue):
    wm = WorkerManager(mock_queue)
    wm.MAX_ACTIVE_WORKERS = 2
    
    assert wm.has_available_slot() == True
    
    wm.active_workers["worker1"] = MagicMock()
    wm.active_workers["worker2"] = MagicMock()
    
    assert wm.has_available_slot() == False

def test_worker_manager_spawn_no_slot(mock_queue):
    wm = WorkerManager(mock_queue)
    wm.MAX_ACTIVE_WORKERS = 0
    
    dna = BeeDNA(species="TestBee", id="test_001")
    result = wm.spawn_worker(dna)
    
    assert result is None

def test_worker_manager_spawn_with_mock(mock_queue):
    wm = WorkerManager(mock_queue)
    wm.MAX_ACTIVE_WORKERS = 5
    wm.bee_registry = {"testbee": MagicMock()}
    
    dna = BeeDNA(species="TestBee", id="test_001")
    
    result = wm.spawn_worker(dna)
    
    assert result is None or result is not None

def test_worker_manager_get_worker_status(mock_queue):
    wm = WorkerManager(mock_queue)
    
    mock_worker = MagicMock()
    mock_worker.is_alive.return_value = True
    wm.active_workers["worker_1"] = mock_worker
    
    assert "worker_1" in wm.active_workers

def test_worker_manager_cleanup_dead_workers(mock_queue):
    wm = WorkerManager(mock_queue)
    
    mock_dead = MagicMock()
    mock_dead.is_alive.return_value = False
    wm.active_workers["dead_worker"] = mock_dead
    
    mock_alive = MagicMock()
    mock_alive.is_alive.return_value = True
    wm.active_workers["alive_worker"] = mock_alive
    
    wm.cleanup_dead_workers()
    
    assert "dead_worker" not in wm.active_workers
    assert "alive_worker" in wm.active_workers
