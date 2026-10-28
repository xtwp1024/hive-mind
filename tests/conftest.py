import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock
from multiprocessing import Queue

sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def mock_queue():
    return Queue()

@pytest.fixture
def mock_dna():
    from hive.core.dna import BeeDNA
    return BeeDNA(
        species="TestBee",
        role="worker",
        id="test_dna_001",
        target_name="TestProject",
        target_url="https://github.com/test/repo",
        learning_rate=0.5
    )

@pytest.fixture
def mock_dna_list():
    from hive.core.dna import BeeDNA
    return [
        BeeDNA(species="TestBee", role="worker", id=f"test_{i}", target_name=f"Project_{i}")
        for i in range(5)
    ]

@pytest.fixture
def mock_bee_registry():
    from hive.bees.base_bee import BaseBee
    return {
        "testbee": BaseBee,
        "miner": BaseBee,
        "architect": BaseBee
    }

@pytest.fixture
def mock_vector_vault():
    vault = MagicMock()
    vault.add_document = MagicMock(return_value=True)
    vault.search = MagicMock(return_value=[])
    vault.get_stats = MagicMock(return_value={"documents": 0, "collections": 0})
    return vault
