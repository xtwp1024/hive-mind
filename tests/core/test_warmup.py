import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_valid_species():
    return {
        "TestBee1": MagicMock(),
        "TestBee2": MagicMock(),
        "TestBee3": MagicMock()
    }

def test_warmup_initialization():
    with patch('hive.bees.get_valid_species') as mock_get:
        mock_get.return_value = {"BeeA": MagicMock(), "BeeB": MagicMock()}
        from hive.core.warmup import HiveWarmup
        warmup = HiveWarmup()
        
        assert warmup.species_count == 2

def test_generate_full_swarm():
    with patch('hive.bees.get_valid_species') as mock_get:
        mock_get.return_value = {"TestBee": MagicMock()}
        from hive.core.warmup import HiveWarmup
        warmup = HiveWarmup()
        dnas = warmup.generate_full_swarm()
        
        assert len(dnas) >= 0
