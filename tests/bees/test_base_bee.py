import pytest
from unittest.mock import MagicMock, patch
from multiprocessing import Queue
from hive.bees.base_bee import BaseBee, validate_git_url
from hive.core.dna import BeeDNA

def test_validate_git_url_valid():
    valid, error = validate_git_url("https://github.com/test/repo")
    assert valid == True
    assert error is None

def test_validate_git_url_invalid_domain():
    valid, error = validate_git_url("https://evil.com/repo")
    assert valid == False
    assert "不允许的 Git 域名" in error

def test_validate_git_url_invalid_chars():
    valid, error = validate_git_url("https://github.com/test;rm -rf")
    assert valid == False
    assert "危险字符" in error

def test_validate_git_url_empty():
    valid, error = validate_git_url("")
    assert valid == False

def test_validate_git_url_ssh():
    valid, error = validate_git_url("git@github.com:test/repo.git")
    assert valid == True

def test_base_bee_initialization():
    dna = BeeDNA(species="TestBee", id="test_001")
    queue = Queue()
    
    with patch('hive.bees.base_bee.get_logger'):
        bee = BaseBee(dna, "bee_001", queue)
        
        assert bee.dna == dna
        assert bee.id == "bee_001"
        assert bee.result_queue == queue

def test_base_bee_daemon():
    dna = BeeDNA(species="TestBee", id="test_001")
    queue = Queue()
    
    with patch('hive.bees.base_bee.get_logger'):
        bee = BaseBee(dna, "bee_001", queue)
        
        assert bee.daemon == True
