import pytest

def test_logger_exists():
    from hive.utils.logger import get_logger
    logger = get_logger("test")
    assert logger is not None
