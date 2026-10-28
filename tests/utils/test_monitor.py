import pytest

def test_monitor_exists():
    from hive.utils import monitor
    assert monitor is not None
