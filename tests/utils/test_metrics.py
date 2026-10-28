import pytest

def test_metrics_collector_exists():
    from hive.utils import metrics
    assert metrics is not None
