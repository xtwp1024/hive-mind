import pytest

def test_settings_exists():
    from hive.config import settings
    assert settings is not None
