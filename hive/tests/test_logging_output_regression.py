"""Exercise real structlog output in isolated subprocesses (no services)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("with_file", [False, True], ids=["console-only", "console-and-file"])
def test_logging_renders_structured_events_once(tmp_path, with_file):
    log_file = tmp_path / "hive.log"
    env = os.environ.copy()
    env.update(
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        HIVE_HOME=str(tmp_path / "home"),
        BASE_DIR=str(tmp_path / "inbound"),
        SHARED_BRAIN_ENABLED="false",
    )
    script = """
import sys
from pathlib import Path
from hive.utils.logger import get_logger, setup_logging

setup_logging(
    log_file=Path(sys.argv[1]) if sys.argv[2] == "1" else None,
    json_output=True,
    enable_colors=False,
)
logger = get_logger("logging-regression")
logger.info("regression-info", request_id="offline")
logger.warning("regression-warning", request_id="offline")
logger.error("regression-error", request_id="offline")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(log_file), str(int(with_file))],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "Logging error" not in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr

    def events(text):
        records = []
        for line in text.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event", "").startswith("regression-"):
                records.append(record)
        assert [record["event"] for record in records] == [
            "regression-info",
            "regression-warning",
            "regression-error",
        ]
        assert [record["level"] for record in records] == ["info", "warning", "error"]
        for record in records:
            assert record["logger"] == "logging-regression"
            assert record["request_id"] == "offline"
            assert record["timestamp"]
        return records

    console_events = events(result.stdout)
    if with_file:
        assert events(log_file.read_text(encoding="utf-8")) == console_events
    else:
        assert not log_file.exists()
