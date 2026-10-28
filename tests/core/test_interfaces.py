import pytest
from hive.core.interfaces import Task, TaskResult

def test_task_creation():
    task = Task(id="task_001", type="test")
    assert task.id == "task_001"
    assert task.type == "test"

def test_task_result_creation():
    result = TaskResult(task_id="task_001", success=True)
    assert result.task_id == "task_001"
    assert result.success == True

def test_task_result_with_error():
    result = TaskResult(task_id="task_001", success=False, error="Error occurred")
    assert result.success == False
    assert result.error == "Error occurred"

def test_task_default_values():
    task = Task(id="default", type="default")
    assert task.priority == 0
    assert task.timeout == 300.0

def test_task_result_default():
    result = TaskResult(task_id="test", success=True)
    assert result.result is None
    assert result.error is None
