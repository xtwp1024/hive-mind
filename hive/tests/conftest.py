# -*- coding: utf-8 -*-
"""测试用工具函数和 fixtures"""

import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from hive.config.settings import settings
from hive.core.dna import DNA, BeeDNA


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录

    Yields:
        临时目录路径
    """
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture
def sample_dna() -> BeeDNA:
    """创建示例 BeeDNA

    Returns:
        示例 BeeDNA 实例
    """
    return BeeDNA(
        species="miner",
        id="test_bee_001",
        role="worker",
        target_name="test_target",
        target_url="https://github.com/example/repo",
    )


@pytest.fixture
def sample_dna_batch() -> list[BeeDNA]:
    """创建一批示例 BeeDNA

    Returns:
        BeeDNA 列表
    """
    species_list = ["miner", "architect", "librarian", "sentinel"]
    return [
        BeeDNA(
            species=species,
            id=f"test_bee_{i:03d}",
            role="worker",
            target_name=f"target_{i}",
        )
        for i, species in enumerate(species_list * 3)
    ]


@pytest.fixture
def mock_settings():
    """模拟配置对象"""
    mock = Mock(spec=settings)
    mock.ENVIRONMENT = "testing"
    mock.LOG_LEVEL = "DEBUG"
    mock.MAX_WORKERS = 2
    mock.MUTATION_RATE = 0.1
    mock.ENABLE_SANDBOX = False
    mock.AUTO_MERGE_BACK = False
    mock.BASE_DIR = Path(tempfile.mkdtemp())
    mock.SANDBOX_DIR = mock.BASE_DIR / "sandbox"
    mock.DATA_DIR = mock.BASE_DIR / "Data"
    mock.LOG_DIR = mock.BASE_DIR / "logs"
    return mock


@pytest.fixture
def temp_project_dir(temp_dir: Path) -> Path:
    """创建包含示例项目的临时目录

    Args:
        temp_dir: 临时目录

    Returns:
        项目目录路径
    """
    project_dir = temp_dir / "test_project"
    project_dir.mkdir()

    # 创建示例文件
    (project_dir / "README.md").write_text("# Test Project\n", encoding="utf-8")
    (project_dir / "main.py").write_text("print('Hello, World!')\n", encoding="utf-8")
    (project_dir / "requirements.txt").write_text("requests>=2.28.0\n", encoding="utf-8")

    # 创建子目录
    src_dir = project_dir / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    (src_dir / "module.py").write_text("def hello():\n    return 'Hi'\n", encoding="utf-8")

    return project_dir


@pytest.fixture
def mock_queue():
    """模拟 multiprocessing.Queue"""
    from unittest.mock import MagicMock

    queue = MagicMock()
    queue.put = MagicMock()
    return queue


class MockProcess:
    """模拟多进程"""

    def __init__(self, target=None, args=()):
        self.target = target
        self.args = args
        self._alive = True
        self._exitcode = None

    def start(self):
        """启动进程"""
        if self.target:
            self.target(*self.args)

    def is_alive(self):
        """检查进程是否存活"""
        return self._alive

    def join(self, timeout=None):
        """等待进程结束"""
        self._alive = False
        self._exitcode = 0

    def terminate(self):
        """终止进程"""
        self._alive = False
        self._exitcode = -1

    @property
    def exitcode(self):
        """退出代码"""
        return self._exitcode


@pytest.fixture
def mock_process():
    """模拟进程 fixture"""
    return MockProcess


def create_test_file(path: Path, content: str = "test content") -> None:
    """创建测试文件

    Args:
        path: 文件路径
        content: 文件内容
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_test_repo(path: Path, files: dict[str, str] = None) -> None:
    """创建测试仓库结构

    Args:
        path: 仓库根目录
        files: 文件映射 {路径: 内容}
    """
    path.mkdir(parents=True, exist_ok=True)

    default_files = {
        "README.md": "# Test Repository",
        "LICENSE": "MIT",
        ".gitignore": "*.pyc\n__pycache__/",
    }

    if files:
        default_files.update(files)

    for file_path, content in default_files.items():
        file_path_obj = path / file_path
        create_test_file(file_path_obj, content)
