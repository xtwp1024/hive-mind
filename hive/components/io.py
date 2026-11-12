import glob
import os
from pathlib import Path
from typing import Any, Dict, List

from hive.core.component import ComponentConfig, HiveComponent


class IOConfig(ComponentConfig):
    root_path: str = "d:/antigravity_scratch"


class IOComponent(HiveComponent):
    """
    Provides File System capabilities.
    Replaces: LibrarianBee
    """

    def __init__(self, config: IOConfig = None):
        super().__init__(config or IOConfig())

    def name(self) -> str:
        return "io_module"

    def capabilities(self) -> List[str]:
        return ["read_file", "write_file", "list_files", "scan_directory"]

    def _validate_path(self, path: str) -> bool:
        """
        Validate path to prevent directory traversal attacks.

        Args:
            path: Path to validate

        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Check for null bytes first
            if "\0" in path:
                return False

            # Resolve the path to its absolute form
            p = Path(path).resolve()
            # Get the root path resolved (ensure it's absolute to prevent bypass)
            root = Path(self.config.root_path).resolve()

            # Ensure the resolved path is within root_path (prevent .. traversal)
            # Use os.path.sep to ensure cross-platform correctness
            root_str = str(root)
            p_str = str(p)
            if not p_str.startswith(
                root_str
                + (
                    ""
                    if root_str.endswith(
                        (
                            "\\",
                            "/",
                        )
                    )
                    else (Path(path).separator)
                )
            ):
                self.logger.warning(f"Path traversal attempt detected: {path} not in {root}")
                return False

            return True
        except Exception as e:
            self.logger.warning(f"Path validation error for {path}: {e}")
            return False

    async def action_read_file(self, path: str) -> str:
        # Security: Validate path before access
        if not self._validate_path(path):
            self.logger.error("Read denied: path traversal attempt", path=path)
            return ""

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            self.logger.error("Read failed", path=path, error=str(e))
            return ""

    async def action_write_file(self, path: str, content: str) -> bool:
        # Security: Validate path before write
        if not self._validate_path(path):
            self.logger.error("Write denied: path traversal attempt", path=path)
            return False

        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            self.logger.error("Write failed", path=path, error=str(e))
            return False

    async def action_list_files(self, directory: str, pattern: str = "*") -> List[str]:
        # Security: Validate directory before listing
        if not self._validate_path(directory):
            self.logger.error("List denied: path traversal attempt", directory=directory)
            return []

        search_path = os.path.join(directory, pattern)
        return glob.glob(search_path)

    async def action_scan_directory(self, directory: str) -> Dict[str, Any]:
        """Deep scan for structural analysis"""
        stats = {"files": 0, "size": 0, "extensions": {}}
        for root, _, files in os.walk(directory):
            for file in files:
                stats["files"] += 1
                try:
                    fp = os.path.join(root, file)
                    sz = os.path.getsize(fp)
                    stats["size"] += sz
                    ext = os.path.splitext(file)[1]
                    stats["extensions"][ext] = stats["extensions"].get(ext, 0) + 1
                except (OSError, PermissionError):
                    pass
        return stats
