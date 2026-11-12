import ast
import re
from typing import Any, Dict, List

from hive.core.component import HiveComponent
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class AnalyzerComponent(HiveComponent):
    """
    Provides Code Analysis capabilities.
    Replaces: MinerBee
    """

    def __init__(self, config=None):
        super().__init__(config)

    def name(self) -> str:
        return "code_analyzer"

    def capabilities(self) -> List[str]:
        return ["analyze_code", "extract_imports", "count_loc"]

    async def action_analyze_code(self, content: str, language: str) -> Dict[str, Any]:
        """Deep analysis of code content"""
        result = {
            "language": language,
            "loc": len(content.splitlines()),
            "complexity": 0,
            "imports": [],
        }

        if language == "python":
            try:
                tree = ast.parse(content)
                result["imports"] = [
                    node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)
                ]
                result["imports"] += [
                    node.module
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module
                ]

                # Rough complexity: count function defs and classes
                funcs = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
                result["complexity"] = funcs + classes
            except SyntaxError as e:
                logger.warning(f"Python syntax error in code analysis: {e}")

        elif language in ["javascript", "typescript", "ts", "js"]:
            # Regex fallback for JS
            imports = re.findall(r'import\s+.*\s+from\s+[\'"](.*)[\'"]', content)
            imports += re.findall(r'require\([\'"](.*)[\'"]\)', content)
            result["imports"] = imports

        return result

    async def action_count_loc(self, content: str) -> int:
        return len(
            [
                line
                for line in content.splitlines()
                if line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith("//")
            ]
        )
