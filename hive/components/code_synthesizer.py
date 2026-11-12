from typing import Any, Dict, List, Optional

from hive.core.component import HiveComponent
from hive.utils.brain_interface import get_brain


class CodeSynthesizerComponent(HiveComponent):
    """
    Synthesizes complete code from specification and mined patterns.
    This is the third stage of the Code God pipeline.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.brain = get_brain()

    def name(self) -> str:
        return "code_synthesizer"

    def capabilities(self) -> list[str]:
        return ["synthesize_code", "generate_module"]

    async def action_synthesize_code(
        self, spec: Dict[str, Any], patterns: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate complete code based on specification and patterns.

        Returns:
            {
                "code": "generated code string",
                "filename": "suggested filename",
                "language": "programming language",
                "tests": "generated test code (optional)"
            }
        """
        # Build context from patterns
        pattern_context = "\n\n---\n\n".join(
            [
                f"# Pattern {i+1} (Relevance: {p.get('relevance', 0):.2f})\n{p.get('content', '')[:500]}"
                for i, p in enumerate(patterns[:5])  # Use top 5 patterns
            ]
        )

        system_prompt = """You are an expert software engineer.
Your task is to generate production-quality code based on specifications and reference patterns.
The code must be:
- Complete and runnable
- Well-documented with docstrings
- Include error handling
- Follow best practices for the language
- Include type hints (for Python)

Return ONLY the code, no markdown formatting, no explanations."""

        prompt = f"""Generate code based on this specification:

SPECIFICATION:
Task Type: {spec.get('task_type', 'function')}
Language: {spec.get('language', 'python')}
Description: {spec.get('description', '')}
Inputs: {spec.get('inputs', [])}
Outputs: {spec.get('outputs', [])}
Constraints: {spec.get('constraints', [])}

REFERENCE PATTERNS (from similar high-quality code):
{pattern_context}

Now generate the complete, production-ready code that fulfills this specification.
Use the reference patterns as inspiration but adapt them to the exact requirements.
"""

        response = self.brain.consult(prompt, system_prompt)

        if not response:
            self.logger.error("Code synthesis failed - no brain response")
            return None

        # Extract code (remove markdown if present)
        code = response.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1])  # Remove first and last line (```)

        # Determine filename
        task_type = spec.get("task_type", "function")
        language = spec.get("language", "python")
        ext_map = {"python": ".py", "javascript": ".js", "typescript": ".ts", "rust": ".rs"}
        ext = ext_map.get(language, ".txt")

        filename = f"generated_{task_type}{ext}"

        result = {"code": code, "filename": filename, "language": language}

        self.logger.info("Code synthesized", filename=filename, lines=len(code.split("\n")))
        return result
