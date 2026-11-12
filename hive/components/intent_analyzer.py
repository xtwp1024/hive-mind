import json
from typing import Any, Dict, Optional

from hive.core.component import HiveComponent
from hive.utils.brain_interface import get_brain


class IntentAnalyzerComponent(HiveComponent):
    """
    Analyzes user's natural language request and extracts structured specification.
    This is the first stage of the Code God pipeline.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.brain = get_brain()

    def name(self) -> str:
        return "intent_analyzer"

    def capabilities(self) -> list[str]:
        return ["analyze_intent", "extract_spec"]

    async def action_analyze_intent(self, user_request: str) -> Optional[Dict[str, Any]]:
        """
        Analyze user's coding request and return structured specification.

        Returns:
            {
                "task_type": "function" | "class" | "module" | "project",
                "language": "python" | "javascript" | "rust" | etc,
                "description": "detailed description",
                "inputs": [...],
                "outputs": [...],
                "constraints": [...],
                "keywords": [...]  # For VectorVault search
            }
        """
        system_prompt = """You are a software specification expert.
Your task is to analyze user requests and extract structured specifications for code generation.
Always respond with valid JSON only, no markdown formatting."""

        prompt = f"""Analyze this coding request and extract a structured specification:

User Request: "{user_request}"

Return a JSON object with these fields:
- task_type: "function", "class", "module", or "project"
- language: primary programming language
- description: detailed technical description
- inputs: list of expected inputs/parameters
- outputs: list of expected outputs/return values
- constraints: list of constraints (performance, security, etc)
- keywords: list of 5-10 technical keywords for code pattern search

Example:
{{
  "task_type": "function",
  "language": "python",
  "description": "Read JSON file and extract value by key path",
  "inputs": ["file_path: str", "key_path: str"],
  "outputs": ["value: Any"],
  "constraints": ["handle missing keys gracefully", "support nested keys"],
  "keywords": ["json", "file io", "dict access", "error handling", "pathlib"]
}}

Now analyze the user request above."""

        response = self.brain.consult(prompt, system_prompt)

        if not response:
            self.logger.error("Brain consultation failed")
            return None

        try:
            # Extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                spec = json.loads(json_match.group())
                self.logger.info("Intent analyzed successfully", spec=spec)
                return spec
            else:
                self.logger.error("No JSON found in brain response")
                return None
        except Exception as e:
            self.logger.error("Failed to parse specification", error=str(e))
            return None
