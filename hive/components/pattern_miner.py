from pathlib import Path
from typing import Any, Dict, List

from hive.config.settings import get_settings
from hive.core.component import HiveComponent
from hive.core.vector_vault import VectorVault


class PatternMinerComponent(HiveComponent):
    """
    Mines relevant code patterns from VectorVault based on specification keywords.
    This is the second stage of the Code God pipeline.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.vault = None

    async def initialize(self) -> bool:
        """Initialize VectorVault connection"""
        try:
            settings = get_settings()
            vault_path = settings.CHROMA_PERSIST_DIR
            self.vault = VectorVault(str(vault_path))
            self.logger.info("VectorVault connected", count=self.vault.count())
            return True
        except Exception as e:
            self.logger.error("Failed to connect to VectorVault", error=str(e))
            return False

    def name(self) -> str:
        return "pattern_miner"

    def capabilities(self) -> list[str]:
        return ["mine_patterns", "search_code"]

    async def action_mine_patterns(
        self, spec: Dict[str, Any], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search VectorVault for relevant code patterns.

        Args:
            spec: Specification from IntentAnalyzer
            top_k: Number of top results to return

        Returns:
            List of code snippets with metadata
        """
        if not self.vault:
            self.logger.error("VectorVault not initialized")
            return []

        # Build search query from keywords and description
        keywords = spec.get("keywords", [])
        description = spec.get("description", "")
        language = spec.get("language", "python")

        # Construct semantic query
        query = f"{description} {' '.join(keywords)} {language}"

        self.logger.info("Mining patterns", query=query, top_k=top_k)

        try:
            # Search VectorVault (use 'limit' not 'top_k')
            results = self.vault.search(query, limit=top_k)

            patterns = []
            for result in results:
                patterns.append(
                    {
                        "content": result.get("content", ""),
                        "metadata": result.get("metadata", {}),
                        "relevance": result.get("distance", 0.0),
                    }
                )

            self.logger.info("Patterns mined", count=len(patterns))
            return patterns

        except Exception as e:
            self.logger.error("Pattern mining failed", error=str(e))
            return []
