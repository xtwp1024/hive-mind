# -*- coding: utf-8 -*-
"""相关性评分工具

用于仓库/链接的相关性打分与过滤。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable


_WORD_RE = re.compile(r"[a-z0-9_\-]+", re.IGNORECASE)


def _normalize(text: str) -> str:
    return (text or "").lower()


def _tokenize_keywords(keywords: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for kw in keywords:
        for token in _WORD_RE.findall(str(kw).lower()):
            if token:
                tokens.append(token)
    return list(dict.fromkeys(tokens))


@dataclass
class RelevanceConfig:
    min_score: float = 0.6
    strict: bool = True
    preferred_languages: list[str] | None = None


class RelevanceScorer:
    """轻量相关性评分器。"""

    def __init__(self, config: RelevanceConfig | None = None) -> None:
        self._config = config or RelevanceConfig()

    @property
    def min_score(self) -> float:
        return self._config.min_score

    @property
    def strict(self) -> bool:
        return self._config.strict

    def score(
        self,
        *,
        name: str = "",
        description: str = "",
        keywords: Iterable[str] | None = None,
        language: str = "",
        stars: int = 0,
        extra_text: str = "",
    ) -> float:
        """计算相关性分数（0~1）。"""
        keywords = list(keywords or [])
        tokens = _tokenize_keywords(keywords)
        if not tokens:
            return 0.0

        name_text = _normalize(name)
        desc_text = _normalize(description)
        extra_text = _normalize(extra_text)
        combined = " ".join([name_text, desc_text, extra_text]).strip()

        if not combined:
            return 0.0

        match_count = sum(1 for token in tokens if token in combined)
        keyword_score = match_count / max(1, len(tokens))

        name_bonus = 1.0 if any(token in name_text for token in tokens) else 0.0

        stars_score = 0.0
        if stars > 0:
            stars_score = min(1.0, math.log10(stars + 1) / 5)

        language_score = 0.0
        if self._config.preferred_languages:
            if language and language.lower() in [l.lower() for l in self._config.preferred_languages]:
                language_score = 1.0

        score = (
            0.65 * keyword_score
            + 0.15 * name_bonus
            + 0.10 * stars_score
            + 0.10 * language_score
        )

        return max(0.0, min(1.0, score))

    def is_relevant(self, score: float) -> bool:
        return score >= self._config.min_score
