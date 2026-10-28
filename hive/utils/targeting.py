# -*- coding: utf-8 -*-
"""定向吸收关键词构建器

根据项目目标与能力短板生成相关性关键词。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from hive.config.settings import get_settings


def _split_keywords(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    for raw in text.replace("，", ",").replace(";", ",").split(","):
        token = raw.strip()
        if token:
            tokens.append(token)
    return tokens


def _dedupe(items: Iterable[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _load_intel_keywords() -> list[str]:
    strategy_file = Path("intel_strategy.json")
    if not strategy_file.exists():
        return []
    try:
        with strategy_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [kw for kw in data.get("hunting_keywords", []) if isinstance(kw, str)]
    except Exception:
        return []


def load_targeting_keywords(extra_keywords: Iterable[str] | None = None) -> list[str]:
    settings = get_settings()
    keywords: list[str] = []

    keywords.extend(_split_keywords(settings.RELEVANCE_KEYWORDS))
    keywords.extend(_split_keywords(settings.PROJECT_GOALS))
    keywords.extend(_split_keywords(settings.CAPABILITY_GAPS))
    keywords.extend(_load_intel_keywords())

    if extra_keywords:
        for kw in extra_keywords:
            if kw:
                keywords.append(str(kw))

    return _dedupe(keywords)
