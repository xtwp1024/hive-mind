"""
Perceiver - 感知器
==================

功能:
- 意图识别 (Intent Detection)
- 实体提取 (Entity Extraction)
- 情感初步判断 (Sentiment Analysis)
- 紧急程度评估 (Urgency Assessment)

Usage:
    perceiver = Perceiver()
    perception = await perceiver.process("帮我分析这个项目的架构")
    # perception.intent -> "analyze"
    # perception.entities -> [{"type": "project", "value": "项目"}]
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Hive.Brain.Perceiver")


class Intent(str, Enum):
    """预定义意图"""

    # 通用
    UNKNOWN = "unknown"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    HELP = "help"
    CONFIRM = "confirm"
    CANCEL = "cancel"

    # 项目相关
    ANALYZE = "analyze"
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    REVIEW = "review"

    # 执行相关
    EXECUTE = "execute"
    PLAN = "plan"
    DEBUG = "debug"
    TEST = "test"
    DEPLOY = "deploy"

    # 学习相关
    LEARN = "learn"
    EXPLAIN = "explain"
    SUGGEST = "suggest"
    OPTIMIZE = "optimize"

    # Meta
    STATUS = "status"
    METRICS = "metrics"
    STOP = "stop"


@dataclass
class Entity:
    """提取的实体"""

    type: str  # project, file, function, concept, tool, etc.
    value: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Perception:
    """
    感知结果

    包含:
    - original_text: 原始输入
    - intent: 识别的意图
    - entities: 提取的实体
    - sentiment: 情感倾向 (-1 to 1)
    - urgency: 紧急程度 (0-1)
    - has_goal: 是否有明确目标
    - suggested_layers: 建议检索的记忆层
    - metadata: 其他元数据
    """

    original_text: str
    intent: Intent = Intent.UNKNOWN
    entities: List[Entity] = field(default_factory=list)
    sentiment: float = 0.0  # -1 (负面) to 1 (正面)
    urgency: float = 0.5  # 0 (不紧急) to 1 (紧急)
    has_goal: bool = False
    suggested_layers: List[str] = field(default_factory=lambda: ["working", "episodic", "semantic"])
    confidence: float = 0.5  # 整体置信度
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "intent": self.intent.value,
            "entities": [
                {"type": e.type, "value": e.value, "confidence": e.confidence}
                for e in self.entities
            ],
            "sentiment": self.sentiment,
            "urgency": self.urgency,
            "has_goal": self.has_goal,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
        }


class Perceiver:
    """
    感知器

    基于规则的轻量级意图识别和实体提取。
    未来可扩展为LLM-based识别。
    """

    # 意图关键词映射
    INTENT_PATTERNS = {
        Intent.GREETING: [
            r"(你好|您好|hey|hi|hello|greetings)",
            r"早上好|下午好|晚上好",
        ],
        Intent.GOODBYE: [
            r"(再见|bye|exit|quit|离开)",
        ],
        Intent.HELP: [
            r"(帮?助|help|怎么|如何|告诉我)",
        ],
        Intent.READ: [
            r"(查看|阅读|打开|读)",
        ],
        Intent.ANALYZE: [
            r"(分析|检查|审视|评估|审查)",
            r"(看看?|有什么问题|怎么样)",
        ],
        Intent.CREATE: [
            r"(创建|新建|添加|增加|实现|写)",
            r"给我一个",
        ],
        Intent.UPDATE: [
            r"(修改|更新|改进|优化|修复|改变)",
        ],
        Intent.DELETE: [
            r"\b(删除|移除|清除|清理)\b",
        ],
        Intent.SEARCH: [
            r"\b(搜索|查找|找|寻)\b",
            r"在哪",
        ],
        Intent.REVIEW: [
            r"\b(审查|review|审计|回顾)\b",
        ],
        Intent.EXECUTE: [
            r"\b(执行|运行|跑|start|run|launch)\b",
        ],
        Intent.PLAN: [
            r"\b(计划|规划|安排|设计|构建)\b",
        ],
        Intent.DEBUG: [
            r"\b(调试|debug|排除故障|排查)\b",
            r"\b出了?问题\b",
        ],
        Intent.TEST: [
            r"\b(测试|test|检验)\b",
        ],
        Intent.DEPLOY: [
            r"\b(部署|deploy|发布|上线)\b",
        ],
        Intent.LEARN: [
            r"\b(学习|了解|认识|搞懂)\b",
        ],
        Intent.EXPLAIN: [
            r"\b(解释|说明|讲?解|阐述)\b",
        ],
        Intent.SUGGEST: [
            r"\b(建议|推荐|意见|方案)\b",
        ],
        Intent.OPTIMIZE: [
            r"\b(优化|提升|改进性能)\b",
        ],
        Intent.STATUS: [
            r"\b(状态|status|情况|状况)\b",
        ],
        Intent.METRICS: [
            r"\b(指标|metrics|统计|数据)\b",
        ],
        Intent.STOP: [
            r"\b(停止|终止|结束|stop|暂停)\b",
        ],
        Intent.CONFIRM: [
            r"\b(确认|确定|yes|ok|好的|可以)\b",
        ],
        Intent.CANCEL: [
            r"\b(取消|算了|不用了|no)\b",
        ],
    }

    # 实体模式
    ENTITY_PATTERNS = {
        "file": [
            r"\b[\w/]+\.(py|js|ts|json|md|yaml|yml|toml|cfg|ini)\b",
        ],
        "path": [
            r"\b[A-Za-z]:\\[^:\n]+|/\w+/[\w/.-]+\b",
        ],
        "function": [
            r"\b(def|function|fn|async\s+def)\s+(\w+)\b",
        ],
        "class": [
            r"\bclass\s+(\w+)\b",
        ],
        "project": [
            r"项目|project|代码库|codebase",
        ],
        "module": [
            r"模块|module|组件|component",
        ],
        "error": [
            r"(Error|Exception|错误|异常|Failed|失败)",
        ],
        "number": [
            r"\b\d+\b",
        ],
    }

    # 情感词
    POSITIVE_WORDS = {
        "好",
        "不错",
        "优秀",
        "棒",
        "赞",
        "很好",
        "优秀",
        "完美",
        "good",
        "great",
        "excellent",
        "nice",
        "perfect",
    }
    NEGATIVE_WORDS = {
        "坏",
        "差",
        "糟糕",
        "烂",
        "有问题",
        "错误",
        "失败",
        "bad",
        "poor",
        "terrible",
        "awful",
        "error",
        "fail",
    }
    URGENT_WORDS = {
        "紧急",
        "立即",
        "马上",
        "快",
        "急",
        "urgent",
        "immediately",
        "asap",
        "critical",
    }

    def __init__(self, use_llm: bool = False):
        """
        Args:
            use_llm: 是否使用LLM进行更复杂的感知（未来扩展）
        """
        self.use_llm = use_llm
        self._intent_cache: Dict[str, Intent] = {}
        self._stats = {"processed": 0}

    async def process(self, text: str) -> Perception:
        """
        处理输入文本，返回感知结果

        Args:
            text: 输入文本

        Returns:
            Perception: 感知结果
        """
        start = time.time()

        perception = Perception(original_text=text)

        # 1. 意图识别
        perception.intent = self._detect_intent(text)

        # 2. 实体提取
        perception.entities = self._extract_entities(text)

        # 3. 情感分析
        perception.sentiment = self._analyze_sentiment(text)

        # 4. 紧急程度
        perception.urgency = self._assess_urgency(text)

        # 5. 是否有目标
        perception.has_goal = perception.intent not in {
            Intent.GREETING,
            Intent.GOODBYE,
            Intent.HELP,
            Intent.UNKNOWN,
        }

        # 6. 建议检索的记忆层
        perception.suggested_layers = self._suggest_layers(perception)

        # 7. 置信度
        perception.confidence = self._calculate_confidence(perception)

        perception.processing_time_ms = (time.time() - start) * 1000
        self._stats["processed"] += 1

        logger.debug(
            f"🧠 Perceiver: intent={perception.intent.value}, "
            f"entities={len(perception.entities)}, "
            f"processing_time={perception.processing_time_ms:.1f}ms"
        )

        return perception

    def _detect_intent(self, text: str) -> Intent:
        """检测意图"""
        # 缓存检查
        cache_key = text[:100]
        if cache_key in self._intent_cache:
            return self._intent_cache[cache_key]

        text_lower = text.lower()

        # 按优先级匹配
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    self._intent_cache[cache_key] = intent
                    return intent

        return Intent.UNKNOWN

    def _extract_entities(self, text: str) -> List[Entity]:
        """提取实体"""
        entities = []

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    entity = Entity(
                        type=entity_type,
                        value=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.9,
                    )
                    entities.append(entity)

        return entities

    def _analyze_sentiment(self, text: str) -> float:
        """分析情感 (-1 to 1)"""
        text_set = set(text)

        positive = sum(1 for w in self.POSITIVE_WORDS if w in text_set)
        negative = sum(1 for w in self.NEGATIVE_WORDS if w in text_set)

        total = positive + negative
        if total == 0:
            return 0.0

        return (positive - negative) / total

    def _assess_urgency(self, text: str) -> float:
        """评估紧急程度 (0 to 1)"""
        urgent_count = sum(1 for w in self.URGENT_WORDS if w in text)

        # 基础紧急度
        urgency = 0.5

        # 有紧急词
        if urgent_count > 0:
            urgency = min(1.0, 0.5 + urgent_count * 0.2)

        # 有错误/失败相关
        if re.search(r"(error|exception|failed|错误|异常|失败)", text.lower()):
            urgency = max(urgency, 0.8)

        return urgency

    def _suggest_layers(self, perception: Perception) -> List[str]:
        """根据意图建议检索的记忆层"""
        layers = ["working", "episodic"]

        if perception.intent in {Intent.LEARN, Intent.ANALYZE, Intent.EXPLAIN}:
            layers.append("semantic")

        if perception.intent in {Intent.EXECUTE, Intent.CREATE, Intent.UPDATE}:
            layers.append("procedural")

        return layers

    def _calculate_confidence(self, perception: Perception) -> float:
        """计算整体置信度"""
        confidence = 0.5

        # 意图明确
        if perception.intent != Intent.UNKNOWN:
            confidence += 0.2

        # 有实体
        if len(perception.entities) > 0:
            confidence += 0.1

        # 情感明确
        if abs(perception.sentiment) > 0.5:
            confidence += 0.1

        return min(1.0, confidence)

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "cache_size": len(self._intent_cache),
        }
