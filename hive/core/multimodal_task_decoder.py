# -*- coding: utf-8 -*-
"""
Multimodal Task Decoder - 多模态任务解码器

Phase 4: 多模态扩展
根据融合后的多模态表示执行具体任务。

核心功能:
1. 任务类型识别 - 自动识别任务类型
2. 融合表示解码 - 将统一嵌入解码为具体输出
3. 多模态输出生成 - 支持文本、代码、结构化数据等输出
4. 跨模态推理 - 基于多模态信息进行推理

架构设计:
```
┌─────────────────────────────────────────────────────────┐
│              多模态任务解码器                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           任务类型识别器                         │   │
│  │  • 从融合表示推断任务类型                        │   │
│  │  • 支持问答、分类、摘要、生成等                  │   │
│  └─────────────────────────────────────────────────┘   │
│                           │                             │
│                           ▼                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │           融合表示解码器                         │   │
│  │  • 将统一嵌入解码为结构化输出                    │   │
│  │  • 支持文本、代码、JSON 等格式                  │   │
│  └─────────────────────────────────────────────────┘   │
│                           │                             │
│                           ▼                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │           跨模态推理引擎                         │   │
│  │  • 基于多模态信息进行推理                        │   │
│  │  • 支持因果推理、类比推理等                      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
"""

import json
import logging
import math
import threading
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("Hive.TaskDecoder")


class TaskType(Enum):
    """任务类型"""

    QUESTION_ANSWERING = "qa"
    TEXT_CLASSIFICATION = "classification"
    TEXT_SUMMARIZATION = "summarization"
    TEXT_GENERATION = "generation"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    IMAGE_CAPTIONING = "captioning"
    IMAGE_UNDERSTANDING = "image_understanding"
    AUDIO_TRANSCRIPTION = "transcription"
    AUDIO_UNDERSTANDING = "audio_understanding"
    MULTIMODAL_REASONING = "multimodal_reasoning"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    UNKNOWN = "unknown"


class OutputFormat(Enum):
    """输出格式"""

    PLAIN_TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    CODE = "code"
    HTML = "html"


@dataclass
class DecodedOutput:
    """解码输出结果"""

    output_id: str

    task_type: TaskType
    output_format: OutputFormat

    content: Any = None
    text_content: str = ""

    confidence: float = 0.0
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)

    source_modalities: List[str] = field(default_factory=list)
    source_embedding_ids: List[str] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_id": self.output_id,
            "task_type": self.task_type.value,
            "output_format": self.output_format.value,
            "content": self.content,
            "text_content": self.text_content,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "source_modalities": self.source_modalities,
            "source_embedding_ids": self.source_embedding_ids,
            "created_at": self.created_at,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class TaskContext:
    """任务上下文"""

    context_id: str

    prompt: str = ""
    system_prompt: Optional[str] = None

    input_modalities: List[str] = field(default_factory=list)
    input_embeddings: List[Any] = field(default_factory=list)

    expected_output_format: OutputFormat = OutputFormat.PLAIN_TEXT
    expected_task_type: Optional[TaskType] = None

    constraints: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 4096
    temperature: float = 0.7

    created_at: float = field(default_factory=time.time)


@dataclass
class ReasoningStep:
    """推理步骤"""

    step_id: str
    step_type: str

    description: str
    modality_used: str

    evidence: Optional[str] = None
    confidence: float = 0.0

    created_at: float = field(default_factory=time.time)


class TaskTypeClassifier:
    """
    任务类型分类器

    根据输入特征自动识别任务类型。
    """

    def __init__(self):
        self._type_keywords = {
            TaskType.QUESTION_ANSWERING: [
                "what",
                "who",
                "when",
                "where",
                "why",
                "how",
                "什么",
                "谁",
                "什么时候",
                "哪里",
                "为什么",
                "怎么",
                "?",
                "？",
                "question",
                "answer",
                "问答",
                "问题",
            ],
            TaskType.TEXT_SUMMARIZATION: [
                "summary",
                "summarize",
                "summarization",
                "总结",
                "摘要",
                "概括",
                "简述",
                "提炼",
            ],
            TaskType.TEXT_CLASSIFICATION: [
                "classify",
                "classification",
                "category",
                "label",
                "分类",
                "归类",
                "标签",
                "类别",
            ],
            TaskType.CODE_GENERATION: [
                "code",
                "generate",
                "function",
                "class",
                "module",
                "代码",
                "生成",
                "函数",
                "类",
                "模块",
                "编程",
                "implement",
                "实现",
                "write code",
                "写代码",
            ],
            TaskType.CODE_ANALYSIS: [
                "analyze",
                "analysis",
                "review",
                "bug",
                "optimize",
                "分析",
                "代码分析",
                "审查",
                "bug",
                "优化",
                "重构",
            ],
            TaskType.IMAGE_UNDERSTANDING: [
                "image",
                "picture",
                "photo",
                "screenshot",
                "diagram",
                "图像",
                "图片",
                "照片",
                "截图",
                "图表",
                "架构图",
            ],
            TaskType.AUDIO_UNDERSTANDING: [
                "audio",
                "sound",
                "voice",
                "speech",
                "record",
                "音频",
                "声音",
                "语音",
                "录音",
                "音频分析",
            ],
            TaskType.MULTIMODAL_REASONING: [
                "reason",
                "reasoning",
                "infer",
                "inference",
                "推理",
                "推断",
                "逻辑",
                "综合分析",
                "跨模态",
            ],
            TaskType.KNOWLEDGE_EXTRACTION: [
                "extract",
                "extraction",
                "knowledge",
                "insight",
                "提取",
                "知识提取",
                "要点",
                "洞察",
                "关键信息",
            ],
        }

        logger.info("🏷️ [TaskTypeClassifier] 初始化")

    def classify(
        self,
        prompt: str,
        input_modalities: Optional[List[str]] = None,
    ) -> Tuple[TaskType, float]:
        """
        分类任务类型

        Args:
            prompt: 任务提示
            input_modalities: 输入模态列表

        Returns:
            (TaskType, confidence)
        """
        if not prompt and not input_modalities:
            return TaskType.UNKNOWN, 0.0

        scores: Dict[TaskType, float] = defaultdict(float)
        prompt_lower = prompt.lower() if prompt else ""

        for task_type, keywords in self._type_keywords.items():
            for keyword in keywords:
                if keyword.lower() in prompt_lower:
                    scores[task_type] += 1.0

        if input_modalities:
            if "image" in input_modalities:
                scores[TaskType.IMAGE_UNDERSTANDING] += 0.5
                scores[TaskType.MULTIMODAL_REASONING] += 0.3
            if "audio" in input_modalities:
                scores[TaskType.AUDIO_UNDERSTANDING] += 0.5
                scores[TaskType.MULTIMODAL_REASONING] += 0.3
            if "text" in input_modalities and len(input_modalities) == 1:
                scores[TaskType.QUESTION_ANSWERING] += 0.2

        if not scores:
            return TaskType.QUESTION_ANSWERING, 0.3

        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = sorted_types[0]

        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5

        return best_type, confidence

    def detect_output_format(
        self,
        prompt: str,
        task_type: TaskType,
    ) -> OutputFormat:
        """
        检测期望的输出格式

        Args:
            prompt: 任务提示
            task_type: 检测到的任务类型

        Returns:
            OutputFormat
        """
        prompt_lower = prompt.lower()

        json_indicators = ["json", "format", "结构化", "返回json", "return json"]
        for ind in json_indicators:
            if ind in prompt_lower:
                return OutputFormat.JSON

        code_indicators = ["code", "function", "class", "def ", "代码", "函数", "编程"]
        if task_type == TaskType.CODE_GENERATION or task_type == TaskType.CODE_ANALYSIS:
            return OutputFormat.CODE
        for ind in code_indicators:
            if ind in prompt_lower and len(prompt_lower) < 200:
                return OutputFormat.CODE

        markdown_indicators = ["markdown", "md", "标题", "列表", "表格", "##", "###"]
        for ind in markdown_indicators:
            if ind in prompt_lower:
                return OutputFormat.MARKDOWN

        return OutputFormat.PLAIN_TEXT


class FusionDecoder:
    """
    融合表示解码器

    将统一嵌入解码为具体输出。
    """

    def __init__(
        self,
        embedding_dim: int = 128,
    ):
        self.embedding_dim = embedding_dim
        self._brain = None
        self._lock = threading.Lock()

        logger.info(f"🔓 [FusionDecoder] 初始化，嵌入维度: {embedding_dim}")

    def _get_brain(self):
        """懒加载 BrainInterface"""
        if self._brain is None:
            try:
                from hive.utils.brain_interface import get_brain

                self._brain = get_brain()
                logger.info("🔓 [FusionDecoder] BrainInterface 已加载")
            except ImportError:
                logger.warning("🔓 [FusionDecoder] BrainInterface 不可用")
        return self._brain

    def decode(
        self,
        unified_embedding: List[float],
        task_context: TaskContext,
        raw_inputs: Optional[Dict[str, Any]] = None,
    ) -> DecodedOutput:
        """
        解码统一嵌入为输出

        Args:
            unified_embedding: 统一嵌入向量
            task_context: 任务上下文
            raw_inputs: 原始输入（文本、图像路径等）

        Returns:
            DecodedOutput - 解码结果
        """
        import uuid

        start_time = time.time()
        output_id = f"decode_{uuid.uuid4().hex[:12]}"

        brain = self._get_brain()
        output_content = ""
        confidence = 0.5
        reasoning_steps = []

        if brain and raw_inputs:
            try:
                output_content, confidence, reasoning_steps = self._decode_with_llm(
                    task_context, raw_inputs, brain
                )
            except Exception as e:
                logger.warning(f"🔓 [FusionDecoder] LLM 解码失败: {e}")
                output_content = self._decode_fallback(unified_embedding, task_context)
        else:
            output_content = self._decode_fallback(unified_embedding, task_context)

        processing_time = (time.time() - start_time) * 1000

        output = DecodedOutput(
            output_id=output_id,
            task_type=task_context.expected_task_type or TaskType.QUESTION_ANSWERING,
            output_format=task_context.expected_output_format,
            content=output_content,
            text_content=output_content if isinstance(output_content, str) else str(output_content),
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            source_modalities=task_context.input_modalities,
            created_at=time.time(),
            processing_time_ms=processing_time,
            metadata={
                "embedding_dim": len(unified_embedding),
                "max_tokens": task_context.max_tokens,
                "temperature": task_context.temperature,
            },
        )

        return output

    def _decode_with_llm(
        self,
        task_context: TaskContext,
        raw_inputs: Dict[str, Any],
        brain: Any,
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """使用 LLM 进行解码"""
        confidence = 0.8
        reasoning_steps = []

        prompt = task_context.prompt or "基于提供的信息进行分析。"
        system_prompt = task_context.system_prompt

        text_inputs = raw_inputs.get("text", [])
        image_inputs = raw_inputs.get("images", [])
        audio_inputs = raw_inputs.get("audio", [])

        combined_prompt = self._build_combined_prompt(
            prompt, text_inputs, image_inputs, audio_inputs
        )

        if image_inputs:
            reasoning_steps.append(
                {
                    "step": "1",
                    "action": "分析图像内容",
                    "modality": "image",
                    "count": len(image_inputs),
                }
            )

        if audio_inputs:
            reasoning_steps.append(
                {
                    "step": "2",
                    "action": "分析音频转录",
                    "modality": "audio",
                    "count": len(audio_inputs),
                }
            )

        if text_inputs:
            reasoning_steps.append(
                {
                    "step": "3",
                    "action": "分析文本内容",
                    "modality": "text",
                    "count": len(text_inputs),
                }
            )

        reasoning_steps.append(
            {
                "step": "4",
                "action": "综合多模态信息生成输出",
                "modality": "multimodal",
            }
        )

        try:
            if image_inputs and hasattr(brain, "consult_vision"):
                result = brain.consult_vision(
                    prompt=combined_prompt,
                    image_path=image_inputs[0],
                    system_prompt=system_prompt,
                )
            else:
                result = brain.consult(
                    prompt=combined_prompt,
                    system_prompt=system_prompt,
                )

            if result:
                return result, confidence, reasoning_steps

        except Exception as e:
            logger.warning(f"🔓 [FusionDecoder] LLM 调用失败: {e}")

        return "解码失败，无法生成输出。", 0.1, reasoning_steps

    def _build_combined_prompt(
        self,
        base_prompt: str,
        text_inputs: List[str],
        image_inputs: List[str],
        audio_inputs: List[str],
    ) -> str:
        """构建组合提示"""
        parts = []

        if base_prompt:
            parts.append(f"用户请求: {base_prompt}")

        if text_inputs:
            for i, text in enumerate(text_inputs):
                parts.append(f"\n--- 文本输入 {i+1} ---\n{text[:2000]}")

        if audio_inputs:
            for i, audio in enumerate(audio_inputs):
                parts.append(f"\n--- 音频转录 {i+1} ---\n{audio[:2000]}")

        if image_inputs:
            parts.append(f"\n--- 图像 ---\n已提供 {len(image_inputs)} 张图像供分析。")

        return "\n".join(parts)

    def _decode_fallback(
        self,
        unified_embedding: List[float],
        task_context: TaskContext,
    ) -> str:
        """Fallback 解码"""
        import hashlib

        hasher = hashlib.sha256()
        for val in unified_embedding[:32]:
            hasher.update(struct.pack("d", val))

        embedding_hash = hasher.hexdigest()[:8]

        task_type = task_context.expected_task_type or TaskType.QUESTION_ANSWERING
        prompt = task_context.prompt or "无具体请求"

        return (
            f"[Fallback 解码]\n"
            f"任务类型: {task_type.value}\n"
            f"输入模态: {task_context.input_modalities}\n"
            f"原始请求: {prompt[:100]}...\n"
            f"嵌入签名: {embedding_hash}\n"
            f"\n提示: 请配置 BrainInterface 以获取完整解码能力。"
        )


class CrossModalReasoningEngine:
    """
    跨模态推理引擎

    基于多模态信息进行复杂推理。
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self._brain = None

        logger.info(f"🧠 [CrossModalReasoningEngine] 初始化，嵌入维度: {embedding_dim}")

    def _get_brain(self):
        """懒加载 BrainInterface"""
        if self._brain is None:
            try:
                from hive.utils.brain_interface import get_brain

                self._brain = get_brain()
            except ImportError:
                pass
        return self._brain

    def reason(
        self,
        question: str,
        modality_inputs: Dict[str, List[Any]],
        reasoning_type: str = "comprehensive",
    ) -> Dict[str, Any]:
        """
        执行跨模态推理

        Args:
            question: 推理问题
            modality_inputs: 各模态输入 {modality: [inputs]}
            reasoning_type: 推理类型

        Returns:
            推理结果字典
        """
        start_time = time.time()

        modalities_used = list(modality_inputs.keys())

        reasoning_steps = []
        conclusion = ""
        confidence = 0.5

        for modality, inputs in modality_inputs.items():
            if inputs:
                reasoning_steps.append(
                    {
                        "modality": modality,
                        "action": f"分析 {modality} 输入",
                        "count": len(inputs),
                        "status": "processed",
                    }
                )

        brain = self._get_brain()

        if brain:
            try:
                combined_context = self._build_reasoning_context(question, modality_inputs)

                reasoning_prompt = f"""请执行以下推理任务。

问题: {question}

上下文信息:
{combined_context}

请提供:
1. 推理步骤（分点说明）
2. 最终结论
3. 结论的置信度 (0-1)

格式清晰，逻辑严密。"""

                result = brain.consult(
                    prompt=reasoning_prompt,
                    system_prompt="你是一个专业的推理专家，擅长基于多源信息进行逻辑推理。",
                )

                if result:
                    conclusion = result
                    confidence = 0.75
                    reasoning_steps.append(
                        {
                            "modality": "multimodal",
                            "action": "综合推理完成",
                            "status": "success",
                        }
                    )

            except Exception as e:
                logger.warning(f"🧠 [CrossModalReasoning] LLM 推理失败: {e}")
                conclusion = f"推理过程中出错: {str(e)}"
        else:
            conclusion = (
                f"跨模态推理请求\n"
                f"问题: {question}\n"
                f"涉及模态: {modalities_used}\n"
                f"\n提示: 配置 BrainInterface 可获取完整推理能力。"
            )

        processing_time = (time.time() - start_time) * 1000

        return {
            "question": question,
            "modalities_used": modalities_used,
            "reasoning_steps": reasoning_steps,
            "conclusion": conclusion,
            "confidence": confidence,
            "processing_time_ms": processing_time,
        }

    def _build_reasoning_context(
        self,
        question: str,
        modality_inputs: Dict[str, List[Any]],
    ) -> str:
        """构建推理上下文"""
        parts = []

        for modality, inputs in modality_inputs.items():
            for i, inp in enumerate(inputs):
                if isinstance(inp, str) and len(inp) > 0:
                    parts.append(f"[{modality.upper()} 输入 {i+1}]\n{inp[:1500]}\n")

        return "\n".join(parts)

    def causal_reasoning(
        self,
        cause_inputs: List[Any],
        effect_inputs: List[Any],
        question: str = "分析因果关系",
    ) -> Dict[str, Any]:
        """因果推理"""
        return self.reason(
            question=f"{question}。请分析这些输入之间的因果关系。",
            modality_inputs={
                "cause": cause_inputs,
                "effect": effect_inputs,
            },
            reasoning_type="causal",
        )

    def analogical_reasoning(
        self,
        source_inputs: List[Any],
        target_inputs: List[Any],
        question: str = "进行类比推理",
    ) -> Dict[str, Any]:
        """类比推理"""
        return self.reason(
            question=f"{question}。请分析源输入和目标输入之间的相似性和差异。",
            modality_inputs={
                "source": source_inputs,
                "target": target_inputs,
            },
            reasoning_type="analogical",
        )


class MultimodalTaskDecoder:
    """
    多模态任务解码器

    整合任务分类、融合解码、跨模态推理功能。
    """

    def __init__(
        self,
        embedding_dim: int = 128,
    ):
        self.embedding_dim = embedding_dim

        self._classifier = TaskTypeClassifier()
        self._decoder = FusionDecoder(embedding_dim)
        self._reasoning_engine = CrossModalReasoningEngine(embedding_dim)

        self._task_history: deque = deque(maxlen=100)
        self._lock = threading.Lock()

        logger.info(f"🎯 [MultimodalTaskDecoder] 初始化，嵌入维度: {embedding_dim}")

    def decode_task(
        self,
        prompt: str,
        unified_embedding: List[float],
        input_modalities: List[str],
        raw_inputs: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        expected_output_format: Optional[OutputFormat] = None,
        expected_task_type: Optional[TaskType] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> DecodedOutput:
        """
        解码任务

        Args:
            prompt: 任务提示
            unified_embedding: 统一嵌入
            input_modalities: 输入模态列表
            raw_inputs: 原始输入字典 {"text": [...], "images": [...], "audio": [...]}
            system_prompt: 系统提示
            expected_output_format: 期望的输出格式
            expected_task_type: 期望的任务类型
            max_tokens: 最大令牌数
            temperature: 温度参数

        Returns:
            DecodedOutput - 解码输出
        """
        import uuid

        context_id = f"ctx_{uuid.uuid4().hex[:12]}"

        if expected_task_type is None:
            detected_type, confidence = self._classifier.classify(prompt, input_modalities)
            expected_task_type = detected_type
            logger.info(
                f"🎯 [TaskDecoder] 检测到任务类型: {detected_type.value}, 置信度: {confidence:.2f}"
            )

        if expected_output_format is None:
            expected_output_format = self._classifier.detect_output_format(
                prompt, expected_task_type
            )

        task_context = TaskContext(
            context_id=context_id,
            prompt=prompt,
            system_prompt=system_prompt,
            input_modalities=input_modalities,
            expected_output_format=expected_output_format,
            expected_task_type=expected_task_type,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        output = self._decoder.decode(
            unified_embedding=unified_embedding,
            task_context=task_context,
            raw_inputs=raw_inputs,
        )

        with self._lock:
            self._task_history.append(
                {
                    "context_id": context_id,
                    "task_type": expected_task_type.value,
                    "prompt_length": len(prompt),
                    "modalities": input_modalities,
                    "timestamp": time.time(),
                }
            )

        return output

    def reason_with_modalities(
        self,
        question: str,
        text_inputs: Optional[List[str]] = None,
        image_inputs: Optional[List[str]] = None,
        audio_inputs: Optional[List[str]] = None,
        reasoning_type: str = "comprehensive",
    ) -> Dict[str, Any]:
        """
        使用多模态输入进行推理

        Args:
            question: 推理问题
            text_inputs: 文本输入列表
            image_inputs: 图像输入列表
            audio_inputs: 音频输入列表
            reasoning_type: 推理类型

        Returns:
            推理结果字典
        """
        modality_inputs: Dict[str, List[Any]] = {}

        if text_inputs:
            modality_inputs["text"] = text_inputs
        if image_inputs:
            modality_inputs["image"] = image_inputs
        if audio_inputs:
            modality_inputs["audio"] = audio_inputs

        return self._reasoning_engine.reason(
            question=question,
            modality_inputs=modality_inputs,
            reasoning_type=reasoning_type,
        )

    def generate_code_from_multimodal(
        self,
        requirement: str,
        text_inputs: Optional[List[str]] = None,
        image_inputs: Optional[List[str]] = None,
        language: str = "python",
    ) -> Dict[str, Any]:
        """
        从多模态输入生成代码

        Args:
            requirement: 需求描述
            text_inputs: 文本输入（如文档、规范）
            image_inputs: 图像输入（如架构图、UI设计图）
            language: 目标编程语言

        Returns:
            代码生成结果
        """
        prompt = f"""根据以下需求生成 {language} 代码。

需求: {requirement}

请提供:
1. 完整的代码实现
2. 关键注释
3. 使用示例（如果适用）"""

        unified_embedding = self._generate_fallback_embedding(requirement)

        raw_inputs: Dict[str, Any] = {}
        if text_inputs:
            raw_inputs["text"] = text_inputs
        if image_inputs:
            raw_inputs["images"] = image_inputs

        output = self.decode_task(
            prompt=prompt,
            unified_embedding=unified_embedding,
            input_modalities=["text"] + (["image"] if image_inputs else []),
            raw_inputs=raw_inputs,
            expected_task_type=TaskType.CODE_GENERATION,
            expected_output_format=OutputFormat.CODE,
        )

        return {
            "status": "success" if output.confidence > 0.3 else "partial",
            "code": output.text_content,
            "confidence": output.confidence,
            "reasoning_steps": output.reasoning_steps,
            "language": language,
        }

    def extract_knowledge_from_multimodal(
        self,
        text_inputs: Optional[List[str]] = None,
        image_inputs: Optional[List[str]] = None,
        audio_inputs: Optional[List[str]] = None,
        focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        从多模态输入提取知识

        Args:
            text_inputs: 文本输入
            image_inputs: 图像输入
            audio_inputs: 音频输入
            focus: 关注重点

        Returns:
            知识提取结果
        """
        focus_str = f"（关注重点: {focus}）" if focus else ""
        prompt = f"""从以下多模态输入中提取结构化知识{focus_str}。

请以 JSON 格式返回:
{{
  "title": "主题标题",
  "summary": "内容摘要",
  "key_points": ["要点1", "要点2", "要点3"],
  "terms": [["术语", "解释"], ...],
  "insights": ["关键洞察1", "关键洞察2"],
  "action_items": ["可执行建议"]
}}"""

        all_inputs = []
        if text_inputs:
            all_inputs.extend(text_inputs)
        if audio_inputs:
            all_inputs.extend(audio_inputs)

        combined_text = " ".join(all_inputs) if all_inputs else "多模态输入"
        unified_embedding = self._generate_fallback_embedding(combined_text)

        raw_inputs: Dict[str, Any] = {}
        if text_inputs:
            raw_inputs["text"] = text_inputs
        if image_inputs:
            raw_inputs["images"] = image_inputs
        if audio_inputs:
            raw_inputs["audio"] = audio_inputs

        modalities = []
        if text_inputs:
            modalities.append("text")
        if image_inputs:
            modalities.append("image")
        if audio_inputs:
            modalities.append("audio")

        output = self.decode_task(
            prompt=prompt,
            unified_embedding=unified_embedding,
            input_modalities=modalities,
            raw_inputs=raw_inputs,
            expected_task_type=TaskType.KNOWLEDGE_EXTRACTION,
            expected_output_format=OutputFormat.JSON,
        )

        return {
            "status": "success" if output.confidence > 0.3 else "partial",
            "extraction": output.text_content,
            "confidence": output.confidence,
            "modalities_used": modalities,
            "reasoning_steps": output.reasoning_steps,
        }

    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """生成 fallback 嵌入"""
        import hashlib

        hasher = hashlib.sha256(text.encode("utf-8"))
        hash_bytes = hasher.digest()

        embedding = []
        for i in range(0, min(len(hash_bytes), 64), 4):
            chunk = hash_bytes[i : i + 4]
            if len(chunk) < 4:
                chunk = chunk + b"\x00" * (4 - len(chunk))
            value = int.from_bytes(chunk, "big") / 0xFFFFFFFF
            embedding.append(value * 2 - 1)

        while len(embedding) < self.embedding_dim:
            hasher.update(b"salt")
            hash_bytes = hasher.digest()
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i : i + 4]
                if len(chunk) < 4:
                    chunk = chunk + b"\x00" * (4 - len(chunk))
                value = int.from_bytes(chunk, "big") / 0xFFFFFFFF
                embedding.append(value * 2 - 1)
                if len(embedding) >= self.embedding_dim:
                    break

        return embedding[: self.embedding_dim]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            by_type = defaultdict(int)
            for record in self._task_history:
                by_type[record.get("task_type", "unknown")] += 1

            return {
                "total_tasks_processed": len(self._task_history),
                "tasks_by_type": dict(by_type),
                "embedding_dimension": self.embedding_dim,
            }


import struct

_decoder: Optional[MultimodalTaskDecoder] = None
_decoder_lock = threading.Lock()


def get_multimodal_decoder() -> MultimodalTaskDecoder:
    """获取多模态任务解码器单例"""
    global _decoder
    if _decoder is None:
        with _decoder_lock:
            if _decoder is None:
                _decoder = MultimodalTaskDecoder()
    return _decoder
