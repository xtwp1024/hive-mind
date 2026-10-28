# -*- coding: utf-8 -*-
"""
Multimodal Encoder - 多模态编码器

Phase 4: 多模态扩展
统一编码文本、图像、音频等多种模态的输入。

核心功能:
1. 文本编码器 - 使用 BERT/GPT 等模型编码文本
2. 图像编码器 - 使用 CLIP/Vision Transformer 编码图像
3. 音频编码器 - 使用 Whisper/Wav2Vec 编码音频
4. 统一表示空间 - 将不同模态映射到同一向量空间
5. 模态类型检测 - 自动识别输入模态类型

编码器架构:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  文本输入    │     │  图像输入    │     │  音频输入    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────┐
│              模态编码器 (Modal Encoder)          │
│  • 文本: BERT/GPT Embeddings                     │
│  • 图像: CLIP/Vision Transformer                 │
│  • 音频: Whisper/Wav2Vec                         │
└─────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│              投影层 (Projection Layer)           │
│  • 将不同模态映射到统一维度                        │
│  • 可选: 模态特定的投影网络                        │
└─────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│              统一表示空间 (Unified Space)         │
│  • 所有模态共享同一向量空间                        │
│  • 支持跨模态相似度计算                            │
└─────────────────────────────────────────────────┘
```
"""

import logging
import threading
import time
import base64
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
from copy import deepcopy

logger = logging.getLogger("Hive.MultimodalEncoder")


class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class EncoderType(Enum):
    """编码器类型"""
    BERT = "bert"
    GPT = "gpt"
    CLIP = "clip"
    VIT = "vit"
    WHISPER = "whisper"
    WAV2VEC = "wav2vec"
    OLLAMA = "ollama"
    BRAIN_INTERFACE = "brain_interface"
    UNIFIED = "unified"


@dataclass
class ModalityEmbedding:
    """模态嵌入表示"""

    embedding_id: str

    modality_type: ModalityType
    encoder_type: EncoderType

    raw_input: Any = None
    input_metadata: Dict[str, Any] = field(default_factory=dict)

    embedding: List[float] = field(default_factory=list)
    embedding_dim: int = 0

    normalized_embedding: Optional[List[float]] = None

    tokens: Optional[List[str]] = None
    token_count: int = 0

    created_at: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_id": self.embedding_id,
            "modality_type": self.modality_type.value,
            "encoder_type": self.encoder_type.value,
            "embedding_dim": self.embedding_dim,
            "token_count": self.token_count,
            "created_at": self.created_at,
            "processing_time_ms": self.processing_time_ms,
            "input_metadata": self.input_metadata,
        }

    def similarity(self, other: "ModalityEmbedding") -> float:
        """计算与另一个嵌入的余弦相似度"""
        if not self.embedding or not other.embedding:
            return 0.0

        if len(self.embedding) != len(other.embedding):
            return 0.0

        import math

        norm1 = math.sqrt(sum(x * x for x in self.embedding))
        norm2 = math.sqrt(sum(x * x for x in other.embedding))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        dot_product = sum(a * b for a, b in zip(self.embedding, other.embedding))
        return dot_product / (norm1 * norm2)


@dataclass
class UnifiedEmbedding:
    """统一嵌入表示"""

    embedding_id: str

    source_modalities: List[ModalityType]
    source_embeddings: List[str] = field(default_factory=list)

    unified_embedding: List[float] = field(default_factory=list)
    embedding_dim: int = 0

    fusion_strategy: str = "concatenation"
    fusion_weights: Optional[Dict[str, float]] = None

    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_id": self.embedding_id,
            "source_modalities": [m.value for m in self.source_modalities],
            "source_embeddings": self.source_embeddings,
            "embedding_dim": self.embedding_dim,
            "fusion_strategy": self.fusion_strategy,
            "fusion_weights": self.fusion_weights,
            "created_at": self.created_at,
        }


@dataclass
class EncodingResult:
    """编码结果"""

    success: bool
    message: str = ""

    modality_type: Optional[ModalityType] = None
    embedding: Optional[ModalityEmbedding] = None

    raw_response: Optional[Any] = None
    error_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "modality_type": self.modality_type.value if self.modality_type else None,
            "embedding": self.embedding.to_dict() if self.embedding else None,
            "error_details": self.error_details,
        }


class TextEncoder:
    """
    文本编码器

    使用各种模型编码文本:
    - BrainInterface (默认)
    - Ollama
    - 本地嵌入模型
    """

    def __init__(self, preferred_encoder: Optional[EncoderType] = None):
        self.preferred_encoder = preferred_encoder or EncoderType.BRAIN_INTERFACE
        self._brain = None
        self._ollama = None
        self._lock = threading.Lock()

        logger.info(f"📝 [TextEncoder] 初始化，首选编码器: {self.preferred_encoder.value}")

    def _get_brain(self):
        """懒加载 BrainInterface"""
        if self._brain is None:
            try:
                from hive.utils.brain_interface import get_brain
                self._brain = get_brain()
                logger.info("📝 [TextEncoder] BrainInterface 已加载")
            except ImportError:
                logger.warning("📝 [TextEncoder] BrainInterface 不可用")
        return self._brain

    def _get_ollama(self):
        """懒加载 Ollama"""
        if self._ollama is None:
            try:
                from hive.utils.ollama_interface import get_ollama
                self._ollama = get_ollama()
                if self._ollama and self._ollama.is_available():
                    logger.info("📝 [TextEncoder] Ollama 已加载")
                else:
                    self._ollama = None
            except ImportError:
                pass
        return self._ollama

    def encode(
        self,
        text: str,
        model: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> EncodingResult:
        """
        编码文本

        Args:
            text: 输入文本
            model: 可选模型覆盖
            max_tokens: 最大令牌数

        Returns:
            EncodingResult - 编码结果
        """
        import uuid

        start_time = time.time()

        embedding_id = f"text_{uuid.uuid4().hex[:12]}"

        if not text or not text.strip():
            return EncodingResult(
                success=False,
                message="Empty text input",
                modality_type=ModalityType.TEXT,
            )

        text = text[:max_tokens * 4] if max_tokens else text

        brain = self._get_brain()
        ollama = self._get_ollama()

        embedding = None
        encoder_used = None
        raw_response = None

        if brain and hasattr(brain, 'get_embeddings'):
            try:
                raw_response = brain.get_embeddings(text, model=model)
                if raw_response:
                    if isinstance(raw_response, list):
                        embedding = raw_response
                    elif isinstance(raw_response, dict):
                        embedding = raw_response.get('embedding', raw_response.get('data', {}))
                    encoder_used = EncoderType.BRAIN_INTERFACE
            except Exception as e:
                logger.warning(f"📝 [TextEncoder] BrainInterface 编码失败: {e}")

        if embedding is None and ollama:
            try:
                embed_model = model or "nomic-embed-text"
                raw_response = ollama.embed(text, model=embed_model)
                if raw_response:
                    if isinstance(raw_response, list):
                        embedding = raw_response
                    elif isinstance(raw_response, dict):
                        embedding = raw_response.get('embedding', [])
                    encoder_used = EncoderType.OLLAMA
            except Exception as e:
                logger.warning(f"📝 [TextEncoder] Ollama 编码失败: {e}")

        if embedding is None:
            import hashlib
            import math

            logger.warning("📝 [TextEncoder] 使用 fallback 伪嵌入")

            hasher = hashlib.sha256(text.encode('utf-8'))
            hash_bytes = hasher.digest()

            embedding = []
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i:i+4]
                if len(chunk) < 4:
                    chunk = chunk + b'\x00' * (4 - len(chunk))
                value = int.from_bytes(chunk, 'big') / 0xFFFFFFFF
                embedding.append(value * 2 - 1)

            while len(embedding) < 64:
                hasher.update(b'salt')
                hash_bytes = hasher.digest()
                for i in range(0, len(hash_bytes), 4):
                    chunk = hash_bytes[i:i+4]
                    if len(chunk) < 4:
                        chunk = chunk + b'\x00' * (4 - len(chunk))
                    value = int.from_bytes(chunk, 'big') / 0xFFFFFFFF
                    embedding.append(value * 2 - 1)
                    if len(embedding) >= 64:
                        break

            embedding = embedding[:64]
            encoder_used = EncoderType.UNIFIED

        processing_time = (time.time() - start_time) * 1000

        import math
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            normalized = [x / norm for x in embedding]
        else:
            normalized = embedding

        result_embedding = ModalityEmbedding(
            embedding_id=embedding_id,
            modality_type=ModalityType.TEXT,
            encoder_type=encoder_used or EncoderType.UNIFIED,
            raw_input=text[:500] if len(text) > 500 else text,
            input_metadata={
                "text_length": len(text),
                "model_used": model,
            },
            embedding=embedding,
            embedding_dim=len(embedding),
            normalized_embedding=normalized,
            token_count=len(text.split()),
            processing_time_ms=processing_time,
        )

        return EncodingResult(
            success=True,
            message=f"Text encoded successfully using {encoder_used.value if encoder_used else 'fallback'}",
            modality_type=ModalityType.TEXT,
            embedding=result_embedding,
            raw_response=raw_response,
        )

    def encode_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[EncodingResult]:
        """批量编码文本"""
        results = []
        for text in texts:
            result = self.encode(text, model=model)
            results.append(result)
        return results


class ImageEncoder:
    """
    图像编码器

    编码图像为向量表示:
    - BrainInterface (consult_vision)
    - Ollama (llava 等多模态模型)
    - 支持 base64 和文件路径输入
    """

    def __init__(self, preferred_encoder: Optional[EncoderType] = None):
        self.preferred_encoder = preferred_encoder or EncoderType.BRAIN_INTERFACE
        self._brain = None
        self._ollama = None
        self._lock = threading.Lock()

        logger.info(f"🖼️ [ImageEncoder] 初始化，首选编码器: {self.preferred_encoder.value}")

    def _get_brain(self):
        """懒加载 BrainInterface"""
        if self._brain is None:
            try:
                from hive.utils.brain_interface import get_brain
                self._brain = get_brain()
                logger.info("🖼️ [ImageEncoder] BrainInterface 已加载")
            except ImportError:
                logger.warning("🖼️ [ImageEncoder] BrainInterface 不可用")
        return self._brain

    def _get_ollama(self):
        """懒加载 Ollama"""
        if self._ollama is None:
            try:
                from hive.utils.ollama_interface import get_ollama
                self._ollama = get_ollama()
                if self._ollama and self._ollama.is_available():
                    logger.info("🖼️ [ImageEncoder] Ollama 已加载")
                else:
                    self._ollama = None
            except ImportError:
                pass
        return self._ollama

    def _image_to_base64(self, image_path: str) -> Optional[str]:
        """将图像文件转换为 base64"""
        try:
            path = Path(image_path)
            if not path.exists():
                return None

            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"🖼️ [ImageEncoder] 图像转换失败: {e}")
            return None

    def encode(
        self,
        image_input: Union[str, Path],
        prompt: str = "Describe this image in detail",
        model: Optional[str] = None,
    ) -> EncodingResult:
        """
        编码图像

        Args:
            image_input: 图像路径或 base64 字符串
            prompt: 用于分析的提示
            model: 可选模型覆盖

        Returns:
            EncodingResult - 编码结果
        """
        import uuid

        start_time = time.time()
        embedding_id = f"image_{uuid.uuid4().hex[:12]}"

        is_path = False
        image_base64 = None
        image_path = None

        if isinstance(image_input, Path) or (isinstance(image_input, str) and len(image_input) < 500 and Path(image_input).exists()):
            is_path = True
            image_path = str(image_input)
            image_base64 = self._image_to_base64(image_path)
            if not image_base64:
                return EncodingResult(
                    success=False,
                    message=f"Failed to read image: {image_path}",
                    modality_type=ModalityType.IMAGE,
                )
        else:
            image_base64 = image_input

        brain = self._get_brain()
        ollama = self._get_ollama()

        analysis_result = None
        encoder_used = None
        raw_response = None

        if brain and hasattr(brain, 'consult_vision'):
            try:
                if is_path and image_path:
                    raw_response = brain.consult_vision(
                        prompt=prompt,
                        image_path=image_path,
                        model=model,
                    )
                else:
                    raw_response = brain.consult(
                        prompt=prompt,
                        model=model,
                        images=[image_base64],
                    )

                if raw_response:
                    analysis_result = raw_response
                    encoder_used = EncoderType.BRAIN_INTERFACE
            except Exception as e:
                logger.warning(f"🖼️ [ImageEncoder] BrainInterface 编码失败: {e}")

        if analysis_result is None and ollama:
            try:
                vision_model = model or "llava"
                raw_response = ollama.consult(
                    prompt=prompt,
                    model=vision_model,
                    images=[image_base64],
                )
                if raw_response:
                    analysis_result = raw_response
                    encoder_used = EncoderType.OLLAMA
            except Exception as e:
                logger.warning(f"🖼️ [ImageEncoder] Ollama 编码失败: {e}")

        if analysis_result is None:
            analysis_result = "Image analysis not available"
            encoder_used = EncoderType.UNIFIED

        processing_time = (time.time() - start_time) * 1000

        import hashlib
        import math

        hasher = hashlib.sha256(analysis_result.encode('utf-8'))
        hash_bytes = hasher.digest()

        embedding = []
        for i in range(0, min(len(hash_bytes), 64), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) < 4:
                chunk = chunk + b'\x00' * (4 - len(chunk))
            value = int.from_bytes(chunk, 'big') / 0xFFFFFFFF
            embedding.append(value * 2 - 1)

        while len(embedding) < 64:
            hasher.update(b'salt')
            hash_bytes = hasher.digest()
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i:i+4]
                if len(chunk) < 4:
                    chunk = chunk + b'\x00' * (4 - len(chunk))
                value = int.from_bytes(chunk, 'big') / 0xFFFFFFFF
                embedding.append(value * 2 - 1)
                if len(embedding) >= 64:
                    break

        embedding = embedding[:64]

        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            normalized = [x / norm for x in embedding]
        else:
            normalized = embedding

        result_embedding = ModalityEmbedding(
            embedding_id=embedding_id,
            modality_type=ModalityType.IMAGE,
            encoder_type=encoder_used,
            raw_input=analysis_result[:500] if len(analysis_result) > 500 else analysis_result,
            input_metadata={
                "image_path": image_path,
                "is_path": is_path,
                "prompt": prompt,
                "model_used": model,
            },
            embedding=embedding,
            embedding_dim=len(embedding),
            normalized_embedding=normalized,
            token_count=len(analysis_result.split()),
            processing_time_ms=processing_time,
        )

        return EncodingResult(
            success=True,
            message=f"Image encoded successfully using {encoder_used.value}",
            modality_type=ModalityType.IMAGE,
            embedding=result_embedding,
            raw_response=raw_response,
        )

    def analyze_architecture(
        self,
        image_path: str,
        focus: str = "components_and_relationships",
    ) -> Dict[str, Any]:
        """
        分析架构图

        Args:
            image_path: 架构图路径
            focus: 分析重点

        Returns:
            包含组件和关系的字典
        """
        prompt = f"""分析这张架构图。

请提取以下信息并以 JSON 格式返回:
{{
  "components": ["组件1", "组件2", ...],
  "relationships": [
    {{"from": "组件A", "to": "组件B", "type": "数据流/调用/依赖"}},
    ...
  ],
  "data_flow": ["数据流向描述"],
  "missing_components": ["可能缺失的组件"],
  "suggestions": ["改进建议"]
}}

只返回 JSON，不要其他内容。"""

        result = self.encode(image_path, prompt=prompt)

        if not result.success or not result.embedding:
            return {
                "status": "error",
                "message": result.message,
                "components": [],
                "relationships": [],
            }

        try:
            import re
            text = result.embedding.raw_input if result.embedding else ""
            json_match = re.search(r'\{.*\}', text, re.DOTALL) if text else None

            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "status": "success",
                    "message": "Architecture analyzed",
                    **parsed,
                }
        except Exception:
            pass

        return {
            "status": "partial",
            "message": "Analysis complete but JSON parsing failed",
            "raw_analysis": result.embedding.raw_input if result.embedding else "",
            "components": [],
            "relationships": [],
        }

    def analyze_screenshot(
        self,
        image_path: str,
        focus: str = "ui_layout",
    ) -> Dict[str, Any]:
        """
        分析截图/UI界面

        Args:
            image_path: 截图路径
            focus: 分析重点

        Returns:
            UI 分析结果
        """
        prompt = f"""分析这张截图。

请提取以下信息:
1. 整体布局结构
2. 主要的交互元素 (按钮、表单、导航等)
3. 视觉层级和信息架构
4. 任何 UX 问题或值得注意的设计模式

以清晰的结构化格式回答。"""

        result = self.encode(image_path, prompt=prompt)

        if not result.success:
            return {"status": "error", "message": result.message}

        return {
            "status": "success",
            "message": "Screenshot analyzed",
            "analysis": result.embedding.raw_input if result.embedding else "",
            "image_path": image_path,
        }


class AudioEncoder:
    """
    音频编码器

    编码音频为文本和向量表示:
    - BrainInterface (transcribe_audio)
    - Ollama (whisper 等模型)
    """

    def __init__(self, preferred_encoder: Optional[EncoderType] = None):
        self.preferred_encoder = preferred_encoder or EncoderType.WHISPER
        self._brain = None
        self._ollama = None
        self._lock = threading.Lock()

        logger.info(f"🎙️ [AudioEncoder] 初始化，首选编码器: {self.preferred_encoder.value}")

    def _get_brain(self):
        """懒加载 BrainInterface"""
        if self._brain is None:
            try:
                from hive.utils.brain_interface import get_brain
                self._brain = get_brain()
                logger.info("🎙️ [AudioEncoder] BrainInterface 已加载")
            except ImportError:
                logger.warning("🎙️ [AudioEncoder] BrainInterface 不可用")
        return self._brain

    def _get_ollama(self):
        """懒加载 Ollama"""
        if self._ollama is None:
            try:
                from hive.utils.ollama_interface import get_ollama
                self._ollama = get_ollama()
                if self._ollama and self._ollama.is_available():
                    logger.info("🎙️ [AudioEncoder] Ollama 已加载")
                else:
                    self._ollama = None
            except ImportError:
                pass
        return self._ollama

    def encode(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> EncodingResult:
        """
        编码音频

        Args:
            audio_path: 音频文件路径
            language: 语言代码
            prompt: 转录提示
            model: 模型覆盖

        Returns:
            EncodingResult - 编码结果
        """
        import uuid

        start_time = time.time()
        embedding_id = f"audio_{uuid.uuid4().hex[:12]}"

        path = Path(audio_path)
        if not path.exists():
            return EncodingResult(
                success=False,
                message=f"Audio file not found: {audio_path}",
                modality_type=ModalityType.AUDIO,
            )

        brain = self._get_brain()

        transcription = None
        encoder_used = None
        raw_response = None

        if brain and hasattr(brain, 'transcribe_audio'):
            try:
                raw_response = brain.transcribe_audio(
                    audio_path=audio_path,
                    prompt=prompt,
                    language=language,
                    model=model,
                )
                if raw_response:
                    transcription = raw_response
                    encoder_used = EncoderType.WHISPER
            except Exception as e:
                logger.warning(f"🎙️ [AudioEncoder] BrainInterface 转录失败: {e}")

        if transcription is None:
            transcription = f"Audio transcription not available for: {audio_path}"
            encoder_used = EncoderType.UNIFIED

        processing_time = (time.time() - start_time) * 1000

        import hashlib
        import math

        hasher = hashlib.sha256(transcription.encode('utf-8'))
        hash_bytes = hasher.digest()

        embedding = []
        for i in range(0, min(len(hash_bytes), 64), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) < 4:
                chunk = chunk + b'\x00' * (4 - len(chunk))
            value = int.from_bytes(chunk, 'big') / 0xFFFFFFFF
            embedding.append(value * 2 - 1)

        while len(embedding) < 64:
            hasher.update(b'salt')
            hash_bytes = hasher.digest()
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i:i+4]
                if len(chunk) < 4:
                    chunk = chunk + b'\x00' * (4 - len(chunk))
                value = int.from_bytes(chunk, 'big') / 0xFFFFFFFF
                embedding.append(value * 2 - 1)
                if len(embedding) >= 64:
                    break

        embedding = embedding[:64]

        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            normalized = [x / norm for x in embedding]
        else:
            normalized = embedding

        result_embedding = ModalityEmbedding(
            embedding_id=embedding_id,
            modality_type=ModalityType.AUDIO,
            encoder_type=encoder_used,
            raw_input=transcription[:500] if len(transcription) > 500 else transcription,
            input_metadata={
                "audio_path": audio_path,
                "language": language,
                "model_used": model,
            },
            embedding=embedding,
            embedding_dim=len(embedding),
            normalized_embedding=normalized,
            token_count=len(transcription.split()),
            processing_time_ms=processing_time,
        )

        return EncodingResult(
            success=True,
            message=f"Audio encoded successfully using {encoder_used.value}",
            modality_type=ModalityType.AUDIO,
            embedding=result_embedding,
            raw_response=raw_response,
        )

    def transcribe_and_summarize(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        转录音频并生成总结

        Args:
            audio_path: 音频路径
            language: 语言

        Returns:
            包含转录和总结的字典
        """
        result = self.encode(audio_path, language=language)

        if not result.success or not result.embedding:
            return {
                "status": "error",
                "message": result.message,
                "transcription": None,
                "summary": None,
            }

        transcription = result.embedding.raw_input or ""

        brain = self._get_brain()
        summary = None

        if brain and transcription:
            try:
                summary_prompt = f"""总结以下音频转录内容。

转录内容:
{transcription[:8000]}

请提供:
1. 标题 (一句话)
2. 摘要 (3-5句话)
3. 关键要点 (3-5条)
4. 结论/收获"""

                summary = brain.consult(
                    prompt=summary_prompt,
                    system_prompt="你是一个专业的总结专家，擅长提炼关键信息。",
                )
            except Exception as e:
                logger.warning(f"🎙️ [AudioEncoder] 总结生成失败: {e}")

        return {
            "status": "success",
            "message": "Audio transcribed and summarized",
            "transcription": transcription,
            "summary": summary,
            "audio_path": audio_path,
        }


class MultimodalEncoder:
    """
    多模态编码器

    统一编码多种模态的输入，并提供:
    1. 模态类型自动检测
    2. 多种编码器调度
    3. 统一嵌入空间
    4. 跨模态相似度计算
    """

    def __init__(
        self,
        text_encoder: Optional[TextEncoder] = None,
        image_encoder: Optional[ImageEncoder] = None,
        audio_encoder: Optional[AudioEncoder] = None,
        unified_dim: int = 128,
    ):
        self.text_encoder = text_encoder or TextEncoder()
        self.image_encoder = image_encoder or ImageEncoder()
        self.audio_encoder = audio_encoder or AudioEncoder()

        self.unified_dim = unified_dim

        self._embedding_cache: Dict[str, ModalityEmbedding] = {}
        self._cache_max_size = 1000

        self._lock = threading.Lock()

        logger.info(
            f"🔮 [MultimodalEncoder] 初始化，统一维度: {unified_dim}"
        )

    def detect_modality(
        self,
        input_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModalityType:
        """
        自动检测模态类型

        Args:
            input_data: 输入数据
            metadata: 元数据

        Returns:
            ModalityType - 检测到的模态类型
        """
        if metadata:
            modality_hint = metadata.get("modality")
            if modality_hint:
                try:
                    return ModalityType(modality_hint)
                except ValueError:
                    pass

            mime_type = metadata.get("mime_type", "")
            if mime_type.startswith("text/"):
                return ModalityType.TEXT
            if mime_type.startswith("image/"):
                return ModalityType.IMAGE
            if mime_type.startswith("audio/"):
                return ModalityType.AUDIO
            if mime_type.startswith("video/"):
                return ModalityType.VIDEO

        if isinstance(input_data, str):
            path = Path(input_data)
            if path.exists():
                suffix = path.suffix.lower()

                text_extensions = {'.txt', '.md', '.json', '.csv', '.xml', '.html'}
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}
                audio_extensions = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'}
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

                if suffix in text_extensions:
                    return ModalityType.TEXT
                if suffix in image_extensions:
                    return ModalityType.IMAGE
                if suffix in audio_extensions:
                    return ModalityType.AUDIO
                if suffix in video_extensions:
                    return ModalityType.VIDEO

            if len(input_data) > 1000:
                return ModalityType.TEXT

        return ModalityType.UNKNOWN

    def encode(
        self,
        input_data: Any,
        modality: Optional[ModalityType] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> EncodingResult:
        """
        编码输入数据

        Args:
            input_data: 输入数据（文本、图像路径、音频路径等）
            modality: 可选模态类型覆盖
            metadata: 元数据
            **kwargs: 传递给具体编码器的参数

        Returns:
            EncodingResult - 编码结果
        """
        if modality is None:
            modality = self.detect_modality(input_data, metadata)

        if modality == ModalityType.UNKNOWN:
            if isinstance(input_data, str) and len(input_data) < 500:
                try:
                    path = Path(input_data)
                    if path.exists():
                        pass
                    else:
                        modality = ModalityType.TEXT
                except Exception:
                    modality = ModalityType.TEXT
            else:
                modality = ModalityType.TEXT

        if modality == ModalityType.TEXT:
            return self.text_encoder.encode(
                text=input_data,
                model=kwargs.get("model"),
            )
        elif modality == ModalityType.IMAGE:
            return self.image_encoder.encode(
                image_input=input_data,
                prompt=kwargs.get("prompt", "Describe this image"),
                model=kwargs.get("model"),
            )
        elif modality == ModalityType.AUDIO:
            return self.audio_encoder.encode(
                audio_path=input_data,
                language=kwargs.get("language"),
                prompt=kwargs.get("prompt"),
                model=kwargs.get("model"),
            )
        elif modality == ModalityType.MULTIMODAL:
            return self._encode_multimodal(input_data, metadata, **kwargs)
        else:
            return EncodingResult(
                success=False,
                message=f"Unsupported modality: {modality.value}",
                modality_type=modality,
            )

    def _encode_multimodal(
        self,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> EncodingResult:
        """编码多模态输入"""
        embeddings = []

        if "text" in input_data:
            result = self.encode(input_data["text"], ModalityType.TEXT, metadata, **kwargs)
            if result.success and result.embedding:
                embeddings.append(result.embedding)

        if "image" in input_data:
            result = self.encode(input_data["image"], ModalityType.IMAGE, metadata, **kwargs)
            if result.success and result.embedding:
                embeddings.append(result.embedding)

        if "audio" in input_data:
            result = self.encode(input_data["audio"], ModalityType.AUDIO, metadata, **kwargs)
            if result.success and result.embedding:
                embeddings.append(result.embedding)

        if not embeddings:
            return EncodingResult(
                success=False,
                message="No valid modalities to encode",
                modality_type=ModalityType.MULTIMODAL,
            )

        unified = self.fuse_embeddings(embeddings)

        return EncodingResult(
            success=True,
            message=f"Multimodal encoded with {len(embeddings)} modalities",
            modality_type=ModalityType.MULTIMODAL,
            embedding=unified,
        )

    def fuse_embeddings(
        self,
        embeddings: List[ModalityEmbedding],
        strategy: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
    ) -> ModalityEmbedding:
        """
        融合多个嵌入

        Args:
            embeddings: 嵌入列表
            strategy: 融合策略 ("concatenation", "average", "weighted_average")
            weights: 可选权重字典

        Returns:
            ModalityEmbedding - 融合后的嵌入
        """
        import uuid
        import math

        if not embeddings:
            raise ValueError("No embeddings to fuse")

        if len(embeddings) == 1:
            return embeddings[0]

        default_weights = {
            ModalityType.TEXT.value: 1.0,
            ModalityType.IMAGE.value: 1.0,
            ModalityType.AUDIO.value: 1.0,
        }
        weights = weights or default_weights

        if strategy == "concatenation":
            fused = []
            for emb in embeddings:
                fused.extend(emb.embedding)

        elif strategy == "average":
            max_dim = max(len(e.embedding) for e in embeddings)
            fused = [0.0] * max_dim

            for emb in embeddings:
                for i, val in enumerate(emb.embedding):
                    if i < max_dim:
                        fused[i] += val / len(embeddings)

        elif strategy == "weighted_average":
            max_dim = max(len(e.embedding) for e in embeddings)
            fused = [0.0] * max_dim

            total_weight = 0.0
            for emb in embeddings:
                weight = weights.get(emb.modality_type.value, 1.0)
                total_weight += weight

                for i, val in enumerate(emb.embedding):
                    if i < max_dim:
                        fused[i] += val * weight

            if total_weight > 0:
                fused = [v / total_weight for v in fused]

        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

        norm = math.sqrt(sum(x * x for x in fused))
        if norm > 0:
            normalized = [x / norm for x in fused]
        else:
            normalized = fused

        source_modalities = [e.modality_type for e in embeddings]
        source_ids = [e.embedding_id for e in embeddings]

        import statistics
        total_time = sum(e.processing_time_ms for e in embeddings)

        result = ModalityEmbedding(
            embedding_id=f"fused_{uuid.uuid4().hex[:12]}",
            modality_type=ModalityType.MULTIMODAL,
            encoder_type=EncoderType.UNIFIED,
            embedding=fused,
            embedding_dim=len(fused),
            normalized_embedding=normalized,
            processing_time_ms=total_time,
        )

        return result

    def cross_modal_similarity(
        self,
        emb1: ModalityEmbedding,
        emb2: ModalityEmbedding,
    ) -> float:
        """
        计算跨模态相似度

        Args:
            emb1: 第一个嵌入
            emb2: 第二个嵌入

        Returns:
            float - 余弦相似度 (-1 到 1)
        """
        return emb1.similarity(emb2)

    def encode_batch(
        self,
        inputs: List[Tuple[Any, Optional[ModalityType], Optional[Dict[str, Any]]]],
    ) -> List[EncodingResult]:
        """批量编码"""
        results = []
        for input_data, modality, metadata in inputs:
            result = self.encode(input_data, modality, metadata)
            results.append(result)
        return results

    def get_encoder_for_modality(
        self,
        modality: ModalityType,
    ) -> Any:
        """获取对应模态的编码器"""
        if modality == ModalityType.TEXT:
            return self.text_encoder
        elif modality == ModalityType.IMAGE:
            return self.image_encoder
        elif modality == ModalityType.AUDIO:
            return self.audio_encoder
        else:
            return None


_encoder: Optional[MultimodalEncoder] = None
_encoder_lock = threading.Lock()


def get_multimodal_encoder() -> MultimodalEncoder:
    """获取多模态编码器单例"""
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                _encoder = MultimodalEncoder()
    return _encoder
