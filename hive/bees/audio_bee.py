# -*- coding: utf-8 -*-
"""
AudioBee — 音频处理与知识提取专家

Phase 4 P2: 音频处理
职责:
1. 音频转录 — 技术讲座、播客、会议录音 → 文本
2. 知识提取 — 转录文本 → 结构化知识点
3. 要点总结 — 提取关键信息、术语、概念
4. 支持 Whisper API / Ollama whisper / Gemini 音频理解

使用方式:
    audio_bee = AudioBee(dna, "AudioBee-001", result_queue)
    result = audio_bee.transcribe("/path/to/lecture.mp3")
    result = audio_bee.extract_knowledge("/path/to/podcast.wav", topic="Python")
"""

import logging
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA

logger = logging.getLogger("Hive.Audio")


class AudioBee(BaseBee):
    """
    AudioBee — 音频处理与知识提取专家

    与普通 WorkerBee 的区别:
    - 输入是音频文件而非代码
    - 使用 Whisper 等转录模型
    - 输出是结构化知识（知识点、术语、总结）
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)

        # Phase 4 P2: 音频模型配置
        self._preferred_model = self.dna.metadata.get("audio_model", None)
        self._default_language = self.dna.metadata.get("language", None)
        self._brain = None  # 懒加载

        logger.info(f"🎙️ [Audio] AudioBee 已初始化，模型: {self._preferred_model or 'auto'}")

    def _get_brain(self):
        """懒加载 BrainInterface"""
        if self._brain is None:
            from hive.utils.brain_interface import get_brain
            self._brain = get_brain()
        return self._brain

    def specialized_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行音频处理任务

        Args:
            task: 包含以下字段的字典:
                - audio_path: 音频文件路径 (必须)
                - action: 操作类型 ("transcribe" | "extract_knowledge" | "summarize")
                - prompt: 可选提示
                - language: 可选语言代码
                - topic: 可选主题 (用于知识提取)

        Returns:
            处理结果字典
        """
        audio_path = task.get("audio_path", "")
        action = task.get("action", "transcribe")
        prompt = task.get("prompt")
        language = task.get("language") or self._default_language
        topic = task.get("topic")

        if not audio_path:
            return {"status": "error", "message": "Missing audio_path in task"}

        if not Path(audio_path).exists():
            return {"status": "error", "message": f"Audio file not found: {audio_path}"}

        if action == "transcribe":
            return self._do_transcribe(audio_path, prompt, language)
        elif action == "extract_knowledge":
            return self._do_extract_knowledge(audio_path, prompt, language, topic)
        elif action == "summarize":
            return self._do_summarize(audio_path, prompt, language)
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    def _do_transcribe(
        self,
        audio_path: str,
        prompt: Optional[str],
        language: Optional[str],
    ) -> Dict[str, Any]:
        """执行音频转录"""
        brain = self._get_brain()

        transcription = brain.transcribe_audio(
            audio_path=audio_path,
            prompt=prompt,
            language=language,
            model=self._preferred_model,
        )

        if transcription:
            return {
                "status": "success",
                "transcription": transcription,
                "audio_path": audio_path,
                "action": "transcribe",
                "bee_id": self.id
            }
        else:
            return {
                "status": "error",
                "message": "Transcription failed",
                "audio_path": audio_path,
                "bee_id": self.id
            }

    def _do_extract_knowledge(
        self,
        audio_path: str,
        prompt: Optional[str],
        language: Optional[str],
        topic: Optional[str],
    ) -> Dict[str, Any]:
        """从音频中提取结构化知识"""
        # 1. 先转录
        transcription = self._do_transcribe(audio_path, prompt, language)
        if transcription.get("status") != "success":
            return transcription

        text = transcription["transcription"]

        # 2. 用 LLM 提取知识
        extract_prompt = f"""从以下转录文本中提取结构化知识。

{"主题: " + topic if topic else ""}

转录文本:
{text}

请提取并以 JSON 格式返回:
{{
  "title": "讲座/音频标题",
  "summary": "3-5句话总结内容",
  "key_points": ["要点1", "要点2", "要点3"],
  "terms": [["术语", "解释"], ...],
  "concepts": ["核心概念1", "核心概念2"],
  "action_items": ["可执行建议1", ...],
  "questions_raised": ["问题1", "问题2"]
}}

只返回 JSON，不要其他内容。"""

        brain = self._get_brain()
        response = brain.consult(
            prompt=extract_prompt,
            system_prompt="你是一个专业的知识提取专家，擅长从文本中提取结构化知识。",
            model="glm-4"  # 用聪明模型提取
        )

        if not response:
            return {
                "status": "error",
                "message": "Knowledge extraction failed",
                "transcription": text,
                "audio_path": audio_path,
                "bee_id": self.id
            }

        # 3. 解析 JSON
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                knowledge = json.loads(json_match.group())
                return {
                    "status": "success",
                    "transcription": text,
                    "knowledge": knowledge,
                    "audio_path": audio_path,
                    "bee_id": self.id
                }
        except (json.JSONDecodeError, AttributeError):
            pass

        # JSON 解析失败，返回原始文本
        return {
            "status": "partial",
            "transcription": text,
            "extraction_raw": response,
            "audio_path": audio_path,
            "bee_id": self.id
        }

    def _do_summarize(
        self,
        audio_path: str,
        prompt: Optional[str],
        language: Optional[str],
    ) -> Dict[str, Any]:
        """总结音频内容"""
        # 1. 先转录
        transcription = self._do_transcribe(audio_path, prompt, language)
        if transcription.get("status") != "success":
            return transcription

        text = transcription["transcription"]

        # 2. 用 LLM 总结
        summarize_prompt = f"""总结以下音频转录内容。

转录文本:
{text}

请提供:
1. 标题 (一句话)
2. 摘要 (3-5句话)
3. 关键要点 (3-5条)
4. 结论/收获

格式清晰，使用 Markdown。"""

        brain = self._get_brain()
        summary = brain.consult(
            prompt=summarize_prompt,
            system_prompt="你是一个专业的总结专家，擅长提炼关键信息。",
            model="glm-4"
        )

        if summary:
            return {
                "status": "success",
                "transcription": text,
                "summary": summary,
                "audio_path": audio_path,
                "bee_id": self.id
            }
        else:
            return {
                "status": "error",
                "message": "Summarization failed",
                "transcription": text,
                "audio_path": audio_path,
                "bee_id": self.id
            }

    # ===== 快捷方法 =====

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径
            language: 可选语言代码
            prompt: 可选提示

        Returns:
            包含转录文本的结果字典
        """
        return self.specialized_task({
            "audio_path": audio_path,
            "action": "transcribe",
            "language": language,
            "prompt": prompt,
        })

    def extract_knowledge(
        self,
        audio_path: str,
        topic: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从音频中提取结构化知识

        Args:
            audio_path: 音频文件路径
            topic: 可选主题
            language: 可选语言代码

        Returns:
            包含知识点、术语、总结的结果字典
        """
        return self.specialized_task({
            "audio_path": audio_path,
            "action": "extract_knowledge",
            "topic": topic,
            "language": language,
        })

    def summarize(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        总结音频内容

        Args:
            audio_path: 音频文件路径
            language: 可选语言代码

        Returns:
            包含总结的结果字典
        """
        return self.specialized_task({
            "audio_path": audio_path,
            "action": "summarize",
            "language": language,
        })


__all__ = ['AudioBee']
