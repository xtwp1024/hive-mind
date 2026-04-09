# -*- coding: utf-8 -*-
"""
MultimodalBee — 多模态融合专家

Phase 4 P3: 多模态融合
统一调度文本、图像、音频三种模态的输入，输出综合分析结果。

职责:
1. 接收包含多种模态的复杂任务
2. 并行调用 VisionBee / AudioBee / 文本处理
3. 综合各模态结果，输出统一分析
4. 支持 Cross-Modal Reasoning (跨模态推理)

使用方式:
    mm = MultimodalBee(dna, "MultimodalBee-001", result_queue)
    result = mm.analyze(task={
        "prompt": "分析这段技术讲座视频截图",
        "images": ["/path/to/screenshot.png"],
        "audio": "/path/to/lecture.mp3",
    })
"""

import logging
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA

logger = logging.getLogger("Hive.Multimodal")


class MultimodalBee(BaseBee):
    """
    MultimodalBee — 多模态融合专家

    与单一模态专家的区别:
    - 输入可以是 text + image + audio 的任意组合
    - 自动识别模态类型并调度相应处理
    - 输出是跨模态综合分析，而非单一模态结果
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)

        # Phase 4 P3: 多模态配置
        self._brain = None  # 懒加载

        # 内置子专家引用 (实际使用时可替换为真实实例)
        self._vision_model = self.dna.metadata.get("vision_model", None)
        self._audio_model = self.dna.metadata.get("audio_model", None)
        self._multimodal_model = self.dna.metadata.get("multimodal_model", None)

        logger.info(f"🔮 [Multimodal] MultimodalBee 已初始化")

    def _get_brain(self):
        """懒加载 BrainInterface"""
        if self._brain is None:
            from hive.utils.brain_interface import get_brain
            self._brain = get_brain()
        return self._brain

    def specialized_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行多模态分析任务

        Args:
            task: 包含以下字段的字典:
                - prompt: 主要询问 (必须)
                - text: 可选文本片段
                - images: 可选图片路径列表
                - audio: 可选音频路径
                - mode: 分析模式 ("unified" | "sequential" | "parallel")
                    - unified: 直接用 consult_multimodal 统一处理
                    - sequential: 先各模态处理，再综合
                    - parallel: 并行处理各模态 (如果支持)

        Returns:
            多模态综合分析结果
        """
        prompt = task.get("prompt", "")
        text = task.get("text")
        images = task.get("images", [])
        audio = task.get("audio")
        mode = task.get("mode", "unified")

        if not prompt and not text and not images and not audio:
            return {"status": "error", "message": "At least one input required (prompt/text/images/audio)"}

        if mode == "unified":
            return self._do_unified_analysis(prompt, text, images, audio)
        elif mode == "sequential":
            return self._do_sequential_analysis(prompt, text, images, audio)
        elif mode == "parallel":
            return self._do_parallel_analysis(prompt, text, images, audio)
        else:
            return {"status": "error", "message": f"Unknown mode: {mode}"}

    def _do_unified_analysis(
        self,
        prompt: str,
        text: Optional[str],
        images: Optional[list],
        audio: Optional[str]
    ) -> Dict[str, Any]:
        """统一多模态分析 (单次 LLM 调用)"""
        brain = self._get_brain()

        if not hasattr(brain, 'consult_multimodal'):
            logger.warning(f"🔮 [Multimodal] BrainInterface 不支持 consult_multimodal，降级为 sequential")
            return self._do_sequential_analysis(prompt, text, images, audio)

        result = brain.consult_multimodal(
            prompt=prompt,
            text=text,
            images=images,
            audio_path=audio,
            system_prompt="你是一个专业的多模态分析专家，擅长综合分析文本、图像、音频等多种模态的信息。",
            model=self._multimodal_model,
        )

        if result:
            return {
                "status": "success",
                "analysis": result,
                "mode": "unified",
                "modalities": self._count_modalities(text, images, audio),
                "bee_id": self.id
            }
        else:
            return {
                "status": "error",
                "message": "Multimodal analysis failed",
                "mode": "unified",
                "bee_id": self.id
            }

    def _do_sequential_analysis(
        self,
        prompt: str,
        text: Optional[str],
        images: Optional[list],
        audio: Optional[str]
    ) -> Dict[str, Any]:
        """顺序多模态分析 (各模态依次处理，再综合)"""
        modality_results = {}
        all_failed = True

        # 1. 图像分析
        if images:
            vision_results = []
            for img_path in images:
                r = self._analyze_single_image(img_path, prompt)
                if r:
                    vision_results.append(r)
                    all_failed = False
            modality_results["images"] = vision_results

        # 2. 音频处理
        if audio:
            r = self._transcribe_audio(audio)
            if r:
                modality_results["audio"] = r
                all_failed = False

        # 3. 文本直接使用
        if text:
            modality_results["text"] = text
            all_failed = False

        if all_failed:
            return {
                "status": "error",
                "message": "All modalities failed",
                "bee_id": self.id
            }

        # 4. 综合分析
        synthesis = self._synthesize_results(prompt, modality_results)
        return {
            "status": "success",
            "analysis": synthesis,
            "modality_results": modality_results,
            "mode": "sequential",
            "modalities": self._count_modalities(text, images, audio),
            "bee_id": self.id
        }

    def _do_parallel_analysis(
        self,
        prompt: str,
        text: Optional[str],
        images: Optional[list],
        audio: Optional[str]
    ) -> Dict[str, Any]:
        """并行多模态分析 (各模态同时处理)"""
        import concurrent.futures

        modality_results = {}
        futures = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # 图像分析
            if images:
                for i, img_path in enumerate(images):
                    f = executor.submit(self._analyze_single_image, img_path, prompt)
                    futures[f"image_{i}"] = f

            # 音频处理
            if audio:
                f = executor.submit(self._transcribe_audio, audio)
                futures["audio"] = f

            # 等待所有完成
            for name, future in futures.items():
                try:
                    result = future.result(timeout=30)
                    modality_results[name] = result
                except Exception as e:
                    logger.error(f"🔮 [Multimodal] {name} 处理失败: {e}")
                    modality_results[name] = None

        # 文本
        if text:
            modality_results["text"] = text

        if not modality_results:
            return {
                "status": "error",
                "message": "All modalities failed",
                "bee_id": self.id
            }

        # 综合分析
        synthesis = self._synthesize_results(prompt, modality_results)
        return {
            "status": "success",
            "analysis": synthesis,
            "modality_results": modality_results,
            "mode": "parallel",
            "modalities": self._count_modalities(text, images, audio),
            "bee_id": self.id
        }

    def _analyze_single_image(self, image_path: str, question: str) -> Optional[str]:
        """分析单张图片"""
        brain = self._get_brain()
        if hasattr(brain, 'consult_vision'):
            return brain.consult_vision(
                prompt=question,
                image_path=image_path,
                system_prompt="你是一个专业的视觉分析专家。",
                model=self._vision_model,
            )
        return None

    def _transcribe_audio(self, audio_path: str) -> Optional[str]:
        """转录音频"""
        brain = self._get_brain()
        if hasattr(brain, 'transcribe_audio'):
            return brain.transcribe_audio(
                audio_path=audio_path,
                model=self._audio_model,
            )
        return None

    def _synthesize_results(self, prompt: str, modality_results: Dict[str, Any]) -> str:
        """综合各模态结果"""
        brain = self._get_brain()

        # 构建上下文
        context_parts = []
        for modality, result in modality_results.items():
            if result is None:
                continue
            if modality.startswith("image"):
                context_parts.append(f"[图像分析]\n{result}")
            elif modality == "audio":
                context_parts.append(f"[音频转录]\n{result}")
            elif modality == "text":
                context_parts.append(f"[文本内容]\n{result}")

        if not context_parts:
            return "无法获取任何模态的分析结果。"

        combined = "\n\n".join(context_parts)

        synthesis_prompt = f"""基于以下多模态信息，回答用户问题。

用户问题: {prompt}

{"="*60}
{combined}
{"="*60}

请综合以上所有信息，给出全面、连贯的回答。如果某些模态的信息不足以回答问题，请明确指出。"""

        response = brain.consult(
            prompt=synthesis_prompt,
            system_prompt="你是一个专业的多模态综合分析专家，擅长将不同来源的信息整合为连贯的分析。",
            model="glm-4"
        )
        return response or "综合分析失败。"

    def _count_modalities(self, text, images, audio) -> int:
        """统计使用的模态数量"""
        count = 0
        if text:
            count += 1
        if images:
            count += len(images)
        if audio:
            count += 1
        return count

    # ===== 快捷方法 =====

    def analyze(
        self,
        prompt: str,
        text: Optional[str] = None,
        images: Optional[list[str]] = None,
        audio: Optional[str] = None,
        mode: str = "unified"
    ) -> Dict[str, Any]:
        """
        统一分析接口

        Args:
            prompt: 主要询问
            text: 可选文本
            images: 可选图片列表
            audio: 可选音频路径
            mode: "unified" | "sequential" | "parallel"

        Returns:
            多模态分析结果
        """
        return self.specialized_task({
            "prompt": prompt,
            "text": text,
            "images": images or [],
            "audio": audio,
            "mode": mode,
        })

    def analyze_image_text(
        self,
        prompt: str,
        image_path: str,
        text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析图片+文本组合

        Args:
            prompt: 询问
            image_path: 图片路径
            text: 可选文本

        Returns:
            分析结果
        """
        return self.analyze(prompt=prompt, text=text, images=[image_path])

    def analyze_video_frame(
        self,
        prompt: str,
        frame_paths: list[str],
        transcript: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析视频帧序列 (截图+音频转录)

        Args:
            prompt: 询问
            frame_paths: 视频帧截图路径列表
            transcript: 音频转录文本

        Returns:
            分析结果
        """
        return self.analyze(
            prompt=prompt,
            text=transcript,
            images=frame_paths,
            mode="sequential"
        )


__all__ = ['MultimodalBee']
