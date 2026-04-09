# -*- coding: utf-8 -*-
"""
VisionBee — 视觉理解专家

Phase 4 P1: 视觉理解
不依赖外部 API，本地运行 LLM 推理

职责:
1. 分析截图/UI界面，理解用户界面结构
2. 解析架构图，提取组件和连接关系
3. 描述图片内容，辅助认知决策
4. 支持 Claude Vision / Gemini Pro Vision / GPT-4V

使用方式:
    vision_bee = VisionBee(dna, "VisionBee-001", result_queue)
    result = vision_bee.analyze_screenshot("/path/to/screenshot.png", "描述界面布局")
    result = vision_bee.analyze_architecture("/path/to/diagram.png", "提取组件和关系")
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA

logger = logging.getLogger("Hive.Vision")


class VisionBee(BaseBee):
    """
    VisionBee — 视觉理解专家

    与普通 WorkerBee 的区别:
    - 输入是图片而非文字
    - 使用多模态 LLM (Claude Vision / Gemini Pro Vision / GPT-4V)
    - 输出是图片的结构化描述
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)

        # Phase 4 P1: 视觉模型配置
        self._preferred_model = self.dna.metadata.get("vision_model", None)
        self._default_system = self.dna.metadata.get(
            "vision_system_prompt",
            "你是一个专业的视觉分析专家，擅长分析截图、架构图、UI界面，输出简洁的结构化描述。"
        )

        logger.info(f"🖼️ [Vision] VisionBee 已初始化，模型: {self._preferred_model or 'auto'}")

    def specialized_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行视觉分析任务

        Args:
            task: 包含以下字段的字典:
                - image_path: 图片路径 (必须)
                - prompt: 询问内容 (必须)
                - system_prompt: 可选系统提示
                - model: 可选模型覆盖
                - task_type: 任务类型 ("screenshot" | "architecture" | "general")

        Returns:
            包含 "status", "analysis", "model_used" 的字典
        """
        image_path = task.get("image_path", "")
        prompt = task.get("prompt", "")
        system_prompt = task.get("system_prompt") or self._default_system
        model = task.get("model") or self._preferred_model
        task_type = task.get("task_type", "general")

        if not image_path:
            return {"status": "error", "message": "Missing image_path in task"}

        if not prompt:
            return {"status": "error", "message": "Missing prompt in task"}

        # 验证图片存在
        if not Path(image_path).exists():
            return {"status": "error", "message": f"Image not found: {image_path}"}

        # 根据任务类型生成专门的 prompt
        specialized_prompt = self._build_prompt(prompt, task_type)

        # 调用视觉推理
        analysis = self._analyze_image(
            image_path=image_path,
            prompt=specialized_prompt,
            system_prompt=system_prompt,
            model=model,
        )

        if analysis:
            return {
                "status": "success",
                "analysis": analysis,
                "image_path": image_path,
                "task_type": task_type,
                "model_used": model or "auto",
                "bee_id": self.id
            }
        else:
            return {
                "status": "error",
                "message": "Vision analysis failed",
                "image_path": image_path,
                "bee_id": self.id
            }

    def _build_prompt(self, prompt: str, task_type: str) -> str:
        """根据任务类型增强 prompt"""
        if task_type == "screenshot":
            return f"""分析以下截图图片。

问题: {prompt}

请描述:
1. 界面的整体布局和结构
2. 主要的交互元素（按钮、表单、导航等）
3. 界面的视觉层级和信息架构
4. 任何值得注意的设计模式或UX问题

回答格式: 结构化描述，突出关键发现。"""
        elif task_type == "architecture":
            return f"""分析以下架构图或流程图。

问题: {prompt}

请提取:
1. 所有的组件/节点（用列表）
2. 组件之间的关系/连接（用箭头或线条描述）
3. 数据流向（如果适用）
4. 任何缺失的组件或建议的改进

回答格式: 结构化列表，便于后续代码生成使用。"""
        else:
            return prompt

    def _analyze_image(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str,
        model: Optional[str],
    ) -> Optional[str]:
        """调用 BrainInterface 的视觉推理能力"""
        try:
            from hive.utils.brain_interface import get_brain
            brain = get_brain()

            # 优先尝试 consult_vision 方法
            if hasattr(brain, 'consult_vision'):
                result = brain.consult_vision(
                    prompt=prompt,
                    image_path=image_path,
                    system_prompt=system_prompt,
                    model=model,
                )
                if result:
                    return result

            # Fallback: 如果 BrainInterface 没有 vision 方法，尝试其他方式
            logger.warning(f"🖼️ [Vision] BrainInterface 不支持 consult_vision，尝试 Ollama...")
            return self._analyze_image_ollama(image_path, prompt, model)

        except Exception as e:
            logger.error(f"🖼️ [Vision] 视觉分析失败: {e}")
            return None

    def _analyze_image_ollama(
        self,
        image_path: str,
        prompt: str,
        model: Optional[str],
    ) -> Optional[str]:
        """Ollama 多模态模型备选 (如 llava)"""
        try:
            from hive.utils.ollama_interface import get_ollama
            ollama = get_ollama()
            if not ollama or not ollama.is_available():
                logger.warning(f"🖼️ [Vision] Ollama 不可用")
                return None

            # Ollama 多模态模型 (llava): 通过 consult 的 images 参数
            import base64
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")

            effective_model = model or "llava"
            result = ollama.consult(
                prompt=prompt,
                model=effective_model,
                images=[img_base64],
            )
            return result
        except Exception as e:
            logger.error(f"🖼️ [Vision] Ollama 视觉分析失败: {e}")
            return None

    # ===== 快捷分析方法 =====

    def analyze_screenshot(self, image_path: str, question: str = "描述界面布局") -> Dict[str, Any]:
        """
        分析截图

        Args:
            image_path: 截图路径
            question: 要询问的问题

        Returns:
            分析结果字典
        """
        return self.specialized_task({
            "image_path": image_path,
            "prompt": question,
            "task_type": "screenshot",
        })

    def analyze_architecture(self, image_path: str, question: str = "提取组件和关系") -> Dict[str, Any]:
        """
        分析架构图

        Args:
            image_path: 架构图路径
            question: 要询问的问题

        Returns:
            分析结果字典
        """
        return self.specialized_task({
            "image_path": image_path,
            "prompt": question,
            "task_type": "architecture",
        })

    def describe_image(self, image_path: str, focus: str = "") -> Dict[str, Any]:
        """
        通用图片描述

        Args:
            image_path: 图片路径
            focus: 关注的重点

        Returns:
            分析结果字典
        """
        prompt = f"描述这张图片的详细内容。{focus}" if focus else "详细描述这张图片的内容。"
        return self.specialized_task({
            "image_path": image_path,
            "prompt": prompt,
            "task_type": "general",
        })


__all__ = ['VisionBee']
