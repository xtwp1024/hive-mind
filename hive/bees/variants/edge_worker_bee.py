# -*- coding: utf-8 -*-
"""
EdgeWorkerBee — 边缘智能 Worker（基于 Ollama 本地模型）

Phase 2 P4: 边缘智能试点
不依赖外部 API，本地运行 LLM 推理

特点:
- 使用 Ollama 本地模型（llama2, codellama, mistral 等）
- 适合离线/边缘计算场景
- 零网络延迟，无外部依赖
- 可作为蜂群中的"轻量级推理节点"

使用方式:
    1. 安装 Ollama: https://ollama.ai
    2. 启动服务: ollama serve
    3. 拉取模型: ollama pull llama2
    4. 配置: 设置 BRAIN_CONFIG.provider="ollama" 或使用 EdgeWorkerBee
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from hive.bees.async_base_bee import AsyncBaseBee
from hive.core.dna import BeeDNA

logger = logging.getLogger("Hive.EdgeWorker")


class EdgeWorkerBee(AsyncBaseBee):
    """
    边缘智能 Worker — 使用 Ollama 本地模型的异步 Worker

    与普通 WorkerBee 的区别:
    - 不需要外部 API Key
    - 不需要网络连接
    - 延迟更低（无网络往返）
    - 适合安全敏感场景（数据不离开本地）
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)

        # Phase 2 P4: Ollama 接口
        self._ollama = None
        self._preferred_model = self.dna.metadata.get("ollama_model", "llama2")
        self._fallback_to_brain = self.dna.metadata.get("fallback_to_brain", True)

        logger.info(f"🤖 [EdgeWorker] 初始化完成，使用模型: {self._preferred_model}")

    def _get_ollama(self):
        """懒加载 Ollama 接口（首次使用时初始化）"""
        if self._ollama is None:
            try:
                from hive.utils.ollama_interface import get_ollama
                self._ollama = get_ollama(model=self._preferred_model)
                if not self._ollama.is_available():
                    logger.warning(f"🤖 [EdgeWorker] Ollama 服务未运行 (localhost:11434)")
                    return None
                logger.info(f"🤖 [EdgeWorker] Ollama 已连接: {self._ollama.list_models()}")
            except ImportError:
                logger.error(f"🤖 [EdgeWorker] Ollama 接口未安装")
                return None
        return self._ollama

    async def specialized_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行边缘推理任务

        Args:
            task: 任务描述，至少包含 "prompt" 字段

        Returns:
            包含 "status", "result", "model_used" 的字典
        """
        prompt = task.get("prompt", "")
        system_prompt = task.get("system_prompt")
        model = task.get("model", self._preferred_model)
        temperature = task.get("temperature", 0.7)
        max_tokens = task.get("max_tokens", 2048)

        if not prompt:
            return {"status": "error", "message": "Missing prompt in task"}

        # ===== 尝试 Ollama =====
        ollama = self._get_ollama()
        if ollama:
            logger.info(f"🤖 [EdgeWorker] 使用 Ollama 本地推理 (model={model})")
            try:
                result = ollama.consult(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                if result:
                    return {
                        "status": "success",
                        "result": result,
                        "model_used": model,
                        "provider": "ollama",
                        "bee_id": self.id
                    }
            except Exception as e:
                logger.warning(f"🤖 [EdgeWorker] Ollama 推理失败: {e}")

        # ===== Fallback: 尝试 BrainInterface =====
        if self._fallback_to_brain:
            logger.info(f"🤖 [EdgeWorker] 回退到 BrainInterface...")
            try:
                from hive.utils.brain_interface import get_brain
                brain = get_brain()
                if brain:
                    result = brain.consult(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model if model != self._preferred_model else None
                    )
                    if result:
                        return {
                            "status": "success",
                            "result": result,
                            "model_used": model,
                            "provider": "brain",
                            "bee_id": self.id,
                            "fallback": True
                        }
            except Exception as e:
                logger.error(f"🤖 [EdgeWorker] BrainInterface 回退也失败: {e}")

        return {
            "status": "error",
            "message": "Both Ollama and BrainInterface unavailable",
            "bee_id": self.id
        }

    # ===== 快速检查接口 =====

    @staticmethod
    def check_ollama_status() -> Dict[str, Any]:
        """
        静态方法：检查 Ollama 服务状态

        Returns:
            包含 is_running, available_models 的字典
        """
        try:
            from hive.utils.ollama_interface import is_ollama_running, get_ollama
            running = is_ollama_running()
            if running:
                ollama = get_ollama()
                models = ollama.list_models()
                return {"is_running": True, "available_models": models}
            return {"is_running": False, "available_models": []}
        except Exception as e:
            return {"is_running": False, "error": str(e)}


__all__ = ['EdgeWorkerBee']
