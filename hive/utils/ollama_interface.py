# -*- coding: utf-8 -*-
"""
Ollama Interface — 本地大模型推理接口

Phase 2 P4: 边缘智能试点
支持本地部署的 LLM（不依赖外部 API）

API 格式: OpenAI-compatible (http://localhost:11434/v1/chat/completions)
支持模型: llama2, codellama, mistral, mixtral, qwen, deepseek 等
"""

import logging
import requests
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger("Hive.Ollama")


class OllamaInterface:
    """
    Ollama 本地大模型接口

    特点:
    - 本地运行，无需网络和外部 API
    - 支持 llama2, codellama, mistral 等多种模型
    - OpenAI-compatible API 格式
    - 适合边缘计算、低延迟场景
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama2"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        初始化 Ollama 接口

        Args:
            base_url: Ollama 服务地址，默认 http://localhost:11434
            model: 默认模型名，默认 llama2
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        # 缓存可用模型列表（避免每次都请求）
        self._models_cache: Optional[List[str]] = None
        self._models_cache_time: float = 0
        self._models_cache_ttl: float = 300  # 5分钟缓存

        logger.info(f"🤖 [Ollama] 初始化完成: {self.base_url}, 默认模型: {self.model}")

    def is_available(self) -> bool:
        """
        检查 Ollama 服务是否可用

        Returns:
            True 如果服务正在运行
        """
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self, force_refresh: bool = False) -> List[str]:
        """
        获取本地已安装的模型列表

        Args:
            force_refresh: 强制刷新缓存

        Returns:
            模型名称列表
        """
        now = time.time()
        if not force_refresh and self._models_cache and (now - self._models_cache_time) < self._models_cache_ttl:
            return self._models_cache

        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                self._models_cache = models
                self._models_cache_time = now
                logger.info(f"🤖 [Ollama] 发现 {len(models)} 个本地模型: {models}")
                return models
        except Exception as e:
            logger.error(f"🤖 [Ollama] 获取模型列表失败: {e}")

        return self._models_cache or []

    def pull_model(self, model: str, timeout: float = 300.0) -> bool:
        """
        拉取模型（如果本地没有）

        Args:
            model: 模型名称
            timeout: 拉取超时时间（秒）

        Returns:
            True 如果成功
        """
        logger.info(f"🤖 [Ollama] 正在拉取模型: {model}")

        try:
            # Ollama 拉取是流式响应，我们用一个较长的超时
            with self._session.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                stream=True,
                timeout=timeout
            ) as response:
                if response.status_code == 200:
                    # 流式读取（简单实现：等待完成）
                    for line in response.iter_lines():
                        if line:
                            import json
                            try:
                                data = json.loads(line)
                                status = data.get("status", "")
                                if status == "success":
                                    logger.info(f"✅ [Ollama] 模型 {model} 拉取完成")
                                    self._models_cache = None  # 使缓存失效
                                    return True
                                elif status:
                                    logger.debug(f"🤖 [Ollama] 拉取进度: {status}")
                            except json.JSONDecodeError:
                                pass
                    return True
                else:
                    logger.error(f"🤖 [Ollama] 拉取失败: {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"🤖 [Ollama] 拉取异常: {e}")
            return False

    def consult(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Optional[str]:
        """
        向 Ollama 模型咨询（OpenAI-compatible 格式）

        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            model: 模型名（可选，默认使用 self.model）
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数（stop, context 等）

        Returns:
            模型响应文本，失败返回 None
        """
        effective_model = model or self.model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]

                elif response.status_code == 404:
                    # 模型不存在，尝试自动拉取
                    logger.warning(f"🤖 [Ollama] 模型 {effective_model} 不存在，尝试自动拉取...")
                    if self.pull_model(effective_model):
                        continue  # 重试
                    return None

                elif response.status_code == 500:
                    logger.warning(f"🤖 [Ollama] 服务端错误 (500)，重试 {attempt + 1}/{max_retries}")
                    time.sleep(2 ** attempt)
                    continue

                else:
                    logger.error(f"🤖 [Ollama] 请求失败: {response.status_code} - {response.text[:200]}")
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"🤖 [Ollama] 请求超时，重试 {attempt + 1}/{max_retries}")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"🤖 [Ollama] 请求异常: {e}")
                if attempt == max_retries - 1:
                    return None

        logger.error(f"🤖 [Ollama] {max_retries} 次重试后失败")
        return None

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Optional[str]:
        """
        使用 /api/generate 端点（非 chat 格式）

        适用于不支持 chat API 的模型
        """
        effective_model = model or self.model

        payload = {
            "model": effective_model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_prompt,
            **kwargs
        }

        try:
            response = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json().get("response")
            else:
                logger.error(f"🤖 [Ollama] generate 失败: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"🤖 [Ollama] generate 异常: {e}")
            return None

    def embedding(self, text: str, model: Optional[str] = None) -> Optional[List[float]]:
        """
        获取文本的 embedding 向量

        Args:
            text: 文本
            model: 模型名（需要支持 embedding 的模型，如 nomic-embed-text）

        Returns:
            embedding 向量列表，失败返回 None
        """
        effective_model = model or "nomic-embed-text"

        try:
            response = self._session.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": effective_model, "input": text},
                timeout=30
            )

            if response.status_code == 200:
                return response.json().get("data", [{}])[0].get("embedding")
            else:
                logger.error(f"🤖 [Ollama] embedding 失败: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"🤖 [Ollama] embedding 异常: {e}")
            return None

    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        获取模型信息

        Args:
            model: 模型名（默认使用 self.model）

        Returns:
            模型信息字典
        """
        effective_model = model or self.model

        try:
            response = self._session.get(
                f"{self.base_url}/api/show",
                params={"name": effective_model},
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {}

        except Exception:
            return {}

    # ===== Phase 2 P4: 与 BrainInterface 集成 =====

    @classmethod
    def as_brain_provider(cls) -> Dict[str, Any]:
        """
        Phase 2 P4: 返回作为 BrainInterface provider 的配置格式

        用法:
            # 在 brain_config.json 中添加 ollama provider
            # 或者在 BrainInterface 中动态添加
        """
        return {
            "provider": "ollama",
            "base_url": cls.DEFAULT_BASE_URL,
            "model": cls.DEFAULT_MODEL,
        }


# 全局单例
_ollama_instance: Optional[OllamaInterface] = None
_ollama_lock = __import__('threading').Lock()


def get_ollama(
    base_url: Optional[str] = None,
    model: Optional[str] = None
) -> OllamaInterface:
    """
    获取 Ollama 接口单例（线程安全）

    Args:
        base_url: Ollama 服务地址
        model: 默认模型名
    """
    global _ollama_instance
    if _ollama_instance is None:
        with _ollama_lock:
            if _ollama_instance is None:
                _ollama_instance = OllamaInterface(base_url=base_url, model=model)
    return _ollama_instance


def is_ollama_running() -> bool:
    """快速检查 Ollama 是否在运行"""
    try:
        import requests
        response = requests.get(f"{OllamaInterface.DEFAULT_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False
