# -*- coding: utf-8 -*-
import json
import requests
import time
import random
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BrainInterface:
    """大脑接口 - 连接 LLM 服务 (支持 DeepSeek, Gemini, Claude 等)"""
    
    def __init__(self, config_path: str = "brain_config.json"):
        """
        初始化大脑接口
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.provider = self.config.get("provider", "deepseek")
        self.base_url = self.config.get("base_url")
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model", "deepseek-chat")
        
        logger.info(f"🧠 大脑接口初始化: {self.provider} - {self.model}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return {}
    
    def consult(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        咨询大脑 - 带有指数退避重试机制

        Phase 1 动态模型路由支持：
        - model=None: 使用 brain_config.json 中配置的默认模型
        - model="glm-flash": 快速廉价模型（System1/简单任务）
        - model="deepseek-chat": 聪明模型（System2/复杂任务）
        - model="glm-4": 平衡模型（Hybrid）
        """
        # 0. 检查手动覆盖 (Teleoperation)
        override_path = Path("brain_override.md")
        if override_path.exists():
            content = override_path.read_text(encoding='utf-8').strip()
            if content and not content.startswith("# IGNORE"):
                logger.info("🧠 [Neural Link] 检测到手动指令，已覆盖 AI 输出")
                return content

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = None

                # Phase 1 动态模型路由：优先使用指定的模型
                effective_model = model or self.model
                logger.debug(f"🧠 Consulting model: {effective_model}")

                provider = self.provider.lower()
                if provider == "gemini":
                    result = self._consult_gemini(prompt, system_prompt, effective_model)
                elif provider in ["claude", "anthropic"]:
                    result = self._consult_claude(prompt, system_prompt, effective_model)
                elif provider == "ollama":
                    result = self._consult_ollama(prompt, system_prompt, effective_model)
                else:
                    # 默认使用 OpenAI 兼容格式 (DeepSeek, Local LLMs, etc.)
                    result = self._consult_openai_compatible(prompt, system_prompt, effective_model)

                if result:
                    self._log_activity(prompt, result)
                    return result

            except Exception as e:
                error_str = str(e).lower()
                if ("429" in error_str or "rate limit" in error_str) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt + random.random()
                    logger.warning(f"🧠 Rate limit hit (429). Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}")
                    time.sleep(wait_time)
                    continue

                logger.error(f"🧠 大脑咨询异常 (Attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return None

        return None

    def _consult_openai_compatible(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str
    ) -> Optional[str]:
        """OpenAI 兼容接口调用"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
            },
            timeout=45
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 429:
            raise Exception("HTTP 429 Too Many Requests")
        else:
            logger.error(f"OpenAI 接口报错: {response.status_code} - {response.text}")
            return None

    def _consult_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str
    ) -> Optional[str]:
        """
        Phase 2 P4: Ollama 本地模型接口

        Ollama 使用 OpenAI-compatible API，但不需要 Authorization header。
        默认地址: http://localhost:11434/v1/chat/completions
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Ollama 可能使用不同的 base_url（默认 localhost:11434）
        base = self.base_url or "http://localhost:11434"

        response = requests.post(
            f"{base}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 404:
            logger.error(f"🤖 [Ollama] 模型 {model} 未找到，请运行: ollama pull {model}")
            return None
        else:
            logger.error(f"🤖 [Ollama] 接口报错: {response.status_code} - {response.text[:200]}")
            return None

    def _consult_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str
    ) -> Optional[str]:
        """Gemini REST API 调用"""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}]
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

    def _consult_claude(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str
    ) -> Optional[str]:
        """Claude API 调用"""
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": model,
            "max_tokens": 2048,
            "system": system_prompt if system_prompt else "",
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(url, headers=headers, json=data, timeout=45)
        if response.status_code == 200:
            return response.json().get('content', [{}])[0].get('text', '')
        return None

    def _consult_vision(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        Phase 4 P1: 多模态视觉推理

        支持三种视觉provider:
        - claude/anthropic: Claude Vision (base64编码)
        - gemini: Gemini Pro Vision (inline image)
        - openai-compatible: GPT-4V (base64编码)

        Args:
            prompt: 询问内容
            image_path: 图片路径
            system_prompt: 可选系统提示
            model: 视觉模型名 (默认使用配置中的vision模型)

        Returns:
            LLM 对图片的分析结果
        """
        import base64
        import os

        effective_model = model or self.config.get("vision_model", "claude-3-sonnet-20240229")
        provider = self.provider.lower()

        # 读取并编码图片
        if not os.path.exists(image_path):
            logger.error(f"🖼️ [Vision] 图片不存在: {image_path}")
            return None

        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        if provider in ["claude", "anthropic"]:
            return self._consult_claude_vision(prompt, img_base64, system_prompt, effective_model)
        elif provider == "gemini":
            return self._consult_gemini_vision(prompt, img_base64, system_prompt, effective_model)
        else:
            # 默认使用 OpenAI 兼容格式 (支持 GPT-4V, Local LLMs 等)
            return self._consult_openai_vision(prompt, img_base64, system_prompt, effective_model)

    def _consult_claude_vision(
        self,
        prompt: str,
        img_base64: str,
        system_prompt: Optional[str],
        model: str
    ) -> Optional[str]:
        """Claude Vision API (Anthropic)"""
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_base64
                }
            },
            {"type": "text", "text": prompt}
        ]

        data = {
            "model": model,
            "max_tokens": 2048,
            "system": system_prompt or "你是一个专业的代码架构分析专家。",
            "messages": [{"role": "user", "content": content}]
        }

        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result.get('content', [{}])[0].get('text', '')
        else:
            logger.error(f"🖼️ [Claude Vision] API错误: {response.status_code} - {response.text[:200]}")
            return None

    def _consult_gemini_vision(
        self,
        prompt: str,
        img_base64: str,
        system_prompt: Optional[str],
        model: str
    ) -> Optional[str]:
        """Gemini Pro Vision API"""
        import json
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": img_base64}},
                    {"text": prompt}
                ]
            }],
            "system_instruction": {"parts": [{"text": system_prompt or ""}]} if system_prompt else None
        }
        data = {k: v for k, v in data.items() if v is not None}

        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        except Exception as e:
            logger.error(f"🖼️ [Gemini Vision] API错误: {e}")
            return None

    def _consult_openai_vision(
        self,
        prompt: str,
        img_base64: str,
        system_prompt: Optional[str],
        model: str
    ) -> Optional[str]:
        """OpenAI 兼容格式 (GPT-4V 等)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                }
            ]
        })

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 2000
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"🖼️ [OpenAI Vision] API错误: {response.status_code} - {response.text[:200]}")
            return None

    def consult_vision(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        Phase 4 P1: 视觉推理公开接口

        Args:
            prompt: 要询问的问题
            image_path: 图片文件路径
            system_prompt: 可选系统提示
            model: 可选视觉模型覆盖

        Returns:
            LLM 对图片的分析结果字符串
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self._consult_vision(prompt, image_path, system_prompt, model)
                if result:
                    return result
            except Exception as e:
                logger.error(f"🖼️ [Vision] 视觉推理异常 (Attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return None
        return None 

    SENSITIVE_PATTERNS = [
        "api_key", "apikey", "password", "secret", "token", "authorization",
        "sk-", "Bearer ", "x-api-key"
    ]

    def _sanitize_for_log(self, text: str) -> str:
        """从日志中移除敏感信息"""
        import re
        sanitized = text
        for pattern in self.SENSITIVE_PATTERNS:
            sanitized = re.sub(
                rf'({pattern}[=:]\s*)[^\s,"\']+',
                rf'\1[REDACTED]',
                sanitized,
                flags=re.IGNORECASE
            )
        return sanitized

    def _log_activity(self, prompt: str, content: str):
        """记录大脑活动（自动过滤敏感信息）"""
        try:
            log_path = Path("brain_activity.log")
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            safe_prompt = self._sanitize_for_log(prompt)
            safe_content = self._sanitize_for_log(content)
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}] PROMPT: {safe_prompt[:100]}...\n")
                f.write(f"[{timestamp}] BRAIN: {safe_content[:200]}...\n")
                f.write("-" * 20 + "\n")
        except Exception as e:
            logger.debug(f"Failed to write brain activity log: {e}")
    
    def analyze_github_trends(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        分析 GitHub 趋势
        
        Args:
            data: GitHub 数据
            
        Returns:
            分析结果
        """
        system_prompt = """你是 Hive 2.0 的智能分析专家。
你的任务是分析 GitHub 趋势数据,提供战略建议。
请以 JSON 格式返回分析结果。"""
        
        prompt = f"""分析以下 GitHub 数据并提供战略建议:

数据: {json.dumps(data, ensure_ascii=False, indent=2)}

请提供:
1. 趋势分析
2. 推荐的狩猎目标关键词 (3-5个)
3. 优先级排序
4. 风险评估

以 JSON 格式返回,包含: trends, keywords, priorities, risks"""
        
        response = self.consult(prompt, system_prompt)
        
        if response:
            try:
                # 尝试解析 JSON
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except (json.JSONDecodeError, AttributeError) as e:
                logger.debug(f"Failed to parse GitHub trends JSON: {e}")
        
        return None
    
    def evaluate_project(self, project_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        评估项目价值
        
        Args:
            project_info: 项目信息
            
        Returns:
            评估结果
        """
        system_prompt = "你是代码项目评估专家,擅长判断项目的技术价值和学习价值。"
        
        prompt = f"""评估以下项目:

项目信息: {json.dumps(project_info, ensure_ascii=False, indent=2)}

请评估:
1. 技术价值 (1-10分)
2. 学习价值 (1-10分)
3. 代码质量 (1-10分)
4. 推荐理由

以 JSON 格式返回: tech_value, learning_value, code_quality, reason"""
        
        response = self.consult(prompt, system_prompt)
        
        if response:
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except (json.JSONDecodeError, AttributeError) as e:
                logger.debug(f"Failed to parse project evaluation JSON: {e}")
        
        return None

# 全局单例
_brain_instance = None
_brain_lock = __import__('threading').Lock()

def get_brain() -> BrainInterface:
    """获取大脑接口单例"""
    global _brain_instance
    if _brain_instance is None:
        with _brain_lock:
            if _brain_instance is None:
                _brain_instance = BrainInterface()
    return _brain_instance
