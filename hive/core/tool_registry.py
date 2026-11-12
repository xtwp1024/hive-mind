# -*- coding: utf-8 -*-
"""
BeeToolRegistry - 工蜂工具注册中心

借鉴 LangChain @tool 装饰器模式：
- 任何函数加 @bee_tool 后自动注册到全局工具池
- 工蜂可通过 BeeToolRegistry.invoke(name, **kwargs) 跨蜂种调用能力
- 支持工具描述、参数类型提示，便于 LLM 理解可用能力
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("Hive.ToolRegistry")


class ToolDefinition:
    """工具定义"""

    def __init__(self, name: str, description: str, fn: Callable):
        self.name = name
        self.description = description
        self.fn = fn
        self.call_count = 0

    def invoke(self, **kwargs) -> Any:
        """调用工具"""
        self.call_count += 1
        return self.fn(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """转为可描述字典（供 LLM 读取）"""
        import inspect

        sig = inspect.signature(self.fn)
        params = {
            name: str(p.annotation) if p.annotation != inspect.Parameter.empty else "any"
            for name, p in sig.parameters.items()
        }
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
            "call_count": self.call_count,
        }


class _BeeToolRegistry:
    """全局工具注册中心（单例）"""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, fn: Callable) -> None:
        """注册工具"""
        if name in self._tools:
            logger.debug(f"⚙️ [ToolRegistry] 覆盖注册: {name}")
        self._tools[name] = ToolDefinition(name, description, fn)
        logger.info(f"🔧 [ToolRegistry] 注册: {name}")

    def invoke(self, name: str, **kwargs) -> Any:
        """调用工具"""
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"工具未注册: {name}，可用工具: {list(self._tools.keys())}")
        logger.debug(f"⚙️ [ToolRegistry] 调用: {name}")
        return tool.invoke(**kwargs)

    def list_tools(self) -> list[Dict[str, Any]]:
        """列出所有注册工具（供 LLM 规划使用）"""
        return [t.to_dict() for t in self._tools.values()]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def __len__(self):
        return len(self._tools)


# 全局单例
BeeToolRegistry = _BeeToolRegistry()


def bee_tool(name: str, description: str) -> Callable:
    """
    @bee_tool 装饰器 - 将函数注册为全局可调用工具

    用法：
        @bee_tool(name="分析JSON", description="解析JSON文件并返回结构摘要")
        def analyze_json(file_path: str) -> dict:
            ...

        # 其他工蜂调用：
        result = BeeToolRegistry.invoke("分析JSON", file_path="data.json")
    """

    def decorator(fn: Callable) -> Callable:
        BeeToolRegistry.register(name, description, fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator
