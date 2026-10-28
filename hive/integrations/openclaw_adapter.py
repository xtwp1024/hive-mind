#!/usr/bin/env python3
"""
OpenClaw Adapter for Hive Mind
-------------------------------
让 Hive 系统能够与 OpenClaw 通信的适配器模块。

功能:
- 发送消息到 OpenClaw
- 调用 OpenClaw 技能
- 接收 OpenClaw 指令
- 共享数据存储
"""

import os
import json
import requests
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpenClawConfig:
    """OpenClaw 连接配置"""
    base_url: str = "http://localhost:18789"
    token: str = ""
    gateway_token: Optional[str] = None
    timeout: int = 60
    retry_count: int = 3
    
    @classmethod
    def from_env(cls) -> "OpenClawConfig":
        """从环境变量加载配置"""
        return cls(
            base_url=os.getenv("OPENCLAW_URL", "http://localhost:18789"),
            token=os.getenv("OPENCLAW_TOKEN", ""),
            gateway_token=os.getenv("OPENCLAW_GATEWAY_TOKEN"),
            timeout=int(os.getenv("OPENCLAW_TIMEOUT", "60")),
            retry_count=int(os.getenv("OPENCLAW_RETRY", "3"))
        )


class OpenClawAdapter:
    """
    Hive ↔ OpenClaw 双向适配器
    """
    
    def __init__(self, config: Optional[OpenClawConfig] = None):
        self.config = config or OpenClawConfig.from_env()
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        if self.config.token:
            self.session.headers["Authorization"] = f"Bearer {self.config.token}"
        
        if self.config.gateway_token:
            self.session.headers["X-Gateway-Token"] = self.config.gateway_token
        
        logger.info(f"OpenClawAdapter initialized: {self.config.base_url}")
    
    # ═══════════════════════════════════════════════════════════
    # 基础通信方法
    # ═══════════════════════════════════════════════════════════
    
    def send_message(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        发送消息到 OpenClaw
        
        Args:
            message: 消息内容
            context: 附加上下文
            
        Returns:
            OpenClaw 的响应
        """
        payload = {
            "message": message,
            "source": "hive",
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        return self._post("/api/message", payload)
    
    def query_skill(self, skill_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        查询/调用 OpenClaw 的某个技能
        
        Args:
            skill_name: 技能名称 (如 "crypto-price-tracker")
            params: 技能参数
            
        Returns:
            技能执行结果
        """
        payload = {
            "action": "skill_query",
            "skill": skill_name,
            "params": params,
            "source": "hive"
        }
        
        return self._post("/api/skill", payload)
    
    # ═══════════════════════════════════════════════════════════
    # 特定技能快捷调用
    # ═══════════════════════════════════════════════════════════
    
    def get_crypto_price(self, symbol: str) -> Dict[str, Any]:
        """获取加密货币价格"""
        return self.query_skill("crypto-price-tracker", {
            "action": "get_price",
            "symbol": symbol
        })
    
    def search_web(self, query: str, count: int = 5) -> Dict[str, Any]:
        """网络搜索"""
        return self.query_skill("web-search", {
            "action": "search",
            "query": query,
            "count": count
        })
    
    def analyze_market(self, symbol: str, indicators: List[str] = None) -> Dict[str, Any]:
        """技术分析"""
        return self.query_skill("crypto-trading", {
            "action": "technical_analysis",
            "symbol": symbol,
            "indicators": indicators or ["rsi", "macd", "bollinger"]
        })
    
    def execute_shell(self, command: str, timeout: int = 60) -> Dict[str, Any]:
        """执行 shell 命令 (谨慎使用)"""
        return self.query_skill("system-executor", {
            "action": "exec",
            "command": command,
            "timeout": timeout
        })
    
    # ═══════════════════════════════════════════════════════════
    # 文件和数据共享
    # ═══════════════════════════════════════════════════════════
    
    def write_shared_data(self, key: str, data: Any) -> bool:
        r"""
        写入共享数据存储

        数据保存到双方都可以访问的位置:
        D:\Hive_Standalone\Data\shared\
        """
        shared_dir = Path("D:/Hive_Standalone/Data/shared")
        shared_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = shared_dir / f"{key}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                    "source": "hive"
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Shared data written: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to write shared data: {e}")
            return False
    
    def read_shared_data(self, key: str) -> Optional[Dict]:
        """读取共享数据"""
        filepath = Path(f"D:/Hive_Standalone/Data/shared/{key}.json")
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read shared data: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════
    
    def _post(self, endpoint: str, payload: Dict, retry: int = 0) -> Dict[str, Any]:
        """POST 请求 (带重试)"""
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            if retry < self.config.retry_count:
                logger.warning(f"Connection failed, retrying ({retry + 1}/{self.config.retry_count})...")
                import time
                time.sleep(2 ** retry)  # 指数退避（同步睡眠）
                return self._post(endpoint, payload, retry + 1)
            raise
            
        except requests.exceptions.Timeout:
            raise Exception(f"Request to OpenClaw timed out after {self.config.timeout}s")
            
        except requests.exceptions.HTTPError as e:
            raise Exception(f"OpenClaw returned error: {e.response.status_code} - {e.response.text}")
    
    def _get(self, endpoint: str) -> Dict[str, Any]:
        """GET 请求"""
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"GET request failed: {e}")
            raise


class HiveOpenClawBridge:
    """
    高级桥接器: 让 Hive Bee 能够像使用本地功能一样使用 OpenClaw
    """
    
    def __init__(self):
        self.adapter = OpenClawAdapter()
        self.capabilities = {
            "web_search": self.adapter.search_web,
            "price_query": self.adapter.get_crypto_price,
            "market_analysis": self.adapter.analyze_market,
            "shell_exec": self.adapter.execute_shell,
            "data_share": self._share_data_handler
        }
    
    def _share_data_handler(self, action: str, key: str, data: Any = None):
        """数据共享处理"""
        if action == "write":
            return self.adapter.write_shared_data(key, data)
        elif action == "read":
            return self.adapter.read_shared_data(key)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def execute(self, capability: str, **kwargs) -> Any:
        """
        执行 OpenClaw 能力
        
        示例:
            bridge = HiveOpenClawBridge()
            
            # 搜索
            results = bridge.execute("web_search", query="Bitcoin news")
            
            # 查价格
            price = bridge.execute("price_query", symbol="bitcoin")
            
            # 写共享数据
            bridge.execute("data_share", action="write", key="analysis", data={...})
        """
        if capability not in self.capabilities:
            raise ValueError(f"Unknown capability: {capability}. Available: {list(self.capabilities.keys())}")
        
        handler = self.capabilities[capability]
        return handler(**kwargs)


# ═══════════════════════════════════════════════════════════
# 快捷函数 (供 Hive Bees 直接调用)
# ═══════════════════════════════════════════════════════════

_adapters = {}
_adapters_lock = __import__('threading').Lock()

def get_adapter() -> OpenClawAdapter:
    """获取全局适配器实例 (单例，线程安全)"""
    if "default" not in _adapters:
        with _adapters_lock:
            if "default" not in _adapters:
                _adapters["default"] = OpenClawAdapter()
    return _adapters["default"]


def ask_openclaw(message: str, context: Optional[Dict] = None) -> str:
    """
    向 OpenClaw 提问的快捷函数
    
    示例:
        response = ask_openclaw("分析 BTC 当前走势")
    """
    adapter = get_adapter()
    result = adapter.send_message(message, context)
    return result.get("response", "")


def get_price(symbol: str) -> Dict:
    """获取加密货币价格快捷函数"""
    adapter = get_adapter()
    return adapter.get_crypto_price(symbol)


def search(query: str, count: int = 5) -> List[Dict]:
    """网络搜索快捷函数"""
    adapter = get_adapter()
    result = adapter.search_web(query, count)
    return result.get("results", [])


# ═══════════════════════════════════════════════════════════
# 测试代码
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 测试适配器
    print("🧪 Testing OpenClaw Adapter...")
    
    try:
        adapter = OpenClawAdapter()
        
        # 测试 1: 发送消息
        print("\n1. Sending message...")
        response = adapter.send_message("Hello from Hive! 👋")
        print(f"Response: {response}")
        
        # 测试 2: 查询价格
        print("\n2. Querying BTC price...")
        price_data = adapter.get_crypto_price("bitcoin")
        print(f"BTC Price: {price_data}")
        
        # 测试 3: 搜索
        print("\n3. Searching...")
        search_results = adapter.search_web("OpenClaw", count=3)
        print(f"Found {len(search_results.get('results', []))} results")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
