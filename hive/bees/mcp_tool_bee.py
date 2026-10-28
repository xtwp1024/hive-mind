# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
from hive.utils.mcp_client import MCPManager
import os
from pathlib import Path

logger = get_logger(__name__)

# 全局惰性单例
_mcp_manager = None

def get_mcp_manager() -> MCPManager:
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
        # 尝试加载默认的 mcp_servers.json
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "hive" / "config" / "mcp_servers.json"
        _mcp_manager.load_servers(str(config_path))
    return _mcp_manager

class MCPToolBee(BaseBee):
    """
    MCPToolBee (外接工具蜂)
    
    专门负责对接系统中定义好的外部 MCP Server。它就像是一个万能适配器。
    如果 DNA.metadata 里面指定了 `mcp_server` 与 `tool_name`，它就会向外委派任务。
    """
    
    species_name = "mcp_tool"
    description = "通过 MCP (Model Context Protocol) 协议调用外部企业工具或本地代理服务"
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        
    def specialized_task(self, target_path) -> dict:
        """解析 DNA 中的指令并使用 MCPClient 执行"""
        meta = self.dna.metadata or {}
        target_server = meta.get("mcp_server", "")
        tool_name = meta.get("tool_name", "")
        arguments = meta.get("arguments", {})
        
        if not target_server or not tool_name:
            error_msg = f"未提供 MCP Server 名称或 Tool 名称。Meta: {meta}"
            logger.error(error_msg)
            return {"status": "FAILED", "error": error_msg}
            
        logger.info(f"🛠️ MCP 通信尝试: 正在通过 [{target_server}] 调用工具 -> {tool_name}")
        
        manager = get_mcp_manager()
        client = manager.get_client(target_server)
        
        if not client:
            return {"status": "FAILED", "error": f"未注册或未找到该名称的 MCP Server: {target_server}"}
            
        try:
            # 执行 MCP 协议调用
            result = client.call_tool(target_server, tool_name, arguments)
            
            if "error" in result:
                logger.error(f"❌ 工具 {tool_name} 执行失败: {result['error']}")
                return {"status": "FAILED", "mcp_error": result["error"]}
                
            return {
                "status": "SUCCESS",
                "tool_used": tool_name,
                "mcp_result": result
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

