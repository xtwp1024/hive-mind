#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import subprocess
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger("Hive.MCPClient")

class StdIoMCPClient:
    """
    轻量级的简易 MCP (Model Context Protocol) StdIo 客户端。
    用于在不用引入繁重 SDK 的前提下，通过 JSON-RPC 协议与外部命令行工具服务器对接。
    """
    def __init__(self, command: str, args: list, env: dict = None):
        self.command = command
        self.args = args
        self.env = env or {}
        self.process = None
        self._next_id = 1
        self._lock = threading.Lock()

    def start(self):
        """启动 MCP Server 子进程"""
        try:
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                text=True
            )
            logger.info(f"🚀 MCP Server [{self.command}] 已启动. PID: {self.process.pid}")
        except Exception as e:
            logger.error(f"❌ 启动 MCP Server 失败: {e}")
            raise RuntimeError(f"Failed to start MCP Server: {e}")

    def stop(self):
        """终止 MCP Server 子进程"""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            logger.info("🛑 MCP Server 已关闭")

    def call_tool(self, target_server_name: str, tool_name: str, arguments: dict, timeout: float = 30.0) -> Dict[str, Any]:
        """发起调用请求（JSON-RPC 2.0 基础调用）"""
        if not self.process:
            return {"error": "Server not running"}

        import select
        import socket
        import sys

        with self._lock:
            req_id = self._next_id
            self._next_id += 1

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "call_tool",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        try:
            js_str = json.dumps(payload)
            # 发送到进程的 stdin
            self.process.stdin.write(js_str + "\n")
            self.process.stdin.flush()

            # Windows 上 select.select() 不支持管道，只支持 socket
            # 因此在 Windows 上使用不同的超时机制
            if sys.platform == "win32":
                # Windows: 使用带超时的 readline 轮询
                import threading
                result = {"line": None, "error": None}

                def read_line():
                    try:
                        result["line"] = self.process.stdout.readline()
                    except Exception as e:
                        result["error"] = str(e)

                thread = threading.Thread(target=read_line)
                thread.start()
                thread.join(timeout=timeout)
                if thread.is_alive():
                    return {"error": f"Tool call timeout after {timeout}s"}
                if result["error"]:
                    return {"error": f"Read error: {result['error']}"}
                line = result["line"]
            else:
                # Unix: 使用 select 等待响应，带超时避免无限阻塞
                readable, _, _ = select.select([self.process.stdout], [], [], timeout)
                if not readable:
                    return {"error": f"Tool call timeout after {timeout}s"}
                line = self.process.stdout.readline()

            if not line:
                return {"error": "Empty response from server"}

            response = json.loads(line)
            if "error" in response:
                return {"error": response["error"]}
            return response.get("result", {})

        except Exception as e:
            logger.error(f"⚠️ 工具调用异常 [{tool_name}]: {e}")
            return {"error": f"Tool execution failed: {e}"}

class MCPManager:
    """全局 MCP 连接管理池"""
    def __init__(self):
        self.clients: Dict[str, StdIoMCPClient] = {}
        
    def load_servers(self, config_path: str):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for name, conf in data.get("servers", {}).items():
                command = conf.get("command")
                if not command: continue
                # 这里只注册配置，延迟启动
                self.clients[name] = StdIoMCPClient(
                    command=command,
                    args=conf.get("args", []),
                    env=conf.get("env", {})
                )
            logger.info(f"✅ 已加载 {len(self.clients)} 个 MCP Servers 配置")
        except FileNotFoundError:
            logger.warning(f"⚠️ 找不到 MCP 配置文件: {config_path}")
        except Exception as e:
            logger.error(f"❌ 加载 MCP 配置失败: {e}")
            
    def get_client(self, server_name: str) -> Optional[StdIoMCPClient]:
        """获取并确客户端已启动"""
        client = self.clients.get(server_name)
        if client and not client.process:
            client.start()
        return client

