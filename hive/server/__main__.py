#!/usr/bin/env python3
"""
Hive API Server
---------------
为 OpenClaw 提供 RESTful API 接口，允许外部系统与 Hive 交互。

启动:
    python -m hive.server

端点:
    POST   /api/tasks          - 提交新任务
    GET    /api/tasks/<id>     - 查询任务状态
    GET    /api/status         - 获取系统状态
    POST   /api/message        - 发送消息到 Hive
    GET    /api/bees           - 列出所有 bees
    POST   /api/command        - 执行命令
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import logging

# 尝试导入 aiohttp，如果没有则使用标准库
try:
    from aiohttp import web
    ASYNC_MODE = True
except ImportError:
    ASYNC_MODE = False
    from http.server import HTTPServer, BaseHTTPRequestHandler

# Hive 内部导入
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hive.core.queen import QueenBee
from hive.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# 数据存储 (简化版，实际应用应该使用数据库)
# ═══════════════════════════════════════════════════════════

class TaskStore:
    """任务存储"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.results: Dict[str, Any] = {}
    
    def create(self, task_data: Dict) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())[:8]
        task = {
            "id": task_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **task_data
        }
        self.tasks[task_id] = task
        return task_id
    
    def get(self, task_id: str) -> Optional[Dict]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def update(self, task_id: str, updates: Dict):
        """更新任务"""
        if task_id in self.tasks:
            self.tasks[task_id].update(updates)
            self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
    
    def set_result(self, task_id: str, result: Any):
        """设置结果"""
        self.results[task_id] = result
        self.update(task_id, {"status": "completed"})
    
    def list_all(self) -> list:
        """列出所有任务"""
        return list(self.tasks.values())


# 全局存储实例
task_store = TaskStore()


# ═══════════════════════════════════════════════════════════
# aiohttp 实现 (异步模式)
# ═══════════════════════════════════════════════════════════

if ASYNC_MODE:
    
    routes = web.RouteTableDef()
    
    @routes.get('/')
    async def index(request):
        """根路径 - 服务状态"""
        return web.json_response({
            "service": "Hive Mind API",
            "version": "2.0",
            "status": "running",
            "endpoints": [
                "/api/status",
                "/api/tasks",
                "/api/tasks/<id>",
                "/api/bees"
            ]
        })
    
    @routes.get('/api/status')
    async def get_status(request):
        """获取系统状态"""
        status = {
            "status": "healthy",
            "active_bees": 4,  # 应该从 QueenBee 获取
            "pending_tasks": len([t for t in task_store.list_all() if t["status"] == "pending"]),
            "completed_today": len([t for t in task_store.list_all() if t["status"] == "completed"]),
            "timestamp": datetime.now().isoformat(),
            "bees": [
                {"name": "engineer", "status": "idle", "role": "code_generation"},
                {"name": "intel", "status": "idle", "role": "analysis"},
                {"name": "doctor", "status": "idle", "role": "diagnostics"},
                {"name": "alchemist", "status": "idle", "role": "optimization"}
            ]
        }
        return web.json_response(status)
    
    @routes.post('/api/tasks')
    async def create_task(request):
        """创建新任务"""
        try:
            data = await request.json()
            
            # 验证必需字段
            if "type" not in data:
                return web.json_response(
                    {"error": "Missing required field: type"}, 
                    status=400
                )
            
            # 创建任务
            task_id = task_store.create(data)
            
            # 这里应该分发到 QueenBee 处理
            # 简化版：模拟异步处理
            asyncio.create_task(process_task(task_id, data))
            
            return web.json_response({
                "task_id": task_id,
                "status": "pending",
                "message": "Task created successfully",
                "assigned_bee": assign_bee(data["type"])
            }, status=201)
            
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    @routes.get('/api/tasks/{task_id}')
    async def get_task(request):
        """获取任务状态和结果"""
        task_id = request.match_info['task_id']
        task = task_store.get(task_id)
        
        if not task:
            return web.json_response(
                {"error": "Task not found"}, 
                status=404
            )
        
        response = task.copy()
        
        # 如果任务完成，添加结果
        if task_id in task_store.results:
            response["result"] = task_store.results[task_id]
        
        return web.json_response(response)
    
    @routes.get('/api/bees')
    async def list_bees(request):
        """列出所有可用的 bees"""
        bees = [
            {
                "name": "engineer",
                "role": "code_generation",
                "capabilities": ["python", "rust", "solidity", "review"],
                "status": "available"
            },
            {
                "name": "intel", 
                "role": "analysis",
                "capabilities": ["market_analysis", "data_processing", "prediction"],
                "status": "available"
            },
            {
                "name": "doctor",
                "role": "diagnostics", 
                "capabilities": ["debug", "optimize", "refactor"],
                "status": "available"
            },
            {
                "name": "alchemist",
                "role": "optimization",
                "capabilities": ["genetic_algorithm", "hyperparameter_tuning"],
                "status": "available"
            }
        ]
        return web.json_response({"bees": bees})
    
    @routes.post('/api/message')
    async def send_message(request):
        """发送消息到 Hive"""
        try:
            data = await request.json()
            message = data.get("message", "")
            source = data.get("source", "unknown")
            
            logger.info(f"Message from {source}: {message}")
            
            # 这里应该路由到适当的 bee 处理
            # 简化版：返回确认
            return web.json_response({
                "received": True,
                "message_id": str(uuid.uuid4())[:8],
                "source": source,
                "acknowledged": True
            })
            
        except Exception as e:
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    async def process_task(task_id: str, task_data: Dict):
        """异步处理任务 (模拟)"""
        await asyncio.sleep(2)  # 模拟处理时间
        
        task_type = task_data.get("type")
        
        # 根据任务类型生成模拟结果
        if task_type == "analysis":
            result = {
                "analysis": f"Completed analysis for {task_data.get('params', {}).get('symbol', 'unknown')}",
                "indicators": {"rsi": 65, "macd": "bullish", "trend": "upward"},
                "confidence": 0.85
            }
        elif task_type == "code_generation":
            result = {
                "code": "# Generated by Hive Engineer\ndef strategy():\n    pass",
                "language": task_data.get("params", {}).get("language", "python"),
                "tests_included": True
            }
        else:
            result = {"output": "Task completed successfully"}
        
        task_store.set_result(task_id, result)
        logger.info(f"Task {task_id} completed")
    
    def assign_bee(task_type: str) -> str:
        """根据任务类型分配 bee"""
        mapping = {
            "analysis": "intel",
            "code_generation": "engineer",
            "diagnostics": "doctor",
            "optimization": "alchemist",
            "data_processing": "intel"
        }
        return mapping.get(task_type, "engineer")
    
    def init_app():
        """初始化应用"""
        app = web.Application()
        app.add_routes(routes)
        return app


# ═══════════════════════════════════════════════════════════
# 标准库实现 (同步模式，用于 aiohttp 不可用时)
# ═══════════════════════════════════════════════════════════

else:
    
    class HiveHTTPHandler(BaseHTTPRequestHandler):
        """简单的 HTTP 处理器"""
        
        def log_message(self, format, *args):
            """自定义日志"""
            logger.info(f"{self.address_string()} - {format % args}")
        
        def _send_json(self, data: Dict, status: int = 200):
            """发送 JSON 响应"""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        
        def do_GET(self):
            """处理 GET 请求"""
            if self.path == "/":
                self._send_json({
                    "service": "Hive Mind API",
                    "version": "2.0",
                    "status": "running",
                    "mode": "sync"
                })
            
            elif self.path == "/api/status":
                self._send_json({
                    "status": "healthy",
                    "active_bees": 4,
                    "pending_tasks": 0,
                    "mode": "sync"
                })
            
            elif self.path.startswith("/api/tasks/"):
                task_id = self.path.split("/")[-1]
                task = task_store.get(task_id)
                if task:
                    self._send_json(task)
                else:
                    self._send_json({"error": "Task not found"}, 404)
            
            elif self.path == "/api/bees":
                self._send_json({"bees": []})
            
            else:
                self._send_json({"error": "Not found"}, 404)
        
        def do_POST(self):
            """处理 POST 请求"""
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON"}, 400)
                return
            
            if self.path == "/api/tasks":
                task_id = task_store.create(data)
                # 同步处理 (简化版)
                task_store.set_result(task_id, {"status": "processed"})
                
                self._send_json({
                    "task_id": task_id,
                    "status": "completed",
                    "assigned_bee": "engineer"
                }, 201)
            
            elif self.path == "/api/message":
                self._send_json({
                    "received": True,
                    "message_id": str(uuid.uuid4())[:8]
                })
            
            else:
                self._send_json({"error": "Not found"}, 404)
    
    def run_sync_server(host: str = "localhost", port: int = 8080):
        """运行同步服务器"""
        server = HTTPServer((host, port), HiveHTTPHandler)
        logger.info(f"Starting Hive API Server (sync mode) on http://{host}:{port}")
        server.serve_forever()


# ═══════════════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════════════

def main():
    """启动服务器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hive Mind API Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--mode", choices=["auto", "async", "sync"], default="auto",
                       help="Server mode")
    args = parser.parse_args()
    
    if args.mode == "auto":
        mode = "async" if ASYNC_MODE else "sync"
    else:
        mode = args.mode
    
    if mode == "async" and ASYNC_MODE:
        # 使用 aiohttp
        app = init_app()
        web.run_app(app, host=args.host, port=args.port)
    else:
        # 使用标准库
        if not ASYNC_MODE:
            logger.warning("aiohttp not installed, using sync mode")
        run_sync_server(args.host, args.port)


if __name__ == "__main__":
    main()
