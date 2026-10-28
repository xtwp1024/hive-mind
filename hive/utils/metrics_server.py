# -*- coding: utf-8 -*-
"""
MetricsServer - 轻量级 Prometheus 指标 HTTP 端点

使用 Python 内置 http.server，无需额外依赖。
启动后在后台线程运行，暴露 /metrics 和 /health 端点。

用法：
    from hive.utils.metrics_server import start_metrics_server
    start_metrics_server(port=9090)
"""

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from hive.utils.metrics import export_prometheus, get_metrics

logger = logging.getLogger("Hive.MetricsServer")

_server: Optional[HTTPServer] = None
_server_lock = threading.Lock()


class _MetricsHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""

    def do_GET(self):
        if self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/health":
            self._serve_health()
        else:
            self._send(404, "text/plain", b"Not Found")

    def _serve_metrics(self):
        try:
            body = export_prometheus().encode("utf-8")
            self._send(200, "text/plain; version=0.0.4; charset=utf-8", body)
        except Exception as e:
            logger.error(f"[MetricsServer] /metrics 生成失败: {e}")
            self._send(500, "text/plain", b"Internal Server Error")

    def _serve_health(self):
        metrics = get_metrics()
        payload = {
            "status": "ok",
            "uptime_seconds": round(metrics.get_all_metrics().get("uptime_seconds", 0), 1),
            "timestamp": time.time(),
        }
        body = json.dumps(payload).encode("utf-8")
        self._send(200, "application/json", body)

    def _send(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # 静默 access log，避免污染主日志
        pass


def start_metrics_server(port: int = 9090, host: str = "0.0.0.0") -> bool:
    """
    在后台线程启动指标 HTTP 服务器。

    Args:
        port: 监听端口（默认 9090）
        host: 监听地址（默认 0.0.0.0）

    Returns:
        True 表示启动成功，False 表示已在运行或启动失败
    """
    global _server

    with _server_lock:
        if _server is not None:
            logger.warning(f"[MetricsServer] 已在运行，忽略重复启动")
            return False

        try:
            _server = HTTPServer((host, port), _MetricsHandler)
        except OSError as e:
            logger.error(f"[MetricsServer] 端口 {port} 绑定失败: {e}")
            return False

        thread = threading.Thread(
            target=_server.serve_forever,
            name="MetricsServer",
            daemon=True,
        )
        thread.start()
        logger.info(f"[MetricsServer] 已启动: http://{host}:{port}/metrics")
        return True


def stop_metrics_server() -> None:
    """停止指标 HTTP 服务器"""
    global _server
    with _server_lock:
        if _server is not None:
            _server.shutdown()
            _server = None
            logger.info("[MetricsServer] 已停止")
