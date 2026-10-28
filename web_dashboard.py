#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive Web Dashboard
A simple Flask app to monitor the Hive status.
"""

import os
import sys
import time
import json
import psutil
from functools import wraps
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)
CORS(app)

DEBUG_MODE = os.environ.get("HIVE_DASHBOARD_DEBUG", "false").lower() == "true"
BASIC_AUTH_ENABLED = os.environ.get("HIVE_DASHBOARD_AUTH", "true").lower() == "true"

# Security: In production (non-debug mode), require explicit API_KEY
if DEBUG_MODE:
    # Dev mode: allow insecure default for convenience
    API_KEY = os.environ.get("HIVE_DASHBOARD_API_KEY", "dev_key_change_me_insecure")
else:
    # Production: must set explicit key
    API_KEY = os.environ.get("HIVE_DASHBOARD_API_KEY", "")
    if not API_KEY:
        raise ValueError("HIVE_DASHBOARD_API_KEY environment variable must be set in production mode")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not BASIC_AUTH_ENABLED:
            return f(*args, **kwargs)
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

BASE_DIR = Path(__file__).parent
MORGUE_DIR = BASE_DIR / "hive" / "morgue"
LEDGER_PATH = BASE_DIR / "hive" / "mining_ledger.json"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Hive Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #1e1e1e; color: #d4d4d4; margin: 0; padding: 20px; }
        .card { background: #252526; padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        h1, h2 { color: #569cd6; }
        .status-ok { color: #4ec9b0; }
        .status-error { color: #f44747; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #3c3c3c; }
        th { color: #9cdcfe; }
    </style>
</head>
<body>
    <h1>🐝 Hive Standalone Dashboard</h1>
    
    <div class="card">
        <h2>System Status</h2>
        <div id="system-status">Loading...</div>
    </div>

    <div class="card">
        <h2>Active Bees</h2>
        <div id="bees-list">Loading...</div>
    </div>

    <div class="card">
        <h2>Mining Ledger (Last 5)</h2>
        <div id="ledger">Loading...</div>
    </div>

    <div class="card">
        <h2>v2.1 Components</h2>
        <div id="v21-components">Loading...</div>
    </div>

    <script>
        async function fetchStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                // System Status
                document.getElementById('system-status').innerHTML = `
                    <p>CPU Usage: ${data.system.cpu_percent}%</p>
                    <p>Memory Usage: ${data.system.memory_percent}%</p>
                `;

                // Bees
                let beesHtml = '<table><tr><th>PID</th><th>Name</th><th>Status</th></tr>';
                data.bees.forEach(bee => {
                    beesHtml += `<tr><td>${bee.pid}</td><td>${bee.name}</td><td>${bee.status}</td></tr>`;
                });
                beesHtml += '</table>';
                document.getElementById('bees-list').innerHTML = beesHtml;

                // Ledger
                let ledgerHtml = '<ul>';
                data.ledger.forEach(item => {
                    ledgerHtml += `<li><b>${item.project}</b>: ${JSON.stringify(item.stats)}</li>`;
                });
                ledgerHtml += '</ul>';
                document.getElementById('ledger').innerHTML = ledgerHtml;

                // v2.1 Components
                await fetchV21Components();

            } catch (e) {
                console.error(e);
            }
        }
        
        async function fetchV21Components() {
            try {
                const [gateway, scheduler, limiter] = await Promise.all([
                    fetch('/api/gateway').then(r => r.json()),
                    fetch('/api/scheduler').then(r => r.json()),
                    fetch('/api/ratelimiter').then(r => r.json())
                ]);
                
                document.getElementById('v21-components').innerHTML = `
                    <table>
                        <tr><th>Component</th><th>Status</th><th>Details</th></tr>
                        <tr><td>APIGateway</td><td class="status-ok">${gateway.status}</td><td>Routes: ${gateway.routes || 0}, Endpoints: ${gateway.endpoints || 0}</td></tr>
                        <tr><td>Scheduler</td><td class="status-ok">${scheduler.status}</td><td>Tasks: ${scheduler.tasks || 0}, Handlers: ${scheduler.handlers || 0}</td></tr>
                        <tr><td>RateLimiter</td><td class="status-ok">${limiter.status}</td><td>Limiters: ${limiter.limiters || 0}</td></tr>
                    </table>
                `;
            } catch (e) {
                document.getElementById('v21-components').innerHTML = '<p class="status-error">v2.1 components not initialized</p>';
            }
        }
        
        setInterval(fetchStats, 2000);
        fetchStats();
    </script>
</body>
</html>
"""

@app.route('/')
@require_auth
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stats')
@require_auth
def get_stats():
    # System Stats
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    
    # Mock Bees (In a real scenario, this would scan the process list or shared memory)
    bees = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmd = proc.info['cmdline']
                if cmd and any('hive' in c for c in cmd):
                    bees.append({
                        "pid": proc.info['pid'],
                        "name": "Hive Process",
                        "status": "Running"
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Ledger
    ledger_data = []
    if LEDGER_PATH.exists():
        try:
            with open(LEDGER_PATH, 'r', encoding='utf-8') as f:
                ledger_data = json.load(f)[-5:] # Last 5
        except (IOError, json.JSONDecodeError) as e:
            pass  # ledger optional, silently skip

    return jsonify({
        "system": {"cpu_percent": cpu, "memory_percent": mem},
        "bees": bees,
        "ledger": ledger_data
    })

@app.route('/api/gateway')
@require_auth
def get_gateway_status():
    """获取API网关状态"""
    try:
        from main import get_api_gateway
        gateway = get_api_gateway()
        return jsonify({
            "status": "active",
            "routes": len(gateway._routes),
            "endpoints": len(gateway._endpoints)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/scheduler')
@require_auth
def get_scheduler_status():
    """获取调度器状态"""
    try:
        from main import get_scheduler
        scheduler = get_scheduler()
        return jsonify({
            "status": "active",
            "tasks": len(scheduler._tasks),
            "handlers": len(scheduler._handlers)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/ratelimiter')
@require_auth
def get_ratelimiter_status():
    """获取限流器状态"""
    try:
        from main import get_rate_limiter
        limiter = get_rate_limiter()
        return jsonify({
            "status": "active",
            "limiters": len(limiter._limiters)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("HIVE_DASHBOARD_PORT", 5000))
    host = os.environ.get("HIVE_DASHBOARD_HOST", "127.0.0.1")

    if DEBUG_MODE:
        print("WARNING: Debug mode enabled - this may expose sensitive information!")

    if host == "0.0.0.0":
        print("WARNING: Binding to 0.0.0.0 will expose the dashboard on ALL network interfaces!")
        if BASIC_AUTH_ENABLED:
            print("SECURITY ADVISORY: Ensure HIVE_DASHBOARD_API_KEY is set to a strong value.")

    print(f"Starting Hive Dashboard on {host}:{port}")
    print(f"Authentication: {'ENABLED' if BASIC_AUTH_ENABLED else 'DISABLED'}")

    app.run(host=host, port=port, debug=DEBUG_MODE)
