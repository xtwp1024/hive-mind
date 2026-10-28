#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive Web Dashboard V2.5
现代化实时监控控制台

功能:
- SSE 实时事件流
- 响应式网格布局
- Chart.js 可视化
- 任务历史滚动查询
- 蜂群状态拓扑展示
"""

import os
import sys
import time
import json
import psutil
from functools import wraps
from queue import Queue, Empty
from threading import Thread
from flask import Flask, jsonify, render_template_string, request, Response, stream_with_context
from flask_cors import CORS
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)
CORS(app)

DEBUG_MODE = os.environ.get("HIVE_DASHBOARD_DEBUG", "false").lower() == "true"
BASIC_AUTH_ENABLED = os.environ.get("HIVE_DASHBOARD_AUTH", "true").lower() == "true"

if DEBUG_MODE:
    API_KEY = os.environ.get("HIVE_DASHBOARD_API_KEY", "dev_key_change_me_insecure")
else:
    API_KEY = os.environ.get("HIVE_DASHBOARD_API_KEY", "")
    if not API_KEY:
        raise ValueError("HIVE_DASHBOARD_API_KEY environment variable must be set in production mode")

SSE_CONNECTIONS: list = []
SSE_LOCK = __import__('threading').Lock()

BASE_DIR = Path(__file__).parent
MORGUE_DIR = BASE_DIR / "hive" / "morgue"
LEDGER_PATH = BASE_DIR / "hive" / "mining_ledger.json"
EVOLUTION_DIR = BASE_DIR / "hive" / "evolution"
EVOLUTION_GEP_DIR = EVOLUTION_DIR / "gep"


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not BASIC_AUTH_ENABLED:
            return f(*args, **kwargs)
        auth_header = request.headers.get("Authorization", "")
        query_key = request.args.get("api_key", "")
        if auth_header != f"Bearer {API_KEY}" and query_key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def get_message_bus():
    """获取全局消息总线"""
    try:
        from hive.core.dispatcher import get_message_bus
        return get_message_bus()
    except ImportError:
        return None


def get_migrator():
    """获取数据库迁移器"""
    try:
        from hive.evolution.migrations import get_database_migrator
        return get_database_migrator(EVOLUTION_GEP_DIR)
    except ImportError:
        return None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hive Dashboard V2.5</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --accent-primary: #e94560;
            --accent-secondary: #00d9ff;
            --accent-success: #00ff88;
            --accent-warning: #ffaa00;
            --accent-danger: #ff4444;
            --text-primary: #e4e4e4;
            --text-secondary: #a0a0a0;
            --border-color: #2a4a6a;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .header {
            background: var(--bg-card);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header h1 {
            font-size: 1.5rem;
            background: linear-gradient(90deg, var(--accent-secondary), var(--accent-primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header-right {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 20px;
            border: 1px solid var(--accent-success);
        }
        
        .status-badge.disconnected {
            background: rgba(255, 68, 68, 0.1);
            border-color: var(--accent-danger);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-success);
            animation: pulse 2s infinite;
        }
        
        .status-badge.disconnected .status-dot {
            background: var(--accent-danger);
            animation: none;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .container {
            padding: 1.5rem;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .grid-large {
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .card-title {
            font-size: 1.1rem;
            color: var(--accent-secondary);
            font-weight: 600;
        }
        
        .card-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .card-label {
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        .metric-success { color: var(--accent-success); }
        .metric-warning { color: var(--accent-warning); }
        .metric-danger { color: var(--accent-danger); }
        .metric-primary { color: var(--accent-secondary); }
        
        .chart-container {
            position: relative;
            height: 250px;
        }
        
        .bee-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .bee-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        
        .bee-status {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .bee-status.running {
            background: rgba(0, 255, 136, 0.2);
            color: var(--accent-success);
        }
        
        .bee-status.idle {
            background: rgba(0, 217, 255, 0.2);
            color: var(--accent-secondary);
        }
        
        .bee-status.busy {
            background: rgba(255, 170, 0, 0.2);
            color: var(--accent-warning);
        }
        
        .task-history {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .task-item {
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 0.75rem;
            border-left: 4px solid var(--accent-secondary);
        }
        
        .task-item.completed { border-left-color: var(--accent-success); }
        .task-item.failed { border-left-color: var(--accent-danger); }
        .task-item.running { border-left-color: var(--accent-warning); }
        
        .task-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .task-id {
            font-family: monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .task-type {
            font-weight: 600;
            color: var(--accent-secondary);
        }
        
        .task-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .event-log {
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
            font-size: 0.85rem;
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 1rem;
        }
        
        .event-item {
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .event-item:last-child { border-bottom: none; }
        
        .event-time { color: var(--text-secondary); margin-right: 0.5rem; }
        .event-type { font-weight: 600; margin-right: 0.5rem; }
        .event-type.task_started { color: var(--accent-warning); }
        .event-type.task_completed { color: var(--accent-success); }
        .event-type.task_failed { color: var(--accent-danger); }
        .event-type.workflow_started { color: var(--accent-secondary); }
        .event-type.workflow_completed { color: var(--accent-success); }
        .event-type.system_metrics { color: var(--text-secondary); }
        
        .progress-bar {
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-secondary), var(--accent-primary));
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }
        
        .tab {
            padding: 0.5rem 1rem;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            border-radius: 6px 6px 0 0;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        
        .tab:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }
        
        .tab.active {
            color: var(--accent-secondary);
            background: rgba(0, 217, 255, 0.1);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            color: var(--text-secondary);
        }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border-color);
            border-top-color: var(--accent-secondary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .migration-info {
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .migration-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .migration-row:last-child { border-bottom: none; }
        
        .migration-version {
            font-family: monospace;
            font-weight: bold;
            color: var(--accent-secondary);
        }
        
        .migration-status {
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.8rem;
        }
        
        .migration-status.completed {
            background: rgba(0, 255, 136, 0.2);
            color: var(--accent-success);
        }
        
        .migration-status.pending {
            background: rgba(255, 170, 0, 0.2);
            color: var(--accent-warning);
        }
        
        .btn {
            padding: 0.5rem 1.5rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: var(--accent-secondary);
            color: var(--bg-primary);
        }
        
        .btn-primary:hover {
            background: #00b8cc;
        }
        
        .btn-success {
            background: var(--accent-success);
            color: var(--bg-primary);
        }
        
        .btn-success:hover {
            background: #00cc70;
        }
        
        .btn-warning {
            background: var(--accent-warning);
            color: var(--bg-primary);
        }
        
        .btn-warning:hover {
            background: #cc8800;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            
            .header {
                flex-direction: column;
                gap: 1rem;
            }
            
            .header-right {
                width: 100%;
                justify-content: space-between;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>🐝 Hive Dashboard V2.5</h1>
        <div class="header-right">
            <div id="sse-status" class="status-badge">
                <span class="status-dot"></span>
                <span id="sse-text">连接中...</span>
            </div>
            <div id="time-display" style="color: var(--text-secondary);"></div>
        </div>
    </header>
    
    <main class="container">
        <div class="tabs">
            <button class="tab active" data-tab="overview">概览</button>
            <button class="tab" data-tab="tasks">任务</button>
            <button class="tab" data-tab="events">事件</button>
            <button class="tab" data-tab="bees">蜂群</button>
            <button class="tab" data-tab="migration">迁移</button>
        </div>
        
        <div id="tab-overview" class="tab-content active">
            <div class="grid">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">CPU 使用率</span>
                        <span id="cpu-value" class="metric-primary">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div id="cpu-bar" class="progress-fill" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">内存使用率</span>
                        <span id="memory-value" class="metric-warning">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div id="memory-bar" class="progress-fill" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">活跃蜜蜂</span>
                        <span id="bees-count" class="metric-success">0</span>
                    </div>
                    <div class="card-label">当前活跃的工蜂数量</div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">任务总数</span>
                        <span id="tasks-total" class="metric-primary">0</span>
                    </div>
                    <div class="card-label">本次会话处理的任务</div>
                </div>
            </div>
            
            <div class="grid grid-large">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">资源使用趋势</span>
                    </div>
                    <div class="chart-container">
                        <canvas id="resourcesChart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">任务统计</span>
                    </div>
                    <div class="chart-container">
                        <canvas id="tasksChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="tab-tasks" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">任务历史</span>
                    <button class="btn btn-primary" onclick="refreshTasks()">刷新</button>
                </div>
                <div id="task-history" class="task-history">
                    <div class="loading">
                        <span class="spinner"></span>
                        加载中...
                    </div>
                </div>
            </div>
        </div>
        
        <div id="tab-events" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">实时事件流 (SSE)</span>
                    <button class="btn btn-warning" onclick="clearEvents()">清空</button>
                </div>
                <div id="event-log" class="event-log">
                    <div style="color: var(--text-secondary);">等待事件...</div>
                </div>
            </div>
        </div>
        
        <div id="tab-bees" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">蜂群状态</span>
                    <button class="btn btn-primary" onclick="refreshBees()">刷新</button>
                </div>
                <div id="bees-list" class="bee-list">
                    <div class="loading">
                        <span class="spinner"></span>
                        加载中...
                    </div>
                </div>
            </div>
        </div>
        
        <div id="tab-migration" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">数据库迁移状态</span>
                    <button id="migrate-btn" class="btn btn-success" onclick="runMigration()">执行迁移</button>
                </div>
                <div id="migration-info" class="migration-info">
                    <div class="loading">
                        <span class="spinner"></span>
                        加载中...
                    </div>
                </div>
            </div>
        </div>
    </main>
    
    <script>
        const API_BASE = '';
        const API_KEY_PARAM = new URLSearchParams(window.location.search).get('api_key');
        const AUTH_HEADER = API_KEY_PARAM ? '' : '?api_key=' + (new URLSearchParams(window.location.search).get('api_key') || '');
        
        let resourcesChart, tasksChart;
        let cpuHistory = [], memoryHistory = [], timeLabels = [];
        let taskStats = { completed: 0, failed: 0, running: 0, pending: 0 };
        let totalTasks = 0;
        
        function updateTime() {
            document.getElementById('time-display').textContent = new Date().toLocaleString('zh-CN');
        }
        setInterval(updateTime, 1000);
        updateTime();
        
        function initCharts() {
            const resourcesCtx = document.getElementById('resourcesChart').getContext('2d');
            const tasksCtx = document.getElementById('tasksChart').getContext('2d');
            
            resourcesChart = new Chart(resourcesCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'CPU %',
                            data: [],
                            borderColor: '#00d9ff',
                            backgroundColor: 'rgba(0, 217, 255, 0.1)',
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'Memory %',
                            data: [],
                            borderColor: '#ffaa00',
                            backgroundColor: 'rgba(255, 170, 0, 0.1)',
                            fill: true,
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#e4e4e4' }
                        }
                    },
                    scales: {
                        y: {
                            min: 0,
                            max: 100,
                            ticks: { color: '#a0a0a0' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        x: {
                            ticks: { color: '#a0a0a0' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        }
                    }
                }
            });
            
            tasksChart = new Chart(tasksCtx, {
                type: 'doughnut',
                data: {
                    labels: ['已完成', '失败', '运行中', '等待中'],
                    datasets: [{
                        data: [0, 0, 0, 0],
                        backgroundColor: ['#00ff88', '#ff4444', '#ffaa00', '#00d9ff']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: { color: '#e4e4e4' }
                        }
                    }
                }
            });
        }
        
        async function fetchStats() {
            try {
                const response = await fetch(`${API_BASE}/api/stats${AUTH_HEADER ? AUTH_HEADER : ''}`);
                const data = await response.json();
                
                document.getElementById('cpu-value').textContent = data.system.cpu_percent + '%';
                document.getElementById('cpu-bar').style.width = data.system.cpu_percent + '%';
                
                document.getElementById('memory-value').textContent = data.system.memory_percent + '%';
                document.getElementById('memory-bar').style.width = data.system.memory_percent + '%';
                
                document.getElementById('bees-count').textContent = data.bees.length;
                
                const now = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
                timeLabels.push(now);
                cpuHistory.push(data.system.cpu_percent);
                memoryHistory.push(data.system.memory_percent);
                
                if (timeLabels.length > 20) {
                    timeLabels.shift();
                    cpuHistory.shift();
                    memoryHistory.shift();
                }
                
                resourcesChart.data.labels = timeLabels;
                resourcesChart.data.datasets[0].data = cpuHistory;
                resourcesChart.data.datasets[1].data = memoryHistory;
                resourcesChart.update();
                
                return data;
            } catch (e) {
                console.error('Failed to fetch stats:', e);
            }
        }
        
        async function refreshBees() {
            try {
                const response = await fetch(`${API_BASE}/api/stats${AUTH_HEADER ? AUTH_HEADER : ''}`);
                const data = await response.json();
                
                const container = document.getElementById('bees-list');
                if (data.bees.length === 0) {
                    container.innerHTML = '<div style="color: var(--text-secondary); padding: 2rem; text-align: center;">暂无活跃的工蜂进程</div>';
                    return;
                }
                
                container.innerHTML = data.bees.map(bee => `
                    <div class="bee-item">
                        <div>
                            <strong>${bee.name}</strong>
                            <div style="font-size: 0.85rem; color: var(--text-secondary);">
                                PID: ${bee.pid}
                            </div>
                        </div>
                        <span class="bee-status ${bee.status.toLowerCase()}">${bee.status}</span>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to refresh bees:', e);
            }
        }
        
        async function refreshTasks() {
            try {
                const response = await fetch(`${API_BASE}/api/task-history${AUTH_HEADER ? AUTH_HEADER : ''}`);
                const data = await response.json();
                
                const container = document.getElementById('task-history');
                if (data.tasks.length === 0) {
                    container.innerHTML = '<div style="color: var(--text-secondary); padding: 2rem; text-align: center;">暂无任务记录</div>';
                    return;
                }
                
                container.innerHTML = data.tasks.map(task => `
                    <div class="task-item ${task.status}">
                        <div class="task-header">
                            <span class="task-type">${task.type}</span>
                            <span class="task-id">${task.id}</span>
                        </div>
                        <div class="task-meta">
                            <span>状态: ${task.status}</span>
                            ${task.duration ? `<span>耗时: ${task.duration.toFixed(2)}s</span>` : ''}
                            ${task.assigned_bee ? `<span>分配给: ${task.assigned_bee}</span>` : ''}
                        </div>
                        ${task.error ? `<div style="color: var(--accent-danger); margin-top: 0.5rem; font-size: 0.85rem;">错误: ${task.error}</div>` : ''}
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to refresh tasks:', e);
            }
        }
        
        async function refreshMigration() {
            try {
                const response = await fetch(`${API_BASE}/api/migration-status${AUTH_HEADER ? AUTH_HEADER : ''}`);
                const data = await response.json();
                
                const container = document.getElementById('migration-info');
                const btn = document.getElementById('migrate-btn');
                
                let html = `
                    <div style="margin-bottom: 1rem;">
                        <strong>当前版本:</strong> <span class="migration-version">${data.current_version}</span><br>
                        <strong>最新版本:</strong> <span class="migration-version">${data.latest_version}</span><br>
                        <strong>状态:</strong> ${data.is_latest ? '<span style="color: var(--accent-success);">已是最新</span>' : '<span style="color: var(--accent-warning);">有新版本</span>'}
                    </div>
                    <div style="margin-bottom: 0.5rem; font-weight: 600;">迁移记录:</div>
                `;
                
                if (data.applied_migrations && data.applied_migrations.length > 0) {
                    data.applied_migrations.forEach(m => {
                        html += `
                            <div class="migration-row">
                                <div>
                                    <span class="migration-version">${m.version}</span>
                                    <span style="margin-left: 1rem; color: var(--text-secondary);">${m.name}</span>
                                </div>
                                <span class="migration-status ${m.status}">${m.status}</span>
                            </div>
                        `;
                    });
                }
                
                if (data.pending_migrations && data.pending_migrations.length > 0) {
                    html += `<div style="margin-top: 1rem; font-weight: 600; color: var(--accent-warning);">待应用的迁移:</div>`;
                    data.pending_migrations.forEach(m => {
                        html += `
                            <div class="migration-row">
                                <div>
                                    <span class="migration-version">${m.version}</span>
                                    <span style="margin-left: 1rem; color: var(--text-secondary);">${m.name}</span>
                                </div>
                                <span class="migration-status pending">pending</span>
                            </div>
                        `;
                    });
                }
                
                container.innerHTML = html;
                
                if (data.is_latest) {
                    btn.style.display = 'none';
                } else {
                    btn.style.display = 'block';
                }
                
            } catch (e) {
                console.error('Failed to refresh migration:', e);
                document.getElementById('migration-info').innerHTML = '<div style="color: var(--accent-danger);">加载迁移状态失败</div>';
            }
        }
        
        async function runMigration() {
            const btn = document.getElementById('migrate-btn');
            btn.disabled = true;
            btn.textContent = '执行中...';
            
            try {
                const response = await fetch(`${API_BASE}/api/migrate${AUTH_HEADER ? AUTH_HEADER : ''}`);
                const data = await response.json();
                
                alert(`迁移完成!\n应用迁移数: ${data.migrations_applied || 0}\n状态: ${data.status}`);
                await refreshMigration();
                
            } catch (e) {
                alert('迁移失败: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = '执行迁移';
            }
        }
        
        function clearEvents() {
            document.getElementById('event-log').innerHTML = '<div style="color: var(--text-secondary);">等待事件...</div>';
        }
        
        function connectSSE() {
            const statusBadge = document.getElementById('sse-status');
            const statusText = document.getElementById('sse-text');
            
            statusBadge.classList.remove('disconnected');
            statusText.textContent = '连接中...';
            
            try {
                const eventSource = new EventSource(`${API_BASE}/api/events${AUTH_HEADER ? AUTH_HEADER : ''}`);
                
                eventSource.onopen = () => {
                    statusBadge.classList.remove('disconnected');
                    statusText.textContent = '已连接';
                    console.log('SSE connected');
                };
                
                eventSource.onerror = () => {
                    statusBadge.classList.add('disconnected');
                    statusText.textContent = '连接断开';
                    console.log('SSE disconnected');
                    
                    setTimeout(() => {
                        if (statusBadge.classList.contains('disconnected')) {
                            connectSSE();
                        }
                    }, 3000);
                };
                
                eventSource.addEventListener('message', (e) => {
                    try {
                        const data = JSON.parse(e.data);
                        addEvent(data.type, data.data);
                    } catch (err) {
                        console.error('Failed to parse SSE message:', err);
                    }
                });
                
                eventSource.addEventListener('task_started', (e) => {
                    try {
                        const data = JSON.parse(e.data);
                        addEvent('task_started', data);
                        taskStats.running++;
                        updateTasksChart();
                    } catch (err) {
                        console.error(err);
                    }
                });
                
                eventSource.addEventListener('task_completed', (e) => {
                    try {
                        const data = JSON.parse(e.data);
                        addEvent('task_completed', data);
                        taskStats.running = Math.max(0, taskStats.running - 1);
                        taskStats.completed++;
                        totalTasks++;
                        document.getElementById('tasks-total').textContent = totalTasks;
                        updateTasksChart();
                    } catch (err) {
                        console.error(err);
                    }
                });
                
                eventSource.addEventListener('task_failed', (e) => {
                    try {
                        const data = JSON.parse(e.data);
                        addEvent('task_failed', data);
                        taskStats.running = Math.max(0, taskStats.running - 1);
                        taskStats.failed++;
                        totalTasks++;
                        document.getElementById('tasks-total').textContent = totalTasks;
                        updateTasksChart();
                    } catch (err) {
                        console.error(err);
                    }
                });
                
                eventSource.addEventListener('workflow_started', (e) => {
                    try {
                        const data = JSON.parse(e.data);
                        addEvent('workflow_started', data);
                    } catch (err) {
                        console.error(err);
                    }
                });
                
                eventSource.addEventListener('workflow_completed', (e) => {
                    try {
                        const data = JSON.parse(e.data);
                        addEvent('workflow_completed', data);
                    } catch (err) {
                        console.error(err);
                    }
                });
                
                eventSource.addEventListener('system_metrics', (e) => {
                    try {
                        const data = JSON.parse(e.data);
                        addEvent('system_metrics', data);
                    } catch (err) {
                        console.error(err);
                    }
                });
                
            } catch (e) {
                console.error('Failed to create SSE connection:', e);
                statusBadge.classList.add('disconnected');
                statusText.textContent = '连接失败';
            }
        }
        
        function addEvent(type, data) {
            const log = document.getElementById('event-log');
            const time = new Date().toLocaleTimeString('zh-CN');
            
            const eventDiv = document.createElement('div');
            eventDiv.className = 'event-item';
            eventDiv.innerHTML = `
                <span class="event-time">[${time}]</span>
                <span class="event-type ${type}">${type}</span>
                <span>${typeof data === 'string' ? data : (data.task_id || data.workflow_name || JSON.stringify(data).substring(0, 100))}</span>
            `;
            
            const placeholder = log.querySelector('div[style*="等待事件"]');
            if (placeholder) {
                placeholder.remove();
            }
            
            log.appendChild(eventDiv);
            log.scrollTop = log.scrollHeight;
        }
        
        function updateTasksChart() {
            tasksChart.data.datasets[0].data = [
                taskStats.completed,
                taskStats.failed,
                taskStats.running,
                taskStats.pending
            ];
            tasksChart.update();
        }
        
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                tab.classList.add('active');
                document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
                
                if (tab.dataset.tab === 'tasks') refreshTasks();
                if (tab.dataset.tab === 'bees') refreshBees();
                if (tab.dataset.tab === 'migration') refreshMigration();
            });
        });
        
        initCharts();
        fetchStats();
        refreshBees();
        refreshTasks();
        refreshMigration();
        connectSSE();
        
        setInterval(fetchStats, 3000);
    </script>
</body>
</html>
"""

SIMULATED_TASKS = [
    {"id": "task-001", "type": "code_generation", "status": "completed", "duration": 2.5, "assigned_bee": "CodeBee-01", "error": None},
    {"id": "task-002", "type": "file_analysis", "status": "completed", "duration": 1.2, "assigned_bee": "AnalyzerBee-02", "error": None},
    {"id": "task-003", "type": "test_execution", "status": "failed", "duration": 0.8, "assigned_bee": "TestBee-03", "error": "Assertion failed: expected True, got False"},
    {"id": "task-004", "type": "code_review", "status": "running", "duration": None, "assigned_bee": "CodeBee-01", "error": None},
]


@app.route('/')
@require_auth
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/stats')
@require_auth
def get_stats():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    
    bees = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmd = proc.info['cmdline']
                if cmd and any('hive' in str(c).lower() for c in cmd):
                    bees.append({
                        "pid": proc.info['pid'],
                        "name": "Hive Process",
                        "status": "Running"
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    ledger_data = []
    if LEDGER_PATH.exists():
        try:
            with open(LEDGER_PATH, 'r', encoding='utf-8') as f:
                ledger_data = json.load(f)[-5:]
        except (IOError, json.JSONDecodeError):
            pass
    
    return jsonify({
        "system": {"cpu_percent": cpu, "memory_percent": mem},
        "bees": bees,
        "ledger": ledger_data
    })


@app.route('/api/task-history')
@require_auth
def get_task_history():
    return jsonify({
        "tasks": SIMULATED_TASKS,
        "total": len(SIMULATED_TASKS)
    })


@app.route('/api/migration-status')
@require_auth
def get_migration_status():
    migrator = get_migrator()
    if migrator:
        try:
            status = migrator.get_migration_status()
            return jsonify(status)
        except Exception as e:
            return jsonify({
                "current_version": "unknown",
                "latest_version": "unknown",
                "is_latest": True,
                "error": str(e)
            })
    
    return jsonify({
        "current_version": "000",
        "latest_version": "006",
        "is_latest": False,
        "pending_migrations": [
            {"version": "001", "name": "initial_schema"},
            {"version": "002", "name": "schema_migrations_table"},
            {"version": "003", "name": "cluster_tables"},
            {"version": "004", "name": "bee_status_tracking"},
            {"version": "005", "name": "knowledge_graph"},
            {"version": "006", "name": "sse_event_store"},
        ]
    })


@app.route('/api/migrate', methods=['POST'])
@require_auth
def run_migrate():
    migrator = get_migrator()
    if migrator:
        try:
            result = migrator.migrate_to_latest()
            return jsonify(result)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    
    return jsonify({"status": "simulated", "message": "Migration runner not available"})


@app.route('/api/events')
@require_auth
def sse_events():
    """SSE 实时事件流"""
    def generate():
        bus = get_message_bus()
        queue = None
        
        if bus:
            queue = bus.subscribe()
            yield f"data: {json.dumps({'type': 'connected', 'data': 'SSE connected'})}\n\n"
        
        import threading
        stop_event = threading.Event()
        
        def push_system_metrics():
            while not stop_event.is_set():
                try:
                    cpu = psutil.cpu_percent()
                    mem = psutil.virtual_memory().percent
                    yield f"event: system_metrics\n"
                    yield f"data: {json.dumps({'cpu_percent': cpu, 'memory_percent': mem})}\n\n"
                except Exception:
                    pass
                stop_event.wait(5)
        
        if queue:
            try:
                while True:
                    try:
                        msg = queue.get(timeout=1.0)
                        yield f"event: {msg.event_type.value}\n"
                        yield f"data: {json.dumps(msg.data)}\n\n"
                    except Empty:
                        pass
            except GeneratorExit:
                if queue:
                    bus.unsubscribe(queue)
                stop_event.set()
                raise
        else:
            yield f"data: {json.dumps({'type': 'info', 'data': 'Simulated mode - no message bus'})}\n\n"
            
            import time
            event_count = 0
            while True:
                event_count += 1
                yield f"data: {json.dumps({'type': 'heartbeat', 'data': {'count': event_count, 'time': time.time()}})}\n\n"
                time.sleep(10)
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.route('/api/gateway')
@require_auth
def get_gateway_status():
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

    print(f"Starting Hive Dashboard V2.5 on {host}:{port}")
    print(f"Authentication: {'ENABLED' if BASIC_AUTH_ENABLED else 'DISABLED'}")
    print(f"Debug Mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")
    print()
    print(f"Access URL: http://{host}:{port}")
    if BASIC_AUTH_ENABLED and API_KEY and API_KEY != "dev_key_change_me_insecure":
        print(f"With API Key: http://{host}:{port}?api_key={API_KEY}")
    else:
        print(f"Dev Mode: http://{host}:{port}?api_key=dev_key_change_me_insecure")
    print()

    app.run(host=host, port=port, debug=DEBUG_MODE, threaded=True)
