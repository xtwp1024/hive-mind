# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path


class DashboardBee(BaseBee):
    """
    监控仪表板蜜蜂 - 提供系统实时状态可视化
    """
    
    def __init__(self, dna: BeeDNA = None):
        super().__init__(dna)
        self.metrics_history: List[Dict] = []
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "bee_count": 50
        }
        self.report_dir = Path("hive/reports/dashboard")
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    async def execute_task(self, task) -> Dict[str, Any]:
        action = task.data.get("action", "status")
        
        if action == "status":
            return await self._get_system_status()
        elif action == "metrics":
            return await self._get_metrics(task.data.get("period", "1h"))
        elif action == "alerts":
            return await self._get_alerts()
        elif action == "report":
            return await self._generate_report()
        elif action == "configure":
            return await self._configure_alerts(task.data.get("thresholds", {}))
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        from hive.utils.monitor import get_monitor
        
        monitor = get_monitor()
        resources = monitor.get_snapshot()
        
        active_bees = self._get_active_bees()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": resources.get("cpu_percent", 0),
                "memory_percent": resources.get("memory_percent", 0),
            },
            "hive": {
                "active_bees": len(active_bees),
                "bee_types": self._count_bee_types(active_bees),
            },
            "health": self._calculate_health(resources),
        }
        
        return status
    
    def _get_active_bees(self) -> List[str]:
        """获取活跃蜜蜂列表"""
        try:
            from hive.core.queen import get_queen
            queen = get_queen()
            return list(queen.active_workers.keys()) if queen else []
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"Could not get queen bees: {e}")
            return []
    
    def _count_bee_types(self, active_bees: List[str]) -> Dict[str, int]:
        """统计蜜蜂类型"""
        counts = {}
        for bee_id in active_bees:
            bee_type = bee_id.split("_")[0] if "_" in bee_id else bee_id
            counts[bee_type] = counts.get(bee_type, 0) + 1
        return counts
    
    def _calculate_health(self, resources: Dict) -> str:
        """计算健康状态"""
        cpu = resources.get("cpu_percent", 0)
        mem = resources.get("memory_percent", 0)
        
        if cpu > 90 or mem > 95:
            return "critical"
        elif cpu > 75 or mem > 85:
            return "warning"
        else:
            return "healthy"
    
    async def _get_metrics(self, period: str = "1h") -> Dict[str, Any]:
        """获取指标数据"""
        period_map = {
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7)
        }
        
        delta = period_map.get(period, timedelta(hours=1))
        cutoff = datetime.now() - delta
        
        filtered_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.get("timestamp", "2000")) > cutoff
        ]
        
        return {
            "period": period,
            "data_points": len(filtered_metrics),
            "metrics": filtered_metrics[-100:],
            "summary": self._summarize_metrics(filtered_metrics)
        }
    
    def _summarize_metrics(self, metrics: List[Dict]) -> Dict[str, Any]:
        """汇总指标"""
        if not metrics:
            return {}
        
        cpu_values = [m.get("cpu", 0) for m in metrics]
        mem_values = [m.get("memory", 0) for m in metrics]
        
        return {
            "avg_cpu": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "max_cpu": max(cpu_values) if cpu_values else 0,
            "avg_memory": sum(mem_values) / len(mem_values) if mem_values else 0,
            "max_memory": max(mem_values) if mem_values else 0,
        }
    
    async def _get_alerts(self) -> Dict[str, Any]:
        """获取告警列表"""
        status = await self._get_system_status()
        alerts = []
        
        if status["system"]["cpu_percent"] > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "level": "warning",
                "message": f"CPU使用率过高: {status['system']['cpu_percent']:.1f}%",
                "timestamp": datetime.now().isoformat()
            })
        
        if status["system"]["memory_percent"] > self.alert_thresholds["memory_percent"]:
            alerts.append({
                "level": "critical",
                "message": f"内存使用率过高: {status['system']['memory_percent']:.1f}%",
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "alerts": alerts,
            "thresholds": self.alert_thresholds
        }
    
    async def _generate_report(self) -> Dict[str, Any]:
        """生成报告"""
        status = await self._get_system_status()
        metrics = await self._get_metrics("24h")
        alerts = await self._get_alerts()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "status": status,
            "metrics_24h": metrics.get("summary", {}),
            "alerts": alerts.get("alerts", []),
        }
        
        report_file = self.report_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "report_file": str(report_file)
        }
    
    async def _configure_alerts(self, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """配置告警阈值"""
        for key, value in thresholds.items():
            if key in self.alert_thresholds:
                self.alert_thresholds[key] = value
        
        return {
            "status": "success",
            "thresholds": self.alert_thresholds
        }
    
    def record_metrics(self, cpu: float, memory: float):
        """记录指标数据"""
        self.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            "cpu": cpu,
            "memory": memory
        })
        
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-5000:]
