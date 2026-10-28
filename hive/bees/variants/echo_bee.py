#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回声蜂 - EchoBee
信号广播和延迟测试

变异率：11.00%
特殊能力：网络延迟测量 + 信号传播
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)

class EchoBee(BaseBee):
    """
    回声蜂 - 负责信号广播和延迟测试
    
    特殊能力：
    - 测量操作延迟
    - 分析信号传播效率
    - 检测性能瓶颈
    - 生成性能报告
    """
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.latency_measurements = []
        self.signal_metrics = {}
    
    def specialized_task(self, target_path):
        """
        执行信号广播和延迟测试任务
        
        Args:
            target_path: 项目目录路径
        """
        logger.info(f"📡 EchoBee 开始任务: {self.dna.target_name}")
        
        try:
            # 1. 测量文件访问延迟
            file_access_latency = self._measure_file_access_latency(target_path)
            
            # 2. 测量目录遍历性能
            directory_traversal_latency = self._measure_directory_traversal(target_path)
            
            # 3. 分析信号传播模式
            signal_patterns = self._analyze_signal_patterns(target_path)
            
            # 4. 检测性能瓶颈
            performance_bottlenecks = self._detect_bottlenecks(
                file_access_latency, directory_traversal_latency
            )
            
            # 5. 生成性能报告
            performance_report = self._generate_performance_report({
                "project": self.dna.target_name,
                "file_access_latency": file_access_latency,
                "directory_traversal_latency": directory_traversal_latency,
                "signal_patterns": signal_patterns,
                "performance_bottlenecks": performance_bottlenecks,
                "timestamp": time.ctime()
            })
            
            # 6. 保存报告
            self._save_echo_report(performance_report)
            
            logger.info(f"✅ EchoBee 任务完成: {self.dna.target_name}")
            logger.info(f"   平均文件访问延迟: {file_access_latency.get('avg_latency_ms', 0):.2f}ms")
            logger.info(f"   目录遍历耗时: {directory_traversal_latency.get('total_time_ms', 0):.2f}ms")
            logger.info(f"   性能瓶颈: {len(performance_bottlenecks)}")
            
            return {
                "status": "success",
                "avg_latency_ms": file_access_latency.get('avg_latency_ms', 0),
                "traversal_time_ms": directory_traversal_latency.get('total_time_ms', 0),
                "bottlenecks": len(performance_bottlenecks)
            }
            
        except Exception as e:
            logger.error(f"❌ EchoBee 任务失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _measure_file_access_latency(self, target_path: Path) -> Dict[str, Any]:
        """
        测量文件访问延迟
        
        Args:
            target_path: 项目目录路径
            
        Returns:
            延迟测量结果
        """
        latencies = []
        sample_size = 20  # 采样数量
        
        # 获取随机文件进行测试
        files = list(target_path.rglob('*'))
        files = [f for f in files if f.is_file() and f.stat().st_size < 1024 * 1024]  # < 1MB
        
        if not files:
            return {"avg_latency_ms": 0, "max_latency_ms": 0, "min_latency_ms": 0}
        
        # 随机采样
        import random
        sample_files = random.sample(min(files, sample_size), min(len(files), sample_size))
        
        for file_path in sample_files:
            start_time = time.time()
            
            try:
                # 读取文件
                with open(file_path, 'rb') as f:
                    content = f.read(1024)  # 只读1KB
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                logger.debug(f"文件访问测试失败 {file_path}: {e}")
        
        if not latencies:
            return {"avg_latency_ms": 0, "max_latency_ms": 0, "min_latency_ms": 0}
        
        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "samples": len(latencies)
        }
    
    def _measure_directory_traversal(self, target_path: Path) -> Dict[str, Any]:
        """
        测量目录遍历性能
        
        Args:
            target_path: 项目目录路径
            
        Returns:
            遍历性能结果
        """
        start_time = time.time()
        
        files_count = 0
        dirs_count = 0
        total_size = 0
        
        try:
            for item in target_path.rglob('*'):
                if item.is_file():
                    files_count += 1
                    total_size += item.stat().st_size
                elif item.is_dir():
                    dirs_count += 1
                    
        except Exception as e:
            logger.debug(f"目录遍历失败: {e}")
        
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        
        return {
            "total_time_ms": total_time_ms,
            "files_count": files_count,
            "dirs_count": dirs_count,
            "total_size": total_size,
            "files_per_ms": files_count / max(total_time_ms, 1)
        }
    
    def _analyze_signal_patterns(self, target_path: Path) -> Dict[str, Any]:
        """
        分析信号传播模式
        
        Args:
            target_path: 项目目录路径
            
        Returns:
            信号模式分析结果
        """
        patterns = {
            "access_patterns": {},
            "size_distribution": {},
            "depth_distribution": {}
        }
        
        file_sizes = []
        depths = []
        extensions = []
        
        for file_path in target_path.rglob('*'):
            if file_path.is_file():
                # 文件大小
                size = file_path.stat().st_size
                file_sizes.append(size)
                
                # 目录深度
                depth = len(file_path.relative_to(target_path).parts) - 1
                depths.append(depth)
                
                # 文件扩展名
                ext = file_path.suffix.lower()
                extensions.append(ext)
        
        # 分析文件大小分布
        if file_sizes:
            patterns['size_distribution'] = {
                "min": min(file_sizes),
                "max": max(file_sizes),
                "avg": sum(file_sizes) / len(file_sizes),
                "median": sorted(file_sizes)[len(file_sizes) // 2]
            }
        
        # 分析深度分布
        if depths:
            patterns['depth_distribution'] = {
                "min": min(depths),
                "max": max(depths),
                "avg": sum(depths) / len(depths)
            }
        
        # 分析文件类型访问模式
        for ext in extensions:
            patterns['access_patterns'][ext] = patterns['access_patterns'].get(ext, 0) + 1
        
        return patterns
    
    def _detect_bottlenecks(
        self, 
        file_latency: Dict[str, Any], 
        dir_traversal: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        检测性能瓶颈
        
        Args:
            file_latency: 文件延迟数据
            dir_traversal: 目录遍历数据
            
        Returns:
            性能瓶颈列表
        """
        bottlenecks = []
        
        # 检查文件访问延迟
        avg_latency = file_latency.get('avg_latency_ms', 0)
        if avg_latency > 10:  # 超过10ms
            bottlenecks.append({
                "type": "file_access",
                "severity": "high" if avg_latency > 50 else "medium",
                "description": f"文件访问延迟过高: {avg_latency:.2f}ms",
                "recommendation": "考虑使用缓存或优化存储"
            })
        
        # 检查目录遍历性能
        traversal_rate = dir_traversal.get('files_per_ms', 0)
        if traversal_rate < 0.5:  # 每毫秒少于0.5个文件
            bottlenecks.append({
                "type": "directory_traversal",
                "severity": "medium",
                "description": f"目录遍历速度较慢: {traversal_rate:.2f} files/ms",
                "recommendation": "考虑优化目录结构或使用索引"
            })
        
        # 检查文件数量
        total_files = dir_traversal.get('files_count', 0)
        if total_files > 10000:
            bottlenecks.append({
                "type": "file_count",
                "severity": "low",
                "description": f"文件数量过多: {total_files}",
                "recommendation": "考虑文件归档或压缩"
            })
        
        return bottlenecks
    
    def _generate_performance_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成性能报告
        
        Args:
            data: 原始数据
            
        Returns:
            格式化的性能报告
        """
        report = {
            **data,
            "summary": {
                "overall_performance": "good",
                "bottlenecks_count": len(data['performance_bottlenecks']),
                "optimization_priority": "medium"
            }
        }
        
        # 确定整体性能评级
        bottlenecks = data['performance_bottlenecks']
        high_severity = [b for b in bottlenecks if b['severity'] == 'high']
        
        if len(high_severity) > 0:
            report['summary']['overall_performance'] = "poor"
            report['summary']['optimization_priority'] = "high"
        elif len(bottlenecks) > 2:
            report['summary']['overall_performance'] = "fair"
            report['summary']['optimization_priority'] = "medium"
        
        return report
    
    def _save_echo_report(self, report_data: Dict[str, Any]):
        """
        保存性能报告
        
        Args:
            report_data: 报告数据
        """
        try:
            reports_dir = Path("echo_reports")
            reports_dir.mkdir(exist_ok=True)
            
            report_filename = f"{self.dna.target_name.replace('/', '_')}_echo_report.json"
            report_path = reports_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 回声性能报告已保存: {report_path}")
            
        except Exception as e:
            logger.error(f"保存性能报告失败: {e}")
