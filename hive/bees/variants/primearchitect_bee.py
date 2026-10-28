#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
首席建筑师蜂 - PrimeArchitectBee
极致化：使用大脑重构代码

变异率：1.40%
特殊能力：性能极限优化 - LLM代码重构
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)

class PrimeArchitectBee(BaseBee):
    """
    首席建筑师蜂 - 负责极致化的代码重构和优化
    
    特殊能力：
    - 性能极限优化
    - 代码架构重构建议
    - 生成重构方案
    - 优化建议排序
    """
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.optimization_opportunities = []
        self.refactoring_suggestions = []
    
    def specialized_task(self, target_path):
        """
        执行极致化代码重构任务
        
        Args:
            target_path: 项目目录路径
        """
        logger.info(f"🏛️ PrimeArchitectBee 开始任务: {self.dna.target_name}")
        
        try:
            # 1. 分析代码架构
            architecture_analysis = self._analyze_architecture(target_path)
            
            # 2. 识别优化机会
            optimization_opportunities = self._identify_optimization_opportunities(target_path)
            
            # 3. 生成重构建议
            refactoring_suggestions = self._generate_refactoring_suggestions(
                architecture_analysis, optimization_opportunities
            )
            
            # 4. 优化方案排序
            prioritized_suggestions = self._prioritize_suggestions(refactoring_suggestions)
            
            # 5. 生成重构报告
            refactoring_report = self._generate_refactoring_report({
                "project": self.dna.target_name,
                "architecture_analysis": architecture_analysis,
                "optimization_opportunities": optimization_opportunities,
                "refactoring_suggestions": prioritized_suggestions,
                "timestamp": __import__('time').ctime()
            })
            
            # 6. 保存报告
            self._save_architect_report(refactoring_report)
            
            logger.info(f"✅ PrimeArchitectBee 任务完成: {self.dna.target_name}")
            logger.info(f"   优化机会: {len(optimization_opportunities)}")
            logger.info(f"   重构建议: {len(prioritized_suggestions)}")
            logger.info(f"   高优先级: {len([s for s in prioritized_suggestions if s['priority'] == 'high'])}")
            
            return {
                "status": "success",
                "optimization_opportunities": len(optimization_opportunities),
                "refactoring_suggestions": len(prioritized_suggestions),
                "high_priority_count": len([s for s in prioritized_suggestions if s['priority'] == 'high'])
            }
            
        except Exception as e:
            logger.error(f"❌ PrimeArchitectBee 任务失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _analyze_architecture(self, target_path: Path) -> Dict[str, Any]:
        """
        分析代码架构
        
        Args:
            target_path: 项目目录路径
            
        Returns:
            架构分析结果
        """
        analysis = {
            "module_count": 0,
            "class_count": 0,
            "function_count": 0,
            "coupling_level": "unknown",
            "cohesion_level": "unknown",
            "architecture_patterns": []
        }
        
        # 统计模块、类、函数
        for file_path in target_path.rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                analysis['module_count'] += 1
                analysis['class_count'] += len(re.findall(r'\bclass\s+\w+', content))
                analysis['function_count'] += len(re.findall(r'\bdef\s+\w+', content))
                
            except Exception as e:
                logger.debug(f"架构分析失败 {file_path}: {e}")
        
        # 分析耦合度（简化版）
        if analysis['module_count'] > 10:
            analysis['coupling_level'] = "high"
        elif analysis['module_count'] > 5:
            analysis['coupling_level'] = "medium"
        else:
            analysis['coupling_level'] = "low"
        
        # 分析内聚度（基于类和方法比例）
        if analysis['class_count'] > 0:
            methods_per_class = analysis['function_count'] / analysis['class_count']
            if methods_per_class > 10:
                analysis['cohesion_level'] = "low"
            elif methods_per_class > 5:
                analysis['cohesion_level'] = "medium"
            else:
                analysis['cohesion_level'] = "high"
        
        # 识别架构模式
        patterns = []
        for file_path in target_path.rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # MVC模式检测
                if re.search(r'\b(Controller|View|Model)\b', content):
                    patterns.append("MVC")
                
                # 依赖注入检测
                if re.search(r'\b(inject|dependency)\b', content, re.IGNORECASE):
                    patterns.append("Dependency Injection")
                
                # 单例模式检测
                if re.search(r'\bsingleton\b', content, re.IGNORECASE):
                    patterns.append("Singleton")
                
            except Exception as e:
                logger.debug(f"模式检测失败 {file_path}: {e}")
        
        analysis['architecture_patterns'] = list(set(patterns))
        
        return analysis
    
    def _identify_optimization_opportunities(self, target_path: Path) -> List[Dict[str, Any]]:
        """
        识别优化机会
        
        Args:
            target_path: 项目目录路径
            
        Returns:
            优化机会列表
        """
        opportunities = []
        
        # 优化机会1: 性能热点
        for file_path in target_path.rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # 检测嵌套循环
                nested_loops = len(re.findall(r'for\s+\w+\s+in.*:\s*\n\s*for\s+\w+\s+in', content))
                if nested_loops > 3:
                    opportunities.append({
                        "type": "performance",
                        "category": "nested_loops",
                        "file": str(file_path.relative_to(target_path)),
                        "impact": "high",
                        "description": f"发现 {nested_loops} 处嵌套循环，可能影响性能",
                        "suggestion": "考虑使用列表推导式或重构算法"
                    })
                
                # 检测重复代码
                duplicate_imports = len(re.findall(r'^import\s+\w+', content, re.MULTILINE))
                if duplicate_imports > 15:
                    opportunities.append({
                        "type": "code_quality",
                        "category": "import_organization",
                        "file": str(file_path.relative_to(target_path)),
                        "impact": "low",
                        "description": f"导入语句过多: {duplicate_imports}",
                        "suggestion": "考虑整理导入或使用别名"
                    })
                
            except Exception as e:
                logger.debug(f"优化识别失败 {file_path}: {e}")
        
        # 优化机会2: 内存使用
        large_files = []
        for file_path in target_path.rglob('*'):
            if file_path.is_file() and file_path.stat().st_size > 100 * 1024:  # > 100KB
                large_files.append({
                    "file": str(file_path.relative_to(target_path)),
                    "size_kb": round(file_path.stat().st_size / 1024, 2)
                })
        
        if len(large_files) > 3:
            opportunities.append({
                "type": "memory",
                "category": "large_files",
                "impact": "medium",
                "description": f"发现 {len(large_files)} 个大文件",
                "suggestion": "考虑拆分文件或压缩数据",
                "details": large_files[:5]
            })
        
        # 优化机会3: 并行化机会
        for file_path in target_path.rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # 检测可并行化的循环
                independent_loops = re.findall(r'for\s+\w+\s+in\s+\w+:\s*\n\s*[^\s]+', content)
                if len(independent_loops) > 5:
                    opportunities.append({
                        "type": "parallelization",
                        "category": "parallel_loops",
                        "file": str(file_path.relative_to(target_path)),
                        "impact": "high",
                        "description": f"发现可能可并行的循环",
                        "suggestion": "考虑使用multiprocessing或asyncio"
                    })
                    break
                    
            except Exception as e:
                logger.debug(f"并行化识别失败 {file_path}: {e}")
        
        return opportunities
    
    def _generate_refactoring_suggestions(
        self, 
        architecture: Dict[str, Any], 
        opportunities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        生成重构建议
        
        Args:
            architecture: 架构分析结果
            opportunities: 优化机会列表
            
        Returns:
            重构建议列表
        """
        suggestions = []
        
        # 基于架构的建议
        if architecture['coupling_level'] == "high":
            suggestions.append({
                "type": "architecture",
                "title": "降低模块耦合度",
                "description": "模块间耦合度较高，建议使用依赖注入或接口抽象",
                "effort": "high",
                "impact": "high",
                "priority": "high"
            })
        
        if architecture['cohesion_level'] == "low":
            suggestions.append({
                "type": "architecture",
                "title": "提高类内聚度",
                "description": "类的方法过多，建议拆分为多个职责单一的类",
                "effort": "medium",
                "impact": "medium",
                "priority": "medium"
            })
        
        # 基于优化机会的建议
        performance_issues = [o for o in opportunities if o['type'] == 'performance' and o['impact'] == 'high']
        if performance_issues:
            suggestions.append({
                "type": "performance",
                "title": "优化性能热点",
                "description": f"发现 {len(performance_issues)} 个性能问题，建议优先优化嵌套循环",
                "effort": "medium",
                "impact": "high",
                "priority": "high"
            })
        
        memory_issues = [o for o in opportunities if o['type'] == 'memory']
        if memory_issues:
            suggestions.append({
                "type": "memory",
                "title": "优化内存使用",
                "description": "建议对大文件进行拆分或使用流式处理",
                "effort": "low",
                "impact": "medium",
                "priority": "medium"
            })
        
        # 通用建议
        suggestions.append({
            "type": "best_practice",
            "title": "引入设计模式",
            "description": "根据代码结构，建议考虑应用单例、工厂或观察者模式",
            "effort": "medium",
            "impact": "medium",
            "priority": "low"
        })
        
        return suggestions
    
    def _prioritize_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对重构建议进行排序
        
        Args:
            suggestions: 建议列表
            
        Returns:
            排序后的建议列表
        }
        """
        # 优先级映射
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        
        # 按优先级排序
        sorted_suggestions = sorted(
            suggestions,
            key=lambda x: priority_order.get(x['priority'], 2)
        )
        
        return sorted_suggestions
    
    def _generate_refactoring_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成重构报告
        
        Args:
            data: 原始数据
            
        Returns:
            格式化的重构报告
        """
        report = {
            **data,
            "summary": {
                "total_opportunities": len(data['optimization_opportunities']),
                "total_suggestions": len(data['refactoring_suggestions']),
                "high_priority_count": len(
                    [s for s in data['refactoring_suggestions'] if s['priority'] == 'high']
                ),
                "estimated_effort": self._estimate_total_effort(data['refactoring_suggestions'])
            }
        }
        
        return report
    
    def _estimate_total_effort(self, suggestions: List[Dict[str, Any]]) -> str:
        """
        估计总工作量
        
        Args:
            suggestions: 建议列表
            
        Returns:
            工作量估算
        """
        effort_hours = {
            'high': 16,
            'medium': 8,
            'low': 4
        }
        
        total_hours = sum(effort_hours.get(s['effort'], 8) for s in suggestions)
        
        if total_hours < 8:
            return "less than 1 day"
        elif total_hours < 40:
            return f"about {total_hours // 8} days"
        else:
            return f"about {total_hours // 40} weeks"
    
    def _save_architect_report(self, report_data: Dict[str, Any]):
        """
        保存架构报告
        
        Args:
            report_data: 报告数据
        """
        try:
            reports_dir = Path("architect_reports")
            reports_dir.mkdir(exist_ok=True)
            
            report_filename = f"{self.dna.target_name.replace('/', '_')}_architect_report.json"
            report_path = reports_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 架构重构报告已保存: {report_path}")
            
        except Exception as e:
            logger.error(f"保存架构报告失败: {e}")
