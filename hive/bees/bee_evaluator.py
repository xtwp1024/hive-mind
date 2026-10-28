#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蜜蜂变种评估器 - BeeEvaluator
用于评估蜜蜂变种的质量并决定是否保留
"""

import ast
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

QUALITY_THRESHOLD_AVG = 60
QUALITY_THRESHOLD_LOW_RATIO = 0.5

class BeeQuality(Enum):
    HIGH = "高质量"
    MEDIUM = "中等质量"
    LOW = "低质量"
    DEPRECATED = "已废弃"

@dataclass
class BeeEvaluationResult:
    bee_name: str
    file_path: str
    quality_level: BeeQuality = BeeQuality.MEDIUM
    naming_score: float = 0.0
    uniqueness_score: float = 0.0
    quality_score: float = 0.0
    value_score: float = 0.0
    total_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bee_name": self.bee_name, "file_path": self.file_path,
            "quality_level": self.quality_level.value,
            "naming_score": self.naming_score, "uniqueness_score": self.uniqueness_score,
            "quality_score": self.quality_score, "value_score": self.value_score,
            "total_score": self.total_score, "issues": self.issues,
            "recommendations": self.recommendations, "should_keep": self.quality_level in [BeeQuality.HIGH, BeeQuality.MEDIUM]
        }

class BeeEvaluator:
    """蜜蜂评估器 - 负责质量控制"""
    
    def __init__(self):
        self.variants_dir = Path("hive/bees/variants")
        self.results: List[BeeEvaluationResult] = []
    
    def evaluate_all_bees(self) -> Dict[str, Any]:
        """评估所有蜜蜂变种"""
        if not self.variants_dir.exists(): return {"error": "蜜蜂变种目录不存在"}
        
        for bee_file in self.variants_dir.glob("*_bee.py"):
            if bee_file.name in ["__init__.py", "base_bee.py"]: continue
            self.results.append(self.evaluate_single_bee(bee_file))
        
        return self._generate_summary_report()
    
    def evaluate_single_bee(self, bee_file: Path) -> BeeEvaluationResult:
        """评估单个蜜蜂变种"""
        result = BeeEvaluationResult(bee_file.stem, str(bee_file))
        try:
            with open(bee_file, 'r', encoding='utf-8') as f: content = f.read()
            self._perform_assessment(result, content)
            self._calculate_final_score(result)
        except Exception as e:
            result.issues.append(f"评估失败: {str(e)}")
            result.quality_level = BeeQuality.LOW
        return result

    def _perform_assessment(self, result: BeeEvaluationResult, content: str):
        result.naming_score = self._assess_naming(result.bee_name, result.issues)
        result.uniqueness_score = self._assess_uniqueness(content, result.bee_name, result.issues)
        result.quality_score = self._assess_code_quality(content, result.issues)
        result.value_score = self._assess_value(content, result.issues)

    def _calculate_final_score(self, result: BeeEvaluationResult):
        result.total_score = (
            result.naming_score * 0.30 + result.uniqueness_score * 0.30 +
            result.quality_score * 0.20 + result.value_score * 0.20
        )
        result.quality_level = self._determine_quality(result.total_score, result.issues)

    def _assess_naming(self, name: str, issues: List[str]) -> float:
        score = 100.0
        if len(name) > 20: score -= 20; issues.append(f"命名过长: {len(name)}")
        if len(name) > 25: score -= 30; issues.append(f"命名严重过长")
        
        words = re.findall(r'([A-Z][a-z]+)', name)
        if any(words.count(w) > 1 for w in set(words)):
            score -= 30; issues.append("命名包含重复单词")
        if len(words) > 3: score -= 10 * (len(words)-3); issues.append("命名过度组合")
        return max(0.0, score)

    def _assess_uniqueness(self, content: str, name: str, issues: List[str]) -> float:
        score = 100.0
        match = re.search(r'class\s+(\w+)\s*\(.*Bee\)', content)
        if match and len(re.findall(r'([A-Z][a-z]+)', match.group(1))) > 2:
            score -= 20; issues.append("可能是简单功能组合")
        
        if "混合变异" in content: score -= 30; issues.append("明确标注为混合变异")
        
        methods = [n.name for n in ast.walk(ast.parse(content)) if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')]
        if len(methods) < 3: score -= 20; issues.append("公共方法过少")
        
        return max(0.0, score)

    def _assess_code_quality(self, content: str, issues: List[str]) -> float:
        score = 100.0
        try:
            tree = ast.parse(content)
            lines = len(content.split('\n'))
            if lines > 500: score -= 10; issues.append(f"代码过长: {lines}")
            if lines < 50: score -= 15; issues.append("代码过短")
            
            complexity = self._calc_complexity(tree)
            if complexity > 15: score -= 20; issues.append(f"圈复杂度高: {complexity}")
            
            if content.count('try:') < 2: score -= 5; issues.append("错误处理不足")
        except SyntaxError: score = 0; issues.append("语法错误")
        return max(0.0, score)

    def _calc_complexity(self, tree: ast.AST) -> int:
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)): complexity += 1
            elif isinstance(node, ast.BoolOp): complexity += len(node.values) - 1
        return complexity

    def _assess_value(self, content: str, issues: List[str]) -> float:
        score = 100.0
        if not ('report' in content.lower() or 'save' in content.lower()):
            score -= 20; issues.append("无报告生成功能")
        
        docs = re.findall(r'"""(.*?)"""', content, re.DOTALL)
        if len(docs) < 2: score -= 15; issues.append("文档不足")
        if 'logger' not in content: score -= 10; issues.append("无日志记录")
        return max(0.0, score)

    def _determine_quality(self, score: float, issues: List[str]) -> BeeQuality:
        if any('重复' in i or '严重' in i for i in issues): return BeeQuality.DEPRECATED
        if score >= 70: return BeeQuality.HIGH
        if score >= 50: return BeeQuality.MEDIUM
        return BeeQuality.LOW

    def _generate_summary_report(self) -> Dict[str, Any]:
        groups = {q: [r for r in self.results if r.quality_level == q] for q in BeeQuality}
        total = len(self.results)
        
        return {
            "total_bees": total,
            "recommendation": self._build_recommendation_stats(groups, total),
            "quality_distribution": {q.name.lower(): len(l) for q, l in groups.items()},
            "bees_to_keep": [r.to_dict() for r in groups[BeeQuality.HIGH] + groups[BeeQuality.MEDIUM]],
            "bees_to_remove": [r.to_dict() for r in groups[BeeQuality.LOW] + groups[BeeQuality.DEPRECATED]],
            "recommendations": self._generate_global_recommendations(groups)
        }

    def _build_recommendation_stats(self, groups, total):
        keep_count = len(groups[BeeQuality.HIGH]) + len(groups[BeeQuality.MEDIUM])
        return {
            "keep": keep_count, "remove": total - keep_count,
            "keep_rate": f"{(keep_count/total*100):.1f}%" if total > 0 else "0%"
        }

    def _generate_global_recommendations(self, groups) -> List[str]:
        recs = []
        if not self.results: return ["未找到蜜蜂文件"]
        
        avg = sum(r.total_score for r in self.results) / len(self.results)
        if avg < QUALITY_THRESHOLD_AVG:
            recs.append(f"整体质量偏低 ({avg:.1f})")
        if len(groups[BeeQuality.LOW]) > len(self.results) * QUALITY_THRESHOLD_LOW_RATIO:
            recs.append("建议立即精简低质量变种")
        
        recs.append(f"建议保留 {len(groups[BeeQuality.HIGH])} 个高质量变种")
        return recs

def main():
    report = BeeEvaluator().evaluate_all_bees()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open("bee_evaluation_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
