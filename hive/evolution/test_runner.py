#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Runner - 进化后自动测试验证

在应用进化变更后自动运行测试，确保变更不会破坏现有功能。
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Hive.Evolution.TestRunner")


@dataclass
class TestResult:
    """测试结果"""
    success: bool
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration: float = 0.0
    output: str = ""
    failed_tests: List[str] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 1.0
        return self.passed / self.total


class TestRunner:
    """
    测试运行器
    
    在进化变更后自动运行测试套件，验证变更的正确性。
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self._baseline_result: Optional[TestResult] = None
        self._last_result: Optional[TestResult] = None
    
    def run_pytest(
        self,
        test_path: Optional[str] = None,
        markers: Optional[List[str]] = None,
        timeout: int = 300,
        capture: bool = True,
    ) -> TestResult:
        """
        运行 pytest 测试
        
        Args:
            test_path: 测试路径，默认为 tests/
            markers: pytest markers 过滤
            timeout: 超时秒数
            capture: 是否捕获输出
            
        Returns:
            测试结果
        """
        result = TestResult(success=False)
        start_time = time.time()
        
        # 构建命令
        cmd = ["python", "-m", "pytest"]
        
        if test_path:
            cmd.append(test_path)
        else:
            cmd.append(str(self.project_root / "tests"))
        
        # 添加 markers
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # JSON 报告
        cmd.extend(["--tb=short", "-q", "--no-header"])
        
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=capture,
                text=True,
                timeout=timeout,
            )
            
            result.duration = time.time() - start_time
            result.output = proc.stdout + proc.stderr
            
            # 解析输出
            result = self._parse_pytest_output(result, proc.returncode)
            
            logger.info(
                f"🧪 测试完成: {result.passed}/{result.total} 通过 "
                f"({result.pass_rate:.1%}), 耗时 {result.duration:.1f}s"
            )
            
        except subprocess.TimeoutExpired:
            result.duration = timeout
            result.output = f"测试超时 ({timeout}s)"
            logger.error(f"测试超时: {timeout}s")
            
        except Exception as e:
            result.output = str(e)
            logger.error(f"测试运行失败: {e}")
        
        self._last_result = result
        return result
    
    def _parse_pytest_output(self, result: TestResult, returncode: int) -> TestResult:
        """解析 pytest 输出"""
        result.success = returncode == 0
        
        # 尝试从输出解析统计
        output = result.output
        
        # 查找类似 "5 passed, 2 failed, 1 skipped" 的行
        import re
        
        # 匹配 passed
        match = re.search(r"(\d+) passed", output)
        if match:
            result.passed = int(match.group(1))
        
        # 匹配 failed
        match = re.search(r"(\d+) failed", output)
        if match:
            result.failed = int(match.group(1))
        
        # 匹配 skipped
        match = re.search(r"(\d+) skipped", output)
        if match:
            result.skipped = int(match.group(1))
        
        # 匹配 error
        match = re.search(r"(\d+) error", output)
        if match:
            result.failed += int(match.group(1))
        
        result.total = result.passed + result.failed + result.skipped
        
        # 提取失败的测试名
        failed_pattern = re.findall(r"FAILED ([\w/:.]+)", output)
        result.failed_tests = failed_pattern
        
        return result
    
    def run_quick_tests(self) -> TestResult:
        """运行快速测试（单元测试）"""
        return self.run_pytest(
            markers=["not slow", "not integration"],
            timeout=120,
        )
    
    def run_integration_tests(self) -> TestResult:
        """运行集成测试"""
        return self.run_pytest(
            markers=["integration"],
            timeout=600,
        )
    
    def run_evolution_tests(self) -> TestResult:
        """运行进化相关测试"""
        return self.run_pytest(
            test_path=str(self.project_root / "hive" / "evolution"),
            timeout=180,
        )
    
    def establish_baseline(self) -> TestResult:
        """建立测试基线"""
        logger.info("📊 建立测试基线...")
        self._baseline_result = self.run_quick_tests()
        return self._baseline_result
    
    def compare_with_baseline(self, current: Optional[TestResult] = None) -> Dict[str, Any]:
        """
        与基线比较
        
        Returns:
            比较结果，包含是否退化
        """
        current = current or self._last_result
        baseline = self._baseline_result
        
        if not baseline:
            return {
                "has_baseline": False,
                "regression": False,
                "message": "无基线数据",
            }
        
        if not current:
            return {
                "has_baseline": True,
                "regression": True,
                "message": "无当前测试结果",
            }
        
        # 检查退化
        regression = False
        reasons = []
        
        # 通过率下降
        if current.pass_rate < baseline.pass_rate - 0.05:
            regression = True
            reasons.append(
                f"通过率下降: {baseline.pass_rate:.1%} -> {current.pass_rate:.1%}"
            )
        
        # 新增失败
        new_failures = set(current.failed_tests) - set(baseline.failed_tests)
        if new_failures:
            regression = True
            reasons.append(f"新增失败测试: {', '.join(list(new_failures)[:3])}")
        
        return {
            "has_baseline": True,
            "regression": regression,
            "baseline_pass_rate": baseline.pass_rate,
            "current_pass_rate": current.pass_rate,
            "new_failures": list(new_failures),
            "reasons": reasons,
            "message": "; ".join(reasons) if reasons else "测试通过，无退化",
        }
    
    def validate_evolution(self, gene_id: str) -> Tuple[bool, str]:
        """
        验证进化变更
        
        Args:
            gene_id: 基因 ID
            
        Returns:
            (是否通过, 消息)
        """
        logger.info(f"🔬 验证进化变更: {gene_id}")
        
        # 运行快速测试
        result = self.run_quick_tests()
        
        if not result.success:
            return False, f"测试失败: {result.failed} 个测试未通过"
        
        # 与基线比较
        comparison = self.compare_with_baseline(result)
        
        if comparison.get("regression"):
            return False, f"检测到退化: {comparison.get('message')}"
        
        return True, f"验证通过: {result.passed}/{result.total} 测试通过"


# 全局测试运行器
_runner: Optional[TestRunner] = None


def get_test_runner() -> TestRunner:
    """获取全局测试运行器"""
    global _runner
    if _runner is None:
        _runner = TestRunner()
    return _runner
