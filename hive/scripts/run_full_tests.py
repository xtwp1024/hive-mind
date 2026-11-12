# -*- coding: utf-8 -*-
"""
阶段 A: 完整 pytest 测试执行脚本

功能:
1. 执行所有阶段的 pytest 测试
2. 生成详细的测试报告
3. 记录失败项和修复建议
4. 输出汇总结果

使用方法:
    python hive/scripts/run_full_tests.py
    python hive/scripts/run_full_tests.py --phase 3
    python hive/scripts/run_full_tests.py --phase 2,3,4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIP = "SKIP"


@dataclass
class TestResult:
    """单个测试结果"""

    phase: str
    test_file: str
    test_name: str
    status: TestStatus
    duration: float = 0.0
    error_message: Optional[str] = None


@dataclass
class PhaseResult:
    """阶段测试结果"""

    phase: str
    phase_name: str
    test_files: List[str]
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total: int = 0
    duration: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errors == 0


PHASES = {
    "1": {
        "name": "Phase 1: 认知觉醒",
        "files": [
            "hive/tests/test_phase1_cognitive.py",
        ],
        "fast_files": [
            "hive/tests/verify_phase1_2.py",
        ],
    },
    "2": {
        "name": "Phase 2: 群体智慧",
        "files": [
            "hive/tests/test_phase2_communication.py",
            "hive/tests/test_phase2_consensus.py",
            "hive/tests/test_phase2_collaborative.py",
        ],
        "fast_files": [
            "hive/tests/verify_phase1_2.py",
        ],
    },
    "3": {
        "name": "Phase 3: 自主进化能力",
        "files": [
            "hive/tests/test_phase3_evolution.py",
        ],
        "fast_files": [
            "hive/tests/verify_phase3.py",
        ],
    },
    "4": {
        "name": "Phase 4: 多模态扩展",
        "files": [
            "hive/tests/test_phase4_multimodal.py",
        ],
        "fast_files": [
            "hive/tests/verify_phase4.py",
        ],
    },
    "5": {
        "name": "Phase 5: 分布式网络",
        "files": [
            "hive/tests/test_hive_federation_http.py",
            "hive/tests/test_hive_federation_integration.py",
        ],
        "fast_files": [
            "hive/tests/verify_phase5.py",
        ],
    },
    "prd": {
        "name": "PRD UI 控制台",
        "files": [],
        "fast_files": [
            "hive/tests/verify_prd_ui.py",
        ],
    },
}


def run_pytest(test_file: str, verbose: bool = True) -> Tuple[Dict[str, Any], str]:
    """
    运行 pytest 测试

    Returns:
        (result_dict, output_string)
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "-v",
        "--tb=short",
        "--json-report",
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=300,
        )

        duration = time.time() - start_time

        output = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode(
            "utf-8", errors="replace"
        )

        # 解析 JSON 报告
        try:
            # pytest-json-report 输出到 stdout 末尾
            # 尝试在输出中查找 JSON
            import re

            json_match = re.search(r'\{.*"collectors".*\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_result = json.loads(json_str)
            else:
                json_result = {}
        except json.JSONDecodeError:
            json_result = {}

        # 统计结果
        passed = output.count("PASSED")
        failed = output.count("FAILED")
        errors = output.count("ERROR")
        skipped = output.count("SKIPPED")

        total = passed + failed + errors

        success = failed == 0 and errors == 0

        return {
            "success": success,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "total": total,
            "duration": duration,
            "json_result": json_result,
            "returncode": result.returncode,
        }, output

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            "success": False,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "total": 0,
            "duration": duration,
            "error": "Timeout (300s)",
            "returncode": -1,
        }, "Test timed out after 300 seconds"
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "total": 0,
            "duration": duration,
            "error": str(e),
            "returncode": -2,
        }, f"Test error: {e}"


def run_verify_script(test_file: str) -> Tuple[bool, str]:
    """
    运行快速验证脚本 (非 pytest)
    """
    cmd = [sys.executable, test_file]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=60,
        )

        duration = time.time() - start_time
        output = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode(
            "utf-8", errors="replace"
        )

        success = result.returncode == 0

        return success, output

    except subprocess.TimeoutExpired:
        return False, "Test timed out after 60 seconds"
    except Exception as e:
        return False, f"Test error: {e}"


def get_test_files(phase: str, use_fast: bool = False) -> List[str]:
    """获取阶段的测试文件列表"""
    phase_info = PHASES.get(phase, {})
    if use_fast and "fast_files" in phase_info:
        return phase_info.get("fast_files", [])
    return phase_info.get("files", [])


def run_phase_tests(phase: str, use_fast: bool = False, verbose: bool = True) -> PhaseResult:
    """运行单个阶段的所有测试"""
    phase_info = PHASES.get(phase, {})
    phase_name = phase_info.get("name", f"Phase {phase}")

    test_files = get_test_files(phase, use_fast)

    result = PhaseResult(
        phase=phase,
        phase_name=phase_name,
        test_files=test_files,
    )

    print()
    print("=" * 60)
    print(f"运行 {phase_name} 测试")
    print("=" * 60)

    if not test_files:
        print(f"  [跳过] 没有找到测试文件")
        return result

    phase_start = time.time()

    for test_file in test_files:
        test_path = PROJECT_ROOT / test_file

        if not test_path.exists():
            print(f"  [跳过] 文件不存在: {test_file}")
            continue

        print()
        print(f"  测试文件: {test_file}")
        print(f"  {'-' * 50}")

        # 检查是 pytest 还是普通脚本
        if test_file.startswith("hive/tests/test_"):
            # pytest 测试
            test_result, output = run_pytest(test_file, verbose)

            if verbose:
                print(output)

            result.passed += test_result.get("passed", 0)
            result.failed += test_result.get("failed", 0)
            result.errors += test_result.get("errors", 0)
            result.skipped += test_result.get("skipped", 0)
            result.total += test_result.get("total", 0)

            if test_result.get("success"):
                print(f"  [通过] {test_result['passed']} 个测试通过")
            else:
                print(
                    f"  [失败] {test_result.get('failed', 0)} 失败, {test_result.get('errors', 0)} 错误"
                )

                # 记录失败信息
                result.failures.append(
                    {
                        "file": test_file,
                        "passed": test_result.get("passed", 0),
                        "failed": test_result.get("failed", 0),
                        "errors": test_result.get("errors", 0),
                        "output": output if len(output) < 5000 else output[:5000] + "...",
                    }
                )
        else:
            # 普通验证脚本
            success, output = run_verify_script(test_file)

            if verbose:
                print(output)

            result.total += 1

            if success:
                result.passed += 1
                print(f"  [通过] 测试脚本执行成功")
            else:
                result.failed += 1
                print(f"  [失败] 测试脚本执行失败")
                result.failures.append(
                    {
                        "file": test_file,
                        "output": output if len(output) < 5000 else output[:5000] + "...",
                    }
                )

    result.duration = time.time() - phase_start

    print()
    print(f"  阶段汇总: {result.passed} 通过, {result.failed} 失败, {result.errors} 错误")
    print(f"  成功率: {result.success_rate:.1f}%")
    print(f"  耗时: {result.duration:.2f} 秒")

    return result


def generate_report(all_results: List[PhaseResult], output_file: str) -> Dict[str, Any]:
    """
    生成测试报告
    """
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_errors = sum(r.errors for r in all_results)
    total_skipped = sum(r.skipped for r in all_results)
    total = sum(r.total for r in all_results)
    total_duration = sum(r.duration for r in all_results)

    all_failures = []
    for r in all_results:
        for f in r.failures:
            all_failures.append(
                {
                    "phase": r.phase,
                    "phase_name": r.phase_name,
                    **f,
                }
            )

    success_rate = (total_passed / total) * 100 if total > 0 else 0

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_phases": len(all_results),
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "total_skipped": total_skipped,
            "total_tests": total,
            "total_duration": total_duration,
            "success_rate": success_rate,
            "all_passed": total_failed == 0 and total_errors == 0,
        },
        "phases": [
            {
                "phase": r.phase,
                "phase_name": r.phase_name,
                "passed": r.passed,
                "failed": r.failed,
                "errors": r.errors,
                "skipped": r.skipped,
                "total": r.total,
                "duration": r.duration,
                "success_rate": r.success_rate,
                "all_passed": r.all_passed,
                "test_files": r.test_files,
                "failures": r.failures,
            }
            for r in all_results
        ],
        "all_failures": all_failures,
        "recommendations": generate_recommendations(all_failures),
    }

    # 写入 JSON
    json_path = PROJECT_ROOT / output_file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 写入 Markdown
    md_path = json_path.with_suffix(".md")
    md_content = generate_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return report


def generate_recommendations(failures: List[Dict]) -> List[Dict]:
    """
    根据失败项生成修复建议
    """
    recommendations = []

    for failure in failures:
        phase = failure.get("phase", "")
        file = failure.get("file", "")
        output = failure.get("output", "")

        rec = {
            "phase": phase,
            "file": file,
            "issues": [],
            "suggestions": [],
        }

        # 分析输出中的常见错误
        output_lower = output.lower()

        if "import error" in output_lower or "modulenotfound" in output_lower:
            rec["issues"].append("模块导入错误")
            rec["suggestions"].append("检查模块路径和导入语句")
            rec["suggestions"].append("运行 pip install -r requirements.txt")

        if "assertion" in output_lower and "fail" in output_lower:
            rec["issues"].append("断言失败")
            rec["suggestions"].append("检查测试预期值与实际值")
            rec["suggestions"].append("验证测试数据是否正确")

        if "timeout" in output_lower:
            rec["issues"].append("测试超时")
            rec["suggestions"].append("检查测试是否存在死循环")
            rec["suggestions"].append("考虑增加超时时间")

        if "connection" in output_lower and "refused" in output_lower:
            rec["issues"].append("网络连接被拒绝")
            rec["suggestions"].append("检查目标服务是否已启动")
            rec["suggestions"].append("验证端口和主机配置")

        if "attributeerror" in output_lower:
            rec["issues"].append("属性错误")
            rec["suggestions"].append("检查对象是否正确初始化")
            rec["suggestions"].append("验证属性名是否正确")

        if "typeerror" in output_lower:
            rec["issues"].append("类型错误")
            rec["suggestions"].append("检查参数类型是否正确")
            rec["suggestions"].append("验证函数签名")

        if not rec["issues"]:
            rec["issues"].append("未知错误")
            rec["suggestions"].append("查看详细输出了解具体问题")
            rec["suggestions"].append("运行单个测试定位问题")

        recommendations.append(rec)

    return recommendations


def generate_markdown_report(report: Dict) -> str:
    """生成 Markdown 格式的测试报告"""
    summary = report["summary"]
    phases = report["phases"]
    failures = report["all_failures"]
    recommendations = report["recommendations"]

    md = f"""# 📋 Hive 完整测试报告

> 生成时间: {report['generated_at']}

---

## 📊 汇总

| 指标 | 值 |
|------|-----|
| **测试阶段** | {summary['total_phases']} |
| **通过测试** | {summary['total_passed']} |
| **失败测试** | {summary['total_failed']} |
| **错误测试** | {summary['total_errors']} |
| **跳过测试** | {summary['total_skipped']} |
| **总测试数** | {summary['total_tests']} |
| **总耗时** | {summary['total_duration']:.2f} 秒 |
| **成功率** | {summary['success_rate']:.1f}% |

**状态:** {'✅ 全部通过' if summary['all_passed'] else '❌ 存在失败'}

---

## 📈 各阶段结果

"""

    for phase in phases:
        status_emoji = "✅" if phase["all_passed"] else "❌"
        md += f"""### {phase['phase_name']} {status_emoji}

| 指标 | 值 |
|------|-----|
| 通过 | {phase['passed']} |
| 失败 | {phase['failed']} |
| 错误 | {phase['errors']} |
| 跳过 | {phase['skipped']} |
| 总计 | {phase['total']} |
| 耗时 | {phase['duration']:.2f} 秒 |
| 成功率 | {phase['success_rate']:.1f}% |

"""

    if failures:
        md += """---

## ❌ 失败详情

"""

        for i, failure in enumerate(failures, 1):
            md += f"""### 失败 {i}: {failure['phase_name']} - {failure['file']}

```
{failure.get('output', 'N/A')[:3000]}
```

"""

    if recommendations:
        md += """---

## 💡 修复建议

"""

        for rec in recommendations:
            md += f"""### {rec['phase']} - {rec['file']}

**发现的问题:**
"""
            for issue in rec["issues"]:
                md += f"- {issue}\n"

            md += f"""
**建议方案:**
"""
            for suggestion in rec["suggestions"]:
                md += f"- {suggestion}\n"
            md += "\n"

    md += """---

## 🛠️ 快速命令

```bash
# 运行所有快速验证
python hive/scripts/run_full_tests.py --fast

# 运行指定阶段 (如 phase 3)
python hive/scripts/run_full_tests.py --phase 3

# 运行多个阶段
python hive/scripts/run_full_tests.py --phase 2,3,4

# 运行完整 pytest (较慢)
python hive/scripts/run_full_tests.py --full
```

---

*报告生成于: """
    md += report["generated_at"]
    md += """*
"""

    return md


def main():
    """主函数"""
    epilog_text = """
示例:
    # 运行所有快速验证
    python hive/scripts/run_full_tests.py
    
    # 运行所有完整 pytest
    python hive/scripts/run_full_tests.py --full
    
    # 运行指定阶段
    python hive/scripts/run_full_tests.py --phase 3
    python hive/scripts/run_full_tests.py --phase 2,3,4
    
    # 详细输出
    python hive/scripts/run_full_tests.py -v
"""

    parser = argparse.ArgumentParser(
        description="Hive 完整测试执行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text,
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        help="指定阶段 (1,2,3,4,5,prd 或 all)",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="运行完整 pytest (默认使用快速验证)",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="运行快速验证 (默认)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="test_report.json",
        help="输出报告文件名 (JSON 和 MD)",
    )

    args = parser.parse_args()

    # 确定使用快速还是完整测试
    use_fast = not args.full

    # 确定要运行的阶段
    if args.phase == "all":
        phases = list(PHASES.keys())
    else:
        phases = [p.strip() for p in args.phase.split(",")]

    # 过滤不存在的阶段
    phases = [p for p in phases if p in PHASES]

    if not phases:
        print("错误: 没有有效的阶段指定")
        print(f"可用阶段: {list(PHASES.keys())}")
        return 1

    # 打印头部
    print()
    print("=" * 60)
    print("Hive 完整测试执行器")
    print("=" * 60)
    print()
    print(f"测试模式: {'完整 pytest' if not use_fast else '快速验证'}")
    print(f"测试阶段: {', '.join(phases)}")
    print(f"详细输出: {'是' if args.verbose else '否'}")
    print()

    # 运行测试
    all_results = []

    for phase in phases:
        result = run_phase_tests(phase, use_fast, args.verbose)
        all_results.append(result)

    # 生成报告
    print()
    print("=" * 60)
    print("生成测试报告...")
    print("=" * 60)

    report = generate_report(all_results, args.output)

    summary = report["summary"]

    print()
    print("=" * 60)
    print("测试汇总")
    print("=" * 60)
    print()
    print(f"  总阶段: {summary['total_phases']}")
    print(f"  总测试: {summary['total_tests']}")
    print(f"  通过:   {summary['total_passed']}")
    print(f"  失败:   {summary['total_failed']}")
    print(f"  错误:   {summary['total_errors']}")
    print(f"  跳过:   {summary['total_skipped']}")
    print(f"  耗时:   {summary['total_duration']:.2f} 秒")
    print(f"  成功率: {summary['success_rate']:.1f}%")
    print()

    if summary["all_passed"]:
        print("  ✅ 所有测试通过!")
    else:
        print(f"  ❌ 存在 {summary['total_failed'] + summary['total_errors']} 个失败/错误")
        print(f"  📄 查看详情: {args.output} (JSON)")
        print(f"  📄 查看详情: {Path(args.output).with_suffix('.md')} (Markdown)")

    print()
    print("=" * 60)

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
