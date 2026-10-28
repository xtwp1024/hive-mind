# -*- coding: utf-8 -*-
"""
阶段 F: 代码清理 - 自动化脚本

功能:
1. 运行 ruff 检查
2. 运行 black 格式化检查
3. 运行 mypy 类型检查
4. 运行 isort 导入排序检查
5. 生成清理报告

使用方法:
    python hive/scripts/run_cleanup.py
    python hive/scripts/run_cleanup.py --check ruff
    python hive/scripts/run_cleanup.py --check black,mypy
    python hive/scripts/run_cleanup.py --fix
"""

import sys
import os
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class CheckResult:
    """检查结果"""
    name: str
    success: bool
    output: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration: float = 0.0


@dataclass
class CleanupReport:
    """清理报告"""
    timestamp: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[CheckResult]
    summary: str = ""


def run_command(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """运行命令"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or str(PROJECT_ROOT),
            capture_output=True,
            timeout=300,
        )
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        return result.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out after 300 seconds"
    except Exception as e:
        return -1, "", str(e)


def check_tool_available(tool_name: str) -> bool:
    """检查工具是否可用"""
    returncode, _, _ = run_command([sys.executable, "-m", tool_name, "--version"])
    return returncode == 0


def run_ruff_check(fix: bool = False) -> CheckResult:
    """运行 ruff 检查"""
    import time
    
    start_time = time.time()
    
    # 检查工具
    if not check_tool_available("ruff"):
        return CheckResult(
            name="ruff",
            success=False,
            output="ruff not available. Install with: pip install ruff",
            errors=["ruff not installed"],
            duration=time.time() - start_time,
        )
    
    # 构建命令
    cmd = [sys.executable, "-m", "ruff", "check", "hive/"]
    
    if fix:
        cmd.append("--fix")
    
    # 添加常见检查
    cmd.extend([
        "--select=F,E,W",  # Flake8 风格错误
        "--ignore=E501",   # 忽略行长度 (black 会处理)
    ])
    
    returncode, stdout, stderr = run_command(cmd)
    
    output = stdout + stderr
    errors = []
    warnings = []
    
    # 解析输出
    for line in output.split("\n"):
        if "error" in line.lower() and ":" in line:
            errors.append(line)
        elif "warning" in line.lower() and ":" in line:
            warnings.append(line)
    
    success = returncode == 0
    
    return CheckResult(
        name="ruff",
        success=success,
        output=output,
        errors=errors,
        warnings=warnings,
        duration=time.time() - start_time,
    )


def run_black_check(fix: bool = False) -> CheckResult:
    """运行 black 格式化检查"""
    import time
    
    start_time = time.time()
    
    if not check_tool_available("black"):
        return CheckResult(
            name="black",
            success=False,
            output="black not available. Install with: pip install black",
            errors=["black not installed"],
            duration=time.time() - start_time,
        )
    
    cmd = [sys.executable, "-m", "black"]
    
    if fix:
        cmd.append("hive/")
    else:
        cmd.extend(["--check", "hive/"])
    
    # 使用配置中的选项
    cmd.extend(["--line-length", "100"])
    
    returncode, stdout, stderr = run_command(cmd)
    
    output = stdout + stderr
    errors = []
    
    if "would reformat" in output or returncode != 0:
        if not fix:
            errors.append("Code needs formatting")
    
    success = returncode == 0 or (fix and returncode == 0)
    
    return CheckResult(
        name="black",
        success=success,
        output=output,
        errors=errors,
        duration=time.time() - start_time,
    )


def run_mypy_check() -> CheckResult:
    """运行 mypy 类型检查"""
    import time
    
    start_time = time.time()
    
    if not check_tool_available("mypy"):
        return CheckResult(
            name="mypy",
            success=False,
            output="mypy not available. Install with: pip install mypy",
            errors=["mypy not installed"],
            duration=time.time() - start_time,
        )
    
    cmd = [
        sys.executable, "-m", "mypy",
        "hive/",
        "--ignore-missing-imports",
        "--warn-unused-configs",
        "--warn-return-any",
        "--warn-no-return",
        "--check-untyped-defs",
    ]
    
    returncode, stdout, stderr = run_command(cmd)
    
    output = stdout + stderr
    errors = []
    warnings = []
    
    for line in output.split("\n"):
        if ": error:" in line:
            errors.append(line)
        elif ": note:" in line or ": warning:" in line:
            warnings.append(line)
    
    # mypy 对未安装类型提示的包会有很多警告，不算失败
    # 只检查是否有真正的类型错误
    type_errors = [e for e in errors if "cannot find implementation" not in e.lower()]
    
    success = len(type_errors) == 0 or returncode == 0
    
    return CheckResult(
        name="mypy",
        success=success,
        output=output,
        errors=type_errors,
        warnings=warnings,
        duration=time.time() - start_time,
    )


def run_isort_check(fix: bool = False) -> CheckResult:
    """运行 isort 导入排序检查"""
    import time
    
    start_time = time.time()
    
    if not check_tool_available("isort"):
        return CheckResult(
            name="isort",
            success=False,
            output="isort not available. Install with: pip install isort",
            errors=["isort not installed"],
            duration=time.time() - start_time,
        )
    
    cmd = [sys.executable, "-m", "isort"]
    
    if fix:
        cmd.append("hive/")
    else:
        cmd.extend(["--check-only", "hive/"])
    
    # 配置选项 (与 black 兼容)
    cmd.extend([
        "--profile", "black",
        "--line-length", "100",
        "--multi-line-output", "3",
    ])
    
    returncode, stdout, stderr = run_command(cmd)
    
    output = stdout + stderr
    errors = []
    
    if returncode != 0 and not fix:
        errors.append("Import order needs fixing")
    
    success = returncode == 0
    
    return CheckResult(
        name="isort",
        success=success,
        output=output,
        errors=errors,
        duration=time.time() - start_time,
    )


def run_vulture_check() -> CheckResult:
    """运行 vulture 死代码检测"""
    import time
    
    start_time = time.time()
    
    if not check_tool_available("vulture"):
        return CheckResult(
            name="vulture",
            success=True,  # 不算失败，只是警告
            output="vulture not available. Install with: pip install vulture",
            warnings=["vulture not installed"],
            duration=time.time() - start_time,
        )
    
    cmd = [
        sys.executable, "-m", "vulture",
        "hive/",
        "--exclude", "*_test.py,*/tests/*",
    ]
    
    returncode, stdout, stderr = run_command(cmd)
    
    output = stdout + stderr
    warnings = []
    
    for line in output.split("\n"):
        if line.strip() and "unused" in line.lower() or ":" in line:
            warnings.append(line)
    
    # vulture 总是返回 0，不算失败
    return CheckResult(
        name="vulture",
        success=True,
        output=output,
        warnings=warnings,
        duration=time.time() - start_time,
    )


def generate_report(results: List[CheckResult]) -> CleanupReport:
    """生成清理报告"""
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    
    # 生成摘要
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("代码清理报告")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    
    for result in results:
        status = "✅ 通过" if result.success else "❌ 失败"
        summary_lines.append(f"  {result.name}: {status} ({result.duration:.2f}s)")
        
        if result.errors:
            summary_lines.append(f"    错误数: {len(result.errors)}")
            for i, err in enumerate(result.errors[:5]):
                summary_lines.append(f"      - {err[:100]}")
            if len(result.errors) > 5:
                summary_lines.append(f"      ... 还有 {len(result.errors) - 5} 个错误")
        
        if result.warnings:
            summary_lines.append(f"    警告数: {len(result.warnings)}")
    
    summary_lines.append("")
    summary_lines.append(f"总计: {passed}/{total} 通过")
    
    if failed > 0:
        summary_lines.append("")
        summary_lines.append("建议修复以下问题:")
        for result in results:
            if not result.success:
                summary_lines.append(f"  - {result.name}: 需要修复")
    
    summary = "\n".join(summary_lines)
    
    return CleanupReport(
        timestamp=datetime.now().isoformat(),
        total_checks=total,
        passed_checks=passed,
        failed_checks=failed,
        results=results,
        summary=summary,
    )


def save_report(report: CleanupReport, output_file: str):
    """保存报告"""
    # JSON 格式
    json_data = {
        "timestamp": report.timestamp,
        "total_checks": report.total_checks,
        "passed_checks": report.passed_checks,
        "failed_checks": report.failed_checks,
        "results": [
            {
                "name": r.name,
                "success": r.success,
                "errors": r.errors,
                "warnings": r.warnings,
                "duration": r.duration,
            }
            for r in report.results
        ],
    }
    
    output_path = PROJECT_ROOT / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存 JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # 保存 Markdown
    md_path = output_path.with_suffix(".md")
    md_content = f"""# 代码清理报告

> 生成时间: {report.timestamp}

## 摘要

| 指标 | 值 |
|------|-----|
| **总检查数** | {report.total_checks} |
| **通过** | {report.passed_checks} |
| **失败** | {report.failed_checks} |

## 详细结果

"""
    
    for result in report.results:
        status_emoji = "✅" if result.success else "❌"
        md_content += f"### {result.name} {status_emoji}\n\n"
        md_content += f"- 耗时: {result.duration:.2f}s\n"
        
        if result.errors:
            md_content += f"\n**错误 ({len(result.errors)}个):**\n\n"
            for err in result.errors:
                md_content += f"```\n{err}\n```\n\n"
        
        if result.warnings:
            md_content += f"\n**警告 ({len(result.warnings)}个):**\n\n"
            for warn in result.warnings[:10]:
                md_content += f"- {warn}\n"
            if len(result.warnings) > 10:
                md_content += f"- ... 还有 {len(result.warnings) - 10} 个警告\n"
        
        md_content += "\n---\n\n"
    
    md_content += "## 命令汇总\n\n"
    md_content += "```bash\n"
    md_content += "# 运行所有检查\n"
    md_content += "python hive/scripts/run_cleanup.py\n\n"
    md_content += "# 运行特定检查\n"
    md_content += "python hive/scripts/run_cleanup.py --check ruff\n"
    md_content += "python hive/scripts/run_cleanup.py --check black,mypy\n\n"
    md_content += "# 自动修复\n"
    md_content += "python hive/scripts/run_cleanup.py --fix\n"
    md_content += "```\n"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Hive 代码清理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 运行所有检查
    python hive/scripts/run_cleanup.py
    
    # 只运行特定检查
    python hive/scripts/run_cleanup.py --check ruff
    python hive/scripts/run_cleanup.py --check black,mypy
    
    # 自动修复可修复的问题
    python hive/scripts/run_cleanup.py --fix
    
    # 保存报告
    python hive/scripts/run_cleanup.py --output reports/cleanup_report
        """
    )
    
    parser.add_argument(
        "--check",
        type=str,
        default="all",
        help="指定检查类型 (ruff, black, mypy, isort, vulture, all)",
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="自动修复可修复的问题",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="reports/cleanup_report",
        help="报告输出路径 (不含扩展名)",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )
    
    args = parser.parse_args()
    
    # 确定要运行的检查
    check_map = {
        "ruff": lambda: run_ruff_check(fix=args.fix),
        "black": lambda: run_black_check(fix=args.fix),
        "mypy": run_mypy_check,
        "isort": lambda: run_isort_check(fix=args.fix),
        "vulture": run_vulture_check,
    }
    
    if args.check == "all":
        checks_to_run = list(check_map.keys())
    else:
        checks_to_run = [c.strip() for c in args.check.split(",") if c.strip() in check_map]
    
    if not checks_to_run:
        print("错误: 没有有效的检查类型")
        print(f"可用类型: {', '.join(check_map.keys())}")
        return 1
    
    # 打印头部
    print()
    print("=" * 60)
    print("Hive 代码清理工具")
    print("=" * 60)
    print()
    print(f"检查类型: {', '.join(checks_to_run)}")
    print(f"自动修复: {'是' if args.fix else '否'}")
    print()
    
    # 运行检查
    results: List[CheckResult] = []
    
    for check_name in checks_to_run:
        print(f"运行 {check_name} 检查...")
        
        try:
            result = check_map[check_name]()
            results.append(result)
            
            status = "✅ 通过" if result.success else "❌ 失败"
            print(f"  结果: {status}")
            print(f"  耗时: {result.duration:.2f}s")
            
            if args.verbose:
                if result.output:
                    print(f"  输出:\n{result.output[:500]}")
                if result.errors:
                    print(f"  错误: {len(result.errors)} 个")
                if result.warnings:
                    print(f"  警告: {len(result.warnings)} 个")
            
            print()
            
        except Exception as e:
            print(f"  错误: {e}")
            results.append(CheckResult(
                name=check_name,
                success=False,
                output=str(e),
                errors=[str(e)],
            ))
            print()
    
    # 生成报告
    report = generate_report(results)
    
    # 打印摘要
    print(report.summary)
    print()
    
    # 保存报告
    if args.output:
        try:
            save_report(report, args.output)
            print(f"报告已保存到: {args.output}.json / {args.output}.md")
        except Exception as e:
            print(f"保存报告失败: {e}")
    
    # 返回代码
    if report.failed_checks > 0:
        print()
        print("=" * 60)
        print("建议:")
        print("=" * 60)
        print()
        
        for result in results:
            if not result.success:
                if result.name == "ruff":
                    print("  ruff 发现问题:")
                    print("    修复命令: python -m ruff check hive/ --fix")
                elif result.name == "black":
                    print("  black 发现格式问题:")
                    print("    修复命令: python -m black hive/")
                elif result.name == "isort":
                    print("  isort 发现导入顺序问题:")
                    print("    修复命令: python -m isort hive/")
                elif result.name == "mypy":
                    print("  mypy 发现类型问题:")
                    print("    建议: 查看详细输出并修复类型注解")
        
        print()
        print("  或使用:")
        print("    python hive/scripts/run_cleanup.py --fix")
        
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
