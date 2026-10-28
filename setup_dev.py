#!/usr/bin/env python3
"""
开发环境设置脚本

安装 pre-commit hooks 并配置开发环境。
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """运行命令并处理错误

    Args:
        cmd: 命令列表
        description: 命令描述

    Returns:
        是否成功
    """
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description} - 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ 命令未找到: {cmd[0]}")
        print(f"   请先安装必要的工具")
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🐝 Hive Mind Swarm - 开发环境设置")
    print("="*60)

    project_root = Path(__file__).parent
    success_count = 0
    total_steps = 5

    # 1. 检查 Python 版本
    print("\n📋 检查 Python 版本...")
    if sys.version_info < (3, 8):
        print(f"❌ 需要 Python 3.8+，当前版本: {sys.version}")
        sys.exit(1)
    print(f"✅ Python 版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # 2. 安装 pre-commit
    if run_command(
        [sys.executable, "-m", "pip", "install", "-U", "pre-commit"],
        "安装 pre-commit"
    ):
        success_count += 1

    # 3. 安装开发依赖
    if run_command(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        "安装开发依赖"
    ):
        success_count += 1

    # 4. 安装 pre-commit hooks
    if run_command(
        ["pre-commit", "install"],
        "安装 pre-commit hooks"
    ):
        success_count += 1

    # 5. 运行 pre-commit 对所有文件
    if run_command(
        ["pre-commit", "run", "--all-files"],
        "运行 pre-commit 检查（首次可能需要几分钟）"
    ):
        success_count += 1

    # 6. 创建初始 secrets baseline
    if run_command(
        ["detect-secrets", "scan", "--baseline", ".secrets.baseline"],
        "创建 secrets baseline"
    ):
        success_count += 1

    # 总结
    print("\n" + "="*60)
    print(f"📊 设置完成: {success_count}/{total_steps} 个步骤成功")
    print("="*60)

    if success_count == total_steps:
        print("\n✅ 开发环境设置完成！")
        print("\n📝 后续步骤:")
        print("   1. 创建虚拟环境: python -m venv venv")
        print("   2. 激活虚拟环境 (Windows): venv\\Scripts\\activate")
        print("   3. 激活虚拟环境 (Linux/Mac): source venv/bin/activate")
        print("   4. 安装依赖: pip install -r requirements.txt")
        print("   5. 运行测试: pytest")
        print("\n💡 提示: pre-commit hooks 已安装，每次提交时会自动运行")
        return 0
    else:
        print("\n⚠️  部分步骤失败，请手动检查")
        return 1


if __name__ == "__main__":
    sys.exit(main())
