# -*- coding: utf-8 -*-
"""
Hive Security Module - 统一的安全验证模块

集中管理所有安全相关的验证逻辑，消除代码重复。

功能：
1. Git URL 验证 - 验证 Git 仓库 URL 的安全性
2. 命令验证 - 验证 Shell 命令的安全性
3. 输入验证 - 通用的输入清理和验证
4. 危险字符检测 - 检测和阻止注入攻击

使用方式：
    from hive.utils.security import (
        validate_git_url, validate_command,
        sanitize_input, is_safe_gene,
        ALLOWED_GIT_DOMAINS, ALLOWED_BASE_COMMANDS, DANGEROUS_CHARS
    )
"""

import re
from pathlib import Path
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse

ALLOWED_GIT_DOMAINS: Set[str] = {
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "gitee.com",
    "git.opendev.org",
}


ALLOWED_BASE_COMMANDS: Set[str] = {
    "python",
    "python3",
    "node",
    "npm",
    "bun",
    "rustc",
    "cargo",
    "bash",
    "sh",
    "powershell",
    "cmd",
    "git",
    "ls",
    "dir",
    "cd",
    "cat",
    "type",
    "grep",
    "find",
    "echo",
    "pwd",
}


DANGEROUS_CHARS: Set[str] = {
    ";",
    "&",
    "|",
    "`",
    "$",
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    "<",
    ">",
    "\n",
    "\r",
    "\t",
}


DANGEROUS_PATTERNS: List[Tuple[str, str]] = [
    (r"[;&|`$]", "Shell 元字符"),
    (r"\$\([^)]*\)", "命令替换"),
    (r"`[^`]*`", "反引号命令替换"),
    (r">\s*/", "输出重定向到系统目录"),
    (r"rm\s+-rf\s+/", "危险的删除命令"),
    (r"curl.*\|.*sh", "下载并执行脚本"),
    (r"wget.*\|.*sh", "下载并执行脚本"),
    (r"\|\s*sh", "管道到 shell"),
    (r">\s*\|", "输出管道"),
    (r"2>&1", "重定向 stderr"),
    (r"&&", "命令链接"),
    (r"\|\|", "条件执行"),
]


def validate_git_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    验证 Git URL 是否安全

    检查项：
    1. URL 必须是非空字符串
    2. 必须以 http://, https:// 或 git@ 开头
    3. 域名必须在白名单中
    4. 不能包含危险字符

    Args:
        url: 要验证的 Git URL

    Returns:
        (是否安全, 错误信息或 None)
    """
    if not url or not isinstance(url, str):
        return False, "URL 必须是非空字符串"

    if not url.startswith(("http://", "https://", "git@")):
        return False, "URL 必须以 http://, https:// 或 git@ 开头"

    try:
        if url.startswith("http"):
            domain = urlparse(url).netloc.lower()
        else:
            match = re.match(r"git@([^:]+):", url)
            if not match:
                return False, "无效的 SSH Git URL 格式"
            domain = match.group(1).lower()

        if domain not in ALLOWED_GIT_DOMAINS:
            return False, f"不允许的 Git 域名: {domain}"

        if any(c in url for c in DANGEROUS_CHARS):
            return False, "URL 包含危险字符"

        return True, None

    except Exception as e:
        return False, f"URL 解析失败: {e}"


def validate_command(command: str) -> Tuple[bool, Optional[str]]:
    """
    验证命令是否安全，防止命令注入

    检查项：
    1. 命令必须是非空字符串
    2. 不能包含危险的 Shell 模式
    3. 基础命令必须在白名单中
    4. 不能包含路径遍历组件

    Args:
        command: 要验证的命令字符串

    Returns:
        (是否安全, 错误信息或 None)
    """
    if not command or not isinstance(command, str):
        return False, "命令必须是非空字符串"

    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"检测到潜在的危险命令模式: {description}"

    base_cmd = command.split()[0].strip("'\"")
    if base_cmd not in ALLOWED_BASE_COMMANDS:
        return False, f"不允许的命令: {base_cmd}"

    parts = command.split()
    for part in parts[1:]:
        if part.startswith("-") and len(part) > 2:
            continue
        if ".." in part:
            return False, "检测到可疑的路径遍历组件"

    return True, None


def sanitize_input(input_str: str, max_length: int = 1024) -> str:
    """
    清理输入字符串，移除危险字符

    Args:
        input_str: 原始输入字符串
        max_length: 最大长度限制

    Returns:
        清理后的字符串
    """
    if not input_str:
        return ""

    if not isinstance(input_str, str):
        input_str = str(input_str)

    input_str = input_str[:max_length]

    for char in DANGEROUS_CHARS:
        input_str = input_str.replace(char, "")

    input_str = re.sub(r"\s+", " ", input_str).strip()

    return input_str


def is_safe_gene(gene: str) -> bool:
    """
    验证变异基因是否安全

    用于 EngineerBee 中验证 mutation_gene 字段，防止命令注入。

    Args:
        gene: 要验证的基因字符串

    Returns:
        是否安全
    """
    if not gene or not isinstance(gene, str):
        return False

    if any(c in gene for c in DANGEROUS_CHARS):
        return False

    if re.search(r"[^\w\-_\s]", gene):
        return False

    return True


def validate_file_path(path: str, base_dir: Path = None) -> Tuple[bool, Optional[str]]:
    """
    验证文件路径是否安全，防止路径遍历攻击

    Args:
        path: 要验证的路径
        base_dir: 可选的基础目录，路径必须在此目录下

    Returns:
        (是否安全, 错误信息或 None)
    """
    if not path or not isinstance(path, str):
        return False, "路径必须是非空字符串"

    if ".." in path:
        return False, "路径包含危险的遍历组件 (..)"

    try:
        p = Path(path)
        resolved = p.resolve()

        if base_dir:
            base_resolved = base_dir.resolve()
            if not str(resolved).startswith(str(base_resolved)):
                return False, f"路径不在允许的目录下: {base_dir}"

        if len(resolved.parts) < 2:
            return False, "路径过短，可能不安全"

        return True, None

    except (OSError, RuntimeError) as e:
        return False, f"路径验证失败: {e}"


class SecurityValidator:
    """
    安全验证器类 - 提供统一的安全验证接口

    用于需要配置自定义安全规则的场景。

    示例：
        validator = SecurityValidator(
            additional_allowed_domains={'my-git.com'},
            additional_allowed_commands={'mycmd'}
        )
        is_safe, error = validator.validate_git_url(url)
    """

    def __init__(
        self,
        additional_allowed_domains: Set[str] = None,
        additional_allowed_commands: Set[str] = None,
        additional_dangerous_chars: Set[str] = None,
        strict_mode: bool = True,
    ):
        self.allowed_domains = ALLOWED_GIT_DOMAINS.copy()
        self.allowed_commands = ALLOWED_BASE_COMMANDS.copy()
        self.dangerous_chars = DANGEROUS_CHARS.copy()
        self.strict_mode = strict_mode

        if additional_allowed_domains:
            self.allowed_domains.update(additional_allowed_domains)

        if additional_allowed_commands:
            self.allowed_commands.update(additional_allowed_commands)

        if additional_dangerous_chars:
            self.dangerous_chars.update(additional_dangerous_chars)

    def validate_git_url(self, url: str) -> Tuple[bool, Optional[str]]:
        if not url or not isinstance(url, str):
            return False, "URL 必须是非空字符串"

        if not url.startswith(("http://", "https://", "git@")):
            return False, "URL 必须以 http://, https:// 或 git@ 开头"

        try:
            if url.startswith("http"):
                domain = urlparse(url).netloc.lower()
            else:
                match = re.match(r"git@([^:]+):", url)
                if not match:
                    return False, "无效的 SSH Git URL 格式"
                domain = match.group(1).lower()

            if domain not in self.allowed_domains:
                return False, f"不允许的 Git 域名: {domain}"

            if any(c in url for c in self.dangerous_chars):
                return False, "URL 包含危险字符"

            return True, None

        except Exception as e:
            return False, f"URL 解析失败: {e}"

    def validate_command(self, command: str) -> Tuple[bool, Optional[str]]:
        if not command or not isinstance(command, str):
            return False, "命令必须是非空字符串"

        for pattern, description in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"检测到潜在的危险命令模式: {description}"

        base_cmd = command.split()[0].strip("'\"")
        if base_cmd not in self.allowed_commands:
            return False, f"不允许的命令: {base_cmd}"

        parts = command.split()
        for part in parts[1:]:
            if part.startswith("-") and len(part) > 2:
                continue
            if ".." in part:
                return False, "检测到可疑的路径遍历组件"

        return True, None

    def is_safe_gene(self, gene: str) -> bool:
        if not gene or not isinstance(gene, str):
            return False

        if any(c in gene for c in self.dangerous_chars):
            return False

        if self.strict_mode and re.search(r"[^\w\-_\s]", gene):
            return False

        return True


__all__ = [
    "validate_git_url",
    "validate_command",
    "sanitize_input",
    "is_safe_gene",
    "validate_file_path",
    "SecurityValidator",
    "ALLOWED_GIT_DOMAINS",
    "ALLOWED_BASE_COMMANDS",
    "DANGEROUS_CHARS",
    "DANGEROUS_PATTERNS",
]
