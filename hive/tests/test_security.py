# -*- coding: utf-8 -*-
"""测试安全验证功能"""

import pytest

from hive.bees.base_bee import validate_git_url, ALLOWED_GIT_DOMAINS
from hive.components.shell_runner import validate_command, ALLOWED_BASE_COMMANDS


class TestGitURLValidation:
    """测试 Git URL 验证"""

    def test_valid_github_url(self):
        """测试有效的 GitHub URL"""
        is_safe, error = validate_git_url("https://github.com/user/repo.git")
        assert is_safe is True
        assert error is None

    def test_valid_gitlab_url(self):
        """测试有效的 GitLab URL"""
        is_safe, error = validate_git_url("https://gitlab.com/user/repo.git")
        assert is_safe is True
        assert error is None

    def test_valid_ssh_url(self):
        """测试有效的 SSH URL"""
        is_safe, error = validate_git_url("git@github.com:user/repo.git")
        assert is_safe is True
        assert error is None

    def test_invalid_domain(self):
        """测试无效的域名"""
        is_safe, error = validate_git_url("https://evil.com/repo.git")
        assert is_safe is False
        assert "不允许的 Git 域名" in error

    def test_empty_url(self):
        """测试空 URL"""
        is_safe, error = validate_git_url("")
        assert is_safe is False
        assert "非空字符串" in error

    def test_none_url(self):
        """测试 None URL"""
        is_safe, error = validate_git_url(None)
        assert is_safe is False
        assert "非空字符串" in error

    def test_url_with_injection(self):
        """测试包含注入的 URL"""
        is_safe, error = validate_git_url("https://github.com/user/repo.git; rm -rf /")
        assert is_safe is False
        assert "危险字符" in error

    def test_url_with_pipe(self):
        """测试包含管道符的 URL"""
        is_safe, error = validate_git_url("https://github.com/user/repo|curl evil.com")
        assert is_safe is False

    def test_http_url(self):
        """测试 HTTP URL（非 HTTPS）"""
        is_safe, error = validate_git_url("http://github.com/user/repo.git")
        # 应该被允许（虽然不推荐）
        assert is_safe is True

    def test_malformed_ssh_url(self):
        """测试格式错误的 SSH URL"""
        is_safe, error = validate_git_url("git@github.com/user/repo.git")
        # 这个是格式错误的
        assert is_safe is False

    def test_valid_ssh_url(self):
        """测试有效的 SSH URL"""
        is_safe, error = validate_git_url("git@github.com:user/repo.git")
        # 这个是正确的SSH格式
        assert is_safe is True

    def test_invalid_protocol(self):
        """测试无效的协议"""
        is_safe, error = validate_git_url("ftp://github.com/user/repo.git")
        assert is_safe is False
        assert "http://, https:// 或 git@" in error


class TestCommandValidation:
    """测试命令验证"""

    def test_valid_python_command(self):
        """测试有效的 Python 命令"""
        is_safe, error = validate_command("python script.py")
        assert is_safe is True
        assert error is None

    def test_valid_git_command(self):
        """测试有效的 Git 命令"""
        is_safe, error = validate_command("git clone https://github.com/user/repo")
        assert is_safe is True
        assert error is None

    def test_valid_npm_command(self):
        """测试有效的 NPM 命令"""
        is_safe, error = validate_command("npm install")
        assert is_safe is True
        assert error is None

    def test_command_with_semicolon(self):
        """测试包含分号的命令"""
        is_safe, error = validate_command("ls; rm -rf /")
        assert is_safe is False
        assert "危险命令模式" in error

    def test_command_with_pipe(self):
        """测试包含管道符的命令"""
        is_safe, error = validate_command("cat file.py | grep test")
        assert is_safe is False
        assert "危险命令模式" in error

    def test_command_with_backtick(self):
        """测试包含反引号的命令"""
        is_safe, error = validate_command("echo `date`")
        assert is_safe is False
        assert "危险命令模式" in error

    def test_command_with_dollar(self):
        """测试包含美元符号的命令"""
        is_safe, error = validate_command("echo $HOME")
        assert is_safe is False
        assert "危险命令模式" in error

    def test_disallowed_command(self):
        """测试不允许的命令"""
        is_safe, error = validate_command("dangerous_command")
        assert is_safe is False
        assert "不允许的命令" in error

    def test_empty_command(self):
        """测试空命令"""
        is_safe, error = validate_command("")
        assert is_safe is False
        assert "非空字符串" in error

    def test_none_command(self):
        """测试 None 命令"""
        is_safe, error = validate_command(None)
        assert is_safe is False
        assert "非空字符串" in error

    def test_command_with_newline(self):
        """测试包含换行符的命令"""
        is_safe, error = validate_command("ls\nrm -rf /")
        assert is_safe is False
        assert "危险命令模式" in error

    def test_command_with_output_redirect(self):
        """测试包含输出重定向的命令"""
        is_safe, error = validate_command("ls > /etc/passwd")
        assert is_safe is False
        assert "危险命令模式" in error

    def test_complex_benign_command(self):
        """测试复杂的良性命令"""
        is_safe, error = validate_command("python -m pytest tests/test_dna.py -v")
        assert is_safe is True

    def test_command_with_quotes(self):
        """测试包含引号的命令"""
        is_safe, error = validate_command('python "script with spaces.py"')
        # 应该允许，因为引号内的内容被提取出来
        assert is_safe is True

    def test_rustc_command(self):
        """测试 rustc 命令"""
        is_safe, error = validate_command("rustc main.rs -o main")
        assert is_safe is True

    def test_bash_command(self):
        """测试 bash 命令"""
        is_safe, error = validate_command("bash script.sh")
        assert is_safe is True


class TestSecurityConstants:
    """测试安全常量"""

    def test_allowed_git_domains_not_empty(self):
        """测试允许的 Git 域名列表不为空"""
        assert len(ALLOWED_GIT_DOMAINS) > 0
        assert "github.com" in ALLOWED_GIT_DOMAINS

    def test_allowed_base_commands_not_empty(self):
        """测试允许的基础命令列表不为空"""
        assert len(ALLOWED_BASE_COMMANDS) > 0
        assert "python" in ALLOWED_BASE_COMMANDS
        assert "git" in ALLOWED_BASE_COMMANDS

    def test_suspicious_commands_not_allowed(self):
        """测试可疑命令不被允许"""
        suspicious = ["rm", "sudo", "su", "chmod", "chown"]
        for cmd in suspicious:
            assert cmd not in ALLOWED_BASE_COMMANDS
