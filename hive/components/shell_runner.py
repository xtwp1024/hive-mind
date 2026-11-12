import asyncio
import os
import subprocess
import tempfile
from typing import Optional

from hive.core.component import ComponentConfig, HiveComponent
from hive.utils.security import ALLOWED_BASE_COMMANDS as _ALLOWED_BASE_COMMANDS
from hive.utils.security import validate_command as _validate_command

ALLOWED_BASE_COMMANDS = _ALLOWED_BASE_COMMANDS
validate_command = _validate_command


class ShellRunnerConfig(ComponentConfig):
    safe_mode: bool = True
    timeout: int = 30


class ShellRunnerComponent(HiveComponent):
    """
    Component for executing shell commands.
    Provides the 'run_shell' action.
    """

    def __init__(self, config: ShellRunnerConfig = None):
        super().__init__(config or ShellRunnerConfig())

    def name(self) -> str:
        return "shell_runner"

    def capabilities(self) -> list[str]:
        return ["run_shell", "run_script"]

    async def action_run_shell(self, command: str, cwd: str = None) -> tuple[int, str, str]:
        """Execute a shell command"""
        self.logger.info("Executing shell command", command=command, cwd=cwd)

        # 🛡️ Security: 验证命令安全性
        if self.config.safe_mode:
            is_safe, error = validate_command(command)
            if not is_safe:
                self.logger.error("Command validation failed", error=error, command=command)
                return -1, "", f"安全验证失败: {error}"

        try:
            # Use 'cp437' for decoding output on Windows to avoid utf-8 errors with localized messages
            # or errors containing non-utf8 bytes

            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.config.timeout
            )

            rc = process.returncode

            # Robust decoding
            def robust_decode(b):
                try:
                    return b.decode("utf-8")
                except UnicodeDecodeError:
                    return b.decode("cp437", errors="ignore")

            out_str = robust_decode(stdout).strip()
            err_str = robust_decode(stderr).strip()

            if rc != 0:
                self.logger.error("Command failed", rc=rc, stderr=err_str)
            else:
                self.logger.info("Command success", rc=rc)

            return rc, out_str, err_str

        except asyncio.TimeoutError:
            self.logger.error("Command timed out", command=command)
            return -1, "", "Timeout"
        except Exception as e:
            self.logger.error("Execution error", error=str(e))
            return -1, "", str(e)

    async def action_run_script(
        self, script_content: str, extension: str, cwd: str = None
    ) -> tuple[int, str, str]:
        """Write and run a script file (Supports polyglot execution)"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{extension}", delete=False, encoding="utf-8"
        ) as f:
            f.write(script_content)
            script_path = f.name

        self.logger.info("Created temp script", path=script_path)

        try:
            # Determine runner based on extension
            if extension == "py":
                cmd = f"python {script_path}"
            elif extension == "rs":
                exe_path = script_path.replace(".rs", ".exe")
                # Wrap paths in double quotes to handle spaces
                compile_cmd = f'rustc "{script_path}" -o "{exe_path}"'
                rc, out, err = await self.action_run_shell(compile_cmd, cwd)
                if rc != 0:
                    return rc, out, f"Compilation Failed: {err}"
                cmd = f'"{exe_path}"'
            elif extension == "js":
                cmd = f"node {script_path}"
            elif extension == "ts":
                cmd = f"bun {script_path}"  # Prefer bun for faster TS execution if available, else npx tsx
                # Check for bun availability could be added here, assuming standard env for now or fallback
            elif extension == "sh":
                cmd = f"bash {script_path}"
            elif extension == "ps1":
                cmd = f"powershell -File {script_path}"
            else:
                return -1, "", f"Unsupported extension: {extension}"

            return await self.action_run_shell(cmd, cwd)

        finally:
            # Cleanup
            if os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except (OSError, PermissionError):
                    pass
            if extension == "rs":
                exe = script_path.replace(".rs", ".exe")
                pdb = script_path.replace(".rs", ".pdb")
                if os.path.exists(exe):
                    try:
                        os.remove(exe)
                    except (OSError, PermissionError):
                        pass
                if os.path.exists(pdb):
                    try:
                        os.remove(pdb)
                    except (OSError, PermissionError):
                        pass
