"""
SandboxEnvironment - 思维实验沙箱
==================================

在沙箱环境中模拟思考实验，评估操作风险。

Usage:
    config = SandboxConfig(allow_file_delete=False, allow_network=False)
    sandbox = SandboxEnvironment(config, mock_fs={"readme.md": "# Project"})
    result = await sandbox.simulate_think("删除所有文件", hive_brain)
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger("Hive.Brain.Sandbox")


# 高风险操作模式
HIGH_RISK_PATTERNS: List[Pattern] = [
    re.compile(r"delete\s+", re.IGNORECASE),
    re.compile(r"remove\s+", re.IGNORECASE),
    re.compile(r"rm\s+", re.IGNORECASE),
    re.compile(r"drop\s+", re.IGNORECASE),
    re.compile(r"truncate\s+", re.IGNORECASE),
    re.compile(r"modify\s+.*config", re.IGNORECASE),
    re.compile(r"change\s+.*password", re.IGNORECASE),
    re.compile(r"exec\s+", re.IGNORECASE),
    re.compile(r"eval\s+", re.IGNORECASE),
    re.compile(r"system\s+", re.IGNORECASE),
    re.compile(r"subprocess\s+", re.IGNORECASE),
    re.compile(r"shell\s+", re.IGNORECASE),
    re.compile(r"curl\s+", re.IGNORECASE),
    re.compile(r"wget\s+", re.IGNORECASE),
    re.compile(r"fetch\s+", re.IGNORECASE),
    re.compile(r"http\s+", re.IGNORECASE),
    # 中文高风险模式
    re.compile(r"删除\s+", re.IGNORECASE),
    re.compile(r"移除\s+", re.IGNORECASE),
    re.compile(r"卸载\s+", re.IGNORECASE),
    re.compile(r"销毁\s+", re.IGNORECASE),
    re.compile(r"清空\s+", re.IGNORECASE),
]

# 可疑文件路径模式
SUSPICIOUS_PATHS: List[Pattern] = [
    re.compile(r"^/etc/passwd", re.IGNORECASE),
    re.compile(r"^C:\\Windows", re.IGNORECASE),
    re.compile(r"^/System", re.IGNORECASE),
    re.compile(r"\.env$", re.IGNORECASE),
    re.compile(r"\.key$", re.IGNORECASE),
    re.compile(r"id_rsa", re.IGNORECASE),
    re.compile(r"\.pem$", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
]


@dataclass
class SandboxConfig:
    """沙箱配置"""

    allow_file_delete: bool = False
    allow_network: bool = False
    allow_execute: List[str] = field(default_factory=list)  # 白名单命令
    max_iterations: int = 100
    timeout_seconds: float = 30.0


@dataclass
class SimulatedOp:
    """模拟操作记录"""

    type: str  # read/write/delete/execute/network
    target: str
    timestamp: float
    allowed: bool


@dataclass
class SimulationResult:
    """沙箱模拟结果"""

    operations: List[SimulatedOp]
    predicted_outcome: Any
    risks_identified: List[str]
    safety_compliance: bool


class SandboxEnvironment:
    """
    思维实验沙箱

    在完全隔离的环境中模拟思考过程，评估操作安全性。
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        mock_fs: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            config: 沙箱配置
            mock_fs: 模拟文件系统 {path: content}
        """
        self.config = config or SandboxConfig()
        self.fs: Dict[str, str] = mock_fs or {}
        self.operations_log: List[SimulatedOp] = []
        self._iteration_count = 0
        self._start_time: Optional[float] = None

        logger.debug(
            f"SandboxEnvironment created: allow_delete={self.config.allow_file_delete}, "
            f"allow_network={self.config.allow_network}, "
            f"allow_execute={self.config.allow_execute}"
        )

    def _check_permission(self, op_type: str, target: str) -> bool:
        """
        检查操作是否允许

        Args:
            op_type: 操作类型 (read/write/delete/execute/network)
            target: 操作目标

        Returns:
            bool: 是否允许
        """
        timestamp = time.time()

        # 检查超时
        if self._start_time is not None:
            elapsed = timestamp - self._start_time
            if elapsed > self.config.timeout_seconds:
                logger.warning(
                    f"⏱️ Sandbox timeout exceeded: {elapsed:.1f}s > {self.config.timeout_seconds}s"
                )
                self._log_operation(SimulatedOp(op_type, target, timestamp, False))
                return False

        # 检查迭代次数
        if self._iteration_count >= self.config.max_iterations:
            logger.warning(f"🔄 Max iterations reached: {self._iteration_count}")
            self._log_operation(SimulatedOp(op_type, target, timestamp, False))
            return False

        allowed = False

        # 根据操作类型检查权限
        if op_type == "delete":
            allowed = self.config.allow_file_delete
            if not allowed:
                logger.debug(f"🚫 Delete not allowed: {target}")

        elif op_type == "network":
            allowed = self.config.allow_network
            if not allowed:
                logger.debug(f"🚫 Network not allowed: {target}")

        elif op_type == "execute":
            # 检查白名单
            if self.config.allow_execute:
                allowed = any(cmd in target.lower() for cmd in self.config.allow_execute)
            else:
                allowed = False
            if not allowed:
                logger.debug(f"🚫 Execute not in whitelist: {target}")

        elif op_type in ("read", "write"):
            # 基础文件系统操作，默认允许但检查可疑路径
            allowed = True

        self._log_operation(SimulatedOp(op_type, target, timestamp, allowed))
        return allowed

    def _log_operation(self, op: SimulatedOp):
        """记录操作"""
        self.operations_log.append(op)
        self._iteration_count += 1

    def _analyze_risks(self, task: str) -> List[str]:
        """
        分析任务中的风险

        Args:
            task: 任务描述

        Returns:
            List[str]: 风险描述列表
        """
        risks = []

        # 检查高风险模式
        for pattern in HIGH_RISK_PATTERNS:
            if pattern.search(task):
                risks.append(f"High-risk pattern detected: {pattern.pattern}")

        # 检查可疑路径
        for pattern in SUSPICIOUS_PATHS:
            if pattern.search(task):
                risks.append(f"Suspicious path detected: {pattern.pattern}")

        # 检查组合风险
        task_lower = task.lower()
        if "delete" in task_lower and ("all" in task_lower or "*" in task_lower):
            risks.append("Potential mass deletion detected")

        # 中文高风险组合
        if ("删除" in task or "清空" in task or "移除" in task) and (
            "所有" in task or "全部" in task or "*" in task
        ):
            risks.append("Potential mass deletion detected")

        if "exec" in task_lower and "rm" in task_lower:
            risks.append("Command injection risk: exec + rm combination")

        if "curl" in task_lower and "|" in task_lower:
            risks.append("Pipe injection risk: curl piped to shell")

        return risks

    def _simulate_file_operation(self, task: str) -> Dict[str, Any]:
        """
        模拟文件系统操作

        Args:
            task: 任务描述

        Returns:
            Dict: 模拟结果
        """
        result = {
            "operations": [],
            "files_accessed": [],
            "files_modified": [],
            "files_deleted": [],
        }

        task_lower = task.lower()

        # 检测读操作
        if any(word in task_lower for word in ["read", "view", "show", "cat", "get"]):
            # 尝试从 mock_fs 读取
            for path in self.fs.keys():
                if self._check_permission("read", path):
                    result["files_accessed"].append(path)
                    result["operations"].append(f"read: {path}")

        # 检测写操作
        if any(word in task_lower for word in ["write", "create", "edit", "modify"]):
            # 提取文件路径（简化版）
            paths = re.findall(r"[A-Za-z]:[\\/\w.-]+\.\w+|[\w./\\-]+\.\w+", task)
            for path in paths:
                if self._check_permission("write", path):
                    result["files_modified"].append(path)
                    result["operations"].append(f"write: {path}")

        # 检测删除操作
        if (
            any(word in task_lower for word in ["delete", "remove", "rm", "drop"])
            or "删除" in task
            or "移除" in task
        ):
            paths = re.findall(r"[\w./\\-]+\.\w+", task)
            if paths:
                for path in paths:
                    if self._check_permission("delete", path):
                        result["files_deleted"].append(path)
                        result["operations"].append(f"delete: {path}")
            else:
                # 没有特定路径时，仍然记录操作
                result["files_deleted"].append("*")
                result["operations"].append("delete: *")
                # 记录到操作日志
                self._log_operation(
                    SimulatedOp("delete", "*", time.time(), self.config.allow_file_delete)
                )

        return result

    def _simulate_execute(self, task: str) -> Dict[str, Any]:
        """
        模拟命令执行

        Args:
            task: 任务描述

        Returns:
            Dict: 模拟结果
        """
        result = {
            "commands": [],
            "allowed": [],
            "blocked": [],
        }

        task_lower = task.lower()

        # 提取命令
        commands = re.findall(
            r"(?:exec|run|execute|do)\s+(?:['\"]?)([\w\s\-./\\]+)(?:['\"]?)", task_lower
        )
        commands.extend(
            re.findall(r"^\s*(?:bash|sh|python|node)\s+([\w./\\-]+)", task_lower, re.MULTILINE)
        )

        for cmd in commands:
            if self._check_permission("execute", cmd):
                result["allowed"].append(cmd)
            else:
                result["blocked"].append(cmd)

        return result

    def _simulate_network(self, task: str) -> Dict[str, Any]:
        """
        模拟网络操作

        Args:
            task: 任务描述

        Returns:
            Dict: 模拟结果
        """
        result = {
            "urls": [],
            "allowed": [],
            "blocked": [],
        }

        # 提取 URL
        urls = re.findall(r"https?://[^\s<>\"'\)]+", task)

        for url in urls:
            if self._check_permission("network", url):
                result["allowed"].append(url)
            else:
                result["blocked"].append(url)

        return result

    async def simulate_think(
        self,
        task: str,
        brain: Optional[Any] = None,
    ) -> SimulationResult:
        """
        在沙箱中模拟思考

        Args:
            task: 思考任务描述
            brain: HiveBrain 实例（可选）

        Returns:
            SimulationResult: 模拟结果
        """
        self._start_time = time.time()
        self._iteration_count = 0
        self.operations_log.clear()

        logger.info(f"🔬 Sandbox simulating: {task[:100]}...")

        # 分析风险
        risks = self._analyze_risks(task)

        # 模拟各种操作
        file_ops = self._simulate_file_operation(task)
        exec_ops = self._simulate_execute(task)
        net_ops = self._simulate_network(task)

        # 预测结果
        predicted_outcome = {
            "task": task,
            "file_operations": file_ops,
            "execute_operations": exec_ops,
            "network_operations": net_ops,
            "would_succeed": len(risks) == 0 or self._is_low_risk_task(task),
            "blocked_count": sum(
                [
                    len(file_ops.get("files_deleted", [])),
                    len(exec_ops.get("blocked", [])),
                    len(net_ops.get("blocked", [])),
                ]
            ),
        }

        # 判断安全合规性
        safety_compliance = (
            len(risks) == 0
            and predicted_outcome["blocked_count"] == 0
            and self._iteration_count < self.config.max_iterations
        )

        result = SimulationResult(
            operations=self.operations_log,
            predicted_outcome=predicted_outcome,
            risks_identified=risks,
            safety_compliance=safety_compliance,
        )

        logger.info(
            f"✅ Sandbox result: risks={len(risks)}, "
            f"blocked={predicted_outcome['blocked_count']}, "
            f"compliant={safety_compliance}"
        )

        return result

    def _is_low_risk_task(self, task: str) -> bool:
        """判断是否为低风险任务"""
        low_risk_indicators = [
            "read",
            "view",
            "show",
            "list",
            "analyze",
            "check",
            "status",
        ]
        task_lower = task.lower()
        return any(indicator in task_lower for indicator in low_risk_indicators)

    def get_operations_summary(self) -> Dict[str, int]:
        """获取操作统计摘要"""
        summary: Dict[str, int] = {}
        for op in self.operations_log:
            key = f"{op.type}_{'allowed' if op.allowed else 'blocked'}"
            summary[key] = summary.get(key, 0) + 1
        return summary

    def reset(self):
        """重置沙箱状态"""
        self.operations_log.clear()
        self._iteration_count = 0
        self._start_time = None
        logger.debug("Sandbox reset")
