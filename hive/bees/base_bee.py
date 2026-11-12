# -*- coding: utf-8 -*-
import logging
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from hive.core.dna import BeeDNA
from hive.core.interfaces import BeeInterface, Task, TaskResult
from hive.core.service_locator import BeeServiceLocator
from hive.utils.logger import get_logger
from hive.utils.memory_box import MemoryBox
from hive.utils.security import ALLOWED_GIT_DOMAINS as _ALLOWED_GIT_DOMAINS
from hive.utils.security import validate_git_url as _validate_git_url

logger = get_logger(__name__)

ALLOWED_GIT_DOMAINS = _ALLOWED_GIT_DOMAINS
validate_git_url = _validate_git_url


class BaseBee(BeeInterface, multiprocessing.Process):
    """BaseBee: 实现BeeInterface的基础蜜蜂类"""

    def __init__(self, dna: BeeDNA, id: str, result_queue: multiprocessing.Queue):
        BeeInterface.__init__(self, dna, id)
        multiprocessing.Process.__init__(self)
        self.daemon = True  # 防止主进程退出后工蜂成为孤儿进程
        self.result_queue = result_queue
        self.pid_str = "Allocating..."
        # MemoryBox 延迟到 run() 中打开，避免在父进程中持有文件句柄
        self.memory = None
        logger.info(f"[{self.__class__.__name__}-{self.id}] 初始化完成")

    async def execute_task(self, task: Task) -> TaskResult:
        start_time = time.time()
        logger.info(f"[{self.__class__.__name__}-{self.id}] 执行任务: {task.id}")
        mem_key = f"task_{task.id}"  # Define mem_key for the task
        try:
            result = await self._execute_task_internal(task)
            self.update_dna(result)

            # 成功后，将美好的执念铭刻进 Memory 箱体，形成肌肉记忆供未来后代同型号蜂调用
            (
                self.memory.put(
                    mem_key,
                    {
                        "summary": "This path was successful.",
                        "tools_used": getattr(self.dna, "required_tools", []),
                        "outcome_status": "SUCCESS" if result.success else "FAILED",
                    },
                )
                if self.memory
                else None
            )
            return result
        except InterruptedError as int_e:
            # 用户中断通常是可重试的
            logger.warning(f"[WARN] 任务被中断 (可重试): {int_e}")
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"[INTERRUPTED] {int_e}",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"任务执行失败: {e}", exc_info=True)
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def _execute_task_internal(self, task: Task) -> TaskResult:
        start_time = time.time()
        try:
            target_path = task.payload.get("target_path", task.payload.get("target_url"))
            if target_path:
                self.specialized_task(target_path)
            return TaskResult(
                task_id=task.id,
                success=True,
                result={"status": "completed"},
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    def update_dna(self, performance: TaskResult) -> None:
        self.dna.update_performance(
            success=performance.success,
            completion_time=performance.execution_time,
            error=not performance.success,
        )
        logger.debug(f"[{self.__class__.__name__}-{self.id}] DNA已更新", fitness=self.dna.fitness)

    def run(self):
        from hive.utils.logger import setup_logging

        setup_logging()
        self.pid_str = str(os.getpid())
        # 在子进程中打开 MemoryBox，PID 后缀确保每个子进程使用独立的 shelve 文件
        self.memory = MemoryBox.open_box(f"{self.__class__.__name__}_memory_{os.getpid()}")
        bee_type = self.__class__.__name__
        logger.info(f"[{bee_type}-{self.pid_str}] 启动中，目标: {self.dna.id}")
        try:
            try:
                self.devour()
            except Exception as e:
                self._handle_process_crash(e)
                self.return_payload(
                    {"source": self.id, "status": "CRITICAL_FAILURE", "error": str(e)}
                )
        finally:
            # 强制关闭 MemoryBox（Windows multiprocessing 子进程里 __del__ 不可靠）
            if self.memory is not None:
                try:
                    self.memory.close()
                except Exception:
                    pass
                self.memory = None
            logger.info(f"[{bee_type}-{self.pid_str}] 进程结束")

    def _handle_process_crash(self, e: Exception):
        bee_type = self.__class__.__name__
        logger.error(f"[{bee_type}-{self.pid_str}] 进程异常终止: {e}", exc_info=True)
        try:
            from hive.config.settings import get_settings
            from hive.core.diagnostics import create_death_certificate

            cert = create_death_certificate(self, e)
            cert_path = cert.save(get_settings().BASE_DIR / "hive" / "morgue")
            logger.error(f"[DEAD] [{bee_type}-{self.pid_str}] 死亡证明已生成: {cert_path}")
        except Exception as inner_e:
            logger.error(f"[ERR] 无法生成死亡证明: {inner_e}")

    def devour(self) -> None:
        """吞噬循环 - 执行任务的核心流程 (Unified Logic)"""
        try:
            logger.info(
                f"[BEE] {self.__class__.__name__} #{self.id} 开始吞噬: {self.dna.target_name}"
            )
        except Exception as e:
            logger.debug(f"Could not log bee activity: {e}")

        if self._handle_warmup_check():
            return

        result = self._init_result_payload()
        project_path = None
        sandbox_path = None

        # [Memory Box 记忆检索]: 如果我们在自己的专属箱子里找到了同款 task (甚至失败历史)，尝试复用其推盘结果，或者汲取前世教训
        mem_key = f"exp_task_{self.dna.id}"
        past_experience = self.memory.get(mem_key) if self.memory else None
        if past_experience:
            logger.info(
                f"[MEM] [前世记忆浮现!] 识别到此前接触过该任务序列，记忆调取: {past_experience.get('summary', '已阅读')}"
            )
            # 如果存在致命的前世教训，可以写入 context 提供给 LLM 避坑（这里作为示例只打印出来）

        try:
            project_path = self._prepare_project()
            sandbox_path, exec_path = self._setup_sandbox(project_path, result)

            self._run_mission(exec_path, result)
            self._handle_merge_back(project_path, sandbox_path, result)

            result["status"] = "SUCCESS"
            logger.info(f"[OK] 吞噬成功: {self.dna.target_name}")

        except Exception as e:
            self._handle_devour_failure(e, result, mem_key)  # Pass mem_key
        finally:
            self._cleanup_sandbox(sandbox_path)

        self.return_payload(result)

    def _handle_warmup_check(self) -> bool:
        is_warmup = getattr(self.dna, "metadata", {}).get("idle_task", False) or (
            not getattr(self.dna, "target_url", "")
            and not getattr(self.dna, "metadata", {}).get("target_path")
        )
        if is_warmup:
            logger.info(f"[WARM] [预热] {self.__class__.__name__} #{self.id} 待命中")
            self.return_payload(
                {
                    "source": self.id,
                    "dna": self.dna.__dict__,
                    "status": "SUCCESS",
                    "warmup_task": True,
                    "note": "Bee species initialized",
                }
            )
            return True
        return False

    def _init_result_payload(self) -> Dict:
        return {
            "source": self.id,
            "dna": self.dna.__dict__,
            "status": "PENDING",
            "artifacts": [],
            "timestamp": time.time(),
            "sandbox_active": False,
        }

    def _prepare_project(self) -> Path:
        explicit_path = self.dna.metadata.get("target_path")
        if explicit_path and Path(explicit_path).exists():
            logger.info(f"[PATH] [Local] 使用指定本地路径: {explicit_path}")
            return Path(explicit_path)

        project_path = self._clone_repo()
        if not project_path:
            raise Exception("克隆失败")
        if not self._verify_manifest(project_path):
            raise Exception("完整性验证失败")
        return project_path

    def _setup_sandbox(self, project_path: Path, result: Dict) -> tuple[Optional[Path], Path]:
        from hive.config.settings import get_settings

        settings = get_settings()
        if not settings.ENABLE_SANDBOX:
            return None, project_path

        sandbox_path = self._enter_sandbox(project_path)
        if sandbox_path:
            result["sandbox_active"] = True
            logger.info(f"[SBOX] [Shadow Sandbox] 隔离环境就绪: {sandbox_path}")
            return sandbox_path, sandbox_path
        return None, project_path

    def _run_mission(self, exec_path: Path, result: Dict):
        special_result = self.specialized_task(exec_path)
        if special_result:
            result.update(special_result)

    def _handle_merge_back(self, project_path: Path, sandbox_path: Optional[Path], result: Dict):
        from hive.config.settings import get_settings

        settings = get_settings()
        if (
            settings.ENABLE_SANDBOX
            and sandbox_path
            and (settings.AUTO_MERGE_BACK or getattr(self.dna, "can_merge_back", False))
        ):
            logger.info(f"[SBOX] [Shadow Sandbox] 准备导回修改")
            from hive.core.immulink import apply_sandbox_to_source

            if apply_sandbox_to_source(project_path, sandbox_path):
                result["merged_back"] = True
                logger.info("[OK] [ImmuLink] 修改已成功导回")
            else:
                logger.warning("[WARN] [ImmuLink] 导回失败")
                result["merged_back"] = False

    def _handle_devour_failure(self, e: Exception, result: Dict, mem_key: str):  # Added mem_key
        result["status"] = "FAILED"
        result["error"] = str(e)
        logger.error(f"[ERR] 吞噬失败: {e}")

        # 若失败跌倒，痛苦将被作为"前世苦果"封入 Box，警告未来的相同派发任务不要重蹈覆辙 (-1 惩罚度)
        if self.memory:
            self.memory.put(
                mem_key, {"summary": f"FAILED with error: {str(e)[:50]}", "avoid": True}
            )

        # [TheHive SIRP] 新时代网安守则：呼叫军部高空告警塔（取代原来各自为战的报错乱象）
        err_str = str(e)
        sirp = BeeServiceLocator.get_service("TheHiveSIRP")
        if sirp:
            severity = 3
            alert_type = "UNKNOWN_ERROR"
            if "Timeout" in err_str or "403" in err_str or "unauthorized" in err_str.lower():
                severity = 4
                alert_type = "NETWORK_BLOCKADE"
            sirp.raise_alert(
                severity=severity, alert_type=alert_type, bee_id=self.pid_str, payload=err_str
            )
        else:
            # 如果还没挂载风控塔，就退化为原始记录
            logger.warning(f"[WARN] [SIRP Offline] 告警未送达，风控塔离线：{err_str[:30]}")

        try:
            from hive.config.settings import get_settings
            from hive.core.diagnostics import create_death_certificate

            cert = create_death_certificate(self, e)
            cert_path = cert.save(get_settings().BASE_DIR / "hive" / "morgue")
            logger.info(f"[DOC] 诊断报告已归档: {cert_path}")
            from dataclasses import asdict

            result["death_cert"] = asdict(cert)
        except Exception as inner:
            logger.error(f"无法归档诊断报告: {inner}")

    def _cleanup_sandbox(self, sandbox_path: Optional[Path]):
        if sandbox_path:
            self._exit_sandbox(sandbox_path)

    def _enter_sandbox(self, src_path: Path) -> Optional[Path]:
        import fnmatch

        from hive.config.settings import get_settings

        settings = get_settings()
        sandbox_path = settings.SANDBOX_DIR / f"bee_{self.id}_{uuid.uuid4().hex[:8]}"
        try:
            settings.SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
            if src_path.is_dir():
                # 安全检查：预估算复制大小，防止超大仓库耗尽磁盘
                MAX_SANDBOX_SIZE_MB = 500  # 默认500MB上限
                ignore_patterns = (
                    ".git",
                    ".gitignore",
                    "node_modules",
                    "__pycache__",
                    ".venv",
                    "venv",
                    ".env",
                    "*.pyc",
                    "*.pyo",
                    ".DS_Store",
                )

                def should_ignore(path_parts: tuple) -> bool:
                    """检查路径的任一部分或文件名是否匹配忽略模式"""
                    name = path_parts[-1]
                    dir_parts = path_parts[:-1]
                    # 检查目录部分
                    for dp in dir_parts:
                        if dp in (".git", "node_modules", "__pycache__", ".venv", "venv", ".env"):
                            return True
                    # 检查文件名模式
                    for pat in ignore_patterns:
                        if pat.startswith("*") and fnmatch.fnmatch(name, pat):
                            return True
                        elif name == pat:
                            return True
                    return False

                try:
                    total_size = 0
                    for f in src_path.rglob("*"):
                        if f.is_file() and not should_ignore(f.parts):
                            total_size += f.stat().st_size
                    if total_size > MAX_SANDBOX_SIZE_MB * 1024 * 1024:
                        logger.warning(
                            f"[SBOX] 沙箱项目过大 ({total_size / 1024 / 1024:.1f}MB)，拒绝复制"
                        )
                        return None
                except Exception as size_e:
                    logger.warning(f"[SBOX] 无法估算项目大小，继续复制: {size_e}")

                ignore = shutil.ignore_patterns(*ignore_patterns)
                shutil.copytree(src_path, sandbox_path, ignore=ignore)
                return sandbox_path
            return None
        except Exception as e:
            logger.error(f"[SBOX] 沙箱初始化失败: {e}")
            return None

    def _exit_sandbox(self, sandbox_path: Path):
        try:
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path, ignore_errors=True)
                logger.info("[SBOX] [Shadow Sandbox] 隔离环境已销毁")
        except Exception as e:
            logger.warning(f"无法清理沙箱: {e}")

    def _clone_repo(self) -> Optional[Path]:
        """克隆或更新 GitHub 仓库"""
        from hive.config.settings import get_settings

        settings = get_settings()
        download_dir = settings.BASE_DIR
        download_dir.mkdir(parents=True, exist_ok=True)

        target_dir = download_dir / self.dna.target_name
        if target_dir.exists():
            return target_dir

        url = getattr(self.dna, "target_url", None)
        if url:
            is_safe, error = validate_git_url(url)
            if not is_safe:
                logger.error(f"[ERR] Git URL 验证失败: {error}")
                return None

        if not url or "github.com" not in url:
            return None

        return self._perform_git_clone(url, target_dir)

    def _perform_git_clone(self, url: str, target_dir: Path) -> Optional[Path]:
        try:
            logger.info(f"[CLONE] 正在克隆: {url} ...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "--", url, str(target_dir)],
                check=True,
                capture_output=True,
                timeout=300,
            )
            logger.info("[OK] 克隆完成")
            return target_dir
        except Exception as e:
            logger.error(f"[ERR] Git Clone 失败: {e}")
            return None

    def _verify_manifest(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            if not any(path.iterdir()):
                logger.error(f"Manifest check: Directory empty {path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Manifest check error on {path}: {e}")
            return False

    def specialized_task(self, target_path: Path) -> dict:
        return {"note": "No specialized task implemented"}

    def request_human_intervention(
        self, reason: str, risk_level: str = "HIGH", context: dict = None
    ) -> dict:
        """
        [HITL] 人类介入挂起接口。

        任何子类工蜂在执行到敏感、危险或无法自行判断的逻辑时（如资金交易配置、核心文件删除），
        调用此方法。当前进程将会把请求封装并返回给 Queen，由 Queen 派发给 WardenBee 处理，
        或者当前进程直接挂起阻塞等待通信通道（简单实现上返回挂起载荷）。
        """
        context = context or {}
        bee_type = self.__class__.__name__
        logger.warning(
            f"[WAIT] [{bee_type}-{self.id}] 正在申请人类介入! 风险段: {risk_level}, 原因: {reason}"
        )

        intervention_payload = {
            "source_bee": f"{bee_type}-{self.id}",
            "reason": reason,
            "risk_level": risk_level,
            "context": context,
            "status": "AWAITING_HUMAN",
        }

        # 为了不让当前进程继续往下走错误逻辑，抛出特定异常，让 devour() 捕获并挂起/结束
        raise InterruptedError(f"Human intervention requested: {reason}")

    def spawn_sub_tasks(self, strategies: list[dict], target_bee_type: str = "Standard"):
        """
        [HiVA] 递归派发/工蜂繁育请求。
        当当前工蜂遇到超出单次上下文理解或解决尺寸的复杂体时，
        使用该方法将大任务"切割"成子战略，要求统筹端动态繁育出新的工蜂同时去处理。
        """
        logger.info(
            f"[SPAWN] [{self.__class__.__name__}-{self.id}] 触发 HiVA 裂殖！准备繁衍 {len(strategies)} 个子工蜂。"
        )
        for idx, st in enumerate(strategies):
            spawn_payload = {
                "status": "SPAWNED",
                "spawn_request": {
                    "parent_id": self.id,
                    "child_dna_metadata": st,
                    "target_species": target_bee_type,
                },
            }
            # 将孵化请求以特殊的 payload 注射回结果管道，通知女王育种
            self.return_payload(spawn_payload)

    def return_payload(self, nutrient):
        logger.info(
            f"[{self.__class__.__name__}-{self.pid_str}] 返回载荷: {nutrient.get('status', 'unknown')}"
        )
        self.result_queue.put(nutrient)

    def get_status(self) -> dict:
        return {
            "id": self.id,
            "species": self.dna.species,
            "role": self.dna.role,
            "is_active": self.is_active,
            "fitness": self.dna.fitness,
            "success_rate": self.dna.success_rate,
            "task_completion_time": self.dna.task_completion_time,
            "error_count": self.dna.error_count,
            "pid": self.pid_str,
        }
