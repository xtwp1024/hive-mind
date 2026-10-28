# -*- coding: utf-8 -*-
"""
AsyncBaseBee - asyncio-based bee with process pool for CPU-bound tasks

Prototype for async-hybrid worker architecture (Hive v3.0 direction #1).

Key differences from BaseBee:
- Uses asyncio for I/O-bound operations (network, git clone, file I/O)
- Uses ProcessPoolExecutor for CPU-bound sandboxed execution
- Does NOT inherit from multiprocessing.Process (runs in main process)
- Event-loop-driven task execution for higher throughput on I/O-bound workloads

Architecture:
    AsyncBaseBee (main process)
        |
        +-- asyncio event loop (I/O: git clone, file ops)
        |
        +-- ProcessPoolExecutor (CPU: sandbox execution)
        |
        +-- PID-isolated MemoryBox via existing mechanism
"""

import asyncio
import uuid
import os
import time
import shutil
import subprocess
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from hive.core.interfaces import BeeInterface, Task, TaskResult
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
from hive.utils.memory_box import MemoryBox
from hive.core.service_locator import BeeServiceLocator
from hive.utils.security import (
    validate_git_url,
    ALLOWED_GIT_DOMAINS
)

logger = get_logger(__name__)


def _validate_git_url_sync(url: str) -> tuple[bool, Optional[str]]:
    """Validate Git URL (sync version for subprocess) - 委托给统一安全模块"""
    is_safe, error = validate_git_url(url)
    if not is_safe and error:
        error = error.replace("必须是非空字符串", "must be non-empty string")
        error = error.replace("http://, https:// 或 git@", "http://, https:// or git@")
        error = error.replace("无效的 SSH Git URL 格式", "Invalid SSH Git URL format")
        error = error.replace("不允许的 Git 域名", "Disallowed Git domain")
        error = error.replace("URL 包含危险字符", "URL contains dangerous characters")
        error = error.replace("URL 解析失败", "URL parsing failed")
    return is_safe, error


def _cpu_bound_task_executor(task_func: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    CPU-bound task executor running in ProcessPoolExecutor worker.

    This function runs in a separate process for sandboxed CPU execution.
    The task_func string identifies which task to execute.

    Args:
        task_func: Identifier for the task function to execute
        task_data: Dictionary containing task parameters

    Returns:
        Result dictionary with status and data
    """
    pid = os.getpid()
    start_time = time.time()

    try:
        # Simulate CPU-bound work (in real implementation, this would be actual code analysis)
        # Using hash calculation as a realistic CPU-bound benchmark
        data = task_data.get('data', b'x' * 10000)
        iterations = task_data.get('iterations', 1000)

        # CPU-bound computation
        result = 0
        for i in range(iterations):
            result += hash(data + str(i).encode())

        execution_time = time.time() - start_time

        return {
            "success": True,
            "result": {"hash_sum": result, "iterations": iterations},
            "execution_time": execution_time,
            "pid": pid
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "pid": pid
        }


class AsyncBaseBee(BeeInterface):
    """
    AsyncBaseBee - Asyncio-based bee with ProcessPoolExecutor for hybrid workloads.

    Design Philosophy:
    - I/O-bound operations (git clone, file reading) use asyncio for non-blocking execution
    - CPU-bound operations (code analysis, sandbox execution) use ProcessPoolExecutor
    - Single main process with event loop, no multiprocessing.Process inheritance
    - PID-isolated MemoryBox via existing MemoryBox.open_box() mechanism

    Throughput Improvement Potential:
    - BaseBee: Blocks on I/O, one task per process
    - AsyncBaseBee: Overlaps I/O via asyncio, submits CPU work to pool concurrently
    - Expected 3-5x improvement on mixed I/O-CPU workloads
    """

    def __init__(
        self,
        dna: BeeDNA,
        id: str,
        result_queue: Optional[multiprocessing.Queue] = None,
        max_workers: int = 4
    ):
        """
        Initialize AsyncBaseBee.

        Args:
            dna: BeeDNA instance containing bee configuration
            id: Unique bee identifier
            result_queue: Optional multiprocessing queue for returning results
            max_workers: Maximum number of process pool workers
        """
        BeeInterface.__init__(self, dna, id, result_queue)
        self.pid_str = str(os.getpid())
        self.memory: Optional[MemoryBox] = None
        self.max_workers = max_workers
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._active_tasks: Dict[str, Future] = {}
        self._completed_count = 0
        self._failed_count = 0

        logger.info(f"[{self.__class__.__name__}-{self.id}] Initialized (async mode, pid={self.pid_str})")

    @property
    def process_pool(self) -> ProcessPoolExecutor:
        """Lazy-initialize process pool"""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
            logger.debug(f"[{self.__class__.__name__}-{self.id}] ProcessPool created with {self.max_workers} workers")
        return self._process_pool

    def _get_memory_box(self) -> MemoryBox:
        """Get or create PID-isolated MemoryBox"""
        if self.memory is None:
            # Use PID suffix to ensure isolation (mirrors BaseBee pattern)
            box_name = f"{self.__class__.__name__}_memory_{os.getpid()}"
            self.memory = MemoryBox.open_box(box_name)
        return self.memory

    async def execute_task(self, task: Task) -> TaskResult:
        """
        Execute a task using async I/O and process pool.

        Args:
            task: Task to execute

        Returns:
            TaskResult with execution outcome
        """
        start_time = time.time()
        logger.info(f"[{self.__class__.__name__}-{self.id}] Executing task: {task.id}")

        mem_key = f"task_{task.id}"
        task_type = task.type or "default"

        try:
            # Route to appropriate handler based on task type
            if task_type == "cpu_bound":
                result = await self._execute_cpu_bound_task_async(task)
            elif task_type == "io_bound":
                result = await self._execute_io_bound_task_async(task)
            else:
                # Mixed workload: overlap I/O and CPU
                result = await self._execute_mixed_task_async(task)

            # Update DNA with performance
            self.update_dna(result)

            # Record success in memory
            memory = self._get_memory_box()
            memory.put(mem_key, {
                "summary": "This path was successful.",
                "tools_used": getattr(self.dna, 'required_tools', []),
                "outcome_status": "SUCCESS" if result.success else "FAILED"
            })

            self._completed_count += 1
            return result

        except InterruptedError as int_e:
            logger.warning(f"Task interrupted (retryable): {int_e}")
            self._failed_count += 1
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"[INTERRUPTED] {int_e}",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            self._failed_count += 1
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def _execute_cpu_bound_task_async(self, task: Task) -> TaskResult:
        """
        Execute CPU-bound task using ProcessPoolExecutor.

        Args:
            task: Task with cpu_bound type

        Returns:
            TaskResult
        """
        start_time = time.time()

        # Prepare task data for CPU-bound execution
        task_data = {
            "data": task.payload.get('data', b'x' * 10000),
            "iterations": task.payload.get('iterations', 5000),
            "task_id": task.id
        }

        # Submit to process pool (non-blocking via asyncio)
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.process_pool,
            _cpu_bound_task_executor,
            "cpu_analysis",
            task_data
        )

        # Store reference for potential cancellation
        self._active_tasks[task.id] = future

        try:
            # Wait for completion with optional timeout
            result_dict = await asyncio.wait_for(future, timeout=task.timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[{self.__class__.__name__}-{self.id}] Task {task.id} timed out")
            future.cancel()
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"Task timed out after {task.timeout}s",
                execution_time=time.time() - start_time
            )
        finally:
            self._active_tasks.pop(task.id, None)

        return TaskResult(
            task_id=task.id,
            success=result_dict.get("success", False),
            result=result_dict.get("result"),
            error=result_dict.get("error"),
            execution_time=time.time() - start_time
        )

    async def _execute_io_bound_task_async(self, task: Task) -> TaskResult:
        """
        Execute I/O-bound task using asyncio.

        Common I/O-bound operations:
        - Git clone/fetch
        - File reading/writing
        - Network requests

        Args:
            task: Task with io_bound type

        Returns:
            TaskResult
        """
        start_time = time.time()
        target_path = task.payload.get('target_path', task.payload.get('target_url'))

        try:
            if target_path:
                # Check if it's a git URL or local path
                if isinstance(target_path, str) and target_path.startswith(('http://', 'https://', 'git@')):
                    # Async git clone
                    cloned_path = await self._async_clone_repo(target_path)
                    if cloned_path:
                        # Async file operations
                        file_count = await self._async_scan_directory(cloned_path)
                        return TaskResult(
                            task_id=task.id,
                            success=True,
                            result={"path": str(cloned_path), "files_scanned": file_count},
                            execution_time=time.time() - start_time
                        )
                    else:
                        return TaskResult(
                            task_id=task.id,
                            success=False,
                            error="Git clone failed",
                            execution_time=time.time() - start_time
                        )
                elif Path(target_path).exists():
                    # Async directory scan
                    file_count = await self._async_scan_directory(Path(target_path))
                    return TaskResult(
                        task_id=task.id,
                        success=True,
                        result={"path": target_path, "files_scanned": file_count},
                        execution_time=time.time() - start_time
                    )

            return TaskResult(
                task_id=task.id,
                success=True,
                result={"status": "io_task_completed"},
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def _execute_mixed_task_async(self, task: Task) -> TaskResult:
        """
        Execute mixed I/O-CPU workload by overlapping both.

        Strategy:
        1. Start I/O operation (e.g., git clone)
        2. While waiting for I/O, prepare CPU work
        3. Once I/O completes, submit CPU work to pool
        4. Wait for CPU work to complete

        This overlap reduces total wall-clock time.

        Args:
            task: Task with mixed workload

        Returns:
            TaskResult
        """
        start_time = time.time()
        target_path = task.payload.get('target_path', task.payload.get('target_url'))

        try:
            cloned_path = None

            # Phase 1: I/O (non-blocking git clone)
            if target_path and isinstance(target_path, str) and target_path.startswith(('http://', 'https://', 'git@')):
                cloned_path = await self._async_clone_repo(target_path)
                if not cloned_path:
                    return TaskResult(
                        task_id=task.id,
                        success=False,
                        error="Git clone failed",
                        execution_time=time.time() - start_time
                    )

            # Phase 2: CPU-bound work (submitted to process pool)
            # Overlap with any remaining I/O
            task_data = {
                "data": str(target_path or "").encode() * 100,
                "iterations": task.payload.get('iterations', 3000),
                "task_id": task.id
            }

            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                self.process_pool,
                _cpu_bound_task_executor,
                "mixed_analysis",
                task_data
            )

            self._active_tasks[task.id] = future

            try:
                result_dict = await asyncio.wait_for(future, timeout=task.timeout)
            except asyncio.TimeoutError:
                logger.warning(f"[{self.__class__.__name__}-{self.id}] Mixed task {task.id} timed out")
                future.cancel()
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error=f"Task timed out after {task.timeout}s",
                    execution_time=time.time() - start_time
                )
            finally:
                self._active_tasks.pop(task.id, None)

            return TaskResult(
                task_id=task.id,
                success=result_dict.get("success", False),
                result={
                    "path": str(cloned_path) if cloned_path else target_path,
                    "cpu_result": result_dict.get("result")
                },
                error=result_dict.get("error"),
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def _async_clone_repo(self, url: str) -> Optional[Path]:
        """
        Asynchronously clone a git repository.

        Uses asyncio.to_thread to run blocking subprocess in thread pool,
        allowing other async operations to proceed during clone.

        Args:
            url: Git repository URL

        Returns:
            Path to cloned repository, or None on failure
        """
        from hive.config.settings import get_settings

        # Validate URL first (sync, fast)
        is_safe, error = _validate_git_url_sync(url)
        if not is_safe:
            logger.error(f"Git URL validation failed: {error}")
            return None

        settings = get_settings()
        download_dir = settings.BASE_DIR
        target_dir = download_dir / Path(url.split('/')[-1].replace('.git', ''))

        if target_dir.exists():
            logger.info(f"Repository already exists at {target_dir}")
            return target_dir

        # Run git clone in thread pool (non-blocking)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # Use default thread pool
                lambda: subprocess.run(
                    ["git", "clone", "--depth", "1", "--", url, str(target_dir)],
                    check=True,
                    capture_output=True,
                    timeout=300
                )
            )
            logger.info(f"Async clone completed: {url} -> {target_dir}")
            return target_dir
        except subprocess.TimeoutExpired:
            logger.error(f"Git clone timed out: {url}")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e}, stderr: {e.stderr.decode() if e.stderr else ''}")
            return None
        except Exception as e:
            logger.error(f"Git clone error: {e}")
            return None

    async def _async_scan_directory(self, path: Path) -> int:
        """
        Asynchronously scan directory for files.

        Uses asyncio.to_thread for file system operations.

        Args:
            path: Directory path to scan

        Returns:
            Number of files found
        """
        def _scan():
            count = 0
            try:
                for f in path.rglob('*'):
                    if f.is_file() and not any(
                        part in ('.git', 'node_modules', '__pycache__', '.venv')
                        for part in f.parts
                    ):
                        count += 1
            except Exception as e:
                logger.warning(f"Directory scan error on {path}: {e}")
            return count

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _scan)
        except Exception as e:
            logger.warning(f"Async scan failed for {path}: {e}")
            return 0

    def update_dna(self, performance: TaskResult) -> None:
        """
        Update DNA based on task execution performance.

        Args:
            performance: TaskResult from task execution
        """
        self.dna.update_performance(
            success=performance.success,
            completion_time=performance.execution_time,
            error=not performance.success
        )
        logger.debug(
            f"[{self.__class__.__name__}-{self.id}] DNA updated",
            fitness=self.dna.fitness
        )

    async def devour_async(self) -> Dict[str, Any]:
        """
        Async version of devour loop - main entry point for async bee lifecycle.

        Returns:
            Result dictionary with bee execution outcome
        """
        from hive.config.settings import get_settings
        settings = get_settings()

        bee_type = self.__class__.__name__
        logger.info(f"[{bee_type}-{self.pid_str}] Starting async devour: {self.dna.target_name}")

        result = {
            "source": self.id,
            "dna": self.dna.__dict__,
            "status": "PENDING",
            "artifacts": [],
            "timestamp": time.time(),
            "sandbox_active": False
        }

        # Check for warmup/idle task
        is_warmup = (
            getattr(self.dna, 'metadata', {}).get('idle_task', False) or
            (not getattr(self.dna, 'target_url', '') and
             not getattr(self.dna, 'metadata', {}).get('target_path'))
        )

        if is_warmup:
            logger.info(f"[{bee_type}-{self.id}] Warmup task - bee standing by")
            result["status"] = "SUCCESS"
            result["warmup_task"] = True
            self.return_payload(result)
            return result

        project_path = None
        sandbox_path = None

        try:
            # I/O: Prepare project (git clone or local path)
            project_path = await self._prepare_project_async()

            # CPU: Setup sandbox (using process pool)
            sandbox_path, exec_path = await self._setup_sandbox_async(project_path, result)

            # CPU: Run mission (process pool)
            await self._run_mission_async(exec_path, result)

            # I/O: Merge back (async)
            await self._handle_merge_back_async(project_path, sandbox_path, result)

            result["status"] = "SUCCESS"
            logger.info(f"Async devour success: {self.dna.target_name}")

        except Exception as e:
            self._handle_devour_failure_async(e, result)

        finally:
            # Cleanup
            if sandbox_path:
                await self._cleanup_sandbox_async(sandbox_path)

        self.return_payload(result)
        return result

    async def _prepare_project_async(self) -> Path:
        """Async project preparation (clone or use local path)"""
        explicit_path = self.dna.metadata.get('target_path')
        if explicit_path and Path(explicit_path).exists():
            logger.info(f"[Async] Using local path: {explicit_path}")
            return Path(explicit_path)

        url = getattr(self.dna, 'target_url', None)
        if url:
            cloned = await self._async_clone_repo(url)
            if cloned:
                return cloned

        raise Exception("Project preparation failed: no valid path or clone failed")

    async def _setup_sandbox_async(
        self,
        project_path: Path,
        result: Dict
    ) -> tuple[Optional[Path], Path]:
        """Async sandbox setup using process pool for directory copy"""
        from hive.config.settings import get_settings
        settings = get_settings()

        if not settings.ENABLE_SANDBOX:
            return None, project_path

        loop = asyncio.get_event_loop()

        def _copy_to_sandbox():
            sandbox_path = settings.SANDBOX_DIR / f"bee_{self.id}_{uuid.uuid4().hex[:8]}"
            settings.SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

            ignore_patterns = ('.git', '.gitignore', 'node_modules', '__pycache__',
                             '.venv', 'venv', '.env', '*.pyc', '*.pyo', '.DS_Store')
            ignore = shutil.ignore_patterns(*ignore_patterns)

            shutil.copytree(project_path, sandbox_path, ignore=ignore)
            return sandbox_path

        try:
            sandbox_path = await loop.run_in_executor(
                self.process_pool,  # Use process pool for CPU-bound copy
                _copy_to_sandbox
            )
            result["sandbox_active"] = True
            logger.info(f"[Async] Sandbox ready: {sandbox_path}")
            return sandbox_path, sandbox_path
        except Exception as e:
            logger.warning(f"Sandbox setup failed: {e}")
            return None, project_path

    async def _run_mission_async(self, exec_path: Path, result: Dict):
        """Async mission execution"""
        # For now, use CPU-bound task as mission simulation
        loop = asyncio.get_event_loop()

        def _mission_work():
            # Simulate CPU-bound code analysis
            data = str(exec_path).encode() * 1000
            total = 0
            for i in range(2000):
                total += hash(data + str(i).encode())
            return total

        cpu_result = await loop.run_in_executor(self.process_pool, _mission_work)
        result["mission_hash"] = cpu_result
        logger.info(f"[Async] Mission completed with hash: {cpu_result}")

    async def _handle_merge_back_async(
        self,
        project_path: Path,
        sandbox_path: Optional[Path],
        result: Dict
    ):
        """Async merge back operation"""
        from hive.config.settings import get_settings
        settings = get_settings()

        if not settings.ENABLE_SANDBOX or not sandbox_path:
            return

        if settings.AUTO_MERGE_BACK or getattr(self.dna, 'can_merge_back', False):
            logger.info("[Async] Preparing sandbox merge back")

            def _apply_merge():
                from hive.core.immulink import apply_sandbox_to_source
                return apply_sandbox_to_source(project_path, sandbox_path)

            loop = asyncio.get_event_loop()
            merged = await loop.run_in_executor(None, _apply_merge)

            if merged:
                result["merged_back"] = True
                logger.info("[Async] Sandbox merge completed")
            else:
                result["merged_back"] = False
                logger.warning("[Async] Sandbox merge failed")

    async def _cleanup_sandbox_async(self, sandbox_path: Path):
        """Async sandbox cleanup"""
        def _cleanup():
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path, ignore_errors=True)
                logger.info(f"[Async] Sandbox cleaned up: {sandbox_path}")

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _cleanup)
        except Exception as e:
            logger.warning(f"Sandbox cleanup error: {e}")

    def _handle_devour_failure_async(self, e: Exception, result: Dict):
        """Handle async devour failure with SIRP alert and death certificate"""
        result["status"] = "FAILED"
        result["error"] = str(e)
        logger.error(f"Async devour failed: {e}")

        # MemoryBox failure record
        mem_key = f"exp_task_{self.dna.id}"
        if self.memory:
            self.memory.put(mem_key, {
                "summary": f"FAILED with error: {str(e)[:50]}",
                "avoid": True
            })

        # SIRP Alert
        sirp = BeeServiceLocator.get_service("TheHiveSIRP")
        if sirp:
            err_str = str(e)
            severity = 3
            alert_type = "UNKNOWN_ERROR"
            if "Timeout" in err_str or "403" in err_str or "unauthorized" in err_str.lower():
                severity = 4
                alert_type = "NETWORK_BLOCKADE"
            sirp.raise_alert(
                severity=severity,
                alert_type=alert_type,
                bee_id=self.pid_str,
                payload=err_str
            )
        else:
            logger.warning(f"[SIRP Offline] Alert not sent: {str(e)[:30]}")

        # Death certificate
        try:
            from hive.core.diagnostics import create_death_certificate
            from dataclasses import asdict
            cert = create_death_certificate(self, e)
            cert_path = cert.save(get_settings().BASE_DIR / "hive" / "morgue")
            result["death_cert"] = asdict(cert)
            logger.info(f"Death certificate saved: {cert_path}")
        except Exception as inner:
            logger.error(f"Death certificate error: {inner}")

    def return_payload(self, nutrient: Dict[str, Any]):
        """
        Return result payload to queue.

        Args:
            nutrient: Result dictionary to return
        """
        logger.info(
            f"[{self.__class__.__name__}-{self.pid_str}] "
            f"Returning payload: {nutrient.get('status', 'unknown')}"
        )
        if self.result_queue:
            self.result_queue.put(nutrient)

    def get_status(self) -> dict[str, Any]:
        """
        Get bee status.

        Returns:
            Status dictionary
        """
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
            "completed_tasks": self._completed_count,
            "failed_tasks": self._failed_count,
            "async_mode": True
        }

    def shutdown(self):
        """Gracefully shutdown async bee resources"""
        logger.info(f"[{self.__class__.__name__}-{self.id}] Shutting down...")

        # Cancel active tasks
        for task_id, future in self._active_tasks.items():
            if not future.done():
                logger.warning(f"Cancelling active task: {task_id}")
                future.cancel()

        # Shutdown process pool
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            logger.info(f"[{self.__class__.__name__}-{self.id}] Process pool shut down")

        # Close memory box
        if self.memory:
            self.memory.close()

        self.is_active = False
        logger.info(f"[{self.__class__.__name__}-{self.id}] Shutdown complete")
