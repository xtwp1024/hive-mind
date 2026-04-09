#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive Incubator (孵化池)
负责管理代码生命的诞生与优选。

Phase 3 P2: 接入 SwarmEvolutionCoordinator 适应度追踪
- _promote() → 报告 success=True 到进化协调器
- _bury() → 报告 success=False + burial_reason
- 孵化结果实时反馈到适应度计算，形成真正的进化闭环
"""

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

from hive.config.settings import get_settings

logger = logging.getLogger("Hive.Incubator")

class Incubator:
    """
    Incubator (孵化池)
    
    职责:
    1. 监测 `hive/incubation/larvae` 目录下的新生成代码 (幼虫)。
    2. 对幼虫进行"试炼" (Sandboxed Execution)。
    3. 优胜劣汰:
       - 存活且表现良好的 -> 晋升至 `hive/bees/mutants` (成虫/变异体)。
       - 失败或报错的 -> 移入 `hive/incubation/graveyard` (墓地) 供分析。
    """
    def __init__(self):
        self.settings = get_settings()
        self.base_path = Path.cwd() # 使用当前工作目录

        # 定义区域
        self.larvae_dir = self.base_path / "hive" / "incubation" / "larvae"
        self.nursery_dir = self.base_path / "hive" / "incubation" / "nursery"
        self.graveyard_dir = self.base_path / "hive" / "incubation" / "graveyard"
        self.promoted_dir = self.base_path / "hive" / "bees" / "mutants"

        self._ensure_dirs()

        # Phase 3 P2: 注册到进化协调器
        self._evolution: Optional[Any] = None
        self._register_evolution()

        logger.info(f"🥚 孵化池已建立。热源: {self.larvae_dir}")

    def _register_evolution(self):
        """Phase 3 P2: 注册到 SwarmEvolutionCoordinator"""
        try:
            from hive.core.swarm_evolution import get_swarm_evolution_coordinator
            self._evolution = get_swarm_evolution_coordinator()
            self._evolution.register_bee(
                bee_id="Incubator",
                bee_type="Incubator",
                generation=0,
            )
            logger.info(f"🥚 [Incubator] 已注册到进化协调器")
        except ImportError:
            logger.warning(f"🥚 [Incubator] SwarmEvolutionCoordinator 未安装")
        
    def _ensure_dirs(self):
        for p in [self.larvae_dir, self.nursery_dir, self.graveyard_dir, self.promoted_dir]:
            p.mkdir(parents=True, exist_ok=True)
            
        # 确保 mutants 目录下有 __init__.py 以便导入
        if not (self.promoted_dir / "__init__.py").exists():
            (self.promoted_dir / "__init__.py").touch()

    def run_cycle(self):
        """
        执行一次孵化周期
        """
        # 扫描幼虫
        larvae = list(self.larvae_dir.glob("*.py"))
        if not larvae:
            # logger.debug("孵化池空闲...")
            return
            
        logger.info(f"🌡️ 检测到 {len(larvae)} 个生命迹象，开始孵化序列...")
        
        for py_file in larvae:
            self._hatch_one(py_file)
            
    def _hatch_one(self, larvae_path: Path) -> Dict[str, Any]:
        """
        执行一次孵化试炼

        Returns:
            包含 status, success, duration, reason 的结果字典
        """
        logger.info(f"🥚 正在孵化: {larvae_path.name} ...")

        if not self._validate_genetics(larvae_path):
            self._bury(larvae_path, "syntax_error")
            return {"status": "buried", "success": False, "reason": "syntax_error"}

        if not self._run_sandbox_trial(larvae_path):
            # _run_sandbox_trial already calls _bury on failure
            return {"status": "buried", "success": False, "reason": "runtime_error"}

        self._promote(larvae_path)
        return {"status": "promoted", "success": True, "reason": None}

    def _validate_genetics(self, larvae_path: Path) -> bool:
        try:
            compile(larvae_path.read_text(encoding='utf-8'), larvae_path.name, 'exec')
            return True
        except SyntaxError as e:
            logger.error(f"❌ 基因缺陷 (Syntax Error): {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 读取失败: {e}")
            return False

    def _run_sandbox_trial(self, larvae_path: Path) -> bool:
        try:
            cmd = [sys.executable, str(larvae_path)]
            start_time = time.time()
            res = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=10,
                cwd=str(self.nursery_dir) 
            )
            duration = time.time() - start_time
            
            if res.returncode == 0:
                logger.info(f"🐣 破壳成功! (耗时: {duration:.2f}s)")
                logger.debug(f"输出: {res.stdout[:200]}")
                return True
            else:
                logger.warning(f"⚠️ 夭折 (Exit {res.returncode})")
                logger.warning(f"E: {res.stderr[:500]}")
                self._bury(larvae_path, "runtime_error")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ 孵化超时 (Timeout)")
            self._bury(larvae_path, "timeout")
            return False
        except Exception as e:
            logger.error(f"❌ 孵化异常: {e}")
            return False

    def _promote(self, larvae_path: Path):
        """晋升为成虫 (Mutant Bee)"""
        dest = self.promoted_dir / larvae_path.name
        try:
            # 如果存在同名，覆盖或重命名？这里选择覆盖并备份旧的
            if dest.exists():
                backup = dest.with_suffix(f".bak.{int(time.time())}.py")
                shutil.move(str(dest), str(backup))

            shutil.move(str(larvae_path), str(dest))
            logger.info(f"🦋 生命飞升! 已部署至: {dest}")

            # Phase 3 P2: 报告晋升到适应度追踪
            # 从文件名提取 bee_id (去掉 larva_ / healed_ 前缀和时间戳)
            bee_id = self._extract_bee_id(larvae_path.name)
            if self._evolution:
                self._evolution.report_task_result(
                    bee_id=bee_id,
                    success=True,
                    duration=0.0,  # Incubator 不追踪自身耗时
                )
                logger.debug(f"📊 [Incubator] 报告晋升成功: {bee_id}")
        except Exception as e:
            logger.error(f"晋升失败: {e}")

    def _extract_bee_id(self, filename: str) -> str:
        """从幼虫文件名提取 bee_id"""
        # 去掉前缀和时间戳，保留核心标识
        import re
        # larva_xxx_12345.py 或 healed_xxx_12345.py
        cleaned = re.sub(r'^(larva|healed)_', '', filename)
        cleaned = re.sub(r'_\d+\.py$', '', cleaned)
        return f"MutantBee_{cleaned}" if cleaned else f"MutantBee_{filename}"

    def _bury(self, larvae_path: Path, reason: str):
        """移入墓地"""
        try:
            timestamp = int(time.time())
            dest = self.graveyard_dir / f"{larvae_path.stem}_{reason}_{timestamp}{larvae_path.suffix}"
            shutil.move(str(larvae_path), str(dest))
            logger.info(f"🪦 已埋葬: {dest.name}")

            # Phase 3 P2: 报告淘汰到适应度追踪
            bee_id = self._extract_bee_id(larvae_path.name)
            if self._evolution:
                self._evolution.report_task_result(
                    bee_id=bee_id,
                    success=False,
                    duration=0.0,
                    error=f"incubator_bury:{reason}",
                )
                logger.debug(f"📊 [Incubator] 报告淘汰: {bee_id} reason={reason}")
        except Exception as e:
            logger.error(f"埋葬失败: {e}")
