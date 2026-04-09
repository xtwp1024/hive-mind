#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Darwin Bee (达尔文蜂)
负责基于现有代码库和变异指令，生成新的实验性蜂种代码 (Larvae)。
"""

import logging
import re
import random
import time
from pathlib import Path
from typing import Dict, Any, List

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.config.settings import get_settings
from hive.utils.brain_interface import BrainInterface
from hive.core.swarm_communication import get_swarm_communication, MessageType, SwarmMessage

logger = logging.getLogger("Hive.Darwin")

class DarwinBee(BaseBee):
    """
    达尔文蜂 - 进化工程师

    能力:
    1. 读取 VectorVault 中的优秀代码片段。
    2. 使用 LLM (Brain) 杂交/变异这些代码。
    3. 生成完整的 Python 脚本产卵到 `hive/incubation/larvae`。
    4. Phase 2: 通过蜂群网络向 SentinelBee 请求安全扫描
    5. Phase 3: 集成 SwarmEvolutionCoordinator，使用 GEP 适应度引导进化
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue: Any):
        super().__init__(dna, id, result_queue)
        self.settings = get_settings()
        self.larvae_dir = Path.cwd() / "hive" / "incubation" / "larvae"
        self.larvae_dir.mkdir(parents=True, exist_ok=True)

        self.brain = BrainInterface()  # 连接大脑

        # Phase 2: 接入蜂群通信系统
        self._swarm = get_swarm_communication()
        self._swarm.register_bee(self.id, self)
        self._pending_security_results: Dict[str, Any] = {}
        self._swarm.subscribe(MessageType.KNOWLEDGE, self._on_security_knowledge)

        # Phase 3: 注册到进化协调器
        self._evolution = None  # 懒加载
        self._register_evolution()

        logger.info(f"🧬 [Darwin] 已注册到蜂群网络，ID={self.id}")

    def _register_evolution(self):
        """Phase 3: 注册到 SwarmEvolutionCoordinator"""
        try:
            from hive.core.swarm_evolution import get_swarm_evolution_coordinator
            self._evolution = get_swarm_evolution_coordinator()
            self._evolution.register_bee(
                bee_id=self.id,
                bee_type="DarwinBee",
                generation=self._get_generation(),
            )
            self._evolution.start()  # 确保协调器在运行
            logger.info(f"🧬 [Darwin] 已注册到进化协调器")
        except ImportError:
            logger.warning(f"🧬 [Darwin] SwarmEvolutionCoordinator 未安装")

    def _get_generation(self) -> int:
        """从 DNA 中解析代数"""
        if "Gen" in self.dna.id:
            try:
                for part in self.dna.id.split("_"):
                    if part.startswith("Gen") and part[3:].isdigit():
                        return int(part[3:])
            except (ValueError, IndexError):
                pass
        return 0

    def _on_security_knowledge(self, message: SwarmMessage):
        """Phase 2: 接收 SentinelBee 的安全知识消息"""
        if message.content.get("topic") == "security_scan_result":
            task_id = message.content.get("task_id")
            if task_id in self._pending_security_results:
                self._pending_security_results[task_id] = message.content.get("result")

    def specialized_task(self, target_path: Path) -> Dict[str, Any]:
        """
        执行进化任务
        """
        logger.info(f"🧬 [Darwin] 收到进化指令，目标: {target_path}")

        try:
            # 检查是否有医疗任务 (Self-Healing Mission)
            death_cert_path = self.dna.metadata.get("death_cert_path")
            is_medic_mission = bool(death_cert_path)

            import json

            prompt = ""
            source_content = ""

            if is_medic_mission:
                logger.info(f"🚑 [Medic] 这是一次救援行动! 正在分析尸检报告: {death_cert_path}")
                try:
                    with open(death_cert_path, 'r', encoding='utf-8') as f:
                        cert_data = json.load(f)

                    error_msg = cert_data.get('error_message', 'Unknown Error')
                    traceback_str = cert_data.get('traceback_str', '')
                    target_name = cert_data.get('target_name', 'Unknown')

                    if target_path.exists():
                        source_content = target_path.read_text(encoding='utf-8')
                    else:
                        return {"error": "Corpse missing (Source file not found)"}

                    prompt = f"""
                    Task: FIX the python script below which crashed.
                    Role: Expert Python Debugger & Medic.

                    CRASH REPORT:
                    Error Type: {cert_data.get('error_type')}
                    Error Message: {error_msg}
                    Traceback:
                    {traceback_str}

                    BROKEN CODE:
                    {source_content}

                    Requirements:
                    1. Analyze the traceback to find the exact line of failure.
                    2. Apply a minimal, robust fix to prevent this crash.
                    3. Do NOT rewrite the whole file if not necessary. Keep style consistent.
                    4. Ensure `if __name__ == "__main__":` block exists and runs a test.
                    5. Return the FULLY FIXED Python script.
                    """

                except Exception as e:
                    logger.error(f"无法读取尸检报告: {e}")
                    return {"error": f"Medic failed to read cert: {e}"}
            else:
                # 正常进化逻辑
                # 1. 获取灵感 (从 VectorVault 或 Target直接读取)
                if not target_path.exists():
                    return {"error": "Source verification failed"}

                source_content = target_path.read_text(encoding='utf-8')[:3000]

                # 2. Phase 2: 先向 SentinelBee 请求安全扫描
                security_result = self.request_security_review(target_path, timeout=10.0)

                # 将安全扫描结果注入到进化 prompt 中
                security_context = ""
                if security_result and security_result.get("status") == "success":
                    stats = security_result.get("stats", {})
                    issues = security_result.get("issues", [])
                    security_context = f"""
SECURITY SCAN RESULT (from SentinelBee):
- Files scanned: {stats.get('files_scanned', 0)}
- High severity: {stats.get('high', 0)}
- Medium severity: {stats.get('medium', 0)}
- Low severity: {stats.get('low', 0)}
Top issues:
"""
                    for iss in issues[:5]:
                        security_context += f"  [{iss['severity']}] {iss['description']} in {iss['file']}:{iss['line']}\n"
                    security_context += "\nIMPORTANT: Avoid these patterns in the generated code!\n"
                    logger.info(f"🛡️ [Darwin] 安全上下文已准备: H{stats.get('high', 0)} M{stats.get('medium', 0)}")
                elif security_result and security_result.get("status") == "error":
                    security_context = f"\n(Security scan failed: {security_result.get('message', 'unknown')})\n"

                # 3. 向大脑请求变异（注入安全上下文）
                logger.info("🧠 [Darwin] 正在向大脑请求基因编辑...")

                prompt = f"""
Task: Create a new, experimental Python script based on the following code snippet.
Goal: Create a specialized Worker Agent that performs a useful task related to the source code.

Source Code Context:
{source_content}
{security_context}

Requirements:
1. The script MUST be self-contained and runnable as a standalone script.
2. It MUST include a `if __name__ == "__main__":` block that executes a test run.
3. It MUST print "Self-Test Passed" if the test run is successful (Exit 0).
4. If it fails, it SHOULD raise an exception (Exit != 0).
5. ONLY return the Python code block. No markdown checks.

New Agent Name: MutantBee_{int(time.time())}
"""

            if not prompt:
                return {"error": "Prompt generation failed"}

            mutation_response = self.brain.consult(prompt)

            if not mutation_response:
                return {"error": "Brain dead (No response)"}

            # 4. 提取代码
            new_code = self._extract_code(mutation_response)
            if not new_code:
                return {"error": "Failed to extract code from brain response"}

            # 5. 产卵 (Spawn Larva)
            prefix = "healed_" if is_medic_mission else "larva_"
            safe_stem = re.sub(r'[^\w\-]', '_', target_path.stem)
            larva_name = f"{prefix}{safe_stem}_{int(time.time())}.py"
            larva_path = (self.larvae_dir / larva_name).resolve()
            if not str(larva_path).startswith(str(self.larvae_dir.resolve())):
                logger.error(f"[Darwin] 拒绝非法产卵路径: {larva_name}")
                return {"error": "Invalid larva path"}
            larva_path.write_text(new_code, encoding='utf-8')

            # Phase 3: 报告进化成功到适应度追踪
            if self._evolution:
                self._evolution.report_task_result(self.id, success=True, duration=0.0)

            if is_medic_mission:
                logger.info(f"🩹 [Medic] 修复补丁已生成: {larva_name}")
            else:
                logger.info(f"🥚 [Darwin] 产卵成功: {larva_name}")

            return {
                "action": "spawn",
                "larva": str(larva_path),
                "genes_source": target_path.name,
                "is_fix": is_medic_mission,
                "security_context": security_context if not is_medic_mission else None
            }

        except Exception as e:
            # Phase 3: 报告进化失败到适应度追踪
            if self._evolution:
                self._evolution.report_task_result(self.id, success=False, duration=0.0, error=str(e))
            logger.error(f"❌ 进化失败: {e}")
            return {"error": str(e)}

    # ===== Phase 2: 蜂间通信 =====

    def request_security_review(self, target_path: Path, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Phase 2: 通过蜂群网络向 SentinelBee 请求安全扫描

        Args:
            target_path: 要扫描的代码路径
            timeout: 等待结果超时（秒）

        Returns:
            安全扫描结果
        """
        task_id = f"security_review_{int(time.time() * 1000)}"
        self._pending_security_results[task_id] = None

        msg = SwarmMessage(
            id=task_id,
            type=MessageType.TASK_REQUEST,
            from_bee=self.id,
            to_bee="SentinelBee",
            content={
                "task_id": task_id,
                "action": "security_scan",
                "target": str(target_path),
                "requester": self.id,
            },
            priority=8,
        )
        self._swarm.send_message(msg)
        logger.info(f"📨 [Darwin] 向 SentinelBee 请求安全扫描: {target_path}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._pending_security_results.get(task_id) is not None:
                result = self._pending_security_results.pop(task_id)
                logger.info(f"✅ [Darwin] 收到安全扫描结果")
                return result
            time.sleep(0.2)

        logger.warning(f"⏰ [Darwin] 安全扫描请求超时 ({timeout}s)")
        self._pending_security_results.pop(task_id, None)
        return {"status": "timeout", "message": "SentinelBee 未在超时时间内响应"}

    def _extract_code(self, response: str) -> str:
        """从 LLM 响应中提取 Python 代码"""
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            return response.split("```")[1].split("```")[0].strip()
        return response  # 假设全是代码
