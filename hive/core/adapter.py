# -*- coding: utf-8 -*-
"""
Hive V2 Adapter Layer
---------------------
Bridges the gap between V1 Process-based Bees and V2 Async Actors.
"""

import asyncio
import multiprocessing
import time
from typing import Any, Dict, Optional, Type

import structlog

from hive.bees.base_bee import BaseBee
from hive.core.component import HiveActor
from hive.core.dna import BeeDNA
from hive.core.interfaces import Task, TaskResult

logger = structlog.get_logger()

DEFAULT_TIMEOUT_SECONDS = 300
POLL_INTERVAL = 0.1


class BeeActorAdapter(HiveActor):
    """
    Adapter that wraps a V1 BaseBee class into a V2 HiveActor.

    When `perform` is called, it:
    1. Spawns the V1 Bee process (if not running).
    2. Sends the task to the Bee.
    3. Waits for the result (async).
    4. Bridges the result back to V2 format.
    """

    def __init__(self, name: str, bee_class: Type[BaseBee], result_queue: multiprocessing.Queue):
        super().__init__(name=name, role="Adapter")
        self.bee_class = bee_class
        self.result_queue = result_queue
        self.active_process: Optional[BaseBee] = None
        self.dna = BeeDNA(species="Adapter", role="worker", id=name)

    def capabilities(self):
        # Infer capabilities from the Bee class name or docs
        # Implementation Detail: V1 bees don't verify capabilities strictly yet
        return ["legacy_task_execution", self.bee_class.__name__]

    async def perform(self, action: str, **kwargs) -> Any:
        """
        Execute an action using the underlying Bee process.
        """
        self.logger.info("Bridging action to V1 Bee", action=action, bee=self.bee_class.__name__)

        # 1. Prepare Task
        task_id = f"task_{int(time.time()*1000)}"
        task = Task(id=task_id, type=action, payload=kwargs)

        # 2. Spawn Bee if ephemeral (For V1, usually we spawn per task or keep long running)
        # For this adapter, we stick to the Queen's pattern: One Process Per Task is costly,
        # but BaseBee is designed as a Process.

        # We need a mechanism to wait for THIS specific task result from the shared result_queue.
        # Since result_queue is shared by ALL bees, we need to filter for our task_id.
        # However, BaseBee puts results in result_queue but doesn't support async await.

        # CRITICAL: The V1 architecture puts results into a shared queue consumed by Queen.
        # It does NOT return results to the caller directly.
        # To bridge this, we might need to intercept the result or let Queen handle it.

        # Option A: We run the Bee inline (blocking process) ? No, defeats async.
        # Option B: We spawn it, and we poll a dedicated queue?

        # Let's create a dedicated queue for this adapter execution
        local_queue = multiprocessing.Queue()

        # Instantiate V1 Bee
        # We trick the Bee to use our local queue instead of the global Queen queue for this direct execution
        bee = self.bee_class(self.dna, f"{self.name}_proc", local_queue)

        # Override DNA target from params if possible
        if "target_url" in kwargs:
            bee.dna.target_url = kwargs["target_url"]
        if "target_name" in kwargs:
            bee.dna.target_name = kwargs["target_name"]

        self.logger.info("Spawning V1 Bee Process", pid=bee.name)
        bee.start()

        # Wait for result (Non-blocking loop)
        start_time = time.time()
        while bee.is_alive():
            try:
                # Non-blocking get
                result = local_queue.get_nowait()
                # We got a result!
                self.logger.info("Received result from V1 Bee")

                # Cleanup
                bee.join(timeout=1)
                if bee.is_alive():
                    bee.terminate()

                return result
            except Exception as e:
                logger.debug(f"Bee execution error: {e}")
                await asyncio.sleep(POLL_INTERVAL)

            # Timeout
            if time.time() - start_time > DEFAULT_TIMEOUT_SECONDS:
                bee.terminate()
                raise TimeoutError("V1 Bee execution timed out")

        # If process died without result
        raise RuntimeError("V1 Bee process died without returning result")
