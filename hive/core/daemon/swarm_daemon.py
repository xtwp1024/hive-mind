# -*- coding: utf-8 -*-
"""
蜂群后台运行器

让蜂群在后台持续运行。
"""

import asyncio
import signal
import sys
import time
from threading import Thread

from hive.core.swarm import Swarm, SwarmConfig


class SwarmDaemon:
    """蜂群后台守护进程"""

    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()
        self.swarm = None
        self.running = False
        self.thread = None

    def start(self, generations: int = None):
        """启动后台运行"""

        print(f"启动蜂群后台守护进程...")

        # 初始化蜂群
        self.swarm = Swarm(self.config)
        self.running = True

        # 尝试设置信号处理（仅在类 Unix 系统有效）
        try:
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        except (ValueError, OSError):
            pass  # Windows 或非主线程跳过

        # 启动后台线程
        self.thread = Thread(target=self._run_loop, args=(generations,))
        self.thread.daemon = True
        self.thread.start()

        print(f"蜂群已在后台运行")
        print(f"每 3 秒进化一代...")

        return self

    def _run_loop(self, generations: int):
        """运行循环"""
        gen = 0

        while self.running:
            try:
                # 进化一代
                self.swarm.evolve_once()
                gen += 1

                # 打印进度
                if gen % 10 == 0:
                    status = self.swarm.get_status()
                    print(
                        f"Gen {gen}: pop={status['population']}, fitness={status['metrics']['avg_fitness']:.3f}, 知识={status['metrics']['knowledge']}"
                    )

                # 检查是否达到目标代数
                if generations and gen >= generations:
                    print(f"达到目标代数 {generations}，停止")
                    break

                # 等待 3 秒
                time.sleep(3)

            except Exception as e:
                print(f"进化出错: {e}")
                time.sleep(5)

        self.running = False
        print("蜂群后台守护进程已停止")

    def _handle_shutdown(self, signum, frame):
        """处理关闭信号"""
        print("\n收到关闭信号，正在保存状态...")
        self.stop()
        sys.exit(0)

    def stop(self):
        """停止"""
        self.running = False
        if self.swarm:
            # 保存最终状态
            print(f"最终状态: Gen={self.swarm.generation}, pop={len(self.swarm.atoms)}")

    def get_status(self):
        """获取状态"""
        if self.swarm:
            return self.swarm.get_status()
        return None


def run_daemon(generations: int = None):
    """运行后台守护进程"""
    daemon = SwarmDaemon()
    daemon.start(generations)

    # 等待
    try:
        while daemon.running:
            time.sleep(1)
    except KeyboardInterrupt:
        daemon.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="蜂群后台守护进程")
    parser.add_argument("-g", "--generations", type=int, default=None, help="运行代数，默认为无限")
    args = parser.parse_args()

    run_daemon(args.generations)
