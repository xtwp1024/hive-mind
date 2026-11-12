# -*- coding: utf-8 -*-
"""接口定义

定义蜂巢系统中所有组件的接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from hive.core.dna import DNA, BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Task:
    """任务数据结构"""

    id: str
    type: str
    priority: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    timeout: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """任务结果数据结构"""

    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BeeInterface(ABC):
    """蜜蜂接口

    所有蜜蜂类型必须实现此接口。
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue: Any = None):
        """初始化蜜蜂

        Args:
            dna: 蜜蜂DNA
            id: 蜜蜂ID
            result_queue: 结果队列 (用于多进程返回结果)
        """
        self.dna = dna
        self.id = id
        self.result_queue = result_queue
        self.is_active = True

    @abstractmethod
    async def execute_task(self, task: Task) -> TaskResult:
        """执行任务

        Args:
            task: 要执行的任务

        Returns:
            任务结果
        """
        pass

    @abstractmethod
    def update_dna(self, performance: TaskResult) -> None:
        """根据任务执行结果更新DNA

        Args:
            performance: 任务执行结果
        """
        pass

    def get_status(self) -> dict[str, Any]:
        """获取蜜蜂状态

        Returns:
            状态字典
        """
        return {
            "id": self.id,
            "species": self.dna.species,
            "is_active": self.is_active,
            "fitness": self.dna.fitness,
            "role": self.dna.role,
        }

    def deactivate(self) -> None:
        """停用蜜蜂"""
        self.is_active = False
        logger.info("Bee deactivated", id=self.id, species=self.dna.species)

    def activate(self) -> None:
        """激活蜜蜂"""
        self.is_active = True
        logger.info("Bee activated", id=self.id, species=self.dna.species)


class MinisterInterface(ABC):
    """部长接口

    所有部长类型必须实现此接口。
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue: Any = None):
        """初始化部长

        Args:
            dna: 蜜蜂DNA
            id: 蜜蜂ID
            result_queue: 结果队列
        """
        self.dna = dna
        self.id = id
        self.result_queue = result_queue
        self.is_active = True

    @abstractmethod
    async def assign_task(self, task: Task) -> Optional[str]:
        """分配任务给合适的蜜蜂 (单任务)

        Args:
            task: 要分配的任务

        Returns:
            被分配任务的蜜蜂ID，如果没有合适的则返回None
        """
        pass

    @abstractmethod
    def assign_tasks(self, strategy_topics: list[str] = None, capacity: int = 5) -> list[BeeDNA]:
        """批量分配任务 (Core V2 Logic)

        Args:
            strategy_topics: 策略主题列表
            capacity: 容量限制

        Returns:
            生成的蜜蜂DNA列表
        """
        pass

    @abstractmethod
    def get_bee_status(self) -> dict[str, Any]:
        """获取所有管理蜜蜂的状态

        Returns:
            状态字典
        """
        pass

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """获取统计数据

        Returns:
            统计数据字典
        """
        pass

    def activate(self) -> None:
        """激活部长"""
        self.is_active = True
        logger.info("Minister activated", name=self.name)

    def deactivate(self) -> None:
        """停用部长"""
        self.is_active = False
        logger.info("Minister deactivated", name=self.name)


class QueenInterface(ABC):
    """蜂王接口

    蜂王负责协调所有部长并做出全局决策。
    """

    def __init__(self, result_queue: Any, name: str = "Queen"):
        """初始化蜂王

        Args:
            result_queue: 结果队列
            name: 蜂王名称
        """
        self.result_queue = result_queue
        self.name = name
        self.is_active = True

    @abstractmethod
    async def coordinate_ministers(self, task: Task) -> TaskResult:
        """协调部长执行任务

        Args:
            task: 要执行的任务

        Returns:
            任务结果
        """
        pass

    @abstractmethod
    def evolve_species(self) -> bool:
        """进化物种

        Returns:
            是否成功进化
        """
        pass

    @abstractmethod
    def deploy_swarm(self, colony_dna: list[BeeDNA]) -> None:
        """部署蜂群

        Args:
            colony_dna: 蜜蜂DNA列表
        """
        pass

    @abstractmethod
    def consume_honey(self) -> None:
        """消耗并处理蜂群结果"""
        pass

    @abstractmethod
    def get_hive_status(self) -> dict[str, Any]:
        """获取整个蜂巢的状态

        Returns:
            状态字典
        """
        pass

    @abstractmethod
    def get_evolution_status(self) -> dict[str, Any]:
        """获取进化状态

        Returns:
            进化状态字典
        """
        pass

    def activate(self) -> None:
        """激活蜂王"""
        self.is_active = True
        logger.info("Queen activated", name=self.name)

    def deactivate(self) -> None:
        """停用蜂王"""
        self.is_active = False
        logger.info("Queen deactivated", name=self.name)


class HiveInterface(ABC):
    """蜂巢接口

    整个蜂巢系统的主要接口。
    """

    def __init__(self, name: str = "Hive"):
        """初始化蜂巢

        Args:
            name: 蜂巢名称
        """
        self.name = name
        self.is_running = False

    @abstractmethod
    async def start(self) -> bool:
        """启动蜂巢

        Returns:
            是否成功启动
        """
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """停止蜂巢

        Returns:
            是否成功停止
        """
        pass

    @abstractmethod
    async def submit_task(self, task: Task) -> TaskResult:
        """提交任务到蜂巢

        Args:
            task: 要执行的任务

        Returns:
            任务结果
        """
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """获取蜂巢状态

        Returns:
            状态字典
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """获取指标

        Returns:
            指标字典
        """
        pass

    @abstractmethod
    def get_health(self) -> tuple[bool, list[str]]:
        """获取健康状态

        Returns:
            (是否健康, 问题列表)
        """
        pass

    @property
    def is_active(self) -> bool:
        """蜂巢是否活跃"""
        return self.is_running
