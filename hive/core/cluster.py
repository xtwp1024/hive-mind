# -*- coding: utf-8 -*-
"""
Cluster - 涌现结构

动态形成的簇，是蜂群的基本组织单元。
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hive.core.bee_atom import BeeAtom


class ClusterType:
    """簇类型"""
    TASK_ORIENTED = "task"     # 任务导向
    GENE_ORIENTED = "gene"    # 基因导向
    TEMPORARY = "temporary"    # 临时
    ELITE = "elite"           # 精英簇


@dataclass
class Cluster:
    """
    簇 - 蜂群的涌现结构
    
    由自由结合形成的临时组织，完成任务后可能解散。
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = ClusterType.TEMPORARY

    # 成员
    member_ids: set[str] = field(default_factory=set)

    # 状态
    is_active: bool = True
    created_at: int = field(default_factory=lambda: 0)
    last_active_at: int = 0
    
    # 共享知识
    shared_knowledge: list[str] = field(default_factory=list)
    
    # 任务
    current_task: str = ""
    task_progress: float = 0.0
    
    # 统计
    tasks_completed: int = 0
    
    def add_member(self, atom_id: str) -> None:
        """添加成员"""
        self.member_ids.add(atom_id)
        self.last_active_at = time.time()
    
    def remove_member(self, atom_id: str) -> None:
        """移除成员"""
        self.member_ids.discard(atom_id)
    
    @property
    def size(self) -> int:
        """成员数量"""
        return len(self.member_ids)
    
    @property
    def is_empty(self) -> bool:
        """是否为空"""
        return len(self.member_ids) == 0
    
    def add_knowledge(self, knowledge: str) -> None:
        """添加共享知识"""
        if knowledge not in self.shared_knowledge:
            self.shared_knowledge.append(knowledge)
    
    def update_task(self, task: str, progress: float) -> None:
        """更新任务"""
        self.current_task = task
        self.task_progress = progress
        self.last_active_at = time.time()
    
    def complete_task(self) -> None:
        """完成任务"""
        self.tasks_completed += 1
        self.current_task = ""
        self.task_progress = 0.0
    
    def dissolve(self) -> None:
        """解散簇"""
        self.is_active = False
        self.member_ids.clear()
    
    def to_dict(self) -> dict[str, Any]:
        """序列化"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "member_ids": list(self.member_ids),
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "shared_knowledge": self.shared_knowledge,
            "current_task": self.current_task,
            "task_progress": self.task_progress,
            "tasks_completed": self.tasks_completed
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Cluster":
        """反序列化"""
        # 确保 created_at 有有效值（不接受旧数据中的 0 值作为"从未设置"的标志）
        created_at_val = data.get("created_at")
        if not created_at_val:
            import time
            created_at_val = time.time()

        last_active_at_val = data.get("last_active_at")
        if not last_active_at_val:
            import time
            last_active_at_val = time.time()

        cluster = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            type=data.get("type", ClusterType.TEMPORARY),
            is_active=data.get("is_active", True),
            created_at=created_at_val,
            last_active_at=last_active_at_val,
            current_task=data.get("current_task", ""),
            task_progress=data.get("task_progress", 0.0),
            tasks_completed=data.get("tasks_completed", 0)
        )
        cluster.member_ids = set(data.get("member_ids", []))
        cluster.shared_knowledge = data.get("shared_knowledge", [])
        return cluster
    
    def __repr__(self) -> str:
        return f"Cluster(id={self.id[:8]}, type={self.type}, size={self.size})"


class ClusterManager:
    """簇管理器"""

    MAX_CLUSTERS = 100
    CLUSTER_TIMEOUT = 100  # 秒无活动则解散（基于 last_active_at 时间戳）

    def __init__(self):
        self.clusters: dict[str, Cluster] = {}
        self._lock = __import__('threading').Lock()
    
    def create_cluster(self, member_ids: list[str], cluster_type: str = ClusterType.TEMPORARY) -> Cluster:
        """创建簇"""
        with self._lock:
            if len(self.clusters) >= self.MAX_CLUSTERS:
                # 移除最老的簇
                self._remove_oldest_locked()

            cluster = Cluster(
                type=cluster_type,
                member_ids=set(member_ids)
            )
            cluster.created_at = time.time()

            if cluster_type == ClusterType.TASK_ORIENTED:
                cluster.name = f"Task-{cluster.id[:8]}"
            elif cluster_type == ClusterType.GENE_ORIENTED:
                cluster.name = f"Gene-{cluster.id[:8]}"
            elif cluster_type == ClusterType.ELITE:
                cluster.name = f"Elite-{cluster.id[:8]}"

            self.clusters[cluster.id] = cluster
            return cluster

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        """获取簇"""
        with self._lock:
            return self.clusters.get(cluster_id)

    def get_clusters_by_member(self, atom_id: str) -> list[Cluster]:
        """获取某原子所属的簇"""
        with self._lock:
            return [c for c in self.clusters.values() if atom_id in c.member_ids]

    def update_clusters(self, cluster_lists: list[list[str]], generation: int) -> None:
        """根据新的簇列表更新"""
        with self._lock:
            # 标记所有现有簇为非活跃
            for cluster in self.clusters.values():
                cluster.is_active = False

            # 创建或更新簇
            for member_list in cluster_lists:
                if not member_list:
                    continue

                # 尝试找到现有簇
                existing = self._find_existing_cluster_locked(member_list)

                if existing:
                    existing.is_active = True
                    for mid in member_list:
                        existing.add_member(mid)
                else:
                    # 在锁内创建簇需要特殊处理
                    if len(self.clusters) >= self.MAX_CLUSTERS:
                        self._remove_oldest_locked()
                    self._create_cluster_locked(member_list, ClusterType.TEMPORARY)

            # 清理非活跃簇
            self._cleanup_inactive_locked(generation)

    def _find_existing_cluster_locked(self, member_ids: list[str]) -> Cluster | None:
        """查找已有簇（必须在持有锁时调用）"""
        member_set = set(member_ids)

        for cluster in self.clusters.values():
            # 如果有超过 50% 的成员重叠，认为是同一簇
            overlap = len(member_set & cluster.member_ids)
            if overlap > len(member_set) * 0.5:
                return cluster

        return None

    def _create_cluster_locked(self, member_ids: list[str], cluster_type: str) -> Cluster:
        """创建簇（必须在持有锁时调用）"""
        cluster = Cluster(
            type=cluster_type,
            member_ids=set(member_ids)
        )
        self.clusters[cluster.id] = cluster
        return cluster

    def _remove_oldest_locked(self) -> None:
        """移除最老的簇（必须在持有锁时调用）"""
        if not self.clusters:
            return

        oldest = min(self.clusters.values(), key=lambda c: c.created_at)
        del self.clusters[oldest.id]

    def _cleanup_inactive_locked(self, generation: int) -> None:
        """清理非活跃簇（必须在持有锁时调用）"""
        to_remove = []

        for cluster in self.clusters.values():
            if not cluster.is_active:
                to_remove.append(cluster.id)
            elif time.time() - cluster.last_active_at > self.CLUSTER_TIMEOUT:
                to_remove.append(cluster.id)

        for cid in to_remove:
            del self.clusters[cid]
    
    def _cleanup_inactive(self, generation: int) -> None:
        """清理非活跃簇"""
        with self._lock:
            self._cleanup_inactive_locked(generation)

    def get_active_clusters(self) -> list[Cluster]:
        """获取活跃簇"""
        with self._lock:
            return [c for c in self.clusters.values() if c.is_active]

    def get_status(self) -> dict[str, Any]:
        """获取状态"""
        with self._lock:
            return {
                "total_clusters": len(self.clusters),
                "active_clusters": len([c for c in self.clusters.values() if c.is_active]),
                "by_type": self._count_by_type_locked()
            }

    def _count_by_type_locked(self) -> dict[str, int]:
        """按类型统计（必须在持有锁时调用）"""
        counts = {}
        for cluster in self.clusters.values():
            counts[cluster.type] = counts.get(cluster.type, 0) + 1
        return counts
