# -*- coding: utf-8 -*-
"""
AccessTree - 工蜂分层权限树
设计灵感来源于 Hive2Hive 的 PermissionType（OWNER/WRITE/READ）+ 共享文件夹机制。

层级继承规则：
- 子路径自动继承父路径权限（最低权限优先）
- 显式授权可覆盖继承权限
- OWNER > READ_WRITE > READ_ONLY 三级严格序
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger("Hive.AccessTree")


class PermissionType(IntEnum):
    """权限等级（对标 Hive2Hive PermissionType，数值越大权限越高）"""

    READ_ONLY = 1
    READ_WRITE = 2
    OWNER = 3

    def __str__(self) -> str:
        return self.name


class AccessDeniedError(Exception):
    """权限不足异常（对标 Hive2Hive non-allowed operation）"""

    pass


@dataclass
class ACLEntry:
    """单条访问控制记录"""

    agent_id: str
    perm: PermissionType
    granted_by: str = ""


class AccessTree:
    """
    分层权限树。
    每个路径节点可以单独设定访问权限，子路径自动继承父路径权限，可被显式覆盖。
    对标 Hive2Hive 的 fileManager.share(folder, userId, PermissionType)。
    """

    def __init__(self):
        # path → {agent_id: ACLEntry}
        self._acl: dict[str, dict[str, ACLEntry]] = {}
        self._lock = threading.RLock()
        logger.info("[AccessTree] 权限树已初始化")

    # ────────────────────────────────── 授权/撤权 ──

    def grant(
        self, path: str, agent_id: str, perm: PermissionType, granted_by: str = "system"
    ) -> None:
        """
        授权（对标 fileManager.share(folder, userId, PermissionType.WRITE)）。
        OWNER 身份才能向他人授予权限的权限。
        """
        path = self._normalize(path)
        with self._lock:
            self._acl.setdefault(path, {})[agent_id] = ACLEntry(
                agent_id=agent_id,
                perm=perm,
                granted_by=granted_by,
            )
        logger.info(f"[AccessTree] ✅ grant({path!r}, {agent_id}, {perm})")

    def revoke(self, path: str, agent_id: str) -> None:
        """撤销指定路径上某个 agent 的权限（对标内部 ACL 清除）"""
        path = self._normalize(path)
        with self._lock:
            node = self._acl.get(path, {})
            node.pop(agent_id, None)
        logger.info(f"[AccessTree] 🚫 revoke({path!r}, {agent_id})")

    # ────────────────────────────────── 鉴权 ──

    def check(self, path: str, agent_id: str, required: PermissionType) -> bool:
        """
        鉴权：检查 agent 在该路径上是否拥有 required 级别的权限。
        继承规则：向上遍历父路径，取最高匹配权限。
        """
        effective = self._effective_perm(path, agent_id)
        return effective is not None and effective >= required

    def require(self, path: str, agent_id: str, required: PermissionType) -> None:
        """
        强制鉴权：权限不足时抛出 AccessDeniedError。
        对标 Hive2Hive 内部 operation 前的权限校验门卫。
        """
        if not self.check(path, agent_id, required):
            eff = self._effective_perm(path, agent_id)
            raise AccessDeniedError(
                f"[{agent_id}] 在路径 {path!r} 需要 {required}，" f"实际拥有 {eff or '无权限'}"
            )

    def effective_permission(self, path: str, agent_id: str) -> PermissionType | None:
        """查询某 agent 在某路径上的实际有效权限（含继承）"""
        return self._effective_perm(path, agent_id)

    # ────────────────────────────────── 内部辅助 ──

    def _normalize(self, path: str) -> str:
        return "/" + path.strip("/")

    def _effective_perm(self, path: str, agent_id: str) -> PermissionType | None:
        """
        向上遍历路径层级，取最高权限（对标 Hive2Hive 继承式共享文件夹权限）。
        路径 /a/b/c → 检查顺序: /a/b/c → /a/b → /a → /
        """
        path = self._normalize(path)
        highest = None
        parts = path.split("/")

        with self._lock:
            for depth in range(len(parts), 0, -1):
                candidate = "/".join(parts[:depth]) or "/"
                node = self._acl.get(candidate, {})
                entry = node.get(agent_id)
                if entry:
                    if highest is None or entry.perm > highest:
                        highest = entry.perm

        return highest

    def list_permissions(self, path: str) -> dict[str, str]:
        """列举某路径下全部直接授权（不含继承）"""
        path = self._normalize(path)
        with self._lock:
            node = self._acl.get(path, {})
        return {aid: str(e.perm) for aid, e in node.items()}
