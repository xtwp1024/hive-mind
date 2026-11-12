# -*- coding: utf-8 -*-
"""
Node Authentication (节点身份认证)

Phase 5: 分布式网络 - 安全增强

核心职责:
1. 节点身份生成与验证 - 唯一身份、密钥对
2. API Key 管理 - 访问令牌、权限控制
3. 请求签名验证 - 防止中间人攻击
4. 访问控制列表 (ACL) - 细粒度权限控制
5. 安全通信配置 - TLS/SSL 加密

架构设计:
```
┌─────────────────────────────────────────────────────────────┐
│                    节点身份认证系统                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           身份管理器 (IdentityManager)               │   │
│  │  • 生成/验证节点身份                                 │   │
│  │  • 密钥对管理                                       │   │
│  │  • 身份签名                                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           API Key 管理器 (APIKeyManager)            │   │
│  │  • 生成/撤销 API Key                                │   │
│  │  • 权限等级管理                                     │   │
│  │  • Token 过期机制                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           签名验证器 (SignatureVerifier)             │   │
│  │  • HMAC 签名                                        │   │
│  │  • RSA 签名                                         │   │
│  │  • 请求完整性验证                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           访问控制列表 (AccessControlList)           │   │
│  │  • 节点权限配置                                     │   │
│  │  • 角色权限继承                                     │   │
│  │  • 操作权限检查                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

使用方式:
    from hive.core.node_auth import (
        IdentityManager, APIKeyManager, SignatureVerifier, AccessControlList
    )

    # 身份管理
    identity = IdentityManager()
    node_id, public_key = identity.generate_node_identity()

    # API Key 管理
    api_key_mgr = APIKeyManager()
    api_key = api_key_mgr.generate_api_key("admin", permissions=["read", "write", "admin"])

    # 签名验证
    verifier = SignatureVerifier(identity.private_key)
    signature = verifier.sign_request({"action": "sync", "timestamp": 1234567890})
    is_valid = verifier.verify_request({"action": "sync"}, signature)

    # 访问控制
    acl = AccessControlList()
    acl.grant_permission(node_id, "knowledge.read")
    allowed = acl.check_permission(node_id, "knowledge.read")
"""

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import secrets

    HAS_SECRETS = True
except ImportError:
    HAS_SECRETS = False

logger = logging.getLogger("Hive.NodeAuth")


class PermissionLevel(str, Enum):
    """权限等级"""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    NONE = "none"


class SignatureAlgorithm(str, Enum):
    """签名算法"""

    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA512 = "hmac-sha512"


@dataclass
class NodeIdentity:
    """节点身份"""

    node_id: str
    public_key: str
    private_key: str

    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "public_key": self.public_key,
            "private_key": self.private_key,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeIdentity":
        return cls(
            node_id=data["node_id"],
            public_key=data["public_key"],
            private_key=data["private_key"],
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class APIKey:
    """API Key 信息"""

    key_id: str
    key_value: str
    node_id: str

    permissions: List[str] = field(default_factory=list)
    level: PermissionLevel = PermissionLevel.READ

    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_used_at: Optional[float] = None

    is_revoked: bool = False
    revoked_at: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """检查 Key 是否有效"""
        if self.is_revoked:
            return False
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True

    def has_permission(self, permission: str) -> bool:
        """检查是否有特定权限"""
        if self.level == PermissionLevel.ADMIN:
            return True
        if self.level == PermissionLevel.WRITE and permission in ["read", "write"]:
            return True
        if self.level == PermissionLevel.READ and permission == "read":
            return True
        return permission in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "key_value": self.key_value,
            "node_id": self.node_id,
            "permissions": self.permissions,
            "level": self.level.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used_at": self.last_used_at,
            "is_revoked": self.is_revoked,
            "revoked_at": self.revoked_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKey":
        level_str = data.get("level", "read")
        try:
            level = PermissionLevel(level_str)
        except ValueError:
            level = PermissionLevel.READ

        return cls(
            key_id=data["key_id"],
            key_value=data["key_value"],
            node_id=data["node_id"],
            permissions=data.get("permissions", []),
            level=level,
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            last_used_at=data.get("last_used_at"),
            is_revoked=data.get("is_revoked", False),
            revoked_at=data.get("revoked_at"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SignedRequest:
    """已签名的请求"""

    payload: Dict[str, Any]
    signature: str
    timestamp: float
    node_id: str
    key_id: Optional[str] = None
    algorithm: str = "hmac-sha256"


@dataclass
class ACLPermission:
    """ACL 权限条目"""

    node_id: str
    permission: str
    granted_by: str
    granted_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None


class IdentityManager:
    """
    节点身份管理器

    生成和管理节点身份、密钥对。
    """

    def __init__(
        self,
        key_size: int = 32,
        storage_path: Optional[Path] = None,
    ):
        self.key_size = key_size
        self.storage_path = storage_path

        self._identities: Dict[str, NodeIdentity] = {}
        self._lock = threading.RLock()

        self._my_identity: Optional[NodeIdentity] = None

        logger.info(f"🔐 [IdentityManager] 初始化，密钥大小: {key_size}")

    def generate_node_identity(
        self,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NodeIdentity:
        """
        生成新的节点身份

        Args:
            node_id: 可选的节点ID (默认自动生成)
            metadata: 元数据

        Returns:
            NodeIdentity - 节点身份
        """
        if node_id is None:
            node_id = f"hive-{uuid.uuid4().hex[:12]}"

        private_key = self._generate_secret_key(self.key_size)
        public_key = self._derive_public_key(private_key)

        identity = NodeIdentity(
            node_id=node_id,
            public_key=public_key,
            private_key=private_key,
            metadata=metadata or {},
        )

        with self._lock:
            self._identities[node_id] = identity

        self._my_identity = identity
        logger.info(f"🔐 [IdentityManager] 生成新节点身份: {node_id}")

        return identity

    def _generate_secret_key(self, size: int) -> str:
        """生成密钥"""
        if HAS_SECRETS:
            raw = secrets.token_bytes(size)
        else:
            raw = uuid.uuid4().bytes + uuid.uuid4().bytes
            raw = raw[:size]
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def _derive_public_key(self, private_key: str) -> str:
        """从私钥派生公钥 (简化版)"""
        h = hashlib.sha256(private_key.encode("utf-8"))
        h.update(b"hive-public-key-derivation-salt")
        return base64.urlsafe_b64encode(h.digest()).decode("ascii").rstrip("=")

    def get_my_identity(self) -> Optional[NodeIdentity]:
        """获取本节点身份"""
        return self._my_identity

    def get_identity(self, node_id: str) -> Optional[NodeIdentity]:
        """获取指定节点的身份"""
        with self._lock:
            return self._identities.get(node_id)

    def register_peer_identity(
        self,
        node_id: str,
        public_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        注册对等节点身份

        Args:
            node_id: 节点ID
            public_key: 公钥
            metadata: 元数据

        Returns:
            bool - 是否成功
        """
        with self._lock:
            if node_id in self._identities:
                existing = self._identities[node_id]
                if existing.public_key != public_key:
                    logger.warning(f"🔐 [IdentityManager] 节点 {node_id} 公钥变更，拒绝更新")
                    return False
                return True

            self._identities[node_id] = NodeIdentity(
                node_id=node_id,
                public_key=public_key,
                private_key="",
                metadata=metadata or {},
            )

        logger.info(f"🔐 [IdentityManager] 注册对等节点身份: {node_id}")
        return True

    def list_known_nodes(self) -> List[str]:
        """列出所有已知节点"""
        with self._lock:
            return list(self._identities.keys())

    def verify_identity(
        self,
        node_id: str,
        challenge: str,
        signature: str,
    ) -> bool:
        """
        验证节点身份

        Args:
            node_id: 节点ID
            challenge: 挑战字符串
            signature: 签名

        Returns:
            bool - 验证是否通过
        """
        with self._lock:
            identity = self._identities.get(node_id)
            if not identity:
                return False

        verifier = SignatureVerifier(identity.public_key)
        return verifier.verify(challenge, signature)


class APIKeyManager:
    """
    API Key 管理器

    生成、验证、撤销 API Key。
    """

    def __init__(
        self,
        key_size: int = 32,
        default_ttl_days: int = 30,
    ):
        self.key_size = key_size
        self.default_ttl_days = default_ttl_days

        self._api_keys: Dict[str, APIKey] = {}
        self._by_node: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

        logger.info(f"🔑 [APIKeyManager] 初始化，默认有效期: {default_ttl_days} 天")

    def generate_api_key(
        self,
        node_id: str,
        permissions: Optional[List[str]] = None,
        level: PermissionLevel = PermissionLevel.READ,
        ttl_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        生成 API Key

        Args:
            node_id: 节点ID
            permissions: 权限列表
            level: 权限等级
            ttl_days: 有效期天数 (None 表示永不过期)
            metadata: 元数据

        Returns:
            str - API Key 值 (完整格式: {key_id}.{key_value})
        """
        key_id = f"key_{uuid.uuid4().hex[:12]}"
        key_value = self._generate_key_value(self.key_size)

        expires_at = None
        if ttl_days is not None:
            expires_at = time.time() + (ttl_days * 24 * 60 * 60)
        elif self.default_ttl_days > 0:
            expires_at = time.time() + (self.default_ttl_days * 24 * 60 * 60)

        api_key = APIKey(
            key_id=key_id,
            key_value=key_value,
            node_id=node_id,
            permissions=permissions or [],
            level=level,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        full_key = f"{key_id}.{key_value}"

        with self._lock:
            self._api_keys[key_id] = api_key
            self._by_node[node_id].add(key_id)

        logger.info(f"🔑 [APIKeyManager] 生成 API Key: {key_id} for {node_id}")
        return full_key

    def _generate_key_value(self, size: int) -> str:
        """生成 Key 值"""
        if HAS_SECRETS:
            raw = secrets.token_bytes(size)
        else:
            raw = uuid.uuid4().bytes + uuid.uuid4().bytes
            raw = raw[:size]
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def validate_api_key(
        self,
        full_key: str,
        required_permission: Optional[str] = None,
    ) -> Tuple[bool, Optional[APIKey]]:
        """
        验证 API Key

        Args:
            full_key: 完整 Key (key_id.key_value)
            required_permission: 要求的权限

        Returns:
            (bool, APIKey|None) - 是否有效，以及 Key 对象
        """
        try:
            if "." not in full_key:
                return False, None

            key_id, key_value = full_key.split(".", 1)
        except ValueError:
            return False, None

        with self._lock:
            api_key = self._api_keys.get(key_id)

        if not api_key:
            return False, None

        if not api_key.is_valid():
            return False, None

        if api_key.key_value != key_value:
            return False, None

        if required_permission and not api_key.has_permission(required_permission):
            return False, api_key

        with self._lock:
            self._api_keys[key_id].last_used_at = time.time()

        return True, api_key

    def revoke_api_key(self, key_id: str, reason: str = "") -> bool:
        """
        撤销 API Key

        Args:
            key_id: Key ID
            reason: 撤销原因

        Returns:
            bool - 是否成功
        """
        with self._lock:
            api_key = self._api_keys.get(key_id)
            if not api_key:
                return False

            api_key.is_revoked = True
            api_key.revoked_at = time.time()
            api_key.metadata["revoke_reason"] = reason

        logger.info(f"🔑 [APIKeyManager] 撤销 API Key: {key_id}, reason: {reason}")
        return True

    def get_keys_for_node(self, node_id: str) -> List[APIKey]:
        """获取节点的所有 API Key"""
        with self._lock:
            key_ids = self._by_node.get(node_id, set())
            return [self._api_keys[kid] for kid in key_ids if kid in self._api_keys]

    def rotate_api_key(
        self,
        key_id: str,
        ttl_days: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[APIKey]]:
        """
        轮换 API Key (生成新的，撤销旧的)

        Args:
            key_id: 旧 Key ID
            ttl_days: 新 Key 有效期

        Returns:
            (new_full_key, old_key) - 新 Key 和旧 Key
        """
        with self._lock:
            old_key = self._api_keys.get(key_id)
            if not old_key or not old_key.is_valid():
                return None, None

            old_key.is_revoked = True
            old_key.revoked_at = time.time()

        new_full_key = self.generate_api_key(
            node_id=old_key.node_id,
            permissions=old_key.permissions,
            level=old_key.level,
            ttl_days=ttl_days,
            metadata={
                "rotated_from": key_id,
                **old_key.metadata,
            },
        )

        logger.info(f"🔑 [APIKeyManager] 轮换 API Key: {key_id} -> new key")
        return new_full_key, old_key

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            valid_count = sum(1 for k in self._api_keys.values() if k.is_valid())
            revoked_count = sum(1 for k in self._api_keys.values() if k.is_revoked)

            by_level = defaultdict(int)
            for k in self._api_keys.values():
                by_level[k.level.value] += 1

            return {
                "total_keys": len(self._api_keys),
                "valid_keys": valid_count,
                "revoked_keys": revoked_count,
                "by_level": dict(by_level),
                "nodes_with_keys": len(self._by_node),
            }


class SignatureVerifier:
    """
    签名验证器

    提供 HMAC 签名和验证功能。
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256,
    ):
        self.secret_key = secret_key.encode("utf-8")
        self.algorithm = algorithm

        if algorithm == SignatureAlgorithm.HMAC_SHA256:
            self._hash_func = hashlib.sha256
        else:
            self._hash_func = hashlib.sha512

        logger.info(f"✍️ [SignatureVerifier] 初始化，算法: {algorithm.value}")

    def sign(self, data: str) -> str:
        """
        对字符串数据签名

        Args:
            data: 要签名的数据

        Returns:
            str - 签名 (Base64 编码)
        """
        mac = hmac.new(self.secret_key, data.encode("utf-8"), self._hash_func)
        return base64.urlsafe_b64encode(mac.digest()).decode("ascii").rstrip("=")

    def verify(self, data: str, signature: str) -> bool:
        """
        验证签名

        Args:
            data: 原始数据
            signature: 签名

        Returns:
            bool - 签名是否有效
        """
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)

    def sign_request(
        self,
        payload: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> SignedRequest:
        """
        对请求载荷签名

        Args:
            payload: 请求载荷
            timestamp: 时间戳 (默认当前时间)

        Returns:
            SignedRequest - 已签名的请求
        """
        if timestamp is None:
            timestamp = time.time()

        payload_with_ts = {
            **payload,
            "_timestamp": timestamp,
        }

        payload_str = json.dumps(payload_with_ts, sort_keys=True, ensure_ascii=False)
        signature = self.sign(payload_str)

        return SignedRequest(
            payload=payload,
            signature=signature,
            timestamp=timestamp,
            node_id="",
            algorithm=self.algorithm.value,
        )

    def verify_request(
        self,
        payload: Dict[str, Any],
        signature: str,
        timestamp: float,
        max_age_seconds: float = 300.0,
    ) -> Tuple[bool, str]:
        """
        验证请求签名

        Args:
            payload: 请求载荷
            signature: 签名
            timestamp: 时间戳
            max_age_seconds: 最大允许的时间差 (秒)

        Returns:
            (bool, str) - (是否有效, 错误信息)
        """
        now = time.time()
        age = now - timestamp

        if abs(age) > max_age_seconds:
            return False, f"请求已过期: 时间差 {age:.1f}s > {max_age_seconds}s"

        payload_with_ts = {
            **payload,
            "_timestamp": timestamp,
        }

        payload_str = json.dumps(payload_with_ts, sort_keys=True, ensure_ascii=False)

        if not self.verify(payload_str, signature):
            return False, "签名验证失败"

        return True, "验证通过"


class AccessControlList:
    """
    访问控制列表

    细粒度的权限控制。
    """

    STANDARD_PERMISSIONS = {
        "knowledge.read": "读取知识",
        "knowledge.write": "写入知识",
        "knowledge.admin": "知识管理",
        "task.read": "读取任务状态",
        "task.write": "提交任务",
        "task.admin": "任务管理",
        "node.read": "读取节点信息",
        "node.write": "更新节点状态",
        "node.admin": "节点管理",
        "federation.read": "读取联邦状态",
        "federation.write": "加入/离开联邦",
        "federation.admin": "联邦管理",
        "sync.read": "读取同步状态",
        "sync.write": "发起同步",
        "sync.admin": "同步管理",
    }

    ROLES = {
        "guest": ["knowledge.read", "task.read", "node.read"],
        "member": [
            "knowledge.read",
            "knowledge.write",
            "task.read",
            "task.write",
            "node.read",
            "sync.write",
        ],
        "admin": ["*"],
    }

    def __init__(self):
        self._permissions: Dict[str, Dict[str, ACLPermission]] = defaultdict(dict)
        self._roles: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

        logger.info(f"🛂 [AccessControlList] 初始化")

    def grant_permission(
        self,
        node_id: str,
        permission: str,
        granted_by: str = "system",
        ttl_seconds: Optional[float] = None,
    ) -> bool:
        """
        授予权限

        Args:
            node_id: 节点ID
            permission: 权限名称
            granted_by: 授权者
            ttl_seconds: 有效期 (秒)

        Returns:
            bool - 是否成功
        """
        with self._lock:
            expires_at = None
            if ttl_seconds is not None:
                expires_at = time.time() + ttl_seconds

            self._permissions[node_id][permission] = ACLPermission(
                node_id=node_id,
                permission=permission,
                granted_by=granted_by,
                expires_at=expires_at,
            )

        logger.info(f"🛂 [ACL] 授予权限: {node_id} -> {permission} (by {granted_by})")
        return True

    def revoke_permission(self, node_id: str, permission: str) -> bool:
        """
        撤销权限

        Args:
            node_id: 节点ID
            permission: 权限名称

        Returns:
            bool - 是否成功
        """
        with self._lock:
            if node_id not in self._permissions:
                return False
            if permission not in self._permissions[node_id]:
                return False

            del self._permissions[node_id][permission]

        logger.info(f"🛂 [ACL] 撤销权限: {node_id} -> {permission}")
        return True

    def check_permission(
        self,
        node_id: str,
        permission: str,
    ) -> bool:
        """
        检查权限

        Args:
            node_id: 节点ID
            permission: 权限名称

        Returns:
            bool - 是否有权限
        """
        with self._lock:
            roles = self._roles.get(node_id, set())
            for role in roles:
                role_perms = self.ROLES.get(role, [])
                if "*" in role_perms:
                    return True
                for perm in role_perms:
                    if self._match_permission(perm, permission):
                        return True

            perms = self._permissions.get(node_id, {})
            for perm, entry in perms.items():
                if entry.expires_at and time.time() > entry.expires_at:
                    continue
                if self._match_permission(perm, permission):
                    return True

        return False

    def _match_permission(self, pattern: str, permission: str) -> bool:
        """匹配权限模式 (支持通配符 *)"""
        if pattern == "*":
            return True
        if pattern == permission:
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return permission.startswith(prefix)
        return False

    def assign_role(self, node_id: str, role: str) -> bool:
        """
        分配角色

        Args:
            node_id: 节点ID
            role: 角色名

        Returns:
            bool - 是否成功
        """
        if role not in self.ROLES:
            logger.warning(f"🛂 [ACL] 未知角色: {role}")
            return False

        with self._lock:
            self._roles[node_id].add(role)

        logger.info(f"🛂 [ACL] 分配角色: {node_id} -> {role}")
        return True

    def revoke_role(self, node_id: str, role: str) -> bool:
        """撤销角色"""
        with self._lock:
            if node_id not in self._roles:
                return False
            if role not in self._roles[node_id]:
                return False

            self._roles[node_id].remove(role)

        logger.info(f"🛂 [ACL] 撤销角色: {node_id} -> {role}")
        return True

    def get_node_permissions(self, node_id: str) -> Dict[str, Any]:
        """获取节点的所有权限"""
        with self._lock:
            perms = {
                perm: {
                    "granted_by": entry.granted_by,
                    "granted_at": entry.granted_at,
                    "expires_at": entry.expires_at,
                }
                for perm, entry in self._permissions.get(node_id, {}).items()
                if not (entry.expires_at and time.time() > entry.expires_at)
            }

            roles = list(self._roles.get(node_id, set()))

            effective_perms = set()
            for role in roles:
                effective_perms.update(self.ROLES.get(role, []))
            effective_perms.update(perms.keys())

            return {
                "node_id": node_id,
                "roles": roles,
                "direct_permissions": perms,
                "effective_permissions": sorted(effective_perms),
            }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_perms = sum(len(p) for p in self._permissions.values())
            nodes_with_perms = len(self._permissions)
            nodes_with_roles = len(self._roles)

            return {
                "nodes_with_permissions": nodes_with_perms,
                "nodes_with_roles": nodes_with_roles,
                "total_permissions": total_perms,
                "standard_permissions": len(self.STANDARD_PERMISSIONS),
                "available_roles": list(self.ROLES.keys()),
            }


class NodeAuthenticationEngine:
    """
    节点身份认证引擎

    整合所有认证组件的统一接口。
    """

    def __init__(
        self,
        enable_acl: bool = True,
        require_signature: bool = False,
        require_api_key: bool = True,
    ):
        self.enable_acl = enable_acl
        self.require_signature = require_signature
        self.require_api_key = require_api_key

        self._identity_manager = IdentityManager()
        self._api_key_manager = APIKeyManager()
        self._signature_verifiers: Dict[str, SignatureVerifier] = {}
        self._acl = AccessControlList() if enable_acl else None

        self._lock = threading.RLock()

        logger.info(f"🔒 [NodeAuthEngine] 初始化，ACL: {enable_acl}, 签名验证: {require_signature}")

    def initialize_node(
        self,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str]:
        """
        初始化本节点身份

        Args:
            node_id: 可选的节点ID
            metadata: 元数据

        Returns:
            (node_id, public_key, private_key)
        """
        identity = self._identity_manager.generate_node_identity(node_id, metadata)

        with self._lock:
            self._signature_verifiers[identity.node_id] = SignatureVerifier(identity.private_key)

        if self._acl:
            self._acl.assign_role(identity.node_id, "admin")

        logger.info(f"🔒 [NodeAuthEngine] 节点初始化完成: {identity.node_id}")
        return identity.node_id, identity.public_key, identity.private_key

    def register_peer(
        self,
        node_id: str,
        public_key: str,
        role: str = "member",
    ) -> bool:
        """
        注册对等节点

        Args:
            node_id: 节点ID
            public_key: 公钥
            role: 角色

        Returns:
            bool - 是否成功
        """
        if not self._identity_manager.register_peer_identity(node_id, public_key):
            return False

        with self._lock:
            self._signature_verifiers[node_id] = SignatureVerifier(public_key)

        if self._acl and role in AccessControlList.ROLES:
            self._acl.assign_role(node_id, role)

        logger.info(f"🔒 [NodeAuthEngine] 注册对等节点: {node_id} (role: {role})")
        return True

    def authenticate_request(
        self,
        node_id: str,
        payload: Dict[str, Any],
        signature: Optional[str] = None,
        timestamp: Optional[float] = None,
        api_key: Optional[str] = None,
        required_permission: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        认证请求

        Args:
            node_id: 节点ID
            payload: 请求载荷
            signature: 签名 (可选，取决于配置)
            timestamp: 时间戳 (可选)
            api_key: API Key (可选)
            required_permission: 要求的权限 (可选)

        Returns:
            (bool, str) - (是否通过, 消息)
        """
        if self.require_api_key and api_key:
            valid, key_obj = self._api_key_manager.validate_api_key(api_key, required_permission)
            if not valid:
                return False, f"API Key 验证失败"
            if key_obj and key_obj.node_id != node_id:
                return False, "API Key 不属于该节点"

        if self.require_signature:
            if not signature or not timestamp:
                return False, "缺少签名或时间戳"

            verifier = self._signature_verifiers.get(node_id)
            if not verifier:
                return False, f"未知节点: {node_id}"

            valid, msg = verifier.verify_request(payload, signature, timestamp)
            if not valid:
                return False, f"签名验证失败: {msg}"

        if self._acl and required_permission:
            if not self._acl.check_permission(node_id, required_permission):
                return False, f"权限不足: 需要 {required_permission}"

        return True, "认证通过"

    def generate_api_key_for_node(
        self,
        node_id: str,
        level: PermissionLevel = PermissionLevel.READ,
        ttl_days: Optional[int] = None,
    ) -> str:
        """为节点生成 API Key"""
        return self._api_key_manager.generate_api_key(
            node_id=node_id,
            level=level,
            ttl_days=ttl_days,
        )

    def sign_request(
        self,
        payload: Dict[str, Any],
    ) -> SignedRequest:
        """签名请求"""
        my_identity = self._identity_manager.get_my_identity()
        if not my_identity:
            raise ValueError("节点未初始化")

        verifier = self._signature_verifiers.get(my_identity.node_id)
        if not verifier:
            raise ValueError("签名验证器未初始化")

        signed = verifier.sign_request(payload)
        signed.node_id = my_identity.node_id
        return signed

    def get_security_headers(self) -> Dict[str, Any]:
        """获取安全头部信息"""
        my_identity = self._identity_manager.get_my_identity()
        return {
            "node_id": my_identity.node_id if my_identity else None,
            "require_signature": self.require_signature,
            "require_api_key": self.require_api_key,
            "acl_enabled": self.enable_acl,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取认证系统状态"""
        return {
            "my_node_id": (
                self._identity_manager.get_my_identity().node_id
                if self._identity_manager.get_my_identity()
                else None
            ),
            "known_nodes": len(self._identity_manager.list_known_nodes()),
            "api_keys": self._api_key_manager.get_stats(),
            "acl": self._acl.get_stats() if self._acl else None,
            "require_signature": self.require_signature,
            "require_api_key": self.require_api_key,
        }


_auth_engine: Optional[NodeAuthenticationEngine] = None
_auth_lock = threading.Lock()


def get_node_auth_engine() -> NodeAuthenticationEngine:
    """获取节点认证引擎单例"""
    global _auth_engine
    if _auth_engine is None:
        with _auth_lock:
            if _auth_engine is None:
                _auth_engine = NodeAuthenticationEngine()
    return _auth_engine


def init_node_auth(**kwargs) -> NodeAuthenticationEngine:
    """初始化节点认证引擎"""
    global _auth_engine
    with _auth_lock:
        if _auth_engine is not None:
            return _auth_engine
        _auth_engine = NodeAuthenticationEngine(**kwargs)
    return _auth_engine
