# hive.core.node_auth

Node Authentication (节点身份认证) Phase 5: 分布式网络 - 安全增强 核心职责: 1. 节点身份生成与验证 - 唯一身份、密钥对 2. API Key 管理 - 访问令牌、权限控制 3. 请求签名验证 - 防止中间人攻击 4. 访问控制列表 (ACL) - 细粒度权限控制 5. 安全通信配置 - TLS/SSL 加密 架构设计: ``` ┌─────────────────────────────────────────────────────────────┐ │                    节点身份认证系统                            │ ├─────────────────────────────────────────────────────────────┤ │                                                              │ │  ┌─────────────────────────────────────────────────────┐   │ │  │           身份管理器 (IdentityManager)               │   │ │  │  • 生成/验证节点身份                                 │   │ │  │  • 密钥对管理                                       │   │ │  │  • 身份签名                                         │   │ │  └─────────────────────────────────────────────────────┘   │ │                           │                                  │ │                           ▼                                  │ │  ┌─────────────────────────────────────────────────────┐   │ │  │           API Key 管理器 (APIKeyManager)            │   │ │  │  • 生成/撤销 API Key                                │   │ │  │  • 权限等级管理                                     │   │ │  │  • Token 过期机制                                   │   │ │  └─────────────────────────────────────────────────────┘   │ │                           │                                  │ │                           ▼                                  │ │  ┌─────────────────────────────────────────────────────┐   │ │  │           签名验证器 (SignatureVerifier)             │   │ │  │  • HMAC 签名                                        │   │ │  │  • RSA 签名                                         │   │ │  │  • 请求完整性验证                                   │   │ │  └─────────────────────────────────────────────────────┘   │ │                           │                                  │ │                           ▼                                  │ │  ┌─────────────────────────────────────────────────────┐   │ │  │           访问控制列表 (AccessControlList)           │   │ │  │  • 节点权限配置                                     │   │ │  │  • 角色权限继承                                     │   │ │  │  • 操作权限检查                                     │   │ │  └─────────────────────────────────────────────────────┘   │ │                                                              │ └─────────────────────────────────────────────────────────────┘ ``` 使用方式: from hive.core.node_auth import ( IdentityManager, APIKeyManager, SignatureVerifier, AccessControlList ) # 身份管理 identity = IdentityManager() node_id, public_key = identity.generate_node_identity() # API Key 管理 api_key_mgr = APIKeyManager() api_key = api_key_mgr.generate_api_key("admin", permissions=["read", "write", "admin"]) # 签名验证 verifier = SignatureVerifier(identity.private_key) signature = verifier.sign_request({"action": "sync", "timestamp": 1234567890}) is_valid = verifier.verify_request({"action": "sync"}, signature) # 访问控制 acl = AccessControlList() acl.grant_permission(node_id, "knowledge.read") allowed = acl.check_permission(node_id, "knowledge.read")

## 目录

- [类](#类)
  - [ACLPermission](#aclpermission)
  - [APIKey](#apikey)
  - [APIKeyManager](#apikeymanager)
  - [AccessControlList](#accesscontrollist)
  - [IdentityManager](#identitymanager)
  - [NodeAuthenticationEngine](#nodeauthenticationengine)
  - [NodeIdentity](#nodeidentity)
  - [PermissionLevel](#permissionlevel)
  - [SignatureAlgorithm](#signaturealgorithm)
  - [SignatureVerifier](#signatureverifier)
  - [SignedRequest](#signedrequest)
- [函数](#函数)
  - [get_node_auth_engine](#get_node_auth_engine)
  - [init_node_auth](#init_node_auth)

## 类

### ACLPermission

**@dataclass**

ACL 权限条目

#### 属性

- `expires_at`

### APIKey

**@dataclass**

API Key 信息

#### 方法

```python
def from_dict(data: Dict) -> APIKey
```

```python
def has_permission(permission: str) -> bool
```

检查是否有特定权限

```python
def is_valid() -> bool
```

检查 Key 是否有效

```python
def to_dict() -> Dict
```

#### 属性

- `expires_at`
- `is_revoked`
- `last_used_at`
- `level`
- `revoked_at`

### APIKeyManager

API Key 管理器 生成、验证、撤销 API Key。

#### 方法

```python
def generate_api_key(node_id: str, permissions: Optional = None, level: PermissionLevel = PermissionLevel.READ, ttl_days: Optional = None, metadata: Optional = None) -> str
```

生成 API Key

**参数:**

- `node_id`: 节点ID
- `permissions`: 权限列表
- `level`: 权限等级
- `ttl_days`: 有效期天数 (None 表示永不过期)
- `metadata`: 元数据

**返回:** str - API Key 值 (完整格式: {key_id}.{key_value})

```python
def get_keys_for_node(node_id: str) -> List
```

获取节点的所有 API Key

```python
def get_stats() -> Dict
```

获取统计信息

```python
def revoke_api_key(key_id: str, reason: str) -> bool
```

撤销 API Key

**参数:**

- `key_id`: Key ID
- `reason`: 撤销原因

**返回:** bool - 是否成功

```python
def rotate_api_key(key_id: str, ttl_days: Optional = None) -> Tuple
```

轮换 API Key (生成新的，撤销旧的)

**参数:**

- `key_id`: 旧 Key ID
- `ttl_days`: 新 Key 有效期

**返回:** (new_full_key, old_key) - 新 Key 和旧 Key

```python
def validate_api_key(full_key: str, required_permission: Optional = None) -> Tuple
```

验证 API Key

**参数:**

- `full_key`: 完整 Key (key_id.key_value)
- `required_permission`: 要求的权限

**返回:** (bool, APIKey|None) - 是否有效，以及 Key 对象

### AccessControlList

访问控制列表 细粒度的权限控制。

#### 方法

```python
def assign_role(node_id: str, role: str) -> bool
```

分配角色

**参数:**

- `node_id`: 节点ID
- `role`: 角色名

**返回:** bool - 是否成功

```python
def check_permission(node_id: str, permission: str) -> bool
```

检查权限

**参数:**

- `node_id`: 节点ID
- `permission`: 权限名称

**返回:** bool - 是否有权限

```python
def get_node_permissions(node_id: str) -> Dict
```

获取节点的所有权限

```python
def get_stats() -> Dict
```

获取统计信息

```python
def grant_permission(node_id: str, permission: str, granted_by: str = system, ttl_seconds: Optional = None) -> bool
```

授予权限

**参数:**

- `node_id`: 节点ID
- `permission`: 权限名称
- `granted_by`: 授权者
- `ttl_seconds`: 有效期 (秒)

**返回:** bool - 是否成功

```python
def revoke_permission(node_id: str, permission: str) -> bool
```

撤销权限

**参数:**

- `node_id`: 节点ID
- `permission`: 权限名称

**返回:** bool - 是否成功

```python
def revoke_role(node_id: str, role: str) -> bool
```

撤销角色

#### 属性

- `ROLES`
- `STANDARD_PERMISSIONS`

### IdentityManager

节点身份管理器 生成和管理节点身份、密钥对。

#### 方法

```python
def generate_node_identity(node_id: Optional = None, metadata: Optional = None) -> NodeIdentity
```

生成新的节点身份

**参数:**

- `node_id`: 可选的节点ID (默认自动生成)
- `metadata`: 元数据

**返回:** NodeIdentity - 节点身份

```python
def get_identity(node_id: str) -> Optional
```

获取指定节点的身份

```python
def get_my_identity() -> Optional
```

获取本节点身份

```python
def list_known_nodes() -> List
```

列出所有已知节点

```python
def register_peer_identity(node_id: str, public_key: str, metadata: Optional = None) -> bool
```

注册对等节点身份

**参数:**

- `node_id`: 节点ID
- `public_key`: 公钥
- `metadata`: 元数据

**返回:** bool - 是否成功

```python
def verify_identity(node_id: str, challenge: str, signature: str) -> bool
```

验证节点身份

**参数:**

- `node_id`: 节点ID
- `challenge`: 挑战字符串
- `signature`: 签名

**返回:** bool - 验证是否通过

### NodeAuthenticationEngine

节点身份认证引擎 整合所有认证组件的统一接口。

#### 方法

```python
def authenticate_request(node_id: str, payload: Dict, signature: Optional = None, timestamp: Optional = None, api_key: Optional = None, required_permission: Optional = None) -> Tuple
```

认证请求

**参数:**

- `node_id`: 节点ID
- `payload`: 请求载荷
- `signature`: 签名 (可选，取决于配置)
- `timestamp`: 时间戳 (可选)
- `api_key`: API Key (可选)
- `required_permission`: 要求的权限 (可选)

**返回:** (bool, str) - (是否通过, 消息)

```python
def generate_api_key_for_node(node_id: str, level: PermissionLevel = PermissionLevel.READ, ttl_days: Optional = None) -> str
```

为节点生成 API Key

```python
def get_security_headers() -> Dict
```

获取安全头部信息

```python
def get_stats() -> Dict
```

获取认证系统状态

```python
def initialize_node(node_id: Optional = None, metadata: Optional = None) -> Tuple
```

初始化本节点身份

**参数:**

- `node_id`: 可选的节点ID
- `metadata`: 元数据

**返回:** (node_id, public_key, private_key)

```python
def register_peer(node_id: str, public_key: str, role: str = member) -> bool
```

注册对等节点

**参数:**

- `node_id`: 节点ID
- `public_key`: 公钥
- `role`: 角色

**返回:** bool - 是否成功

```python
def sign_request(payload: Dict) -> SignedRequest
```

签名请求

### NodeIdentity

**@dataclass**

节点身份

#### 方法

```python
def from_dict(data: Dict) -> NodeIdentity
```

```python
def to_dict() -> Dict
```

### PermissionLevel

**Enum**

**继承:** `str, Enum`

权限等级

#### 属性

- `ADMIN`
- `NONE`
- `READ`
- `WRITE`
- `name`
- `value`

### SignatureAlgorithm

**Enum**

**继承:** `str, Enum`

签名算法

#### 属性

- `HMAC_SHA256`
- `HMAC_SHA512`
- `name`
- `value`

### SignatureVerifier

签名验证器 提供 HMAC 签名和验证功能。

#### 方法

```python
def sign(data: str) -> str
```

对字符串数据签名

**参数:**

- `data`: 要签名的数据

**返回:** str - 签名 (Base64 编码)

```python
def sign_request(payload: Dict, timestamp: Optional = None) -> SignedRequest
```

对请求载荷签名

**参数:**

- `payload`: 请求载荷
- `timestamp`: 时间戳 (默认当前时间)

**返回:** SignedRequest - 已签名的请求

```python
def verify(data: str, signature: str) -> bool
```

验证签名

**参数:**

- `data`: 原始数据
- `signature`: 签名

**返回:** bool - 签名是否有效

```python
def verify_request(payload: Dict, signature: str, timestamp: float, max_age_seconds: float = 300.0) -> Tuple
```

验证请求签名

**参数:**

- `payload`: 请求载荷
- `signature`: 签名
- `timestamp`: 时间戳
- `max_age_seconds`: 最大允许的时间差 (秒)

**返回:** (bool, str) - (是否有效, 错误信息)

### SignedRequest

**@dataclass**

已签名的请求

#### 属性

- `algorithm`
- `key_id`

## 函数

### get_node_auth_engine

```python
def get_node_auth_engine() -> NodeAuthenticationEngine
```

获取节点认证引擎单例

### init_node_auth

```python
def init_node_auth(kwargs) -> NodeAuthenticationEngine
```

初始化节点认证引擎

---

*文档生成于: 2026-04-23T01:21:36.671532*
