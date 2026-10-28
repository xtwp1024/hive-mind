# 阶段 F: 代码清理 - 检查清单

> 创建日期: 2026-04-22
> 版本: v1.0

---

## 📋 快速命令

```bash
# 1. 运行所有检查
python hive/scripts/run_cleanup.py

# 2. 运行 ruff 检查
ruff check hive/

# 3. 运行 black 格式化检查
black --check hive/

# 4. 运行 mypy 类型检查
mypy hive/ --ignore-missing-imports

# 5. 运行 isort 导入排序检查
isort --check-only hive/

# 6. 检测死代码
vulture hive/ --exclude=*_test.py,*/tests/*
```

---

## 📝 详细检查清单

### F1: 重复代码检测

| 检查项 | 状态 | 备注 |
|--------|------|------|
| 日志初始化代码是否重复 | ⬜ | `logger = logger.bind(...)` 模式 |
| 错误处理模式是否一致 | ⬜ | try/except 模式 |
| 配置读取逻辑是否重复 | ⬜ | 环境变量读取 |
| 线程安全锁模式是否一致 | ⬜ | `with self._lock:` |
| 时间戳处理是否重复 | ⬜ | `time.time()` vs `datetime` |

**工具使用:**
```bash
# 安装检测工具
pip install simian

# 检测重复代码 (需要 Java)
# 或者手动检查常见模式
```

**常见重复模式位置:**
```
hive/core/
├── hive_federation.py    # 可能与其他网络代码重复
├── node_auth.py          # 认证模式
├── memory_store.py       # 存储模式
└── dispatcher.py         # 日志模式
```

---

### F2: 注释和 Docstring 完善

#### F2.1 模块级别

| 文件 | 模块 docstring | 文件头部注释 | 状态 |
|------|-----------------|--------------|------|
| `hive/core/dispatcher.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/hive_federation.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/multimodal_encoder.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/cross_modal_association.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/multimodal_task_decoder.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/capability_gap_analyzer.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/species_blueprint.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/auto_species_generator.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/resource_allocator.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/dna_adaptive.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/node_auth.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/conflict_resolution.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/load_balancer.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/core/memory_store.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `hive/evolution/migrations.py` | ✅ 已有 | ✅ 已有 | ✅ |
| `web_dashboard_v25.py` | ✅ 已有 | ✅ 已有 | ✅ |

#### F2.2 类级别

检查每个公共类是否有 docstring:

| 类名 | 文件 | docstring | 示例代码 | 状态 |
|------|------|-----------|----------|------|
| `MessageBus` | dispatcher.py | ✅ | ⬜ | ⚠️ |
| `TaskQueue` | dispatcher.py | ✅ | ⬜ | ⚠️ |
| `HiveDispatcher` | dispatcher.py | ✅ | ⬜ | ⚠️ |
| `HiveFederation` | hive_federation.py | ✅ | ⬜ | ⚠️ |
| `KnowledgeSync` | knowledge_sync.py | ✅ | ⬜ | ⚠️ |
| `MultimodalEncoder` | multimodal_encoder.py | ✅ | ⬜ | ⚠️ |
| `CrossModalAttention` | cross_modal_association.py | ✅ | ⬜ | ⚠️ |
| `UnifiedRepresentationSpace` | cross_modal_association.py | ✅ | ⬜ | ⚠️ |
| `MultimodalTaskDecoder` | multimodal_task_decoder.py | ✅ | ⬜ | ⚠️ |
| `CapabilityGapAnalyzer` | capability_gap_analyzer.py | ✅ | ⬜ | ⚠️ |
| `SpeciesBlueprint` | species_blueprint.py | ✅ | ⬜ | ⚠️ |
| `AutoSpeciesGenerator` | auto_species_generator.py | ✅ | ⬜ | ⚠️ |
| `DynamicResourceAllocator` | resource_allocator.py | ✅ | ⬜ | ⚠️ |
| `DNAAdaptiveEngine` | dna_adaptive.py | ✅ | ⬜ | ⚠️ |
| `IdentityManager` | node_auth.py | ✅ | ⬜ | ⚠️ |
| `APIKeyManager` | node_auth.py | ✅ | ⬜ | ⚠️ |
| `SignatureVerifier` | node_auth.py | ✅ | ⬜ | ⚠️ |
| `AccessControlList` | node_auth.py | ✅ | ⬜ | ⚠️ |
| `ConflictDetector` | conflict_resolution.py | ✅ | ⬜ | ⚠️ |
| `ConflictResolver` | conflict_resolution.py | ✅ | ⬜ | ⚠️ |
| `LoadBalancer` | load_balancer.py | ✅ | ⬜ | ⚠️ |
| `HealthChecker` | load_balancer.py | ✅ | ⬜ | ⚠️ |
| `FailoverManager` | load_balancer.py | ✅ | ⬜ | ⚠️ |
| `MemoryStore` | memory_store.py | ✅ | ⬜ | ⚠️ |
| `DatabaseMigrator` | migrations.py | ✅ | ⬜ | ⚠️ |
| `MigrationRegistry` | migrations.py | ✅ | ⬜ | ⚠️ |

#### F2.3 方法级别

**标准 docstring 格式:**
```python
def function_name(param1: Type, param2: Type = default) -> ReturnType:
    """
    函数简短描述.

    详细描述（可选）.

    Args:
        param1: 参数1描述
        param2: 参数2描述

    Returns:
        返回值描述

    Example:
        >>> function_name("test", 123)
        'result'

    Raises:
        ValueError: 参数无效时
    """
```

**需要完善的关键方法:**

| 模块 | 方法 | docstring | 状态 |
|------|------|-----------|------|
| dispatcher.py | `MessageBus.subscribe()` | ✅ | ✅ |
| dispatcher.py | `MessageBus.publish()` | ✅ | ✅ |
| dispatcher.py | `HiveDispatcher.run_workflow()` | ✅ | ✅ |
| hive_federation.py | `HiveFederation.register_node()` | ✅ | ✅ |
| hive_federation.py | `HiveFederation.broadcast_knowledge()` | ✅ | ✅ |
| hive_federation.py | `HiveFederation.delegate_task()` | ✅ | ✅ |
| memory_store.py | `MemoryStore.memorize()` | ✅ | ✅ |
| memory_store.py | `MemoryStore.retrieve_rag()` | ✅ | ✅ |
| memory_store.py | `MemoryStore.save()` | ✅ | ✅ |
| memory_store.py | `MemoryStore.load()` | ✅ | ✅ |

---

### F3: 类型注解完善

#### F3.1 使用工具检查

```bash
# 安装类型检查工具
pip install mypy

# 运行类型检查
mypy hive/ --ignore-missing-imports --warn-unused-configs

# 更严格的检查
mypy hive/ --ignore-missing-imports \
    --warn-unused-configs \
    --warn-return-any \
    --warn-no-return \
    --check-untyped-defs
```

#### F3.2 类型注解标准

**函数参数:**
```python
# ❌ 不好
def process(data, callback):
    pass

# ✅ 好
from typing import TypeVar, Callable, Optional

T = TypeVar('T')

def process(
    data: dict[str, Any],
    callback: Callable[[T], None],
    transform: Optional[Callable[[dict[str, Any]], T]] = None
) -> T:
    pass
```

**数据类:**
```python
# ❌ 不好
@dataclass
class MyData:
    id = ""
    values = []

# ✅ 好
@dataclass
class MyData:
    id: str = ""
    values: list[float] = field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None
```

**列表/字典类型:**
```python
# ❌ 不好
def get_items() -> list:
    pass

def get_config() -> dict:
    pass

# ✅ 好
def get_items() -> list[Item]:
    pass

def get_config() -> dict[str, Any]:
    pass
```

#### F3.3 需要检查的模块

| 模块 | 类型注解 | 状态 |
|------|----------|------|
| `hive/core/dispatcher.py` | ✅ 部分 | ⚠️ |
| `hive/core/hive_federation.py` | ✅ 部分 | ⚠️ |
| `hive/core/memory_store.py` | ✅ 部分 | ⚠️ |
| `hive/core/multimodal_encoder.py` | ✅ 部分 | ⚠️ |
| `hive/core/cross_modal_association.py` | ✅ 部分 | ⚠️ |
| `hive/core/node_auth.py` | ✅ 部分 | ⚠️ |
| `hive/core/conflict_resolution.py` | ✅ 部分 | ⚠️ |
| `hive/core/load_balancer.py` | ✅ 部分 | ⚠️ |
| `hive/evolution/migrations.py` | ✅ 部分 | ⚠️ |

---

### F4: 无用代码清理

#### F4.1 检查清单

| 检查项 | 工具/命令 | 状态 |
|--------|-----------|------|
| 未使用的导入 | `ruff check --select=F401` | ⬜ |
| 未使用的变量 | `ruff check --select=F841` | ⬜ |
| 未使用的函数/类 | `vulture` | ⬜ |
| 注释掉的代码块 | 手动检查 | ⬜ |
| 过时的 TODO/FIXME | 手动检查 | ⬜ |
| 调试打印语句 | 手动检查 | ⬜ |

#### F4.2 常用命令

```bash
# 检查未使用的导入
ruff check hive/ --select=F401

# 检查未使用的变量
ruff check hive/ --select=F841

# 检查未定义的名称
ruff check hive/ --select=F821

# 检查所有 F 类错误 (flake8 风格)
ruff check hive/ --select=F

# 自动修复可修复的问题
ruff check hive/ --select=F401,F841 --fix

# 检测死代码
pip install vulture
vulture hive/ --exclude=*_test.py,*/tests/*
```

#### F4.3 需要检查的常见模式

```python
# ❌ 应该删除

# 1. 调试打印
print("Debug:", value)           # 应该删除或使用 logger

# 2. 注释掉的代码
# def old_function():            # 应该删除或取消注释
#     pass

# 3. 过时的 TODO
# TODO: 修复这个问题 (2023年)  # 应该完成或删除

# 4. 未使用的导入
import unused_module             # 应该删除

# 5. 未使用的变量
result = some_function()         # 应该使用或删除
# ... 没有使用 result

# 6. 空的 except 块
try:
    do_something()
except:                          # 应该指定异常类型
    pass                         # 应该记录或处理

# ✅ 正确做法

# 1. 使用 logger
logger.debug("Debug: %s", value)

# 2. 保留或删除
def old_function():              # 取消注释或删除
    pass

# 3. 完成或删除
# TODO: 修复这个问题 (当前迭代)

# 4. 删除未使用的

# 5. 使用变量
result = some_function()
process(result)

# 6. 正确处理异常
try:
    do_something()
except SpecificError as e:
    logger.error("Error occurred: %s", e)
    raise  # 或处理
```

---

## 🛠️ 自动化脚本

### 快速检查脚本

已创建: `hive/scripts/run_cleanup.py`

```bash
# 运行所有检查
python hive/scripts/run_cleanup.py

# 只运行特定检查
python hive/scripts/run_cleanup.py --check ruff
python hive/scripts/run_cleanup.py --check black
python hive/scripts/run_cleanup.py --check mypy
```

### 自动修复

```bash
# 自动格式化代码
black hive/

# 自动排序导入
isort hive/

# 自动修复 ruff 可修复的问题
ruff check hive/ --fix
```

---

## 📊 清理进度跟踪

| 任务 | 开始日期 | 完成日期 | 状态 |
|------|----------|----------|------|
| F1: 重复代码检测 | - | - | ⬜ 待开始 |
| F2: 注释/docstring 完善 | - | - | ⬜ 待开始 |
| F3: 类型注解 | - | - | ⬜ 待开始 |
| F4: 无用代码清理 | - | - | ⬜ 待开始 |

---

## ✅ 验收标准

- [ ] `ruff check hive/` 无 F401/F841/F821 错误
- [ ] `black --check hive/` 通过
- [ ] `mypy hive/` 无关键类型错误
- [ ] 所有公共类有完整 docstring
- [ ] 所有公共方法有完整 docstring
- [ ] 无明显重复代码块 (> 20 行)
- [ ] 无调试打印语句
- [ ] 无注释掉的代码块
