# hive.evolution.migrations

Hive 数据库版本化迁移系统 基于 GEPStore 的 Schema Versioning 实现。 确保数据库平滑升级，数据不丢失。 设计原则: 1. 每个迁移是一个独立的版本 2. 支持升级和降级 3. 记录迁移历史 4. 幂等操作（重复执行安全）

## 目录

- [类](#类)
  - [DatabaseMigrator](#databasemigrator)
  - [MigrationRecord](#migrationrecord)
  - [MigrationRegistry](#migrationregistry)
  - [MigrationStatus](#migrationstatus)
  - [MigrationStep](#migrationstep)
- [函数](#函数)
  - [ensure_latest_schema](#ensure_latest_schema)
  - [get_database_migrator](#get_database_migrator)

## 类

### DatabaseMigrator

数据库迁移器 负责: 1. 追踪当前数据库版本 2. 执行迁移 3. 记录迁移历史 4. 支持回滚

#### 方法

```python
def close() -> None
```

关闭连接

```python
def get_applied_migrations() -> List
```

获取已应用的迁移列表

```python
def get_current_version() -> str
```

获取当前数据库版本

```python
def get_migration_status() -> Dict
```

获取迁移状态

```python
def migrate_to(target_version: str) -> Dict
```

迁移到指定版本

```python
def migrate_to_latest() -> Dict
```

迁移到最新版本

**返回:** 迁移结果摘要

### MigrationRecord

**@dataclass**

迁移记录

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `duration_seconds`
- `error_message`
- `rollback_applied`

### MigrationRegistry

迁移注册中心

#### 方法

```python
def get_all() -> List
```

获取所有迁移（按版本排序）

```python
def get_by_version(version: str) -> Optional
```

按版本获取迁移

```python
def get_latest_version() -> Optional
```

获取最新版本号

```python
def get_versions_between(from_version: str, to_version: str) -> List
```

获取两个版本之间的迁移

```python
def register(migration: MigrationStep) -> None
```

注册迁移

### MigrationStatus

**Enum**

**继承:** `Enum`

迁移状态

#### 属性

- `COMPLETED`
- `FAILED`
- `PENDING`
- `ROLLED_BACK`
- `RUNNING`
- `name`
- `value`

### MigrationStep

**@dataclass**

单个迁移步骤

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `down_fn`
- `up_fn`

## 函数

### ensure_latest_schema

```python
def ensure_latest_schema(data_dir: Optional = None) -> Dict
```

确保数据库 Schema 是最新的 这是一个便捷函数，可以在应用启动时调用。

### get_database_migrator

```python
def get_database_migrator(data_dir: Optional = None) -> DatabaseMigrator
```

获取全局数据库迁移器

---

*文档生成于: 2026-04-23T01:21:36.810841*
