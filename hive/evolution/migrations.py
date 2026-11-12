# -*- coding: utf-8 -*-
"""
Hive 数据库版本化迁移系统

基于 GEPStore 的 Schema Versioning 实现。
确保数据库平滑升级，数据不丢失。

设计原则:
1. 每个迁移是一个独立的版本
2. 支持升级和降级
3. 记录迁移历史
4. 幂等操作（重复执行安全）
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("Hive.DB.Migration")


class MigrationStatus(Enum):
    """迁移状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationRecord:
    """迁移记录"""

    version: str
    name: str
    description: str
    status: MigrationStatus
    applied_at: float = field(default_factory=time.time)
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    rollback_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "applied_at": self.applied_at,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "rollback_applied": self.rollback_applied,
        }


@dataclass
class MigrationStep:
    """单个迁移步骤"""

    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    up_fn: Optional[Callable] = None
    down_fn: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "up_sql": self.up_sql[:100] + "..." if len(self.up_sql) > 100 else self.up_sql,
        }


class MigrationRegistry:
    """迁移注册中心"""

    _migrations: List[MigrationStep] = []
    _by_version: Dict[str, MigrationStep] = {}

    @classmethod
    def register(cls, migration: MigrationStep) -> None:
        """注册迁移"""
        if migration.version in cls._by_version:
            logger.warning(f"Migration version {migration.version} already registered, skipping")
            return
        cls._migrations.append(migration)
        cls._by_version[migration.version] = migration
        # 按版本排序
        cls._migrations.sort(key=lambda m: m.version)
        logger.info(f"Registered migration: {migration.version} - {migration.name}")

    @classmethod
    def get_all(cls) -> List[MigrationStep]:
        """获取所有迁移（按版本排序）"""
        return cls._migrations.copy()

    @classmethod
    def get_by_version(cls, version: str) -> Optional[MigrationStep]:
        """按版本获取迁移"""
        return cls._by_version.get(version)

    @classmethod
    def get_latest_version(cls) -> Optional[str]:
        """获取最新版本号"""
        if cls._migrations:
            return cls._migrations[-1].version
        return None

    @classmethod
    def get_versions_between(cls, from_version: str, to_version: str) -> List[MigrationStep]:
        """获取两个版本之间的迁移"""
        result = []
        for m in cls._migrations:
            if from_version < m.version <= to_version:
                result.append(m)
        return result


# ============================================================
# 迁移定义
# ============================================================

# V1: 初始 Schema
MigrationRegistry.register(
    MigrationStep(
        version="001",
        name="initial_schema",
        description="创建初始表结构: genes, capsules, events",
        up_sql="""
        CREATE TABLE IF NOT EXISTS genes (
            id          TEXT PRIMARY KEY,
            data        TEXT NOT NULL,
            updated_at  REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS capsules (
            id          TEXT PRIMARY KEY,
            data        TEXT NOT NULL,
            updated_at  REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id          TEXT PRIMARY KEY,
            timestamp   REAL NOT NULL,
            type        TEXT NOT NULL,
            data        TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
    """,
        down_sql="""
        DROP INDEX IF EXISTS idx_events_ts;
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS capsules;
        DROP TABLE IF EXISTS genes;
    """,
    )
)

# V2: 添加版本追踪表
MigrationRegistry.register(
    MigrationStep(
        version="002",
        name="schema_migrations_table",
        description="创建 schema_migrations 表用于追踪迁移历史",
        up_sql="""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version         TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            description     TEXT,
            status          TEXT NOT NULL DEFAULT 'completed',
            applied_at      REAL NOT NULL,
            duration        REAL,
            error_message   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_migrations_applied ON schema_migrations(applied_at);
    """,
        down_sql="""
        DROP INDEX IF EXISTS idx_migrations_applied;
        DROP TABLE IF EXISTS schema_migrations;
    """,
    )
)

# V3: 添加集群状态表
MigrationRegistry.register(
    MigrationStep(
        version="003",
        name="cluster_tables",
        description="创建集群状态和任务历史表",
        up_sql="""
        CREATE TABLE IF NOT EXISTS clusters (
            id              TEXT PRIMARY KEY,
            name            TEXT,
            status          TEXT NOT NULL DEFAULT 'active',
            leader_id       TEXT,
            member_count    INTEGER DEFAULT 0,
            capabilities    TEXT,
            created_at      REAL NOT NULL,
            updated_at      REAL NOT NULL,
            disbanded_at    REAL
        );
        
        CREATE TABLE IF NOT EXISTS task_history (
            id              TEXT PRIMARY KEY,
            task_type       TEXT NOT NULL,
            status          TEXT NOT NULL,
            assigned_bee    TEXT,
            cluster_id      TEXT,
            priority        INTEGER DEFAULT 0,
            created_at      REAL NOT NULL,
            started_at      REAL,
            completed_at    REAL,
            duration        REAL,
            result_summary  TEXT,
            error_message   TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_task_history_type ON task_history(task_type);
        CREATE INDEX IF NOT EXISTS idx_task_history_status ON task_history(status);
        CREATE INDEX IF NOT EXISTS idx_task_history_created ON task_history(created_at);
    """,
        down_sql="""
        DROP INDEX IF EXISTS idx_task_history_created;
        DROP INDEX IF EXISTS idx_task_history_status;
        DROP INDEX IF EXISTS idx_task_history_type;
        DROP TABLE IF EXISTS task_history;
        DROP TABLE IF EXISTS clusters;
    """,
    )
)

# V4: 添加蜂群状态追踪
MigrationRegistry.register(
    MigrationStep(
        version="004",
        name="bee_status_tracking",
        description="创建蜜蜂状态和系统指标表",
        up_sql="""
        CREATE TABLE IF NOT EXISTS bee_status (
            bee_id          TEXT PRIMARY KEY,
            bee_type        TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'idle',
            current_task    TEXT,
            capabilities    TEXT,
            load_factor     REAL DEFAULT 0.0,
            last_heartbeat  REAL NOT NULL,
            created_at      REAL NOT NULL,
            retired_at      REAL
        );
        
        CREATE TABLE IF NOT EXISTS system_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       REAL NOT NULL,
            cpu_percent     REAL,
            memory_percent  REAL,
            active_bees     INTEGER,
            active_clusters INTEGER,
            pending_tasks   INTEGER,
            running_tasks   INTEGER,
            completed_tasks INTEGER,
            failed_tasks    INTEGER
        );
        
        CREATE INDEX IF NOT EXISTS idx_bee_status_type ON bee_status(bee_type);
        CREATE INDEX IF NOT EXISTS idx_bee_status_status ON bee_status(status);
        CREATE INDEX IF NOT EXISTS idx_bee_status_heartbeat ON bee_status(last_heartbeat);
        CREATE INDEX IF NOT EXISTS idx_system_metrics_ts ON system_metrics(timestamp);
    """,
        down_sql="""
        DROP INDEX IF EXISTS idx_system_metrics_ts;
        DROP INDEX IF EXISTS idx_bee_status_heartbeat;
        DROP INDEX IF EXISTS idx_bee_status_status;
        DROP INDEX IF EXISTS idx_bee_status_type;
        DROP TABLE IF EXISTS system_metrics;
        DROP TABLE IF EXISTS bee_status;
    """,
    )
)

# V5: 添加知识图谱表
MigrationRegistry.register(
    MigrationStep(
        version="005",
        name="knowledge_graph",
        description="创建知识图谱相关表",
        up_sql="""
        CREATE TABLE IF NOT EXISTS knowledge_items (
            id              TEXT PRIMARY KEY,
            content         TEXT NOT NULL,
            source_type     TEXT,
            source_id       TEXT,
            importance      REAL DEFAULT 0.5,
            access_count    INTEGER DEFAULT 0,
            created_at      REAL NOT NULL,
            updated_at      REAL NOT NULL,
            last_accessed   REAL
        );
        
        CREATE TABLE IF NOT EXISTS knowledge_relations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            from_item_id    TEXT NOT NULL,
            to_item_id      TEXT NOT NULL,
            relation_type   TEXT NOT NULL,
            strength        REAL DEFAULT 0.5,
            created_at      REAL NOT NULL,
            UNIQUE(from_item_id, to_item_id, relation_type)
        );
        
        CREATE TABLE IF NOT EXISTS knowledge_tags (
            item_id         TEXT NOT NULL,
            tag             TEXT NOT NULL,
            weight          REAL DEFAULT 1.0,
            created_at      REAL NOT NULL,
            PRIMARY KEY(item_id, tag)
        );
        
        CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_items(source_type, source_id);
        CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge_items(created_at);
        CREATE INDEX IF NOT EXISTS idx_relations_from ON knowledge_relations(from_item_id);
        CREATE INDEX IF NOT EXISTS idx_relations_to ON knowledge_relations(to_item_id);
        CREATE INDEX IF NOT EXISTS idx_tags_tag ON knowledge_tags(tag);
    """,
        down_sql="""
        DROP INDEX IF EXISTS idx_tags_tag;
        DROP INDEX IF EXISTS idx_relations_to;
        DROP INDEX IF EXISTS idx_relations_from;
        DROP INDEX IF EXISTS idx_knowledge_created;
        DROP INDEX IF EXISTS idx_knowledge_source;
        DROP TABLE IF EXISTS knowledge_tags;
        DROP TABLE IF EXISTS knowledge_relations;
        DROP TABLE IF EXISTS knowledge_items;
    """,
    )
)

# V6: 添加 SSE 事件存储
MigrationRegistry.register(
    MigrationStep(
        version="006",
        name="sse_event_store",
        description="创建 SSE 事件持久化表",
        up_sql="""
        CREATE TABLE IF NOT EXISTS sse_events (
            id              TEXT PRIMARY KEY,
            event_type      TEXT NOT NULL,
            data            TEXT NOT NULL,
            timestamp       REAL NOT NULL,
            acknowledged    BOOLEAN DEFAULT FALSE,
            acknowledged_at REAL
        );
        
        CREATE INDEX IF NOT EXISTS idx_sse_events_type ON sse_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_sse_events_ts ON sse_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_sse_events_ack ON sse_events(acknowledged);
    """,
        down_sql="""
        DROP INDEX IF EXISTS idx_sse_events_ack;
        DROP INDEX IF EXISTS idx_sse_events_ts;
        DROP INDEX IF EXISTS idx_sse_events_type;
        DROP TABLE IF EXISTS sse_events;
    """,
    )
)


class DatabaseMigrator:
    """
    数据库迁移器

    负责:
    1. 追踪当前数据库版本
    2. 执行迁移
    3. 记录迁移历史
    4. 支持回滚
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        logger.info(f"Database migrator initialized: {db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_migrations_table(self) -> None:
        """确保迁移表存在"""
        conn = self._get_conn()

        # 检查 schema_migrations 表是否存在
        cursor = conn.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='schema_migrations'
        """
        )
        exists = cursor.fetchone() is not None

        if not exists:
            # 执行 V1 和 V2 迁移（初始 Schema）
            self._execute_initial_migrations()

    def _execute_initial_migrations(self) -> None:
        """执行初始迁移"""
        conn = self._get_conn()

        # 执行 V1: 初始 Schema
        v1 = MigrationRegistry.get_by_version("001")
        if v1:
            conn.executescript(v1.up_sql)
            conn.commit()
            logger.info("Applied migration 001: initial_schema")

        # 执行 V2: 迁移表
        v2 = MigrationRegistry.get_by_version("002")
        if v2:
            conn.executescript(v2.up_sql)
            conn.commit()
            logger.info("Applied migration 002: schema_migrations_table")

        # 记录 V1 和 V2
        now = time.time()
        for version in ["001", "002"]:
            m = MigrationRegistry.get_by_version(version)
            if m:
                conn.execute(
                    """
                    INSERT INTO schema_migrations 
                    (version, name, description, status, applied_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (m.version, m.name, m.description, "completed", now),
                )
        conn.commit()

    def get_current_version(self) -> str:
        """获取当前数据库版本"""
        self._ensure_migrations_table()
        conn = self._get_conn()

        cursor = conn.execute(
            """
            SELECT version FROM schema_migrations 
            WHERE status = 'completed'
            ORDER BY version DESC LIMIT 1
        """
        )
        row = cursor.fetchone()

        if row:
            return row[0]
        return "000"  # 初始状态

    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """获取已应用的迁移列表"""
        self._ensure_migrations_table()
        conn = self._get_conn()

        cursor = conn.execute(
            """
            SELECT version, name, description, status, applied_at, duration, error_message
            FROM schema_migrations
            ORDER BY version
        """
        )

        return [
            {
                "version": row[0],
                "name": row[1],
                "description": row[2],
                "status": row[3],
                "applied_at": row[4],
                "duration": row[5],
                "error_message": row[6],
            }
            for row in cursor.fetchall()
        ]

    def migrate_to_latest(self) -> Dict[str, Any]:
        """
        迁移到最新版本

        Returns:
            迁移结果摘要
        """
        current_version = self.get_current_version()
        latest_version = MigrationRegistry.get_latest_version() or "000"

        if current_version >= latest_version:
            return {
                "status": "already_latest",
                "current_version": current_version,
                "latest_version": latest_version,
                "migrations_applied": 0,
            }

        # 获取需要应用的迁移
        migrations = MigrationRegistry.get_versions_between(current_version, latest_version)

        if not migrations:
            return {
                "status": "no_migrations",
                "current_version": current_version,
                "latest_version": latest_version,
                "migrations_applied": 0,
            }

        conn = self._get_conn()
        applied = []
        failed = None

        for migration in migrations:
            start_time = time.time()
            logger.info(f"Applying migration {migration.version}: {migration.name}")

            try:
                # 执行迁移
                conn.execute("BEGIN IMMEDIATE")

                # 执行 SQL
                conn.executescript(migration.up_sql)

                # 执行自定义函数（如果有）
                if migration.up_fn:
                    migration.up_fn(conn)

                # 记录迁移
                duration = time.time() - start_time
                conn.execute(
                    """
                    INSERT INTO schema_migrations 
                    (version, name, description, status, applied_at, duration)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        migration.version,
                        migration.name,
                        migration.description,
                        "completed",
                        start_time,
                        duration,
                    ),
                )

                conn.execute("COMMIT")
                applied.append(migration.version)
                logger.info(f"Migration {migration.version} applied in {duration:.2f}s")

            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Migration {migration.version} failed: {e}")
                failed = {
                    "version": migration.version,
                    "error": str(e),
                }
                break

        return {
            "status": "completed" if failed is None else "partial",
            "current_version": current_version,
            "target_version": latest_version,
            "migrations_applied": len(applied),
            "applied_versions": applied,
            "failed_migration": failed,
        }

    def migrate_to(self, target_version: str) -> Dict[str, Any]:
        """迁移到指定版本"""
        current_version = self.get_current_version()

        if current_version == target_version:
            return {
                "status": "already_at_version",
                "version": current_version,
            }

        if current_version < target_version:
            # 升级
            migrations = MigrationRegistry.get_versions_between(current_version, target_version)
        else:
            # 降级 - 暂不支持
            return {
                "status": "rollback_not_supported",
                "message": "Automatic rollback is not supported yet",
            }

        conn = self._get_conn()
        applied = []

        for migration in migrations:
            start_time = time.time()
            logger.info(f"Applying migration {migration.version}: {migration.name}")

            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.executescript(migration.up_sql)
                if migration.up_fn:
                    migration.up_fn(conn)

                duration = time.time() - start_time
                conn.execute(
                    """
                    INSERT INTO schema_migrations 
                    (version, name, description, status, applied_at, duration)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        migration.version,
                        migration.name,
                        migration.description,
                        "completed",
                        start_time,
                        duration,
                    ),
                )
                conn.execute("COMMIT")
                applied.append(migration.version)

            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Migration failed: {e}")
                return {
                    "status": "failed",
                    "error": str(e),
                    "applied_versions": applied,
                }

        return {
            "status": "completed",
            "migrations_applied": len(applied),
            "applied_versions": applied,
        }

    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        current = self.get_current_version()
        latest = MigrationRegistry.get_latest_version()
        applied = self.get_applied_migrations()
        all_migrations = MigrationRegistry.get_all()

        pending = []
        for m in all_migrations:
            is_applied = any(a["version"] == m.version for a in applied)
            if not is_applied:
                pending.append(m.to_dict())

        return {
            "current_version": current,
            "latest_version": latest,
            "is_latest": current >= (latest or "000"),
            "total_migrations": len(all_migrations),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "pending_migrations": pending,
            "applied_migrations": applied,
        }

    def close(self) -> None:
        """关闭连接"""
        if self._conn:
            self._conn.close()
            self._conn = None


# 全局单例
_global_migrator: Optional[DatabaseMigrator] = None


def get_database_migrator(data_dir: Optional[Path] = None) -> DatabaseMigrator:
    """获取全局数据库迁移器"""
    global _global_migrator

    if _global_migrator is None:
        if data_dir is None:
            # 默认位置
            data_dir = Path(__file__).parent.parent / "evolution" / "gep"
        _global_migrator = DatabaseMigrator(data_dir / "gep.db")

    return _global_migrator


def ensure_latest_schema(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    确保数据库 Schema 是最新的

    这是一个便捷函数，可以在应用启动时调用。
    """
    migrator = get_database_migrator(data_dir)
    return migrator.migrate_to_latest()


if __name__ == "__main__":
    # 命令行模式：运行迁移
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Hive Database Migration Tool")
    print("=" * 60)
    print()

    migrator = get_database_migrator()

    # 获取状态
    status = migrator.get_migration_status()
    print(f"当前版本: {status['current_version']}")
    print(f"最新版本: {status['latest_version']}")
    print(f"是否最新: {'是' if status['is_latest'] else '否'}")
    print(f"已应用迁移数: {status['applied_count']}")
    print(f"待应用迁移数: {status['pending_count']}")
    print()

    if not status["is_latest"]:
        print("待应用的迁移:")
        for m in status["pending_migrations"]:
            print(f"  - {m['version']}: {m['name']}")
        print()

        # 执行迁移
        result = migrator.migrate_to_latest()
        print(f"迁移结果: {result['status']}")
        print(f"应用迁移数: {result['migrations_applied']}")
        if result.get("failed_migration"):
            print(f"失败迁移: {result['failed_migration']}")
    else:
        print("数据库已是最新版本，无需迁移。")

    print()
    print("=" * 60)
