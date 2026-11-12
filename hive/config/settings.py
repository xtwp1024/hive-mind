# -*- coding: utf-8 -*-
"""
蜂巢系统配置管理

使用Pydantic进行类型验证和环境变量加载
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from hive.paths import resolve_hive_home

logger = logging.getLogger("Hive.Settings")


class HiveSettings(BaseSettings):
    """
    蜂巢系统配置

    所有配置项都通过环境变量或.env文件加载
    """

    # ========== 路径配置 ==========
    # BASE_DIR 是「hive 根/Data/Inbound」的派生投影；根锚点只由
    # hive.paths.resolve_hive_home 解析（P1-7 单一真源），此处不再
    # 硬编码盘符 / 拼 cwd / 自己读 HIVE_HOME。
    def _default_base_dir() -> Path:
        return resolve_hive_home() / "Data" / "Inbound"

    BASE_DIR: Path = Field(default_factory=_default_base_dir)

    # 子目录默认相对于 BASE_DIR, 除非单独设置了环境变量
    @property
    def VAULT_DIR(self) -> Path:
        return Path(os.getenv("VAULT_DIR", self.BASE_DIR / "vault"))

    @property
    def REPORT_DIR(self) -> Path:
        return Path(os.getenv("REPORT_DIR", self.BASE_DIR / "reports"))

    @property
    def CHROMA_PERSIST_DIR(self) -> Path:
        return Path(os.getenv("CHROMA_PERSIST_DIR", self.BASE_DIR / "chroma_db"))

    @property
    def LOG_DIR(self) -> Path:
        return Path(os.getenv("LOG_DIR", self.BASE_DIR / "logs"))

    @property
    def SANDBOX_DIR(self) -> Path:
        return Path(os.getenv("SANDBOX_DIR", self.BASE_DIR / "sandbox"))

    @property
    def INBOUND_DIR(self) -> Path:
        return Path(os.getenv("INBOUND_DIR", self.BASE_DIR / "Inbound"))

    ENABLE_SANDBOX: bool = Field(default=True, description="是否启用影子沙箱")
    AUTO_MERGE_BACK: bool = Field(default=False, description="是否自动向原项目导回修改 (两步验证)")

    # ========== API密钥配置 ==========
    GITHUB_TOKEN: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # ========== 性能配置 ==========
    MAX_WORKERS: int = Field(default=10, ge=1, le=50, description="最大工蜂数量")
    BATCH_SIZE: int = Field(default=1000, ge=100, description="批处理大小")
    TIMEOUT_SECONDS: int = Field(default=300, ge=10, description="超时时间（秒）")
    MAX_CONCURRENT_WORKERS: int = Field(default=10, ge=1, description="最大并发工蜂")

    # ========== 相关性过滤配置 ==========
    GITHUB_RELEVANCE_MIN_SCORE: float = Field(
        default=0.0, ge=0.0, le=1.0, description="GitHub 仓库相关性最低分"
    )
    GITHUB_RELEVANCE_STRICT: bool = Field(default=False, description="是否严格过滤低相关仓库")
    RELEVANCE_KEYWORDS: str = Field(
        default="ai agent,llm,multi-agent,framework,orchestration",
        description="相关性关键词（逗号分隔）",
    )
    RELEVANCE_LANGUAGES: str = Field(
        default="python,rust,go,typescript,javascript,java",
        description="偏好语言（逗号分隔）",
    )
    GITHUB_MAX_KEYWORDS_PER_HUNT: int = Field(
        default=5, ge=1, le=20, description="每轮狩猎最大关键词数"
    )
    GITHUB_STARS_MIN: int = Field(default=0, ge=0, description="GitHub 搜索最低 stars 阈值")
    GITHUB_SEARCH_SLEEP_SECONDS: float = Field(
        default=6.0, ge=0.5, description="GitHub 搜索间隔秒数"
    )
    GITHUB_RATE_LIMIT_COOLDOWN_SECONDS: int = Field(
        default=120, ge=10, description="触发限流后的冷却秒数"
    )
    GITHUB_USE_LANGUAGE_FILTER: bool = Field(
        default=False, description="是否在 GitHub 搜索中强制 language 过滤"
    )
    GITHUB_QUERY_IN: str = Field(
        default="name,description,readme,topics",
        description="GitHub 搜索 in: 范围（逗号分隔，可为空）",
    )
    PROJECT_GOALS: str = Field(
        default=",".join(
            [
                "distributed systems",
                "web ui",
                "monitoring dashboard",
                "api docs",
                "performance",
                "distributed",
                "web ui",
                "observability",
                "document generation",
            ]
        ),
        description="项目目标关键词（逗号分隔）",
    )
    CAPABILITY_GAPS: str = Field(
        default=",".join(
            [
                "queenbee refactor",
                "modular architecture",
                "ci/cd",
                "scheduler",
                "workflow",
                "knowledge ingestion",
                "vector db",
            ]
        ),
        description="能力短板关键词（逗号分隔）",
    )

    # ========== 策略配置 ==========
    COOLDOWN_CYCLES: int = Field(default=12, ge=1, description="策略冷却周期数")
    MUTATION_RATE: float = Field(default=0.20, ge=0.0, le=1.0, description="变异率")
    SIMILARITY_THRESHOLD: float = Field(default=0.85, ge=0.0, le=1.0, description="相似度阈值")
    REPORT_BATCH_SIZE: int = Field(default=20, ge=1, description="报告生成批次大小")

    # ========== 资源限制 ==========
    MIN_FREE_MEMORY_MB: int = Field(default=500, ge=100, description="最小可用内存（MB）")
    MAX_CPU_PERCENT: int = Field(default=90, ge=50, le=100, description="最大CPU使用率（%）")

    # ========== 日志配置 ==========
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FORMAT: str = Field(default="json", description="日志格式（json/text）")
    LOG_FILE: Optional[str] = Field(default=None, description="日志文件路径")

    # ========== 共享大脑日志配置 ==========
    SHARED_BRAIN_ENABLED: bool = Field(default=True, description="是否启用共享大脑事件日志")
    SHARED_BRAIN_EVENTS_DIR: Path = Field(
        default_factory=lambda: Path(os.getenv("SHARED_BRAIN_EVENTS_DIR", "shared-brain/events")),
        description="共享大脑事件日志目录",
    )

    # ========== 环境配置 ==========
    ENVIRONMENT: str = Field(default="development", description="运行环境")
    DEBUG: bool = Field(default=False, description="调试模式")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    def get_path(self, name: str) -> Path:
        """
        获取配置路径

        Args:
            name: 路径名称（如"base", "vault", "report"）

        Returns:
            Path对象
        """
        attr_name = f"{name}_dir"
        if hasattr(self, attr_name.upper()):
            return getattr(self, attr_name.upper())
        return Path(".")

    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.ENVIRONMENT.lower() == "production"

    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.ENVIRONMENT.lower() == "development"

    def is_testing(self) -> bool:
        """是否为测试环境"""
        return self.ENVIRONMENT.lower() == "testing"

    def ensure_directories(self):
        """确保所有必需的目录存在"""
        dirs = [
            self.BASE_DIR,
            self.VAULT_DIR,
            self.REPORT_DIR,
            self.CHROMA_PERSIST_DIR,
            self.LOG_DIR,
            self.SANDBOX_DIR,
            self.INBOUND_DIR,
        ]

        if self.SHARED_BRAIN_ENABLED:
            dirs.append(self.SHARED_BRAIN_EVENTS_DIR)

        for dir_path in dirs:
            # 安全检查：防止路径遍历攻击
            try:
                resolved = dir_path.resolve()
                # 检查路径是否为绝对路径且不会穿越到系统根以上的目录
                # 允许创建在用户目录下的路径，如 /home/user/hive, C:\Users\user\hive 等
                # 但阻止直接创建在根目录如 /, C:\ 等
                if len(resolved.parts) >= 2:
                    # 额外检查：确保所有路径部分都在预期的base目录下
                    # 防止 /../../etc/passwd 类型的攻击
                    if resolved.parts[0] == ".." or any(p == ".." for p in resolved.parts):
                        logger.warning(f"路径遍历检测失败: {dir_path}")
                        continue
                    dir_path.mkdir(parents=True, exist_ok=True)
                else:
                    logger.warning(f"目录路径过短，可能不安全: {dir_path}")
            except (OSError, RuntimeError) as e:
                logger.warning(f"目录创建跳过 {dir_path}: {e}")


_settings = None
_settings_lock = threading.Lock()


def get_settings(env_file: Optional[str] = None) -> HiveSettings:
    """
    获取配置单例（线程安全）

    Args:
        env_file: 可选的.env文件路径

    Returns:
        HiveSettings实例
    """
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = HiveSettings(_env_file=env_file)
                if not _settings.is_testing():
                    _settings.ensure_directories()
    return _settings


# 为了向后兼容，创建一个延迟加载的属性
class _SettingsProxy:
    """延迟加载的配置代理"""

    def __getattr__(self, name):
        return getattr(get_settings(), name)


settings = _SettingsProxy()
