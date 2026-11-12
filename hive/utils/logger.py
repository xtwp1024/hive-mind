# -*- coding: utf-8 -*-
"""结构化日志系统配置

使用structlog提供结构化日志记录，支持JSON格式和彩色控制台输出。
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import structlog
from structlog.types import Processor

# 防止重复包装 stdout/stderr 的守护标志
_stdout_wrapped = False


def _add_log_level(logger: Any, method_name: str, event_dict: dict) -> dict:
    """添加日志级别"""
    event_dict["level"] = method_name.upper()
    return event_dict


def _add_logger_name(logger: Any, method_name: str, event_dict: dict) -> dict:
    """添加logger名称"""
    event_dict["logger"] = logger.name
    return event_dict


def _exc_info_filter(logger: Any, method_name: str, event_dict: dict) -> dict:
    """过滤异常信息，避免敏感数据泄露"""
    if "exception" in event_dict:
        # 可以在这里添加敏感数据过滤逻辑
        exc_str = str(event_dict["exception"])
        # 移除可能的文件路径信息
        if "\\" in exc_str or "/" in exc_str:
            # 保留异常类型和消息，移除完整路径
            lines = exc_str.split("\n")
            if lines:
                # 更彻底地清理路径信息
                import re

                # 移除常见的路径模式 C:\, D:\, /home/, /Users/ 等
                cleaned_lines = []
                for line in lines:
                    # 移除 Windows 路径模式 (如 C:\Users\...)
                    line = re.sub(
                        r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+(?:\\[^\\/:*?"<>|\r\n]+)*)?',
                        "[PATH]",
                        line,
                    )
                    # 移除 Unix 路径模式 (如 /home/user/... 或 /Users/...)
                    line = re.sub(r"/(?:[^/:\r\n]+/)*[^/:\r\n]*", "[PATH]", line)
                    cleaned_lines.append(line)
                event_dict["exception"] = "\n".join(cleaned_lines)
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_output: bool = False,
    enable_colors: bool = True,
    shared_brain_enabled: bool = False,
    shared_brain_dir: Optional[Path] = None,
    shared_brain_source: str = "hive",
) -> None:
    """配置structlog日志系统

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径（可选）
        json_output: 是否输出JSON格式日志
        enable_colors: 是否启用彩色输出（仅控制台模式）
        shared_brain_enabled: 是否启用共享大脑事件日志
        shared_brain_dir: 共享大脑事件目录（启用时必填）
        shared_brain_source: 事件来源标识
    """
    # Windows控制台UTF-8编码修复（仅执行一次，防止重复嵌套包装）
    global _stdout_wrapped
    if sys.platform == "win32" and not _stdout_wrapped:
        import io

        if not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if not isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        _stdout_wrapped = True

    # 配置标准logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        stream=sys.stdout,
    )

    # 配置处理器
    processors: list[Processor] = [
        # 添加时间戳
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        # 异常处理
        structlog.processors.UnicodeDecoder(),
    ]

    if shared_brain_enabled and shared_brain_dir:
        from hive.utils.shared_brain import SharedBrainEventWriter, SharedBrainLogProcessor

        writer = SharedBrainEventWriter(
            events_dir=shared_brain_dir,
            enabled=True,
            source=shared_brain_source,
        )
        processors.append(SharedBrainLogProcessor(writer))

    # 控制台输出处理器
    if json_output:
        # JSON格式输出
        console_processor = structlog.processors.JSONRenderer(ensure_ascii=False)
    elif enable_colors:
        # 彩色控制台输出
        console_processor = structlog.dev.ConsoleRenderer(
            colors=enable_colors,
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:
        # 纯文本输出
        console_processor = structlog.processors.JSONRenderer(ensure_ascii=False)

    # 文件输出处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_processor = structlog.processors.JSONRenderer(ensure_ascii=False)

        processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)
    else:
        file_processor = None

    # 配置structlog
    shared_processors = processors.copy()

    # 标准输出配置
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 配置格式化器
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=console_processor,
    )

    # 获取root logger并添加处理器
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # 如果配置了文件输出
    if log_file and file_processor:
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processor=file_processor,
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """获取指定名称的logger

    Args:
        name: Logger名称，通常使用__name__

    Returns:
        配置好的structlog logger实例
    """
    return structlog.get_logger(name)


# 默认日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
