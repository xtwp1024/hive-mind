# -*- coding: utf-8 -*-
"""重试装饰器和工具

为不稳定的外部操作提供指数退避重试机制。
"""

import functools
import random
import time
from typing import Any, Callable, Optional, Type, Union

from hive.utils.logger import get_logger

logger = get_logger(__name__)


class RetryConfig:
    """重试配置类"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: tuple[Type[Exception], ...] = (Exception,),
        stop_on: tuple[Type[Exception], ...] = (),
    ):
        """初始化重试配置

        Args:
            max_attempts: 最大重试次数
            base_delay: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            exponential_base: 指数退避的基数
            jitter: 是否添加随机抖动以避免惊群效应
            retry_on: 需要重试的异常类型元组
            stop_on: 不需要重试的异常类型元组（立即失败）
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on
        self.stop_on = stop_on


def retry(config: Optional[RetryConfig] = None, **config_kwargs) -> Callable:
    """重试装饰器

    为函数添加指数退避重试机制。

    Args:
        config: 重试配置对象，如果为None则使用config_kwargs创建
        **config_kwargs: 传递给RetryConfig的参数

    Example:
        # 使用默认配置
        @retry()
        def unstable_function():
            pass

        # 自定义配置
        @retry(max_attempts=5, base_delay=2.0)
        def network_call():
            pass

        # 使用RetryConfig对象
        config = RetryConfig(max_attempts=3, retry_on=(TimeoutError, ConnectionError))
        @retry(config=config)
        def specific_retry():
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 合并配置
            nonlocal config
            if config is None:
                config = RetryConfig(**config_kwargs)

            last_exception: Optional[Exception] = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.stop_on as e:
                    # 立即失败，不重试
                    logger.error(
                        "Stop exception encountered",
                        function=func.__name__,
                        exception=type(e).__name__,
                        message=str(e),
                    )
                    raise
                except config.retry_on as e:
                    last_exception = e

                    if attempt == config.max_attempts:
                        # 最后一次尝试失败
                        logger.error(
                            "Max retry attempts reached",
                            function=func.__name__,
                            attempts=attempt,
                            exception=type(e).__name__,
                            message=str(e),
                        )
                        break

                    # 计算延迟时间
                    delay = min(
                        config.base_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay,
                    )

                    # 添加抖动
                    if config.jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        "Retrying after failure",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=config.max_attempts,
                        delay_seconds=round(delay, 2),
                        exception=type(e).__name__,
                        message=str(e),
                    )

                    time.sleep(delay)

            # 所有重试都失败
            if last_exception is not None:
                raise last_exception

        return wrapper

    return decorator


def async_retry(config: Optional[RetryConfig] = None, **config_kwargs) -> Callable:
    """异步重试装饰器

    为异步函数添加指数退避重试机制。

    Args:
        config: 重试配置对象
        **config_kwargs: 传递给RetryConfig的参数
    """
    import asyncio

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 合并配置
            nonlocal config
            if config is None:
                config = RetryConfig(**config_kwargs)

            last_exception: Optional[Exception] = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.stop_on as e:
                    logger.error(
                        "Stop exception encountered",
                        function=func.__name__,
                        exception=type(e).__name__,
                        message=str(e),
                    )
                    raise
                except config.retry_on as e:
                    last_exception = e

                    if attempt == config.max_attempts:
                        logger.error(
                            "Max retry attempts reached",
                            function=func.__name__,
                            attempts=attempt,
                            exception=type(e).__name__,
                            message=str(e),
                        )
                        break

                    delay = min(
                        config.base_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay,
                    )

                    if config.jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        "Retrying after failure",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=config.max_attempts,
                        delay_seconds=round(delay, 2),
                        exception=type(e).__name__,
                        message=str(e),
                    )

                    await asyncio.sleep(delay)

            if last_exception is not None:
                raise last_exception

        return wrapper

    return decorator
