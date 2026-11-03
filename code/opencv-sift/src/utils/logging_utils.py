"""日志初始化模块。

这个文件负责为单图处理和批处理任务创建统一日志器，
支持控制台输出和文件输出两种形式。

输入：
- 日志配置
- 日志文件路径

输出：
- 已配置完成的 `logging.Logger`
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.schema import LoggingConfig


def setup_logger(logging_config: LoggingConfig, log_file_path: Path) -> logging.Logger:
    """根据配置初始化并返回项目日志器。"""

    logger = logging.getLogger(logging_config.logger_name)
    logger.setLevel(_resolve_level(logging_config.level))
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if logging_config.console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logger.level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if logging_config.file_enabled:
        resolved_log_path = log_file_path.expanduser().resolve()
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _resolve_level(level_name: str) -> int:
    """把字符串日志等级转换为 `logging` 常量。"""

    normalized = level_name.upper()
    return getattr(logging, normalized, logging.INFO)
