"""运行上下文构建模块。

这个文件负责根据配置准备一次运行所需的上下文信息，
包括输出目录和统一日志器。

输入：
- 顶层应用配置

输出：
- `RunContext`
"""

from __future__ import annotations

from src.config.schema import AppConfig
from src.io.output_writer import prepare_output_paths
from src.types import RunContext
from src.utils.logging_utils import setup_logger


def build_run_context(config: AppConfig) -> RunContext:
    """根据顶层配置构建运行上下文。"""

    output_paths = prepare_output_paths(config.output)
    log_file_path = output_paths.logs_dir / config.logging.filename
    logger = setup_logger(config.logging, log_file_path)
    return RunContext(output_paths=output_paths, logger=logger)
