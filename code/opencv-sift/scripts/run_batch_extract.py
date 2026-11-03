"""批处理命令行入口。

这个脚本用于对一个或多个输入目录进行递归扫描，并批量执行指纹预处理、
SIFT 特征提取、模板保存和可视化输出。

输入：
- `--config`：YAML 或 JSON 配置文件路径
- `--input`：一个或多个输入目录，会覆盖配置文件中的 `input.input_roots`
- `--output`：可选，覆盖配置中的 `output.output_root`

输出：
- 终端打印批处理汇总信息
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.loader import load_config
from src.constants import DEFAULT_CONFIG_PATH
from src.pipeline.batch_runner import process_batch


def build_parser() -> argparse.ArgumentParser:
    """构建批处理脚本的命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="Run batch fingerprint SIFT extraction.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML 或 JSON 配置文件路径。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=None,
        help="一个或多个输入目录，会覆盖配置中的 input.input_roots。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选，覆盖配置中的 output.output_root。",
    )
    return parser


def main() -> None:
    """按命令行参数执行批处理流程。"""

    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.input:
        config.input.input_roots = list(args.input)
    if args.output is not None:
        config.output.output_root = args.output

    if not config.input.input_roots:
        raise ValueError("批处理模式下必须提供至少一个输入目录。")

    batch_result = process_batch(config)
    print(f"Total: {batch_result.total_images}")
    print(f"Succeeded: {batch_result.succeeded}")
    print(f"Failed: {batch_result.failed}")
    print(f"Summary: {batch_result.summary_path}")


if __name__ == "__main__":
    main()
