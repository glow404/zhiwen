"""单图 MVP 命令行入口。

这个文件负责把命令行参数转换为程序配置，并调用单张图指纹特征提取流程。

输入：
- `--config`：YAML 或 JSON 配置文件路径
- `--image`：可选，单张指纹图路径，会覆盖配置中的 `input.single_image_path`

输出：
- 终端打印本次处理的图片路径、关键点数量、模板路径、可视化路径
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.loader import load_config
from src.constants import DEFAULT_CONFIG_PATH
from src.pipeline.single_image import process_single_image


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    输出：
    - 返回 `argparse.ArgumentParser` 对象，供主函数解析参数使用
    """

    parser = argparse.ArgumentParser(description="Run single-image fingerprint SIFT extraction.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML or JSON config file.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image path that overrides input.single_image_path in config.",
    )
    return parser


def main() -> None:
    """按命令行参数执行单图指纹 SIFT 提取流程。

    输入：
    - 来自命令行的配置文件路径与图片路径

    输出：
    - 无直接返回值
    - 将处理结果打印到标准输出
    """

    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    image_path = args.image or config.input.single_image_path
    if image_path is None:
        raise ValueError("A single input image path must be provided by config or --image.")

    result = process_single_image(image_path, config)
    print(f"Image: {result.image_path}")
    print(f"Keypoints: {result.keypoint_count}")
    print(f"Template: {result.template_path}")
    print(f"Visualization: {result.visualization_path}")


if __name__ == "__main__":
    main()
