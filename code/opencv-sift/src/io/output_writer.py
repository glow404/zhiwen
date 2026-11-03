"""输出结果写入模块。

这个文件负责创建输出目录，并把图像结果、JSON 汇总等文件落盘。

输入：
- 输出配置
- 待保存图像
- 待保存 JSON 数据

输出：
- 已创建的输出目录对象
- 已保存文件的绝对路径
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.config.schema import OutputConfig
from src.exceptions import OutputWriteError
from src.types import OutputPaths


def prepare_output_paths(output_config: OutputConfig) -> OutputPaths:
    """根据输出配置创建本次运行所需的目录结构。"""

    root_dir = output_config.output_root.expanduser().resolve()
    templates_dir = root_dir / output_config.template_dirname
    visualizations_dir = root_dir / output_config.visualization_dirname
    intermediate_dir = root_dir / output_config.intermediate_dirname
    reports_dir = root_dir / output_config.report_dirname
    logs_dir = root_dir / output_config.log_dirname

    for path in (root_dir, templates_dir, visualizations_dir, intermediate_dir, reports_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    return OutputPaths(
        root_dir=root_dir,
        templates_dir=templates_dir,
        visualizations_dir=visualizations_dir,
        intermediate_dir=intermediate_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
    )


def save_image(image: np.ndarray, output_path: Path) -> Path:
    """把一张图像写入磁盘并返回实际保存路径。"""

    resolved_path = output_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(resolved_path), image)
    if not ok:
        raise OutputWriteError(f"Failed to write image: {resolved_path}")
    return resolved_path


def save_json(data: dict, output_path: Path) -> Path:
    """把一个 JSON 对象写入磁盘并返回实际保存路径。"""

    resolved_path = output_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OutputWriteError(f"Failed to write JSON: {resolved_path}") from exc
    return resolved_path
