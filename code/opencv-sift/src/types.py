"""项目核心数据结构定义。

这个文件定义各阶段之间传递的强类型数据对象，避免使用无结构字典导致字段混乱。

输入：
- 来自图像读取、预处理、ROI、特征提取等模块的中间结果

输出：
- 供流水线和 I/O 模块复用的数据类对象
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class PreprocessArtifacts:
    """保存预处理阶段产生的中间图像。

    输出字段：
    - `input_image`：原始输入图
    - `resized_image`：尺寸调整后的图
    - `normalized_image`：灰度归一化后的图
    - `enhanced_image`：对比度增强后的图
    - `denoised_image`：去噪后的图
    - `variance_map`：局部方差图，用于 ROI 提取
    - `roi_mask_raw`：原始 ROI 掩膜
    - `roi_mask_refined`：形态学清理后的 ROI 掩膜
    - `background_suppressed_image`：背景抑制后的最终图像
    """

    input_image: np.ndarray
    resized_image: np.ndarray
    normalized_image: np.ndarray
    enhanced_image: np.ndarray
    denoised_image: np.ndarray
    variance_map: np.ndarray | None = None
    roi_mask_raw: np.ndarray | None = None
    roi_mask_refined: np.ndarray | None = None
    background_suppressed_image: np.ndarray | None = None


@dataclass(slots=True)
class ROIResult:
    """表示 ROI 提取阶段的输出。

    输出字段：
    - `roi_mask`：与原图同尺寸的前景掩膜
    - `roi_bbox`：ROI 的矩形框，格式为 `(x0, y0, x1, y1)`
    - `roi_image`：裁剪后的 ROI 图像
    - `foreground_ratio`：前景区域占整图的比例
    """

    roi_mask: np.ndarray
    roi_bbox: tuple[int, int, int, int]
    roi_image: np.ndarray
    foreground_ratio: float


@dataclass(slots=True)
class PreprocessResult:
    """表示预处理阶段的完整输出。

    输出字段：
    - `output_image`：预处理后的最终图像
    - `artifacts`：预处理中间结果集合
    - `roi_result`：ROI 提取结果
    - `debug_output_dir`：调试图片保存目录
    - `debug_image_paths`：调试中间图路径映射
    """

    output_image: np.ndarray
    artifacts: PreprocessArtifacts
    roi_result: ROIResult
    debug_output_dir: Path | None = None
    debug_image_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureResult:
    """表示单张图的局部特征提取结果。

    输出字段：
    - `keypoints`：OpenCV 关键点列表
    - `descriptors`：描述子矩阵，可能为空
    - `keypoint_count`：关键点数量
    - `descriptor_dim`：描述子维度
    - `descriptor_dtype`：描述子数据类型
    - `keypoints_data`：结构化关键点数据，便于序列化保存
    """

    keypoints: list[Any]
    descriptors: np.ndarray | None
    keypoint_count: int
    descriptor_dim: int
    descriptor_dtype: str = ""
    keypoints_data: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class FingerprintTemplate:
    """表示可落盘保存的指纹模板对象。

    输出字段：
    - 包含样本标识、图像信息、预处理元数据、关键点元数据、描述子和配置快照
    """

    template_version: str
    image_id: str
    source_path: str
    image_shape: tuple[int, int]
    roi_bbox: tuple[int, int, int, int]
    preprocess_metadata: dict[str, Any]
    timestamp: str
    keypoints_x: np.ndarray
    keypoints_y: np.ndarray
    keypoints_size: np.ndarray
    keypoints_angle: np.ndarray
    keypoints_response: np.ndarray
    keypoints_octave: np.ndarray
    keypoints_class_id: np.ndarray
    descriptors: np.ndarray | None
    descriptor_dtype: str
    config_snapshot: dict[str, Any]


@dataclass(slots=True)
class OutputPaths:
    """保存一次运行过程中使用到的输出目录。"""

    root_dir: Path
    templates_dir: Path
    visualizations_dir: Path
    intermediate_dir: Path
    reports_dir: Path
    logs_dir: Path


@dataclass(slots=True)
class ProcessResult:
    """表示单张图流水线执行完成后的结果。

    输出字段：
    - `image_path`：原始图片路径
    - `template_path`：模板文件路径
    - `mnt_path`：文本细节点导出路径
    - `visualization_path`：可视化图路径
    - `intermediate_paths`：中间图路径列表
    - `keypoint_count`：提取到的关键点数量
    - `status`：处理状态
    - `error_message`：失败时的错误信息
    - `summary_path`：当前样本 summary 文件路径
    """

    image_path: Path
    template_path: Path | None = None
    mnt_path: Path | None = None
    visualization_path: Path | None = None
    intermediate_paths: list[Path] = field(default_factory=list)
    keypoint_count: int = 0
    status: str = "success"
    error_message: str | None = None
    summary_path: Path | None = None


@dataclass(slots=True)
class BatchResult:
    """表示一次批处理任务的汇总结果。"""

    input_roots: list[Path]
    total_images: int
    succeeded: int
    failed: int
    item_results: list[ProcessResult] = field(default_factory=list)
    summary_path: Path | None = None


@dataclass(slots=True)
class RunContext:
    """保存一次运行共享的输出目录和日志器。"""

    output_paths: OutputPaths
    logger: logging.Logger
