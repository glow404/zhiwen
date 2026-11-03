"""特征提取结果可视化模块。

这个文件负责把关键点和 ROI 框绘制到图像上，生成便于人工检查的可视化结果。

输入：
- 灰度图
- 关键点列表
- ROI 边界框
- 可视化配置

输出：
- 带关键点和 ROI 标注的可视化图像
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config.schema import VisualizationConfig
from src.features.sift_extractor import draw_keypoints as draw_sift_keypoints


def draw_keypoints(
    image: np.ndarray,
    keypoints: list,
    draw_rich_keypoints: bool,
) -> np.ndarray:
    """在灰度图上绘制关键点。"""

    return draw_sift_keypoints(
        image=image,
        keypoints=keypoints,
        draw_rich_keypoints=draw_rich_keypoints,
    )


def draw_roi_bbox(image: np.ndarray, roi_bbox: tuple[int, int, int, int]) -> np.ndarray:
    """在图像上叠加 ROI 矩形框。"""

    canvas = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 2
        else image.copy()
    )
    x0, y0, x1, y1 = roi_bbox
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 0, 0), 2)
    return canvas


def draw_feature_overview(
    image: np.ndarray,
    keypoints: list,
    roi_bbox: tuple[int, int, int, int],
    config: VisualizationConfig,
) -> np.ndarray:
    """构建当前版本的总览可视化图。

    输入：
    - 预处理后的图像
    - 关键点列表
    - ROI 边界框
    - 可视化配置

    输出：
    - 可直接保存到磁盘的 BGR 图像
    """

    keypoint_image = draw_keypoints(
        image=image,
        keypoints=keypoints,
        draw_rich_keypoints=config.draw_rich_keypoints,
    )
    if not config.draw_roi_bbox:
        return keypoint_image

    return draw_roi_bbox(keypoint_image, roi_bbox)
