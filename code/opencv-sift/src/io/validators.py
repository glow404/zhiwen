"""基础校验模块。

这个文件负责对图像和描述子等关键输入做基本合法性检查，
避免非法数据继续流入后续模块。

输入：
- 灰度图像数组
- 描述子数组

输出：
- 无返回值
- 校验失败时抛出明确异常
"""

from __future__ import annotations

import numpy as np

from src.exceptions import FeatureExtractionError, ImageLoadError


def validate_gray_image(image: np.ndarray) -> None:
    """校验输入是否为非空灰度图。"""

    if image is None:
        raise ImageLoadError("Loaded image is None.")
    if image.size == 0:
        raise ImageLoadError("Loaded image is empty.")
    if image.ndim != 2:
        raise ImageLoadError("Expected a grayscale image with shape [H, W].")


def validate_descriptors(descriptors: np.ndarray | None) -> None:
    """校验 OpenCV 返回的描述子数组是否合法。"""

    if descriptors is None:
        return
    if descriptors.ndim != 2:
        raise FeatureExtractionError("Descriptors must be a 2D array.")
