"""图像读取工具。

这个文件负责从磁盘读取单张图像，并统一输出为灰度 `uint8` 数组，
供后续预处理、ROI 和特征提取模块直接使用。

输入：
- 图像路径

输出：
- 二维灰度 `numpy.ndarray`
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.io.validators import validate_gray_image


def load_image(path: Path, force_gray: bool = True) -> np.ndarray:
    """读取单张图像并返回灰度 `uint8` 数组。

    输入：
    - `path`：图像文件路径
    - `force_gray`：是否强制按灰度图读取

    输出：
    - 形状为 `[H, W]` 的灰度图像数组
    """

    image_path = path.expanduser().resolve()
    read_flag = cv2.IMREAD_GRAYSCALE if force_gray else cv2.IMREAD_UNCHANGED
    image = cv2.imread(str(image_path), read_flag)
    validate_gray_image(image)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image
