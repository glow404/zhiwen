"""指纹灰度图的基础预处理模块。

这个文件负责完成单图基础预处理，包括：
- 可选缩放
- 灰度归一化
- CLAHE 对比度增强
- 去噪

输入：
- 单张灰度图像
- 预处理配置

输出：
- 预处理增强阶段的中间结果
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config.schema import PreprocessConfig
from src.exceptions import PreprocessError
from src.types import PreprocessArtifacts


def resize_if_needed(image: np.ndarray, enabled: bool, max_size: int) -> np.ndarray:
    """按最长边限制对图像进行等比例缩放。

    输入：
    - `image`：输入灰度图
    - `enabled`：是否启用缩放
    - `max_size`：最长边最大尺寸

    输出：
    - 缩放后的图像；若不需要缩放则返回副本
    """

    if not enabled:
        return image.copy()

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_size:
        return image.copy()

    scale = max_size / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def normalize_intensity(
    image: np.ndarray,
    enabled: bool,
    mean_target: float,
    std_target: float,
) -> np.ndarray:
    """把图像灰度归一化到目标均值和标准差。

    输入：
    - 输入灰度图
    - 归一化开关
    - 目标均值和目标标准差

    输出：
    - 归一化后的 `uint8` 图像
    """

    if not enabled:
        return image.copy()

    image_float = image.astype(np.float32)
    current_mean = float(image_float.mean())
    current_std = float(image_float.std())
    if current_std < 1e-6:
        raise PreprocessError("Image standard deviation is too small for normalization.")

    normalized = (image_float - current_mean) / current_std
    normalized = normalized * std_target + mean_target
    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_clahe(
    image: np.ndarray,
    enabled: bool,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    """在启用时对图像执行 CLAHE 增强。"""

    if not enabled:
        return image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def denoise_image(
    image: np.ndarray,
    method: str,
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    """按配置选择简单去噪算子。

    当前支持：
    - `none`
    - `gaussian`
    - `median`
    """

    if method == "none":
        return image.copy()
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
    if method == "median":
        return cv2.medianBlur(image, kernel_size)
    raise PreprocessError(f"Unsupported denoise method: {method}")


def run_preprocess(image: np.ndarray, config: PreprocessConfig) -> PreprocessArtifacts:
    """执行当前版本的基础增强流水线。

    输入：
    - `image`：原始灰度图
    - `config`：预处理配置

    输出：
    - `PreprocessArtifacts`，包含各阶段中间图
    """

    resized = resize_if_needed(
        image=image,
        enabled=config.resize.enabled,
        max_size=config.resize.max_size,
    )
    normalized = normalize_intensity(
        image=resized,
        enabled=config.normalize.enabled,
        mean_target=config.normalize.mean_target,
        std_target=config.normalize.std_target,
    )
    enhanced = apply_clahe(
        image=normalized,
        enabled=config.clahe.enabled,
        clip_limit=config.clahe.clip_limit,
        tile_grid_size=config.clahe.tile_grid_size,
    )
    denoised = denoise_image(
        image=enhanced,
        method=config.denoise.method,
        kernel_size=config.denoise.kernel_size,
        sigma=config.denoise.sigma,
    )

    return PreprocessArtifacts(
        input_image=image.copy(),
        resized_image=resized,
        normalized_image=normalized,
        enhanced_image=enhanced,
        denoised_image=denoised,
    )
