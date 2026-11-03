"""ROI 提取模块的当前基线实现。

这个文件负责给后续特征提取阶段提供 ROI 信息。
当前版本支持两种方式：
- `full_image`：把整张图近似视为 ROI
- `block_variance`：通过局部方差估计指纹前景区域

输入：
- 预处理后的灰度图
- ROI 配置

输出：
- `ROIResult`，包含掩膜、边界框、裁剪图和前景比例
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config.schema import ROIConfig
from src.exceptions import ROIExtractionError
from src.types import ROIResult


def extract_roi(image: np.ndarray, config: ROIConfig) -> ROIResult:
    """根据当前配置从图像中提取 ROI。

    输入：
    - `image`：预处理后的灰度图
    - `config`：ROI 配置

    输出：
    - `ROIResult`
    """

    if not config.enabled or config.method == "full_image":
        return build_full_image_roi(image, margin=config.margin if config.enabled else 0)

    if config.method != "block_variance":
        raise ROIExtractionError(f"不支持的 ROI 提取方法: {config.method}")

    variance_map = estimate_variance_map(image, config.block_size)
    raw_mask = threshold_variance_map(variance_map, config.variance_threshold)
    refined_mask = refine_roi_mask(
        raw_mask,
        morph_kernel=config.morph_kernel,
        close_iters=config.close_iters,
        open_iters=config.open_iters,
        use_largest_component=config.use_largest_component,
    )
    return build_roi_result(
        image=image,
        roi_mask=refined_mask,
        margin=config.margin,
        min_area_ratio=config.min_area_ratio,
    )


def estimate_variance_map(image: np.ndarray, block_size: int) -> np.ndarray:
    """估计图像的局部方差图。

    指纹脊线区域通常比纯背景具有更强的局部灰度变化，
    因此局部方差可作为前景估计的简单基线。
    """

    image_float = image.astype(np.float32)
    mean = cv2.blur(image_float, (block_size, block_size))
    sqmean = cv2.blur(image_float * image_float, (block_size, block_size))
    variance = sqmean - mean * mean
    return np.clip(variance, 0.0, None).astype(np.float32)


def threshold_variance_map(variance_map: np.ndarray, variance_threshold: float) -> np.ndarray:
    """根据局部方差阈值得到原始前景掩膜。"""

    mask = np.zeros(variance_map.shape, dtype=np.uint8)
    mask[variance_map >= variance_threshold] = 255
    return mask


def refine_roi_mask(
    mask: np.ndarray,
    morph_kernel: int,
    close_iters: int,
    open_iters: int,
    use_largest_component: bool,
) -> np.ndarray:
    """对原始 ROI 掩膜做形态学清理。"""

    refined = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))

    if close_iters > 0:
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    if open_iters > 0:
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=open_iters)
    if use_largest_component:
        refined = keep_largest_component(refined)
    return refined


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """保留掩膜中的最大连通域，抑制离散噪声块。"""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask.copy()

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(component_areas)) + 1
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255
    return largest_mask


def compute_roi_bbox(
    roi_mask: np.ndarray,
    margin: int,
    min_area_ratio: float,
) -> tuple[int, int, int, int]:
    """根据最终掩膜计算 ROI 包围框。"""

    foreground_pixels = int(np.count_nonzero(roi_mask))
    foreground_ratio = foreground_pixels / float(roi_mask.size)
    if foreground_pixels == 0 or foreground_ratio < min_area_ratio:
        raise ROIExtractionError("ROI 前景区域过小，无法稳定提取有效前景。")

    ys, xs = np.where(roi_mask > 0)
    height, width = roi_mask.shape[:2]

    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(width, int(xs.max()) + 1 + margin)
    y1 = min(height, int(ys.max()) + 1 + margin)

    if x1 <= x0 or y1 <= y0:
        raise ROIExtractionError("ROI 包围框非法。")
    return x0, y0, x1, y1


def crop_to_roi(image: np.ndarray, roi_bbox: tuple[int, int, int, int]) -> np.ndarray:
    """根据 ROI 包围框裁剪图像。"""

    x0, y0, x1, y1 = roi_bbox
    return image[y0:y1, x0:x1].copy()


def build_roi_result(
    image: np.ndarray,
    roi_mask: np.ndarray,
    margin: int,
    min_area_ratio: float,
) -> ROIResult:
    """根据掩膜构建 `ROIResult`。"""

    roi_bbox = compute_roi_bbox(roi_mask, margin=margin, min_area_ratio=min_area_ratio)
    roi_image = crop_to_roi(image, roi_bbox)
    foreground_ratio = float(np.count_nonzero(roi_mask) / float(roi_mask.size))
    return ROIResult(
        roi_mask=roi_mask,
        roi_bbox=roi_bbox,
        roi_image=roi_image,
        foreground_ratio=foreground_ratio,
    )


def build_full_image_roi(image: np.ndarray, margin: int) -> ROIResult:
    """使用整张图构建一个保守的 ROI 基线结果。"""

    height, width = image.shape[:2]
    x0 = max(0, margin)
    y0 = max(0, margin)
    x1 = max(x0, width - margin)
    y1 = max(y0, height - margin)

    roi_image = image[y0:y1, x0:x1].copy()
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[y0:y1, x0:x1] = 255

    return ROIResult(
        roi_mask=roi_mask,
        roi_bbox=(x0, y0, x1, y1),
        roi_image=roi_image,
        foreground_ratio=float(roi_mask.mean() / 255.0),
    )
