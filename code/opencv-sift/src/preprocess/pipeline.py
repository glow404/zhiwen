"""指纹预处理总入口模块。

这个文件负责把指纹图的预处理各个步骤串联起来，形成适合 SIFT 特征提取的稳定输入：
图像读取与校验 -> 灰度增强 -> ROI 提取 -> 背景抑制 -> debug 中间图保存。

输入：
- 图像路径或灰度图像数组
- 预处理配置
- ROI 配置

输出：
- `PreprocessResult`，包含最终预处理图像、ROI mask、各阶段中间结果和 debug 输出路径
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config.schema import PreprocessConfig, ROIConfig
from src.exceptions import ROIExtractionError
from src.io.image_reader import load_image
from src.io.output_writer import save_image
from src.preprocess.normalize import run_preprocess as run_signal_preprocess
from src.preprocess.roi import (
    build_full_image_roi,
    build_roi_result,
    estimate_variance_map,
    refine_roi_mask,
    threshold_variance_map,
)
from src.types import PreprocessArtifacts, PreprocessResult, ROIResult
from src.utils.paths import build_image_id


def preprocess_fingerprint_path(
    image_path: Path,
    preprocess_config: PreprocessConfig,
    roi_config: ROIConfig,
) -> PreprocessResult:
    """从图像路径开始执行完整预处理流程。"""

    resolved_image_path = image_path.expanduser().resolve()
    image = load_image(resolved_image_path)
    return preprocess_fingerprint_image(
        image=image,
        preprocess_config=preprocess_config,
        roi_config=roi_config,
        image_id=build_image_id(resolved_image_path),
    )


def preprocess_fingerprint_image(
    image: np.ndarray,
    preprocess_config: PreprocessConfig,
    roi_config: ROIConfig,
    image_id: str = "image",
) -> PreprocessResult:
    """对单张灰度图执行完整预处理流程。"""

    artifacts = run_signal_preprocess(image, preprocess_config)
    roi_result, variance_map, raw_mask, refined_mask = _run_roi_stage(
        artifacts.denoised_image,
        roi_config,
    )
    background_suppressed = suppress_background(
        image=artifacts.denoised_image,
        roi_mask=roi_result.roi_mask,
        enabled=preprocess_config.background_suppression.enabled,
        fill_value=preprocess_config.background_suppression.fill_value,
        blur_kernel=preprocess_config.background_suppression.blur_kernel,
    )

    artifacts.variance_map = variance_map
    artifacts.roi_mask_raw = raw_mask
    artifacts.roi_mask_refined = refined_mask
    artifacts.background_suppressed_image = background_suppressed

    debug_output_dir: Path | None = None
    debug_image_paths: dict[str, Path] = {}
    if preprocess_config.debug.enabled and preprocess_config.debug.save_intermediate:
        debug_output_dir, debug_image_paths = save_debug_artifacts(
            artifacts=artifacts,
            roi_result=roi_result,
            save_dir=preprocess_config.debug.save_dir,
            image_id=image_id,
        )

    return PreprocessResult(
        output_image=background_suppressed,
        artifacts=artifacts,
        roi_result=roi_result,
        debug_output_dir=debug_output_dir,
        debug_image_paths=debug_image_paths,
    )


def suppress_background(
    image: np.ndarray,
    roi_mask: np.ndarray,
    enabled: bool,
    fill_value: int,
    blur_kernel: int,
) -> np.ndarray:
    """根据 ROI mask 抑制背景区域，尽量保留前景脊线细节。"""

    if not enabled:
        return image.copy()

    mask_float = roi_mask.astype(np.float32) / 255.0
    if blur_kernel > 1:
        mask_float = cv2.GaussianBlur(mask_float, (blur_kernel, blur_kernel), sigmaX=0.0)

    image_float = image.astype(np.float32)
    suppressed = image_float * mask_float + float(fill_value) * (1.0 - mask_float)
    return np.clip(suppressed, 0, 255).astype(np.uint8)


def save_debug_artifacts(
    artifacts: PreprocessArtifacts,
    roi_result: ROIResult,
    save_dir: Path,
    image_id: str,
) -> tuple[Path, dict[str, Path]]:
    """把预处理各阶段的中间图保存到 debug 目录。"""

    debug_dir = save_dir.expanduser().resolve() / image_id
    debug_image_paths: dict[str, Path] = {}

    debug_images = {
        "01_input": artifacts.input_image,
        "02_resized": artifacts.resized_image,
        "03_normalized": artifacts.normalized_image,
        "04_enhanced": artifacts.enhanced_image,
        "05_denoised": artifacts.denoised_image,
        "06_variance_map": _to_debug_image(artifacts.variance_map),
        "07_roi_mask_raw": _to_debug_image(artifacts.roi_mask_raw),
        "08_roi_mask_refined": _to_debug_image(artifacts.roi_mask_refined),
        "09_background_suppressed": _to_debug_image(artifacts.background_suppressed_image),
        "10_roi_crop": roi_result.roi_image,
    }

    for name, debug_image in debug_images.items():
        if debug_image is None:
            continue
        debug_image_paths[name] = save_image(debug_image, debug_dir / f"{name}.png")

    return debug_dir, debug_image_paths


def _run_roi_stage(
    image: np.ndarray,
    roi_config: ROIConfig,
) -> tuple[ROIResult, np.ndarray | None, np.ndarray, np.ndarray]:
    """执行 ROI 提取并返回调试所需的中间结果。"""

    if not roi_config.enabled or roi_config.method == "full_image":
        roi_result = build_full_image_roi(image, margin=roi_config.margin if roi_config.enabled else 0)
        return roi_result, None, roi_result.roi_mask.copy(), roi_result.roi_mask.copy()

    variance_map = estimate_variance_map(image, roi_config.block_size)
    raw_mask = threshold_variance_map(variance_map, roi_config.variance_threshold)
    refined_mask = refine_roi_mask(
        raw_mask,
        morph_kernel=roi_config.morph_kernel,
        close_iters=roi_config.close_iters,
        open_iters=roi_config.open_iters,
        use_largest_component=roi_config.use_largest_component,
    )

    try:
        roi_result = build_roi_result(
            image=image,
            roi_mask=refined_mask,
            margin=roi_config.margin,
            min_area_ratio=roi_config.min_area_ratio,
        )
    except ROIExtractionError:
        roi_result = build_full_image_roi(image, margin=0)
        refined_mask = roi_result.roi_mask.copy()

    return roi_result, variance_map, raw_mask, refined_mask


def _to_debug_image(image: np.ndarray | None) -> np.ndarray | None:
    """把任意中间结果转换为适合保存的 `uint8` 图像。"""

    if image is None:
        return None
    if image.dtype == np.uint8:
        return image
    normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)
