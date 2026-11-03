"""指纹预处理模块的最小测试用例。

这个文件重点验证两件事：
- 预处理流程能输出 ROI mask 和背景抑制结果
- debug 模式能把中间图保存到指定目录
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config.schema import (
    BackgroundSuppressionConfig,
    CLAHEConfig,
    DebugConfig,
    DenoiseConfig,
    NormalizeConfig,
    PreprocessConfig,
    ROIConfig,
    ResizeConfig,
)
from src.preprocess.pipeline import preprocess_fingerprint_image


def test_preprocess_pipeline_returns_roi_and_suppressed_image(tmp_path: Path) -> None:
    """验证预处理流程能返回 ROI mask 和背景抑制后的图像。"""

    image = _build_synthetic_fingerprint_image()
    preprocess_config = PreprocessConfig(
        resize=ResizeConfig(enabled=False, max_size=256),
        normalize=NormalizeConfig(enabled=True, mean_target=128.0, std_target=40.0),
        clahe=CLAHEConfig(enabled=True, clip_limit=2.0, tile_grid_size=(8, 8)),
        denoise=DenoiseConfig(method="gaussian", kernel_size=3, sigma=0.0),
        background_suppression=BackgroundSuppressionConfig(enabled=True, fill_value=0, blur_kernel=5),
        debug=DebugConfig(enabled=False, save_dir=tmp_path / "debug", save_intermediate=True),
    )
    roi_config = ROIConfig(
        enabled=True,
        method="block_variance",
        margin=0,
        block_size=11,
        variance_threshold=8.0,
        morph_kernel=5,
        close_iters=1,
        open_iters=1,
        min_area_ratio=0.01,
        use_largest_component=True,
    )

    result = preprocess_fingerprint_image(image, preprocess_config, roi_config, image_id="synthetic")

    assert result.output_image.shape == image.shape
    assert result.roi_result.roi_mask.shape == image.shape
    assert np.count_nonzero(result.roi_result.roi_mask) > 0
    assert result.artifacts.background_suppressed_image is not None
    assert int(result.output_image[5, 5]) <= int(result.artifacts.denoised_image[5, 5])


def test_preprocess_pipeline_saves_debug_images(tmp_path: Path) -> None:
    """验证 debug 模式下能保存中间图。"""

    image = _build_synthetic_fingerprint_image()
    preprocess_config = PreprocessConfig(
        resize=ResizeConfig(enabled=False, max_size=256),
        normalize=NormalizeConfig(enabled=True, mean_target=128.0, std_target=40.0),
        clahe=CLAHEConfig(enabled=False, clip_limit=2.0, tile_grid_size=(8, 8)),
        denoise=DenoiseConfig(method="none", kernel_size=3, sigma=0.0),
        background_suppression=BackgroundSuppressionConfig(enabled=True, fill_value=0, blur_kernel=3),
        debug=DebugConfig(enabled=True, save_dir=tmp_path / "debug", save_intermediate=True),
    )
    roi_config = ROIConfig(
        enabled=True,
        method="block_variance",
        margin=0,
        block_size=9,
        variance_threshold=5.0,
        morph_kernel=3,
        close_iters=1,
        open_iters=1,
        min_area_ratio=0.01,
        use_largest_component=True,
    )

    result = preprocess_fingerprint_image(image, preprocess_config, roi_config, image_id="debug_case")

    assert result.debug_output_dir is not None
    assert result.debug_output_dir.exists()
    assert "01_input" in result.debug_image_paths
    assert result.debug_image_paths["01_input"].exists()


def _build_synthetic_fingerprint_image() -> np.ndarray:
    """构造一张具有前景纹理和背景区域的合成指纹样图。"""

    image = np.full((160, 160), 30, dtype=np.uint8)
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.ellipse(mask, center=(80, 80), axes=(45, 60), angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

    for offset in range(-80, 81, 6):
        start = (0, 80 + offset)
        end = (160, 40 + offset)
        cv2.line(image, start, end, 170, 1)

    image[mask == 0] = 30
    image = cv2.GaussianBlur(image, (3, 3), 0.0)
    return image
