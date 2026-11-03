"""批处理主流程的最小测试。

这个文件验证：
- 批处理能扫描输入目录并处理多张图
- 会生成批处理 summary
- 会生成日志文件
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config.schema import (
    AppConfig,
    AppMetaConfig,
    BackgroundSuppressionConfig,
    CLAHEConfig,
    DebugConfig,
    DenoiseConfig,
    InputConfig,
    LoggingConfig,
    NormalizeConfig,
    OutputConfig,
    PreprocessConfig,
    ResizeConfig,
    ROIConfig,
    SIFTConfig,
    VisualizationConfig,
)
from src.pipeline.batch_runner import process_batch


def test_process_batch_creates_summary_and_log(tmp_path: Path) -> None:
    """验证批处理入口能生成 summary 和日志文件。"""

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(input_dir / "sample_1.png"), _build_checkerboard_image())
    cv2.imwrite(str(input_dir / "sample_2.png"), _build_checkerboard_image())

    config = AppConfig(
        app=AppMetaConfig(name="test_batch", template_version="v0.1"),
        input=InputConfig(
            single_image_path=None,
            input_roots=[input_dir],
            recursive=True,
            patterns=[".png"],
        ),
        output=OutputConfig(
            output_root=tmp_path / "outputs",
            template_dirname="templates",
            visualization_dirname="visualizations",
            intermediate_dirname="intermediate",
            report_dirname="reports",
            log_dirname="logs",
            overwrite=True,
            save_template=True,
            save_visualization=True,
            save_intermediate=True,
        ),
        preprocess=PreprocessConfig(
            resize=ResizeConfig(enabled=False, max_size=512),
            normalize=NormalizeConfig(enabled=True, mean_target=128.0, std_target=50.0),
            clahe=CLAHEConfig(enabled=True, clip_limit=2.0, tile_grid_size=(8, 8)),
            denoise=DenoiseConfig(method="gaussian", kernel_size=3, sigma=0.0),
            background_suppression=BackgroundSuppressionConfig(enabled=True, fill_value=0, blur_kernel=5),
            debug=DebugConfig(enabled=False, save_dir=tmp_path / "debug", save_intermediate=True),
        ),
        roi=ROIConfig(
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
        ),
        sift=SIFTConfig(
            nfeatures=64,
            n_octave_layers=3,
            contrast_threshold=0.04,
            edge_threshold=10.0,
            sigma=1.6,
            descriptor_type="default",
        ),
        visualization=VisualizationConfig(enabled=True, draw_rich_keypoints=True, draw_roi_bbox=True),
        logging=LoggingConfig(
            level="INFO",
            console_enabled=False,
            file_enabled=True,
            filename="pipeline.log",
            logger_name="test_batch_logger",
        ),
    )

    batch_result = process_batch(config)

    assert batch_result.total_images == 2
    assert batch_result.summary_path is not None
    assert batch_result.summary_path.exists()
    assert (config.output.output_root / config.output.log_dirname / "pipeline.log").exists()


def test_process_batch_preserves_nested_input_structure_in_outputs(tmp_path: Path) -> None:
    """验证批处理输出会保留输入目录中的相对层级结构。"""

    input_dir = tmp_path / "inputs"
    nested_dir = input_dir / "bu_Rgd1237_Rgd1245" / "bu" / "fys_R0" / "Rgd1237"
    nested_dir.mkdir(parents=True, exist_ok=True)
    image_path = nested_dir / "sample_1.303782_pair0.png"
    cv2.imwrite(str(image_path), _build_checkerboard_image())

    config = AppConfig(
        app=AppMetaConfig(name="test_batch", template_version="v0.1"),
        input=InputConfig(
            single_image_path=None,
            input_roots=[input_dir],
            recursive=True,
            patterns=[".png"],
        ),
        output=OutputConfig(
            output_root=tmp_path / "outputs",
            template_dirname="templates",
            visualization_dirname="visualizations",
            intermediate_dirname="intermediate",
            report_dirname="reports",
            log_dirname="logs",
            overwrite=True,
            save_template=True,
            save_visualization=True,
            save_intermediate=True,
        ),
        preprocess=PreprocessConfig(
            resize=ResizeConfig(enabled=False, max_size=512),
            normalize=NormalizeConfig(enabled=True, mean_target=128.0, std_target=50.0),
            clahe=CLAHEConfig(enabled=True, clip_limit=2.0, tile_grid_size=(8, 8)),
            denoise=DenoiseConfig(method="gaussian", kernel_size=3, sigma=0.0),
            background_suppression=BackgroundSuppressionConfig(enabled=True, fill_value=0, blur_kernel=5),
            debug=DebugConfig(enabled=False, save_dir=tmp_path / "debug", save_intermediate=True),
        ),
        roi=ROIConfig(
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
        ),
        sift=SIFTConfig(
            nfeatures=64,
            n_octave_layers=3,
            contrast_threshold=0.04,
            edge_threshold=10.0,
            sigma=1.6,
            descriptor_type="default",
        ),
        visualization=VisualizationConfig(enabled=True, draw_rich_keypoints=True, draw_roi_bbox=True),
        logging=LoggingConfig(
            level="INFO",
            console_enabled=False,
            file_enabled=True,
            filename="pipeline.log",
            logger_name="test_batch_nested_logger",
        ),
    )

    batch_result = process_batch(config)

    assert batch_result.total_images == 1
    item = batch_result.item_results[0]
    relative_stem = Path("bu_Rgd1237_Rgd1245") / "bu" / "fys_R0" / "Rgd1237" / "sample_1.303782_pair0"
    assert item.template_path == (
        config.output.output_root
        / "templates"
        / relative_stem.parent
        / f"{relative_stem.name}.npz"
    )
    assert item.mnt_path == (
        config.output.output_root
        / "templates"
        / relative_stem.parent
        / f"{relative_stem.name}.mnt"
    )
    assert item.visualization_path == (
        config.output.output_root
        / "visualizations"
        / relative_stem.parent
        / "sample_1.303782_pair0_keypoints.png"
    )
    assert item.summary_path == (
        config.output.output_root
        / "reports"
        / relative_stem.parent
        / "sample_1.303782_pair0_summary.json"
    )
    assert item.intermediate_paths == [
        config.output.output_root
        / "intermediate"
        / relative_stem.parent
        / "sample_1.303782_pair0_preprocessed.png"
    ]


def _build_checkerboard_image() -> np.ndarray:
    """构造一张带局部纹理的测试图像。"""

    image = np.zeros((96, 96), dtype=np.uint8)
    step = 12
    for y in range(0, 96, step):
        for x in range(0, 96, step):
            if (x // step + y // step) % 2 == 0:
                image[y : y + step, x : x + step] = 210
            else:
                image[y : y + step, x : x + step] = 40
    return image
