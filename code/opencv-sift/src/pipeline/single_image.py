"""单张图指纹特征提取主流程。

这个文件负责把各个基础模块串起来，形成当前 MVP 的单图处理流水线：
图像读取 -> 预处理 -> ROI -> SIFT -> 模板保存 -> 可视化 -> summary 输出。

输入：
- 单张图路径
- 顶层应用配置

输出：
- `ProcessResult`，包含模板路径、可视化路径、中间图路径和关键点数量
"""

from __future__ import annotations

from pathlib import Path
import traceback

from src.config.loader import config_to_dict
from src.config.schema import AppConfig
from src.exceptions import AppError
from src.features.sift_extractor import extract_sift_features
from src.io.output_writer import save_image, save_json
from src.io.template_store import build_preprocess_metadata, build_template, save_template
from src.pipeline.context import build_run_context
from src.preprocess.pipeline import preprocess_fingerprint_path
from src.types import ProcessResult, RunContext
from src.utils.paths import build_relative_output_stem
from src.visualization.draw import draw_feature_overview


def process_single_image(
    image_path: Path,
    config: AppConfig,
    run_context: RunContext | None = None,
    raise_on_error: bool = True,
) -> ProcessResult:
    """执行单张图的基础指纹特征提取流程。

    输入：
    - `image_path`：单张图路径
    - `config`：顶层配置对象
    - `run_context`：可选，共享运行上下文
    - `raise_on_error`：失败时是否继续向上抛异常

    输出：
    - `ProcessResult`
    """

    resolved_image_path = image_path.expanduser().resolve()
    shared_context = run_context or build_run_context(config)
    output_stem = build_relative_output_stem(resolved_image_path, config.input.input_roots)
    logger = shared_context.logger

    try:
        logger.info("开始处理图像: %s", resolved_image_path.as_posix())
        preprocess_result = preprocess_fingerprint_path(
            image_path=resolved_image_path,
            preprocess_config=config.preprocess,
            roi_config=config.roi,
        )
        feature_result = extract_sift_features(
            image=preprocess_result.output_image,
            roi_mask=preprocess_result.roi_result.roi_mask,
            config=config.sift,
        )

        process_result = ProcessResult(
            image_path=resolved_image_path,
            keypoint_count=feature_result.keypoint_count,
            status="success",
        )

        if config.output.save_intermediate:
            intermediate_path = shared_context.output_paths.intermediate_dir / output_stem.parent / (
                f"{output_stem.name}_preprocessed.png"
            )
            process_result.intermediate_paths.append(
                save_image(preprocess_result.output_image, intermediate_path)
            )

        if config.output.save_template:
            template = build_template(
                image_path=resolved_image_path,
                image_shape=tuple(int(v) for v in preprocess_result.output_image.shape),
                roi_bbox=preprocess_result.roi_result.roi_bbox,
                feature_result=feature_result,
                template_version=config.app.template_version,
                config_snapshot=config_to_dict(config),
                preprocess_metadata=build_preprocess_metadata(preprocess_result),
            )
            template_path = (
                shared_context.output_paths.templates_dir
                / output_stem.parent
                / f"{output_stem.name}.npz"
            )
            process_result.template_path = save_template(
                template,
                template_path,
                format_type="npz_json",
            )
            process_result.mnt_path = process_result.template_path.with_suffix(".mnt")

        if config.visualization.enabled and config.output.save_visualization:
            overview_image = draw_feature_overview(
                image=preprocess_result.output_image,
                keypoints=feature_result.keypoints,
                roi_bbox=preprocess_result.roi_result.roi_bbox,
                config=config.visualization,
            )
            visualization_path = (
                shared_context.output_paths.visualizations_dir
                / output_stem.parent
                / f"{output_stem.name}_keypoints.png"
            )
            process_result.visualization_path = save_image(overview_image, visualization_path)

        summary_path = shared_context.output_paths.reports_dir / output_stem.parent / (
            f"{output_stem.name}_summary.json"
        )
        process_result.summary_path = save_json(
            {
                "status": process_result.status,
                "image_path": resolved_image_path.as_posix(),
                "foreground_ratio": preprocess_result.roi_result.foreground_ratio,
                "keypoint_count": feature_result.keypoint_count,
                "descriptor_dim": feature_result.descriptor_dim,
                "debug_output_dir": preprocess_result.debug_output_dir.as_posix()
                if preprocess_result.debug_output_dir
                else None,
                "template_path": process_result.template_path.as_posix()
                if process_result.template_path
                else None,
                "mnt_path": process_result.mnt_path.as_posix() if process_result.mnt_path else None,
                "visualization_path": process_result.visualization_path.as_posix()
                if process_result.visualization_path
                else None,
                "error_message": None,
            },
            summary_path,
        )
        logger.info(
            "处理完成: %s | keypoints=%s",
            resolved_image_path.as_posix(),
            feature_result.keypoint_count,
        )
        return process_result
    except AppError as exc:
        return _handle_process_error(
            image_path=resolved_image_path,
            output_stem=output_stem,
            context=shared_context,
            logger=logger,
            error=exc,
            raise_on_error=raise_on_error,
        )
    except Exception as exc:
        return _handle_process_error(
            image_path=resolved_image_path,
            output_stem=output_stem,
            context=shared_context,
            logger=logger,
            error=exc,
            raise_on_error=raise_on_error,
            include_traceback=True,
        )


def _handle_process_error(
    image_path: Path,
    output_stem: Path,
    context: RunContext,
    logger,
    error: Exception,
    raise_on_error: bool,
    include_traceback: bool = False,
) -> ProcessResult:
    """统一处理单图流程中的失败情况。"""

    if include_traceback:
        logger.error("处理失败: %s\n%s", image_path.as_posix(), traceback.format_exc())
    else:
        logger.error("处理失败: %s | %s", image_path.as_posix(), str(error))

    failed_result = ProcessResult(
        image_path=image_path,
        status="failed",
        error_message=str(error),
    )
    summary_path = context.output_paths.reports_dir / output_stem.parent / (
        f"{output_stem.name}_summary.json"
    )
    failed_result.summary_path = save_json(
        {
            "status": failed_result.status,
            "image_path": image_path.as_posix(),
            "keypoint_count": 0,
            "descriptor_dim": 0,
            "template_path": None,
            "mnt_path": None,
            "visualization_path": None,
            "error_message": failed_result.error_message,
        },
        summary_path,
    )

    if raise_on_error:
        raise error
    return failed_result
