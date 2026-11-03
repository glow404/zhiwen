"""批处理主流程模块。

这个文件负责递归扫描输入目录，逐张处理指纹图像，并输出批处理汇总结果。

输入：
- 顶层应用配置
- 可选共享运行上下文

输出：
- `BatchResult`
- 批处理 summary JSON
"""

from __future__ import annotations

from pathlib import Path

from src.config.schema import AppConfig
from src.io.discover import discover_images
from src.io.output_writer import save_json
from src.pipeline.context import build_run_context
from src.pipeline.single_image import process_single_image
from src.types import BatchResult, RunContext


def process_batch(
    config: AppConfig,
    run_context: RunContext | None = None,
) -> BatchResult:
    """执行目录级批处理指纹特征提取。"""

    shared_context = run_context or build_run_context(config)
    logger = shared_context.logger
    input_roots = [path.expanduser().resolve() for path in config.input.input_roots]

    logger.info("开始批处理，共 %s 个输入目录。", len(input_roots))
    image_paths = discover_images(
        input_roots=input_roots,
        recursive=config.input.recursive,
        patterns=config.input.patterns,
    )

    logger.info("发现待处理图像数量: %s", len(image_paths))
    item_results = [
        process_single_image(
            image_path=image_path,
            config=config,
            run_context=shared_context,
            raise_on_error=False,
        )
        for image_path in image_paths
    ]

    succeeded = sum(1 for item in item_results if item.status == "success")
    failed = sum(1 for item in item_results if item.status != "success")
    batch_result = BatchResult(
        input_roots=input_roots,
        total_images=len(image_paths),
        succeeded=succeeded,
        failed=failed,
        item_results=item_results,
    )
    batch_result.summary_path = write_batch_summary(batch_result, shared_context)
    logger.info("批处理结束：成功 %s，失败 %s。", succeeded, failed)
    return batch_result


def write_batch_summary(batch_result: BatchResult, run_context: RunContext) -> Path:
    """把批处理汇总结果写入 JSON。"""

    summary_path = run_context.output_paths.reports_dir / "batch_summary.json"
    summary_data = {
        "input_roots": [path.as_posix() for path in batch_result.input_roots],
        "total_images": batch_result.total_images,
        "succeeded": batch_result.succeeded,
        "failed": batch_result.failed,
        "items": [
            {
                "image_path": item.image_path.as_posix(),
                "status": item.status,
                "keypoint_count": item.keypoint_count,
                "template_path": item.template_path.as_posix() if item.template_path else None,
                "mnt_path": item.mnt_path.as_posix() if item.mnt_path else None,
                "visualization_path": (
                    item.visualization_path.as_posix() if item.visualization_path else None
                ),
                "summary_path": item.summary_path.as_posix() if item.summary_path else None,
                "error_message": item.error_message,
            }
            for item in batch_result.item_results
        ],
    }
    return save_json(summary_data, summary_path)
