"""指纹模板构建、存储与加载模块。

这个文件负责把单个指纹样本整理为统一模板对象，并提供统一的保存与加载接口。

当前支持两种模板格式：
- `npz_json`：数值数组保存在 `.npz`，元信息保存在同名 `.json`
- `directory`：目录中分别保存 `descriptors.npy`、`keypoints.npz`、`metadata.json`

此外，`npz_json` 模式会额外导出一个同名 `.mnt` 文本文件，用于和
细节点/关键点文本格式工具对接。

输入：
- 图片路径或样本标识
- 图像尺寸
- 预处理元数据
- ROI 信息
- 特征提取结果
- 配置快照

输出：
- `FingerprintTemplate` 对象
- 可供后续匹配模块直接读取的模板文件或模板目录
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.exceptions import TemplateStoreError
from src.features.sift_extractor import serialize_keypoints
from src.types import FeatureResult, FingerprintTemplate, PreprocessResult
from src.utils.paths import build_image_id


def build_template(
    image_path: Path,
    image_shape: tuple[int, int],
    roi_bbox: tuple[int, int, int, int],
    feature_result: FeatureResult,
    template_version: str,
    config_snapshot: dict[str, Any],
    preprocess_metadata: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> FingerprintTemplate:
    """根据提取到的 SIFT 特征构建可序列化模板。

    输入：
    - 原图路径、图像尺寸、ROI 边界框
    - 特征提取结果
    - 模板版本号
    - 配置快照
    - 预处理元数据
    - 时间戳

    输出：
    - `FingerprintTemplate` 模板对象
    """

    keypoints_data = (
        feature_result.keypoints_data
        if feature_result.keypoints_data
        else serialize_keypoints(feature_result.keypoints)
    )

    return FingerprintTemplate(
        template_version=template_version,
        image_id=build_image_id(image_path),
        source_path=image_path.as_posix(),
        image_shape=image_shape,
        roi_bbox=roi_bbox,
        preprocess_metadata=preprocess_metadata or {},
        timestamp=timestamp or generate_timestamp(),
        keypoints_x=keypoints_data["x"],
        keypoints_y=keypoints_data["y"],
        keypoints_size=keypoints_data["size"],
        keypoints_angle=keypoints_data["angle"],
        keypoints_response=keypoints_data["response"],
        keypoints_octave=keypoints_data["octave"],
        keypoints_class_id=keypoints_data["class_id"],
        descriptors=feature_result.descriptors,
        descriptor_dtype=feature_result.descriptor_dtype,
        config_snapshot=config_snapshot,
    )


def build_preprocess_metadata(preprocess_result: PreprocessResult) -> dict[str, Any]:
    """从预处理结果中提取适合写入模板的轻量元数据。"""

    return {
        "foreground_ratio": preprocess_result.roi_result.foreground_ratio,
        "roi_bbox": list(preprocess_result.roi_result.roi_bbox),
        "debug_output_dir": preprocess_result.debug_output_dir.as_posix()
        if preprocess_result.debug_output_dir
        else None,
        "debug_image_paths": {
            key: value.as_posix()
            for key, value in preprocess_result.debug_image_paths.items()
        },
        "artifacts": {
            "input_shape": list(preprocess_result.artifacts.input_image.shape),
            "resized_shape": list(preprocess_result.artifacts.resized_image.shape),
            "normalized_shape": list(preprocess_result.artifacts.normalized_image.shape),
            "enhanced_shape": list(preprocess_result.artifacts.enhanced_image.shape),
            "denoised_shape": list(preprocess_result.artifacts.denoised_image.shape),
            "has_variance_map": preprocess_result.artifacts.variance_map is not None,
            "has_roi_mask_raw": preprocess_result.artifacts.roi_mask_raw is not None,
            "has_roi_mask_refined": preprocess_result.artifacts.roi_mask_refined is not None,
            "has_background_suppressed_image": (
                preprocess_result.artifacts.background_suppressed_image is not None
            ),
        },
    }


def save_template(
    template: FingerprintTemplate,
    save_path: Path,
    format_type: str = "npz_json",
) -> Path:
    """统一保存模板接口。

    输入：
    - `template`：待保存模板
    - `save_path`：目标模板路径
    - `format_type`：模板存储格式，支持 `npz_json`、`directory` 和 `mnt`

    输出：
    - 实际保存后的模板主路径
    """

    if format_type == "npz_json":
        return save_template_npz_json(template, save_path)
    if format_type == "directory":
        return save_template_directory(template, save_path)
    if format_type == "mnt":
        return save_template_mnt(template, save_path)
    raise TemplateStoreError(f"不支持的模板存储格式: {format_type}")


def load_template(path: Path) -> FingerprintTemplate:
    """统一加载模板接口。"""

    resolved_path = path.expanduser().resolve()
    if resolved_path.is_dir():
        return load_template_directory(resolved_path)
    if resolved_path.suffix.lower() == ".npz":
        return load_template_npz_json(resolved_path)
    if resolved_path.suffix.lower() == ".json":
        return load_template_npz_json(resolved_path.with_suffix(".npz"))
    raise TemplateStoreError(f"无法识别模板路径格式: {resolved_path}")


def save_template_npz_json(template: FingerprintTemplate, save_path: Path) -> Path:
    """把模板保存为 `.npz + .json` 组合格式。"""

    resolved_path = save_path.expanduser().resolve()
    if resolved_path.suffix.lower() != ".npz":
        resolved_path = resolved_path.with_suffix(".npz")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = resolved_path.with_suffix(".json")

    try:
        np.savez_compressed(
            resolved_path,
            image_shape=np.asarray(template.image_shape, dtype=np.int32),
            roi_bbox=np.asarray(template.roi_bbox, dtype=np.int32),
            keypoints_x=template.keypoints_x,
            keypoints_y=template.keypoints_y,
            keypoints_size=template.keypoints_size,
            keypoints_angle=template.keypoints_angle,
            keypoints_response=template.keypoints_response,
            keypoints_octave=template.keypoints_octave,
            keypoints_class_id=template.keypoints_class_id,
            descriptors=_descriptor_array(template.descriptors),
            has_descriptors=np.asarray([template.descriptors is not None], dtype=np.uint8),
        )
        _write_metadata_json(metadata_path, build_template_metadata(template, format_type="npz_json"))
        save_template_mnt(template, resolved_path.with_suffix(".mnt"))
    except OSError as exc:
        raise TemplateStoreError(f"保存模板失败: {resolved_path}") from exc

    return resolved_path


def save_template_mnt(template: FingerprintTemplate, save_path: Path) -> Path:
    """把关键点以 `.mnt` 文本格式保存。"""

    resolved_path = save_path.expanduser().resolve()
    if resolved_path.suffix.lower() != ".mnt":
        resolved_path = resolved_path.with_suffix(".mnt")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    image_height, image_width = (int(value) for value in template.image_shape)
    point_count = int(len(template.keypoints_x))
    x_coords = np.rint(template.keypoints_x).astype(np.int32)
    y_coords = np.rint(template.keypoints_y).astype(np.int32)
    orientations = np.deg2rad(template.keypoints_angle.astype(np.float64))
    lines = [template.image_id, f"{point_count} {image_height} {image_width}"]
    lines.extend(
        f"{int(x)} {int(y)} {orientation:.6f}"
        for x, y, orientation in zip(x_coords, y_coords, orientations, strict=False)
    )

    try:
        with resolved_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write("\n".join(lines))
            file.write("\n")
    except OSError as exc:
        raise TemplateStoreError(f"保存 MNT 模板失败: {resolved_path}") from exc

    return resolved_path


def load_template_npz_json(path: Path) -> FingerprintTemplate:
    """从 `.npz + .json` 组合格式加载模板。"""

    resolved_path = path.expanduser().resolve()
    metadata_path = resolved_path.with_suffix(".json")

    try:
        data = np.load(resolved_path, allow_pickle=False)
    except OSError as exc:
        raise TemplateStoreError(f"读取模板失败: {resolved_path}") from exc

    metadata = _read_metadata_json(metadata_path)
    return _build_template_from_arrays_and_metadata(data, metadata)


def save_template_directory(template: FingerprintTemplate, save_path: Path) -> Path:
    """把模板保存为目录格式。"""

    resolved_dir = save_path.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    descriptors_path = resolved_dir / "descriptors.npy"
    keypoints_path = resolved_dir / "keypoints.npz"
    metadata_path = resolved_dir / "metadata.json"

    try:
        np.save(descriptors_path, _descriptor_array(template.descriptors))
        np.savez_compressed(
            keypoints_path,
            keypoints_x=template.keypoints_x,
            keypoints_y=template.keypoints_y,
            keypoints_size=template.keypoints_size,
            keypoints_angle=template.keypoints_angle,
            keypoints_response=template.keypoints_response,
            keypoints_octave=template.keypoints_octave,
            keypoints_class_id=template.keypoints_class_id,
        )
        _write_metadata_json(metadata_path, build_template_metadata(template, format_type="directory"))
    except OSError as exc:
        raise TemplateStoreError(f"保存目录模板失败: {resolved_dir}") from exc

    return resolved_dir


def load_template_directory(path: Path) -> FingerprintTemplate:
    """从目录格式加载模板。"""

    resolved_path = path.expanduser().resolve()
    descriptors_path = resolved_path / "descriptors.npy"
    keypoints_path = resolved_path / "keypoints.npz"
    metadata_path = resolved_path / "metadata.json"

    metadata = _read_metadata_json(metadata_path)
    try:
        descriptors = np.load(descriptors_path, allow_pickle=False)
        keypoints_data = np.load(keypoints_path, allow_pickle=False)
    except OSError as exc:
        raise TemplateStoreError(f"读取目录模板失败: {resolved_path}") from exc

    arrays = {
        "image_shape": np.asarray(metadata["image_shape"], dtype=np.int32),
        "roi_bbox": np.asarray(metadata["roi_bbox"], dtype=np.int32),
        "keypoints_x": keypoints_data["keypoints_x"],
        "keypoints_y": keypoints_data["keypoints_y"],
        "keypoints_size": keypoints_data["keypoints_size"],
        "keypoints_angle": keypoints_data["keypoints_angle"],
        "keypoints_response": keypoints_data["keypoints_response"],
        "keypoints_octave": keypoints_data["keypoints_octave"],
        "keypoints_class_id": keypoints_data["keypoints_class_id"],
        "descriptors": descriptors,
        "has_descriptors": np.asarray([metadata["has_descriptors"]], dtype=np.uint8),
    }
    return _build_template_from_arrays_and_metadata(arrays, metadata)


def save_template_npz(template: FingerprintTemplate, save_path: Path) -> Path:
    """兼容旧调用方式的保存接口。"""

    return save_template_npz_json(template, save_path)


def load_template_npz(path: Path) -> FingerprintTemplate:
    """兼容旧调用方式的加载接口。"""

    return load_template_npz_json(path)


def build_template_metadata(
    template: FingerprintTemplate,
    format_type: str,
) -> dict[str, Any]:
    """构建可写入 JSON 的模板元信息。"""

    return {
        "template_version": template.template_version,
        "image_id": template.image_id,
        "sample_id": template.image_id,
        "image_path": template.source_path,
        "source_path": template.source_path,
        "image_shape": list(template.image_shape),
        "roi_bbox": list(template.roi_bbox),
        "preprocess_metadata": template.preprocess_metadata,
        "timestamp": template.timestamp,
        "config_snapshot": template.config_snapshot,
        "descriptor_dtype": template.descriptor_dtype,
        "has_descriptors": template.descriptors is not None,
        "storage_format": format_type,
    }


def generate_timestamp() -> str:
    """生成 ISO 8601 UTC 时间戳。"""

    return datetime.now(timezone.utc).isoformat()


def _build_template_from_arrays_and_metadata(
    arrays: Any,
    metadata: dict[str, Any],
) -> FingerprintTemplate:
    """根据数组数据和元信息重建模板对象。"""

    has_descriptors = bool(int(np.asarray(arrays["has_descriptors"]).reshape(-1)[0]))
    descriptors = arrays["descriptors"] if has_descriptors else None

    return FingerprintTemplate(
        template_version=str(metadata["template_version"]),
        image_id=str(metadata["image_id"]),
        source_path=str(metadata.get("source_path") or metadata.get("image_path") or ""),
        image_shape=tuple(int(value) for value in arrays["image_shape"]),
        roi_bbox=tuple(int(value) for value in arrays["roi_bbox"]),
        preprocess_metadata=dict(metadata.get("preprocess_metadata", {})),
        timestamp=str(metadata["timestamp"]),
        keypoints_x=arrays["keypoints_x"],
        keypoints_y=arrays["keypoints_y"],
        keypoints_size=arrays["keypoints_size"],
        keypoints_angle=arrays["keypoints_angle"],
        keypoints_response=arrays["keypoints_response"],
        keypoints_octave=arrays["keypoints_octave"],
        keypoints_class_id=arrays["keypoints_class_id"],
        descriptors=descriptors,
        descriptor_dtype=str(metadata.get("descriptor_dtype", "")),
        config_snapshot=dict(metadata.get("config_snapshot", {})),
    )


def _write_metadata_json(path: Path, metadata: dict[str, Any]) -> None:
    """把模板元信息写入 JSON 文件。"""

    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def _read_metadata_json(path: Path) -> dict[str, Any]:
    """从 JSON 文件中读取模板元信息。"""

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except OSError as exc:
        raise TemplateStoreError(f"读取模板元信息失败: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TemplateStoreError(f"模板元信息 JSON 非法: {path}") from exc


def _descriptor_array(descriptors: np.ndarray | None) -> np.ndarray:
    """把描述子转换为适合保存的数组。"""

    if descriptors is None:
        return np.empty((0, 0), dtype=np.float32)
    return descriptors
