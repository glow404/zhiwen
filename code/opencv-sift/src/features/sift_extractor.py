"""SIFT 特征提取与序列化模块。

这个文件负责：
- 创建 OpenCV SIFT 提取器
- 在预处理后的指纹图像上提取关键点与描述子
- 把关键点转换为可保存的结构化数据
- 保存 / 加载特征文件
- 绘制关键点可视化图

输入：
- 预处理后的灰度图
- 可选 ROI mask
- SIFT 参数配置

输出：
- `FeatureResult`
- 关键点可视化图
- 可保存到磁盘的特征文件
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config.schema import SIFTConfig
from src.exceptions import FeatureExtractionError
from src.io.validators import validate_descriptors
from src.types import FeatureResult


def create_sift(config: SIFTConfig) -> Any:
    """根据配置创建一个 OpenCV SIFT 提取器。

    输入：
    - `config`：SIFT 配置对象

    输出：
    - OpenCV SIFT 提取器实例
    """

    if not hasattr(cv2, "SIFT_create"):
        raise FeatureExtractionError("当前 OpenCV 构建不支持 SIFT_create。")

    descriptor_type = resolve_descriptor_type(config.descriptor_type)
    kwargs = dict(
        nfeatures=config.nfeatures,
        nOctaveLayers=config.n_octave_layers,
        contrastThreshold=config.contrast_threshold,
        edgeThreshold=config.edge_threshold,
        sigma=config.sigma,
    )

    if descriptor_type is None:
        return cv2.SIFT_create(**kwargs)

    try:
        return cv2.SIFT_create(**kwargs, descriptorType=descriptor_type)
    except TypeError as exc:
        raise FeatureExtractionError(
            "当前 OpenCV Python 接口不支持 descriptorType 参数。"
        ) from exc


def resolve_descriptor_type(descriptor_type: str) -> int | None:
    """把配置中的描述子类型字符串映射为 OpenCV 常量。"""

    normalized = descriptor_type.lower()
    if normalized == "default":
        return None
    if normalized == "float32":
        return int(cv2.CV_32F)
    if normalized == "uint8":
        return int(cv2.CV_8U)
    raise FeatureExtractionError(f"不支持的 descriptor_type: {descriptor_type}")


def extract_features(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    config: SIFTConfig | None = None,
) -> FeatureResult:
    """在单张图像上检测 SIFT 关键点并计算描述子。

    输入：
    - `image`：待处理灰度图
    - `mask`：可选掩膜，限制特征检测区域
    - `config`：SIFT 参数；为空时使用默认配置

    输出：
    - `FeatureResult`
    """

    prepared_mask = prepare_feature_mask(image, mask)
    sift_config = config or SIFTConfig()
    sift = create_sift(sift_config)
    keypoints, descriptors = sift.detectAndCompute(image, prepared_mask)
    validate_descriptors(descriptors)

    serialized_keypoints = serialize_keypoints(keypoints or [])
    descriptor_dim = 0 if descriptors is None else int(descriptors.shape[1])
    descriptor_dtype = "" if descriptors is None else str(descriptors.dtype)
    return FeatureResult(
        keypoints=keypoints or [],
        descriptors=descriptors,
        keypoint_count=len(keypoints or []),
        descriptor_dim=descriptor_dim,
        descriptor_dtype=descriptor_dtype,
        keypoints_data=serialized_keypoints,
    )


def extract_sift_features(
    image: np.ndarray,
    roi_mask: np.ndarray | None,
    config: SIFTConfig,
) -> FeatureResult:
    """兼容旧调用方式的别名函数。"""

    return extract_features(image=image, mask=roi_mask, config=config)


def serialize_keypoints(keypoints: list[Any]) -> dict[str, np.ndarray]:
    """把 OpenCV 关键点列表转换为可保存的结构化数组。"""

    if not keypoints:
        empty_float = np.empty((0,), dtype=np.float32)
        empty_int = np.empty((0,), dtype=np.int32)
        return {
            "x": empty_float,
            "y": empty_float,
            "size": empty_float,
            "angle": empty_float,
            "response": empty_float,
            "octave": empty_int,
            "class_id": empty_int,
        }

    return {
        "x": np.asarray([keypoint.pt[0] for keypoint in keypoints], dtype=np.float32),
        "y": np.asarray([keypoint.pt[1] for keypoint in keypoints], dtype=np.float32),
        "size": np.asarray([keypoint.size for keypoint in keypoints], dtype=np.float32),
        "angle": np.asarray([keypoint.angle for keypoint in keypoints], dtype=np.float32),
        "response": np.asarray([keypoint.response for keypoint in keypoints], dtype=np.float32),
        "octave": np.asarray([keypoint.octave for keypoint in keypoints], dtype=np.int32),
        "class_id": np.asarray([keypoint.class_id for keypoint in keypoints], dtype=np.int32),
    }


def deserialize_keypoints(keypoints_data: dict[str, np.ndarray]) -> list[Any]:
    """把结构化关键点数组重建为 OpenCV KeyPoint 列表。"""

    point_count = len(keypoints_data["x"])
    keypoints: list[Any] = []
    for index in range(point_count):
        keypoints.append(
            cv2.KeyPoint(
                float(keypoints_data["x"][index]),
                float(keypoints_data["y"][index]),
                float(keypoints_data["size"][index]),
                float(keypoints_data["angle"][index]),
                float(keypoints_data["response"][index]),
                int(keypoints_data["octave"][index]),
                int(keypoints_data["class_id"][index]),
            )
        )
    return keypoints


def save_features(
    feature_result: FeatureResult,
    save_path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """把提取结果保存为可复用的 `.npz` 特征文件。"""

    resolved_path = save_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    keypoints_data = (
        feature_result.keypoints_data
        if feature_result.keypoints_data
        else serialize_keypoints(feature_result.keypoints)
    )
    descriptors = feature_result.descriptors
    descriptor_array = (
        descriptors
        if descriptors is not None
        else np.empty((0, 0), dtype=np.float32)
    )
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    try:
        np.savez_compressed(
            resolved_path,
            keypoints_x=keypoints_data["x"],
            keypoints_y=keypoints_data["y"],
            keypoints_size=keypoints_data["size"],
            keypoints_angle=keypoints_data["angle"],
            keypoints_response=keypoints_data["response"],
            keypoints_octave=keypoints_data["octave"],
            keypoints_class_id=keypoints_data["class_id"],
            descriptors=descriptor_array,
            has_descriptors=np.asarray([descriptors is not None], dtype=np.uint8),
            descriptor_dim=np.asarray([feature_result.descriptor_dim], dtype=np.int32),
            descriptor_dtype=np.asarray([feature_result.descriptor_dtype], dtype="<U32"),
            metadata_json=np.asarray([metadata_json], dtype="<U4096"),
        )
    except OSError as exc:
        raise FeatureExtractionError(f"保存特征文件失败: {resolved_path}") from exc
    return resolved_path


def load_features(save_path: Path) -> tuple[FeatureResult, dict[str, Any]]:
    """从 `.npz` 特征文件中加载关键点、描述子和附加元数据。"""

    resolved_path = save_path.expanduser().resolve()
    try:
        data = np.load(resolved_path, allow_pickle=False)
    except OSError as exc:
        raise FeatureExtractionError(f"读取特征文件失败: {resolved_path}") from exc

    keypoints_data = {
        "x": data["keypoints_x"],
        "y": data["keypoints_y"],
        "size": data["keypoints_size"],
        "angle": data["keypoints_angle"],
        "response": data["keypoints_response"],
        "octave": data["keypoints_octave"],
        "class_id": data["keypoints_class_id"],
    }
    keypoints = deserialize_keypoints(keypoints_data)
    has_descriptors = bool(int(data["has_descriptors"][0]))
    descriptors = data["descriptors"] if has_descriptors else None
    descriptor_dim = int(data["descriptor_dim"][0])
    descriptor_dtype = str(data["descriptor_dtype"][0])
    metadata = json.loads(str(data["metadata_json"][0]))

    return (
        FeatureResult(
            keypoints=keypoints,
            descriptors=descriptors,
            keypoint_count=len(keypoints),
            descriptor_dim=descriptor_dim,
            descriptor_dtype=descriptor_dtype,
            keypoints_data=keypoints_data,
        ),
        metadata,
    )


def draw_keypoints(
    image: np.ndarray,
    keypoints: list[Any],
    draw_rich_keypoints: bool = True,
) -> np.ndarray:
    """在灰度图上绘制关键点，返回可保存的 BGR 图像。"""

    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_rich_keypoints else 0
    return cv2.drawKeypoints(
        image=image,
        keypoints=keypoints,
        outImage=None,
        color=(0, 255, 0),
        flags=flags,
    )


def prepare_feature_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
    """校验并规范化用于特征提取的掩膜。"""

    if mask is None:
        return None
    if mask.shape[:2] != image.shape[:2]:
        raise FeatureExtractionError("特征提取 mask 与输入图像尺寸不一致。")
    if mask.dtype == np.uint8:
        return mask
    return np.clip(mask, 0, 255).astype(np.uint8)
