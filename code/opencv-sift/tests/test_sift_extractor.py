"""SIFT 特征提取模块的最小测试。

这个文件验证：
- 关键点结构化数据是否完整
- 特征保存 / 加载是否能处理 descriptors 为空的情况
- 关键点绘制函数是否返回可保存的彩色图像
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config.schema import SIFTConfig
from src.features.sift_extractor import (
    draw_keypoints,
    extract_features,
    load_features,
    save_features,
    serialize_keypoints,
)
from src.types import FeatureResult


def test_serialize_keypoints_contains_required_fields() -> None:
    """验证关键点结构化结果包含保存所需的全部字段。"""

    keypoints = [
        cv2.KeyPoint(10.0, 20.0, 3.0, 45.0, 0.5, 1, 2),
        cv2.KeyPoint(30.0, 40.0, 5.0, 90.0, 0.8, 2, 3),
    ]

    keypoints_data = serialize_keypoints(keypoints)

    assert set(keypoints_data) == {"x", "y", "size", "angle", "response", "octave", "class_id"}
    assert keypoints_data["x"].shape == (2,)
    assert keypoints_data["class_id"][1] == 3


def test_save_and_load_features_supports_none_descriptors(tmp_path: Path) -> None:
    """验证 descriptors 为 None 时也能正常保存和加载。"""

    keypoints = [cv2.KeyPoint(12.0, 18.0, 4.0, 0.0, 0.2, 0, -1)]
    feature_result = FeatureResult(
        keypoints=keypoints,
        descriptors=None,
        keypoint_count=1,
        descriptor_dim=0,
        descriptor_dtype="",
        keypoints_data=serialize_keypoints(keypoints),
    )

    feature_path = tmp_path / "features_none_desc.npz"
    save_features(feature_result, feature_path, metadata={"image_id": "sample"})
    loaded_feature_result, metadata = load_features(feature_path)

    assert loaded_feature_result.keypoint_count == 1
    assert loaded_feature_result.descriptors is None
    assert metadata["image_id"] == "sample"


def test_extract_features_returns_structured_keypoint_data() -> None:
    """验证提取结果中包含结构化关键点字段。"""

    image = _build_checkerboard_image()
    config = SIFTConfig(
        nfeatures=64,
        n_octave_layers=3,
        contrast_threshold=0.04,
        edge_threshold=10.0,
        sigma=1.6,
        descriptor_type="default",
    )

    feature_result = extract_features(image=image, mask=None, config=config)

    assert set(feature_result.keypoints_data) == {
        "x",
        "y",
        "size",
        "angle",
        "response",
        "octave",
        "class_id",
    }
    assert feature_result.keypoint_count == len(feature_result.keypoints_data["x"])


def test_draw_keypoints_returns_color_image() -> None:
    """验证关键点绘制结果为三通道图像。"""

    image = np.zeros((64, 64), dtype=np.uint8)
    keypoints = [cv2.KeyPoint(20.0, 20.0, 5.0)]

    visualization = draw_keypoints(image, keypoints, draw_rich_keypoints=True)

    assert visualization.ndim == 3
    assert visualization.shape[2] == 3


def _build_checkerboard_image() -> np.ndarray:
    """构造一张便于产生局部特征的棋盘格测试图。"""

    image = np.zeros((128, 128), dtype=np.uint8)
    step = 16
    for y in range(0, 128, step):
        for x in range(0, 128, step):
            if (x // step + y // step) % 2 == 0:
                image[y : y + step, x : x + step] = 220
            else:
                image[y : y + step, x : x + step] = 30
    return image
