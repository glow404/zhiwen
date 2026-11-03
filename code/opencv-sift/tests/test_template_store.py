"""指纹模板存储与加载模块测试。

这个文件验证：
- 模板能以 `npz + json` 格式保存和加载
- 模板能以目录格式保存和加载
- 模板中包含预处理元数据、时间戳和配置快照
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.io.template_store import build_template, load_template, save_template
from src.types import FeatureResult


def test_save_and_load_template_npz_json(tmp_path: Path) -> None:
    """验证 `npz + json` 模板格式能正确往返。"""

    template = _build_sample_template(tmp_path)
    template_path = tmp_path / "sample_template.npz"

    saved_path = save_template(template, template_path, format_type="npz_json")
    loaded = load_template(saved_path)

    assert saved_path.exists()
    assert saved_path.with_suffix(".json").exists()
    assert saved_path.with_suffix(".mnt").exists()
    assert loaded.image_id == template.image_id
    assert loaded.preprocess_metadata["foreground_ratio"] == 0.42
    assert loaded.timestamp == template.timestamp
    assert loaded.descriptors is not None
    assert loaded.descriptors.shape == (2, 128)


def test_save_template_mnt_format(tmp_path: Path) -> None:
    """验证 `.mnt` 文件格式与角度单位转换正确。"""

    template = _build_sample_template(tmp_path)
    template_path = tmp_path / "sample_template.npz"

    saved_path = save_template(template, template_path, format_type="npz_json")
    mnt_lines = saved_path.with_suffix(".mnt").read_text(encoding="utf-8").splitlines()

    assert mnt_lines == [
        "sample_pair0",
        "2 128 128",
        "10 20 0.785398",
        "30 40 1.570796",
    ]


def test_save_and_load_template_directory(tmp_path: Path) -> None:
    """验证目录模板格式能正确往返。"""

    template = _build_sample_template(tmp_path)
    template_dir = tmp_path / "template_dir"

    saved_dir = save_template(template, template_dir, format_type="directory")
    loaded = load_template(saved_dir)

    assert saved_dir.is_dir()
    assert (saved_dir / "metadata.json").exists()
    assert (saved_dir / "descriptors.npy").exists()
    assert (saved_dir / "keypoints.npz").exists()
    assert loaded.image_shape == template.image_shape
    assert loaded.config_snapshot["sift"]["nfeatures"] == 64
    assert loaded.keypoints_class_id.shape == (2,)


def _build_sample_template(tmp_path: Path):
    """构造一个测试用模板对象。"""

    keypoints = [
        cv2.KeyPoint(10.0, 20.0, 3.0, 45.0, 0.5, 1, 2),
        cv2.KeyPoint(30.0, 40.0, 5.0, 90.0, 0.8, 2, 3),
    ]
    keypoints_data = {
        "x": np.asarray([10.0, 30.0], dtype=np.float32),
        "y": np.asarray([20.0, 40.0], dtype=np.float32),
        "size": np.asarray([3.0, 5.0], dtype=np.float32),
        "angle": np.asarray([45.0, 90.0], dtype=np.float32),
        "response": np.asarray([0.5, 0.8], dtype=np.float32),
        "octave": np.asarray([1, 2], dtype=np.int32),
        "class_id": np.asarray([2, 3], dtype=np.int32),
    }
    feature_result = FeatureResult(
        keypoints=keypoints,
        descriptors=np.ones((2, 128), dtype=np.float32),
        keypoint_count=2,
        descriptor_dim=128,
        descriptor_dtype="float32",
        keypoints_data=keypoints_data,
    )

    return build_template(
        image_path=tmp_path / "sample_pair0.png",
        image_shape=(128, 128),
        roi_bbox=(10, 10, 100, 100),
        feature_result=feature_result,
        template_version="v0.1",
        preprocess_metadata={
            "foreground_ratio": 0.42,
            "roi_bbox": [10, 10, 100, 100],
        },
        timestamp="2026-04-08T00:00:00+00:00",
        config_snapshot={
            "sift": {
                "nfeatures": 64,
                "contrast_threshold": 0.04,
            }
        },
    )
