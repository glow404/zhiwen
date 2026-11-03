"""细节点 NMS 后处理测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.io.template_store import load_template, save_template
from src.postprocess.minutiae_nms import filter_npz
from src.types import FingerprintTemplate


def test_filter_npz_uses_response_scores_and_keeps_orientation_distinct_points(tmp_path: Path) -> None:
    """验证 NPZ 模板 NMS 会优先保留高响应点，并输出派生文件。"""

    template = FingerprintTemplate(
        template_version="v0.1",
        image_id="sample.303782_pair0",
        source_path="",
        image_shape=(128, 96),
        roi_bbox=(0, 0, 96, 128),
        preprocess_metadata={},
        timestamp="2026-01-01T00:00:00+00:00",
        keypoints_x=np.asarray([10.0, 12.0, 12.0, 50.0], dtype=np.float32),
        keypoints_y=np.asarray([10.0, 11.0, 11.0, 60.0], dtype=np.float32),
        keypoints_size=np.asarray([3.0, 3.0, 3.0, 3.0], dtype=np.float32),
        keypoints_angle=np.rad2deg(np.asarray([0.0, 0.05, 1.570796, 0.1], dtype=np.float32)),
        keypoints_response=np.asarray([0.9, 0.4, 0.8, 0.3], dtype=np.float32),
        keypoints_octave=np.asarray([0, 0, 0, 0], dtype=np.int32),
        keypoints_class_id=np.asarray([0, 0, 0, 0], dtype=np.int32),
        descriptors=np.arange(4 * 128, dtype=np.float32).reshape(4, 128),
        descriptor_dtype="float32",
        config_snapshot={},
    )
    npz_path = save_template(template, tmp_path / "sample.303782_pair0.npz", format_type="npz_json")

    output_path = tmp_path / "filtered.npz"
    saved_npz_path, saved_mnt_path, visualization_path, before_count, after_count = filter_npz(
        input_path=npz_path,
        output_path=output_path,
        radius=4.0,
        angle_threshold_deg=20.0,
    )
    filtered = load_template(saved_npz_path)
    visualization = cv2.imread(str(visualization_path), cv2.IMREAD_COLOR)

    assert before_count == 4
    assert after_count == 3
    assert saved_npz_path.exists()
    assert saved_mnt_path.exists()
    assert visualization_path.exists()
    assert visualization is not None
    assert visualization.shape == (128, 96, 3)
    assert int(np.count_nonzero(visualization)) > 0
    assert filtered.image_id == "sample.303782_pair0"
    assert filtered.keypoints_x.tolist() == [10.0, 12.0, 50.0]
    assert filtered.keypoints_y.tolist() == [10.0, 11.0, 60.0]
    assert filtered.descriptors.shape == (3, 128)
    assert filtered.keypoints_response.tolist() == [0.9, 0.8, 0.3]
    assert np.allclose(filtered.keypoints_angle, np.rad2deg([0.0, 1.570796, 0.1]), atol=1e-4)
    assert filtered.keypoints_octave.tolist() == [0, 0, 0]
    assert filtered.keypoints_class_id.tolist() == [0, 0, 0]
    assert filtered.roi_bbox == (0, 0, 96, 128)
    assert filtered.image_shape == (128, 96)
    assert np.allclose(filtered.descriptors[0], template.descriptors[0])
    assert np.allclose(filtered.descriptors[1], template.descriptors[2])
    assert np.allclose(filtered.descriptors[2], template.descriptors[3])
    assert saved_mnt_path.read_text(encoding="utf-8").splitlines()[:2] == [
        "sample.303782_pair0",
        "3 128 96",
    ]
