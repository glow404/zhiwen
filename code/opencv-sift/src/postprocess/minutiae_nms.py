"""基于极大值抑制的细节点筛选工具。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import cv2
import numpy as np

from src.io.image_reader import load_image
from src.io.output_writer import save_image
from src.io.template_store import load_template, save_template
from src.types import FingerprintTemplate


@dataclass(slots=True)
class MNTRecord:
    """表示单个 `.mnt` 文件中的细节点记录。"""

    image_id: str
    point_count: int
    image_height: int
    image_width: int
    points: np.ndarray


def load_mnt(path: Path) -> MNTRecord:
    """读取 `.mnt` 文件并解析为结构化记录。"""

    resolved_path = path.expanduser().resolve()
    lines = [line.strip() for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"MNT 文件格式不完整: {resolved_path}")

    image_id = lines[0]
    header = lines[1].split()
    if len(header) != 3:
        raise ValueError(f"MNT 头部格式非法: {resolved_path}")

    point_count, image_height, image_width = (int(value) for value in header)
    raw_points: list[tuple[float, float, float]] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"MNT 点位格式非法: {resolved_path}")
        raw_points.append((float(parts[0]), float(parts[1]), float(parts[2])))

    points = np.asarray(raw_points, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float64)
    if len(points) != point_count:
        raise ValueError(
            f"MNT 头部点数与实际点数不一致: header={point_count}, actual={len(points)}, path={resolved_path}"
        )

    return MNTRecord(
        image_id=image_id,
        point_count=point_count,
        image_height=image_height,
        image_width=image_width,
        points=points,
    )


def save_mnt(record: MNTRecord, output_path: Path) -> Path:
    """把筛选后的细节点重新保存为 `.mnt` 文件。"""

    resolved_path = output_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        record.image_id,
        f"{len(record.points)} {record.image_height} {record.image_width}",
    ]
    lines.extend(
        f"{int(round(point[0]))} {int(round(point[1]))} {float(point[2]):.6f}"
        for point in record.points
    )
    resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return resolved_path


def build_visualization_output_path(output_path: Path) -> Path:
    """为 NMS 后的 `.mnt` 构建配套可视化图路径。"""

    resolved_path = output_path.expanduser().resolve()
    return resolved_path.with_name(f"{resolved_path.stem}_vis.png")


def draw_mnt_visualization(
    record: MNTRecord,
    background_image: np.ndarray | None = None,
    direction_length: int = 10,
) -> np.ndarray:
    """把 `.mnt` 点位及其方向绘制成一张 BGR 可视化图。"""

    if background_image is None:
        canvas = np.zeros((record.image_height, record.image_width, 3), dtype=np.uint8)
    elif background_image.ndim == 2:
        canvas = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = background_image.copy()

    for point in record.points:
        x_coord, y_coord, orientation = (float(value) for value in point)
        center = (int(round(x_coord)), int(round(y_coord)))
        end_point = (
            int(round(x_coord + direction_length * math.cos(orientation))),
            int(round(y_coord + direction_length * math.sin(orientation))),
        )
        cv2.circle(canvas, center, 2, (0, 255, 0), thickness=-1)
        cv2.arrowedLine(
            canvas,
            center,
            end_point,
            (0, 0, 255),
            thickness=1,
            line_type=cv2.LINE_AA,
            tipLength=0.35,
        )
    return canvas


def try_load_visualization_background(mnt_path: Path, record: MNTRecord) -> np.ndarray | None:
    """尝试从同名元信息中定位原图，作为 `.mnt` 可视化背景。"""

    metadata_path = mnt_path.expanduser().resolve().with_suffix(".json")
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    source_path = metadata.get("source_path") or metadata.get("image_path")
    if not source_path:
        return None

    try:
        background = load_image(Path(str(source_path)))
    except Exception:
        return None

    expected_shape = (record.image_height, record.image_width)
    if background.shape[:2] != expected_shape:
        background = cv2.resize(
            background,
            (record.image_width, record.image_height),
            interpolation=cv2.INTER_LINEAR,
        )
    return background


def save_mnt_visualization(
    input_path: Path,
    output_path: Path,
    record: MNTRecord,
) -> Path:
    """为筛选后的 `.mnt` 生成带方向箭头的可视化图。"""

    visualization_path = build_visualization_output_path(output_path)
    background = try_load_visualization_background(input_path, record)
    visualization = draw_mnt_visualization(record, background_image=background)
    return save_image(visualization, visualization_path)


def discover_npz_files(path: Path) -> list[Path]:
    """发现输入路径下的全部模板 `.npz` 文件。"""

    resolved_path = path.expanduser().resolve()
    if resolved_path.is_file():
        if resolved_path.suffix.lower() != ".npz":
            raise ValueError(f"输入文件不是 .npz: {resolved_path}")
        return [resolved_path]
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {resolved_path}")
    return sorted(
        item
        for item in resolved_path.rglob("*.npz")
        if item.is_file() and item.name != "keypoints.npz"
    )


def build_default_output_npz_path(input_path: Path, input_root: Path) -> Path:
    """为输入 `.npz` 文件或目录构建默认输出路径。"""

    resolved_input = input_path.expanduser().resolve()
    resolved_root = input_root.expanduser().resolve()
    if resolved_root.is_file():
        return resolved_input.with_name(f"{resolved_input.stem}_nms.npz")

    output_root = resolved_root.parent / f"{resolved_root.name}_nms"
    relative_path = resolved_input.relative_to(resolved_root)
    return output_root / relative_path


def build_template_points(template: FingerprintTemplate) -> np.ndarray:
    """从模板中提取供 NMS 使用的 `(x, y, angle_rad)` 数组。"""

    point_count = int(len(template.keypoints_x))
    if point_count == 0:
        return np.empty((0, 3), dtype=np.float64)

    return np.column_stack(
        [
            np.asarray(template.keypoints_x, dtype=np.float64).reshape(-1),
            np.asarray(template.keypoints_y, dtype=np.float64).reshape(-1),
            np.deg2rad(np.asarray(template.keypoints_angle, dtype=np.float64).reshape(-1)),
        ]
    )


def build_filtered_template(
    template: FingerprintTemplate,
    kept_indices: np.ndarray,
) -> FingerprintTemplate:
    """根据保留索引重建筛选后的模板。"""

    descriptors = template.descriptors
    if descriptors is not None:
        if len(descriptors) != len(template.keypoints_x):
            raise ValueError("模板描述子数量与关键点数量不一致。")
        descriptors = descriptors[kept_indices]

    return replace(
        template,
        keypoints_x=template.keypoints_x[kept_indices],
        keypoints_y=template.keypoints_y[kept_indices],
        keypoints_size=template.keypoints_size[kept_indices],
        keypoints_angle=template.keypoints_angle[kept_indices],
        keypoints_response=template.keypoints_response[kept_indices],
        keypoints_octave=template.keypoints_octave[kept_indices],
        keypoints_class_id=template.keypoints_class_id[kept_indices],
        descriptors=descriptors,
    )


def draw_template_visualization(
    template: FingerprintTemplate,
    background_image: np.ndarray | None = None,
    direction_length: int = 10,
) -> np.ndarray:
    """把模板关键点及其方向绘制成一张 BGR 可视化图。"""

    image_height, image_width = (int(value) for value in template.image_shape)
    if background_image is None:
        canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    elif background_image.ndim == 2:
        canvas = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = background_image.copy()

    x0, y0, x1, y1 = (int(value) for value in template.roi_bbox)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 0, 0), 1)

    for x_coord, y_coord, angle_deg in zip(
        template.keypoints_x,
        template.keypoints_y,
        template.keypoints_angle,
        strict=False,
    ):
        x_value = float(x_coord)
        y_value = float(y_coord)
        angle_rad = math.radians(float(angle_deg))
        center = (int(round(x_value)), int(round(y_value)))
        end_point = (
            int(round(x_value + direction_length * math.cos(angle_rad))),
            int(round(y_value + direction_length * math.sin(angle_rad))),
        )
        cv2.circle(canvas, center, 2, (0, 255, 0), thickness=-1)
        cv2.arrowedLine(
            canvas,
            center,
            end_point,
            (0, 0, 255),
            thickness=1,
            line_type=cv2.LINE_AA,
            tipLength=0.35,
        )
    return canvas


def try_load_template_background(template: FingerprintTemplate) -> np.ndarray | None:
    """尝试从模板记录的原图路径读取可视化背景。"""

    if not template.source_path:
        return None

    try:
        background = load_image(Path(template.source_path))
    except Exception:
        return None

    expected_shape = tuple(int(value) for value in template.image_shape)
    if background.shape[:2] != expected_shape:
        background = cv2.resize(
            background,
            (expected_shape[1], expected_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return background


def save_template_visualization(
    template: FingerprintTemplate,
    output_path: Path,
) -> Path:
    """为筛选后的模板生成基于 `.npz` 的关键点可视化图。"""

    visualization_path = build_visualization_output_path(output_path)
    background = try_load_template_background(template)
    visualization = draw_template_visualization(template, background_image=background)
    return save_image(visualization, visualization_path)


def filter_npz(
    input_path: Path,
    output_path: Path,
    radius: float = 8.0,
    angle_threshold_deg: float = 20.0,
    suppress_same_orientation_only: bool = True,
) -> tuple[Path, Path, Path, int, int]:
    """读取一个 `.npz` 模板并输出经过 NMS 筛选的新模板与可视化图。"""

    template = load_template(input_path)
    points = build_template_points(template)
    scores = np.asarray(template.keypoints_response, dtype=np.float64).reshape(-1)
    kept_indices = apply_minutiae_nms(
        points=points,
        scores=scores,
        radius=radius,
        angle_threshold_deg=angle_threshold_deg,
        suppress_same_orientation_only=suppress_same_orientation_only,
    )
    filtered_template = build_filtered_template(template, kept_indices)
    saved_path = save_template(filtered_template, output_path, format_type="npz_json")
    mnt_path = saved_path.with_suffix(".mnt")
    visualization_path = save_template_visualization(filtered_template, saved_path)
    return saved_path, mnt_path, visualization_path, len(template.keypoints_x), len(kept_indices)


def try_load_response_scores(mnt_path: Path) -> np.ndarray | None:
    """尝试读取同名 `.npz` 中的关键点响应值，作为 NMS 分数。"""

    npz_path = mnt_path.expanduser().resolve().with_suffix(".npz")
    if not npz_path.exists():
        return None

    with np.load(npz_path, allow_pickle=False) as data:
        if "keypoints_response" not in data:
            return None
        responses = np.asarray(data["keypoints_response"], dtype=np.float64).reshape(-1)
    return responses


def apply_minutiae_nms(
    points: np.ndarray,
    scores: np.ndarray | None,
    radius: float,
    angle_threshold_deg: float,
    suppress_same_orientation_only: bool = True,
) -> np.ndarray:
    """对细节点执行基于空间距离和方向约束的 NMS。"""

    if len(points) == 0:
        return np.empty((0,), dtype=np.int32)

    if scores is not None and len(scores) != len(points):
        raise ValueError("NMS 分数长度与细节点数量不一致。")

    if scores is None:
        order = np.arange(len(points), dtype=np.int32)
    else:
        order = np.argsort(-scores, kind="stable")

    radius_sq = float(radius) * float(radius)
    angle_threshold_rad = math.radians(angle_threshold_deg)
    kept_indices: list[int] = []

    for candidate_idx in order.tolist():
        candidate = points[candidate_idx]
        suppressed = False
        for kept_idx in kept_indices:
            kept_point = points[kept_idx]
            dx = float(candidate[0] - kept_point[0])
            dy = float(candidate[1] - kept_point[1])
            if dx * dx + dy * dy > radius_sq:
                continue

            if suppress_same_orientation_only:
                angle_diff = _angular_distance_rad(float(candidate[2]), float(kept_point[2]))
                if angle_diff > angle_threshold_rad:
                    continue

            suppressed = True
            break

        if not suppressed:
            kept_indices.append(candidate_idx)

    kept_indices.sort()
    return np.asarray(kept_indices, dtype=np.int32)


def filter_mnt(
    input_path: Path,
    output_path: Path,
    radius: float = 8.0,
    angle_threshold_deg: float = 20.0,
    suppress_same_orientation_only: bool = True,
    require_npz_scores: bool = False,
) -> tuple[Path, Path, int, int]:
    """读取一个 `.mnt` 并输出经过 NMS 筛选的新 `.mnt`。"""

    record = load_mnt(input_path)
    scores = try_load_response_scores(input_path)
    if require_npz_scores and scores is None:
        raise FileNotFoundError(f"未找到同名 NPZ 响应值文件: {input_path.with_suffix('.npz')}")

    kept_indices = apply_minutiae_nms(
        points=record.points,
        scores=scores,
        radius=radius,
        angle_threshold_deg=angle_threshold_deg,
        suppress_same_orientation_only=suppress_same_orientation_only,
    )
    filtered_record = MNTRecord(
        image_id=record.image_id,
        point_count=len(kept_indices),
        image_height=record.image_height,
        image_width=record.image_width,
        points=record.points[kept_indices],
    )
    saved_path = save_mnt(filtered_record, output_path)
    visualization_path = save_mnt_visualization(input_path, saved_path, filtered_record)
    return saved_path, visualization_path, len(record.points), len(filtered_record.points)


def discover_mnt_files(path: Path) -> list[Path]:
    """发现输入路径下的全部 `.mnt` 文件。"""

    resolved_path = path.expanduser().resolve()
    if resolved_path.is_file():
        if resolved_path.suffix.lower() != ".mnt":
            raise ValueError(f"输入文件不是 .mnt: {resolved_path}")
        return [resolved_path]
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {resolved_path}")
    return sorted(item for item in resolved_path.rglob("*.mnt") if item.is_file())


def build_default_output_path(input_path: Path, input_root: Path) -> Path:
    """为输入文件或目录构建默认输出路径。"""

    resolved_input = input_path.expanduser().resolve()
    resolved_root = input_root.expanduser().resolve()
    if resolved_root.is_file():
        return resolved_input.with_name(f"{resolved_input.stem}_nms.mnt")

    output_root = resolved_root.parent / f"{resolved_root.name}_nms"
    relative_path = resolved_input.relative_to(resolved_root)
    return output_root / relative_path


def _angular_distance_rad(angle_a: float, angle_b: float) -> float:
    """计算两个弧度角的最小环形角距离。"""

    diff = abs(angle_a - angle_b) % (2.0 * math.pi)
    return min(diff, 2.0 * math.pi - diff)
