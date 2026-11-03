"""对 `.npz` 模板文件执行 NMS 筛选。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.postprocess.minutiae_nms import (  # noqa: E402
    build_default_output_npz_path,
    discover_npz_files,
    filter_npz,
)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="对 NPZ 模板关键点执行基于极大值抑制的筛选。")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入的 .npz 文件或包含 .npz 文件的目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件或目录。未提供时，文件模式输出为 *_nms.npz，目录模式输出到同级 *_nms 目录。",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=6.0,
        help="NMS 抑制半径，单位为像素。",
    )
    parser.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=10.0,
        help="仅在方向差不超过该阈值时互相抑制，单位为度。",
    )
    parser.add_argument(
        "--ignore-angle",
        action="store_true",
        help="忽略方向差，只按空间距离执行抑制。",
    )
    return parser


def main() -> None:
    """按命令行参数执行 NPZ 模板 NMS 筛选。"""

    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    npz_files = discover_npz_files(input_path)
    if not npz_files:
        print("未找到任何 .npz 文件。")
        return

    total_before = 0
    total_after = 0
    for npz_path in npz_files:
        if args.output is None:
            output_path = build_default_output_npz_path(npz_path, input_path)
        else:
            resolved_output = args.output.expanduser().resolve()
            if input_path.is_file():
                output_path = resolved_output
            else:
                relative_path = npz_path.relative_to(input_path)
                output_path = resolved_output / relative_path

        saved_path, mnt_path, visualization_path, before_count, after_count = filter_npz(
            input_path=npz_path,
            output_path=output_path,
            radius=args.radius,
            angle_threshold_deg=args.angle_threshold_deg,
            suppress_same_orientation_only=not args.ignore_angle,
        )
        total_before += before_count
        total_after += after_count
        print(
            f"{npz_path} -> {saved_path} | mnt={mnt_path} | vis={visualization_path} | "
            f"{before_count} -> {after_count}"
        )

    print(f"Done: files={len(npz_files)}, keypoints={total_before} -> {total_after}")


if __name__ == "__main__":
    main()
