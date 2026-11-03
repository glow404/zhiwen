"""路径处理辅助模块。

这个文件统一提供和 `pathlib` 相关的常用小工具，避免在各模块中重复写路径处理逻辑。

输入：
- 用户提供的路径
- 目标文件路径
- 输入根目录与样本图路径

输出：
- 规范化后的绝对路径
- 已创建父目录的目标路径环境
- 适合复用到输出目录下的相对路径
"""

from __future__ import annotations

from pathlib import Path


def resolve_project_path(path_like: str | Path) -> Path:
    """把用户输入的路径解析为绝对 `Path` 对象。"""

    return Path(path_like).expanduser().resolve()


def build_image_id(image_path: Path) -> str:
    """基于原图文件名生成样本标识，保留完整主文件名。"""

    filename = image_path.name
    suffix = image_path.suffix
    if suffix:
        return filename[: -len(suffix)]
    return filename


def ensure_parent_dir(file_path: Path) -> None:
    """在需要时创建目标文件的父目录。"""

    file_path.parent.mkdir(parents=True, exist_ok=True)


def build_relative_output_stem(image_path: Path, input_roots: list[Path]) -> Path:
    """基于输入根目录构建输出目录下使用的相对路径骨架。"""

    resolved_image_path = image_path.expanduser().resolve()
    resolved_roots = [root.expanduser().resolve() for root in input_roots]
    sorted_roots = sorted(resolved_roots, key=lambda path: len(path.parts), reverse=True)

    for root in sorted_roots:
        try:
            relative_path = resolved_image_path.relative_to(root)
        except ValueError:
            continue
        return relative_path.with_suffix("")

    return Path(build_image_id(resolved_image_path))
