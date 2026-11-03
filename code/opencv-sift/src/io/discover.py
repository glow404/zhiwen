"""目录扫描模块。

这个文件负责从一个或多个输入目录中递归或非递归地发现图像文件。
当前主要为后续批处理流程做准备。

输入：
- 输入目录列表
- 是否递归扫描
- 文件后缀白名单

输出：
- 满足条件的图像路径列表
"""

from __future__ import annotations

from pathlib import Path


def discover_images(
    input_roots: list[Path],
    recursive: bool,
    patterns: list[str],
) -> list[Path]:
    """仅使用 `pathlib` 从目录中发现图像文件。"""

    normalized_patterns = {pattern.lower() for pattern in patterns}
    discovered: list[Path] = []

    for root in input_roots:
        resolved_root = root.expanduser().resolve()
        iterator = resolved_root.rglob("*") if recursive else resolved_root.glob("*")
        for path in iterator:
            if path.is_file() and path.suffix.lower() in normalized_patterns:
                discovered.append(path)

    return sorted(set(discovered))
