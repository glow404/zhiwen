import numpy as np
from typing import Tuple, Any

def p(img: np.ndarray, x: int, y: int, i: int) -> Tuple[Any, int, int]:
    """
    返回邻域像素值及其坐标。

    参数:
        img - 2D numpy array (灰度) 或 2D-like 可索引数组，索引格式 img[row, col]
        x, y - 中心像素的列、行坐标，使用 0-based 索引
        i - 邻域位置编号:
            4 | 3 | 2
            5 |   | 1
            6 | 7 | 8
          1 和 9 等价

    返回:
        j - 邻域像素值（越界时返回 0）
        X - 邻域像素的列坐标（0-based，已裁剪到图像范围）
        Y - 邻域像素的行坐标（0-based，已裁剪到图像范围）
    """
    img = np.asarray(img)
    h, w = img.shape[:2]

    # 目标偏移 (dx, dy): 列偏移, 行偏移
    offsets = {
        1: (1, 0),
        2: (1, -1),
        3: (0, -1),
        4: (-1, -1),
        5: (-1, 0),
        6: (-1, 1),
        7: (0, 1),
        8: (1, 1),
        9: (1, 0),
    }
    if i not in offsets:
        raise ValueError(f"i must be in 1..9, got {i}")

    dx, dy = offsets[i]
    X = x + dx
    Y = y + dy

    # 裁剪到图像范围
    if X < 0 or X >= w or Y < 0 or Y >= h:
        # 越界时返回 0 值并将坐标裁剪到合法范围
        X_clamped = min(max(X, 0), w - 1)
        Y_clamped = min(max(Y, 0), h - 1)
        return 0, X_clamped, Y_clamped

    # 取值（若多通道，返回整像素向量）
    val = img[Y, X]
    return val, X, Y

# 示例:
# import numpy as np
# im = np.arange(25).reshape(5,5)
# j, X, Y = p(im, 2, 2, 1)  # 取 (x+1,y)
# print(j, X, Y)