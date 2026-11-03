import numpy as np

def compute_coherence(oimg, N=2):
    """
    计算方向图的局部相干性（coherence）。
    输入:
        oimg - 2D array (方向图，弧度)
        N    - 局部窗口半径（默认 2，对应 (2N+1)x(2N+1) 窗口）
    输出:
        cimg - 2D numpy array，与 oimg 大小相同，值域 [0,1]
    """
    oimg = np.asarray(oimg, dtype=float)
    if oimg.ndim != 2:
        raise ValueError("oimg must be a 2D array")

    h, w = oimg.shape

    # pad rows: top flip of first N rows, bottom flip of last N rows
    top = np.flipud(oimg[:N, :])
    bottom = np.flipud(oimg[h - N:h, :])
    o_p = np.vstack((top, oimg, bottom))

    # pad cols: left flip of first N cols, right flip of last N cols (operate on the padded rows)
    left = np.fliplr(o_p[:, :N])
    right = np.fliplr(o_p[:, w - N:w])
    o_p = np.hstack((left, o_p, right))

    cimg = np.zeros((h, w), dtype=float)
    k = 2 * N + 1
    for i in range(N, N + h):
        for j in range(N, N + w):
            th = o_p[i, j]
            blk = o_p[i - N:i + N + 1, j - N:j + N + 1]
            cimg[i - N, j - N] = np.mean(np.abs(np.cos(blk - th)))

    return cimg

# 示例:
# import numpy as np
# o = np.random.rand(100,100)*np.pi  # 随机方向图（0..pi 亦可）
# c = compute_coherence(o)