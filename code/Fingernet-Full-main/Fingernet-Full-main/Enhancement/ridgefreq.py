import numpy as np

# 导入 freqest（同目录）
try:
    from .freqest import freqest
except Exception:
    from freqest import freqest

def ridgefreq(im, mask, orient, blksze, windsze, minWaveLength, maxWaveLength):
    """
    Python 移植的 ridgefreq.m

    参数:
        im            : 2D ndarray, 灰度图像
        mask          : 2D ndarray, 掩码（0/1 或 bool），与 im 同尺寸
        orient        : 2D ndarray, 方向场（弧度）
        blksze        : int, 块大小
        windsze       : int, 用于 freqest 的窗口长度（奇数）
        minWaveLength : float, 最小波长（像素）
        maxWaveLength : float, 最大波长（像素）

    返回:
        freq       : 2D ndarray (float), 每像素频率（无效或未估计处为0）
        medianfreq : float, 所有有效频率的中位数（无有效频率时返回0.0）
    """
    im = np.asarray(im, dtype=float)
    mask = np.asarray(mask)
    orient = np.asarray(orient, dtype=float)

    rows, cols = im.shape
    freq = np.zeros_like(im, dtype=float)

    # MATLAB 中循环 r = 1:blksze:rows-blksze (1-based)，等价的 0-based Python 范围:
    # range(0, rows-blksze+1, blksze)
    r_limit = rows - blksze + 1
    c_limit = cols - blksze + 1
    if r_limit <= 0 or c_limit <= 0:
        # 图像比一个块还小，返回全零
        medianfreq = 0.0
        return freq, medianfreq

    for r in range(0, r_limit, blksze):
        for c in range(0, c_limit, blksze):
            blkim = im[r:r+blksze, c:c+blksze]
            blkor = orient[r:r+blksze, c:c+blksze]
            # 调用 freqest，返回与块同尺寸的频率块
            freq_block = freqest(blkim, blkor, windsze, minWaveLength, maxWaveLength)
            # 直接写回对应位置
            freq[r:r+blksze, c:c+blksze] = freq_block

    # 用掩码屏蔽非脊区域（确保掩码为 0/1）
    try:
        freq = freq * (mask.astype(bool))
    except Exception:
        freq = freq * mask

    # 计算有效频率的中位数
    valid = freq > 0
    if np.any(valid):
        medianfreq = float(np.median(freq[valid]))
    else:
        medianfreq = 0.0

    return freq, medianfreq