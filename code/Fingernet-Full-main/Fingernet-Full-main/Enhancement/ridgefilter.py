import numpy as np
from scipy.ndimage import rotate

def ridgefilter(im, orient, freq, kx, ky, showfilter=0):
    """
    Python 实现的 ridgefilter (基于 Peter Kovesi 的 MATLAB 版本)

    参数:
        im       : 2D ndarray (图像，任意数值类型)
        orient   : 2D ndarray (方向场，弧度)
        freq     : 2D ndarray (频率图，频率=1/波长，freq<=0 表示无效)
        kx, ky   : float (滤波器 sigma 与波长的比例)
        showfilter: int (是否显示滤波器，未实现可为 0)

    返回:
        newim : 2D ndarray (增强后的图像, dtype=float64)
    """
    angleInc = 3  # 角度步进 (度)，应整除180

    im = np.asarray(im, dtype=float)
    rows, cols = im.shape
    newim = np.zeros((rows, cols), dtype=float)

    # 有效频率点
    validr, validc = np.nonzero(freq > 0)
    if validr.size == 0:
        return newim

    # 四舍五入频率到 0.01
    freq_copy = freq.copy()
    inds = (validr, validc)
    freq_copy[inds] = np.round(freq_copy[inds] * 100) / 100.0

    # distinct frequencies
    unfreq = np.unique(freq_copy[inds])
    # 构建 freq -> index 映射 (以 round(freq*100) 为键)
    freq_map = {int(round(f * 100)): idx for idx, f in enumerate(unfreq)}

    # 生成每个频率对应的基滤波器并对不同角度预旋转
    nAngles = int(180 / angleInc)
    filters = []  # filters[k][o] -> ndarray
    sze_list = np.zeros(len(unfreq), dtype=int)

    for k_idx, fval in enumerate(unfreq):
        # 若 fval == 0 跳过（不过按构造应非0）
        if fval <= 0:
            filters.append([np.zeros((1,1)) for _ in range(nAngles)])
            sze_list[k_idx] = 0
            continue
        sigmax = (1.0 / fval) * kx
        sigmay = (1.0 / fval) * ky
        s = int(round(3.0 * max(sigmax, sigmay)))
        sze_list[k_idx] = s
        # meshgrid: x 横向(列), y 纵向(行)
        x = np.arange(-s, s+1)
        y = np.arange(-s, s+1)
        X, Y = np.meshgrid(x, y)
        # reference filter: gaussian envelope * cosine along x
        reffilter = np.exp(-((X**2) / (sigmax**2) + (Y**2) / (sigmay**2)) / 2.0) * np.cos(2.0 * np.pi * fval * X)
        # 生成旋转版本；注意 MATLAB 中 orientation 是沿脊方向，因此使用 -(o*angleInc+90)
        filt_orients = []
        for o in range(1, nAngles+1):
            angle = -(o * angleInc + 90.0)  # degrees
            # scipy.ndimage.rotate: positive = CCW; reshape=False 相当于 'crop'
            rotated = rotate(reffilter, angle, reshape=False, order=1, mode='reflect')
            filt_orients.append(rotated)
        filters.append(filt_orients)

    # 最大滤波器半径（用于保证不越界）
    maxsze = int(np.max(sze_list)) if sze_list.size > 0 else 0

    # 选择那些距离边界足够远的有效点
    final_mask = (validr > maxsze) & (validr < rows - maxsze) & (validc > maxsze) & (validc < cols - maxsze)
    final_r = validr[final_mask]
    final_c = validc[final_mask]
    if final_r.size == 0:
        return newim

    # orient 到 index (MATLAB 中 orient/pi*180/angleInc, 取整并修正到 1..maxorientindex)
    maxorientindex = nAngles
    # convert radians->degrees along 0..180 range
    orient_deg = (orient / np.pi) * 180.0
    orientindex = np.rint(orient_deg / angleInc).astype(int)  # still may be 0..n
    # 修正使得 1..maxorientindex 对应 MATLAB 行为
    # 在 MATLAB: orientindex < 1 => +maxorientindex ; >maxorientindex => -maxorientindex
    # 我们在 Python 中需要 0-based 索引 -> subtract 1 later
    orientindex = orientindex.copy()
    orientindex[orientindex < 1] += maxorientindex
    orientindex[orientindex > maxorientindex] -= maxorientindex
    orientindex0 = orientindex - 1  # 0-based

    # 最终过滤并计算
    for r, c in zip(final_r, final_c):
        key = int(round(freq_copy[r, c] * 100))
        if key not in freq_map:
            continue
        filterindex = freq_map[key]  # 0-based index into unfreq / filters
        s = int(sze_list[filterindex])
        # 可能出现 orientindex0[r,c] 超出范围 (数值误差) -> 保证在 0..nAngles-1
        oi = int(orientindex0[r, c]) % nAngles
        local_filter = filters[filterindex][oi]
        # 卷积区域
        r1 = r - s
        r2 = r + s + 1
        c1 = c - s
        c2 = c + s + 1
        patch = im[r1:r2, c1:c2]
        if patch.shape != local_filter.shape:
            # 若尺寸不匹配（极少数边缘情况），跳过
            continue
        newim[r, c] = np.sum(patch * local_filter)

    return newim