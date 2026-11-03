import numpy as np

# 导入之前移植的 normalise 函数（确保 Enhancement/normalise.py 可用）
try:
    from .normalise import normalise
except Exception:
    from normalise import normalise

def ridgesegment(im, blksze, thresh):
    """
    等价于 MATLAB ridgesegment(im, blksze, thresh)

    输入:
      im     - 灰度图像 (ndarray)
      blksze - 块大小 (int)
      thresh - 标准差阈值 (float)

    输出:
      normim  - 对脊线区域重新归一化后的图像 (float64)
      mask    - 布尔掩码, 脊线区域为 True (ndarray, bool)
      maskind - 一维索引数组, mask 中为 True 的平铺线性索引 (ndarray, int)
    """
    img = np.asarray(im, dtype=float)

    # 将整幅图像先归一化为均值 0，方差 1（对应 MATLAB normalise(im,0,1)）
    try:
        img_normed = normalise(img, 0, 1)
    except TypeError:
        # 兼容如果 normalise 只接受单参的情况
        img_normed = normalise(img)

    rows, cols = img_normed.shape
    stddevim = np.zeros_like(img_normed, dtype=float)

    # 对图像按非重叠块计算块内标准差，并在对应块内填充该标准差值
    for r in range(0, rows, blksze):
        r_end = min(r + blksze, rows)
        for c in range(0, cols, blksze):
            c_end = min(c + blksze, cols)
            block = img_normed[r:r_end, c:c_end]
            # 若 block 全为常数，std = 0
            s = float(np.std(block))
            stddevim[r:r_end, c:c_end] = s

    # 阈值化得到掩码
    mask = stddevim > thresh
    maskind = np.flatnonzero(mask)  # 0-based linear indices

    # 若没有检测到任何脊块，返回归一化图像并空掩码
    if maskind.size == 0:
        # 返回已全图零均值单位方差的图像作为 normim
        return img_normed.astype(np.float64), mask.astype(bool), maskind

    # 对检测到的脊线像素部分重新计算均值和标准差，并据此对图像做再归一化
    vals = img_normed.flat[maskind]
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    if std_val <= 1e-8:
        # 若脊线区域标准差为 0，直接返回减去均值的图像（避免除零）
        normim = (img_normed - mean_val).astype(np.float64)
    else:
        normim = ((img_normed - mean_val) / std_val).astype(np.float64)

    return normim, mask.astype(bool), maskind