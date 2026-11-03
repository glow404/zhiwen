import numpy as np
import cv2

# 导入已移植的函数（尝试相对导入或顶层导入）
try:
    from .ridgesegment import ridgesegment
    from .ridgeorient import ridgeorient
    from .ridgefreq import ridgefreq
    from .ridgefilter import ridgefilter
    from .show import show
except Exception:
    try:
        from ridgesegment import ridgesegment
        from ridgeorient import ridgeorient
        from ridgefreq import ridgefreq
        from ridgefilter import ridgefilter
        from show import show
    except Exception:
        # 如果 show 不存在，定义空占位
        def show(*a, **k): pass

def testfin_IIITD(im, filename=None):
    """
    Python 移植的 testfin_IIITD.m

    参数:
        im       - 输入图像数组或图像路径（灰度或彩色）
        filename - 可选，若提供则把 binim（二值 0/255）保存为该文件

    返回:
        newim, binim, mask, reliability
    """
    # 读取图像（支持路径）
    if isinstance(im, str):
        img = cv2.imread(im, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {im}")
    else:
        img = np.asarray(im)

    # 灰度化并转为 float
    if img.ndim == 3 and img.shape[2] >= 3:
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    else:
        im_gray = img.astype(float)

    # 1) Identify ridge-like regions and normalise image
    blksze = 16
    thresh = 0.1
    normim, mask = ridgesegment(im_gray, blksze, thresh)

    # 2) Determine ridge orientations
    orientim, reliability = ridgeorient(normim, 1, 5, 5)

    # 3) Determine ridge frequency values across the image
    blksze_freq = 36
    freq, medfreq = ridgefreq(normim, mask, orientim, blksze_freq, 5, 5, 15)

    # 使用整个指纹的中位频率（MATLAB 原代码）
    if np.isscalar(medfreq):
        freq = medfreq * mask.astype(float)
    else:
        try:
            mf = np.asarray(medfreq, dtype=float)
            if mf.shape == mask.shape:
                freq = mf * mask.astype(float)
            else:
                freq = float(np.median(mf)) * mask.astype(float)
        except Exception:
            freq = float(np.median(medfreq)) * mask.astype(float)

    # 4) Apply filters to enhance the ridge pattern
    newim = ridgefilter(normim, orientim, freq, 0.5, 0.5, 1)

    # 5) Binarise: ridge/valley threshold is medfreq
    try:
        if np.isscalar(medfreq):
            binim = newim > medfreq
        else:
            binim = newim > np.asarray(medfreq)
    except Exception:
        binim = newim > np.median(newim)

    # 显示二值图（与 MATLAB show(binim,5) 等效，若 show 可用）
    try:
        show(binim.astype(np.uint8) * 255, 5)
    except Exception:
        pass

    # 保存 binim 至 filename（MATLAB 做法）
    if filename is not None:
        try:
            out = (binim.astype(np.uint8) * 255)
            cv2.imwrite(filename, out)
        except Exception:
            pass

    # 6) relim = binim .* mask .* (reliability>0.4) （保留以便外部使用，如需返回可另行处理）
    relim = np.logical_and.reduce([binim, mask.astype(bool), (reliability > 0.4)])

    return newim, binim, mask, reliability