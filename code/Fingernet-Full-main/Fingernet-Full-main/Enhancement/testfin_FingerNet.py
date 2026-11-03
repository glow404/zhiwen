import numpy as np
import cv2

# ...existing code...
# 尝试相对或顶层导入已移植函数
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
        # 若 show 不可用也不影响核心功能
        def show(*a, **k): pass

def testfin_FingerNet(im, filename=None):
    """
    等价于 MATLAB testfin_FingerNet.m
    参数:
      im: numpy array 或 图像路径
      filename: 可选，保存最终二值可靠输出 (0/255)
    返回:
      newim, binim, mask, reliability
    """
    # 读取图像（支持路径或数组）
    if isinstance(im, str):
        img = cv2.imread(im, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {im}")
    else:
        img = np.asarray(im)

    # 转为灰度 float
    if img.ndim == 3 and img.shape[2] == 3:
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    elif img.ndim == 2:
        im_gray = img.astype(float)
    else:
        # 若含 alpha 或多通道，取第一通道
        im_gray = img[..., 0].astype(float)

    # 1) Identify ridge-like regions and normalise image
    blksze = 16
    thresh = 0.5
    normim, mask = ridgesegment(im_gray, blksze, thresh)

    # 2) Determine ridge orientations
    orientim, reliability = ridgeorient(normim, 1, 5, 5)

    # 3) Determine ridge frequency values across the image
    blksze_freq = 36
    freq, medfreq = ridgefreq(normim, mask, orientim, blksze_freq, 5, 5, 15)

    # Use median frequency across image as in MATLAB
    # medfreq expected scalar; if array, take median
    if np.isscalar(medfreq):
        freq_for_filter = medfreq * mask.astype(float)
    else:
        try:
            mf = np.asarray(medfreq, dtype=float)
            if mf.shape == mask.shape:
                freq_for_filter = mf * mask.astype(float)
            else:
                freq_for_filter = float(np.median(mf)) * mask.astype(float)
        except Exception:
            freq_for_filter = float(np.median(medfreq)) * mask.astype(float)

    # 4) Apply filters to enhance the ridge pattern
    newim = ridgefilter(normim, orientim, freq_for_filter, 0.5, 0.5, 1)

    # 可视化（如果需要）
    try:
        show(newim, 4)
    except Exception:
        pass

    # 5) Binarise: ridge/valley threshold is medfreq
    try:
        if np.isscalar(medfreq):
            binim = newim > medfreq
        else:
            binim = newim > np.asarray(medfreq)
    except Exception:
        binim = newim > (np.median(newim) if newim.size else 0)

    # 6) final result: binim .* mask .* (reliability>0.5)
    relim = np.logical_and.reduce([binim, mask.astype(bool), (reliability > 0.5)])

    # 保存结果文件（若提供）
    if filename is not None:
        try:
            out_img = (relim.astype(np.uint8) * 255)
            cv2.imwrite(filename, out_img)
        except Exception:
            pass

    return newim, binim, mask, reliability
# ...existing code...