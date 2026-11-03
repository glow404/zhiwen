import numpy as np
import cv2

# 尝试相对导入已移植函数
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
        # 最少保证核心功能可用（show 可缺省）
        def show(*a, **k): pass

def testfin_FVC(im, filename=None):
    """
    Python 等价于 MATLAB testfin_FVC.m

    参数:
        im       - 输入图像（ndarray）或图像路径（str）
        filename - 可选，保存最终可靠二值图到该路径（0/255）

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

    # 灰度化并转 float
    if img.ndim == 3 and img.shape[2] == 3:
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    elif img.ndim == 2:
        im_gray = img.astype(float)
    else:
        # 多通道取第一通道
        im_gray = img[..., 0].astype(float)

    # 1) Identify ridge-like regions and normalise image
    blksze = 16
    thresh = 0.1
    normim, mask = ridgesegment(im_gray, blksze, thresh)

    # 2) Determine ridge orientations
    orientim, reliability = ridgeorient(normim, 1, 5, 5)

    # 3) Determine ridge frequency values across the image
    blksze_freq = 36
    freq, medfreq = ridgefreq(normim, mask, orientim, blksze_freq, 5, 5, 15)

    # Use median frequency across mask (MATLAB 中用 scalar medfreq * mask)
    if np.isscalar(medfreq):
        freq_for_filter = medfreq * mask.astype(float)
    else:
        mf = np.asarray(medfreq, dtype=float)
        if mf.shape == mask.shape:
            freq_for_filter = mf * mask.astype(float)
        else:
            freq_for_filter = float(np.median(mf)) * mask.astype(float)

    # 4) Apply filters to enhance the ridge pattern
    newim = ridgefilter(normim, orientim, freq_for_filter, 0.5, 0.5, 1)

    # 5) Binarise: ridge/valley threshold is medfreq (MATLAB used newim > medfreq)
    if np.isscalar(medfreq):
        binim = newim > medfreq
    else:
        try:
            binim = newim > np.asarray(medfreq)
        except Exception:
            binim = newim > np.median(newim)

    # 6) final reliable region: binim .* mask .* (reliability>0.5)
    relim = np.logical_and.reduce([binim, mask.astype(bool), (reliability > 0.5)])

    # MATLAB 显示 ~relim；这里也可显示反相以便检查
    try:
        show((~relim).astype(np.uint8) * 255, 7)
    except Exception:
        pass

    # 保存结果（若提供文件名）
    if filename is not None:
        try:
            out_img = (relim.astype(np.uint8) * 255)
            cv2.imwrite(filename, out_img)
        except Exception:
            pass

    return newim, binim, mask, reliability