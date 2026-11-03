# ...existing code...
import numpy as np
import cv2

def _import_helpers():
    """尝试导入项目中已移植的函数（若不存在则抛出 ImportError）"""
    try:
        from ridgesegment import ridgesegment
        from ridgeorient import ridgeorient
        from ridgefreq import ridgefreq
        from ridgefilter import ridgefilter
        from show import show
    except Exception:
        try:
            from ridgesegment import ridgesegment
            from ridgeorient import ridgeorient
            from ridgefreq import ridgefreq
            from ridgefilter import ridgefilter
            from show import show
        except Exception as e:
            raise ImportError("需要 ridgesegment/ridgeorient/ridgefreq/ridgefilter/show 已移植的 Python 实现") from e
    return ridgesegment, ridgeorient, ridgefreq, ridgefilter, show

def testfin(im=None, filename=None):
    """
    Python 移植的 testfin.m

    参数:
      im       - numpy ndarray（图像）或 图像文件路径（str）；如果为 None，会尝试读取当前目录下 'finger.png'
      filename - 可选，若提供则把最终显示的二值可靠区域保存为该路径（0/255）

    返回:
      newim, binim, mask, reliability, orientim
    """
    ridgesegment, ridgeorient, ridgefreq, ridgefilter, show = _import_helpers()

    # 读取图像
    if im is None:
        img = cv2.imread('finger.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError("输入图像为空且无法读取默认文件 'finger.png'")
    elif isinstance(im, str):
        img = cv2.imread(im, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {im}")
    else:
        img = np.asarray(im)

    # 灰度并转为 float
    if img.ndim == 3 and img.shape[2] >= 3:
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    else:
        im_gray = img.astype(float)

    # 1) Identify ridge-like regions and normalise image
    blksze = 5
    thresh = 0.085
    normim, mask = ridgesegment(im_gray, blksze, thresh)

    # 2) Determine ridge orientations
    orientim, reliability = ridgeorient(normim, 1, 3, 3)

    # 3) Determine ridge frequency values across the image
    # MATLAB: [freq, medfreq] = ridgefreq(normim, mask, orientim, 32, 5, 5, 15);
    freq, medfreq = ridgefreq(normim, mask, orientim, 32, 5, 5, 15)

    # Use median frequency across whole fingerprint as in original
    if np.isscalar(medfreq):
        freq_for_filter = medfreq * mask.astype(float)
    else:
        mf = np.asarray(medfreq, dtype=float)
        if mf.shape == mask.shape:
            freq_for_filter = mf * mask.astype(float)
        else:
            freq_for_filter = float(np.median(mf)) * mask.astype(float)

    # 4) Apply oriented filters to enhance the ridge pattern
    newim = ridgefilter(normim, orientim, freq_for_filter, 0.5, 0.5, 1)

    # 5) Binarise, ridge/valley threshold is 0 (MATLAB used newim>0)
    binim = newim > 0

    # 6) Display/save binary image for masked & reliable regions
    relim = np.logical_and.reduce([binim, mask.astype(bool), (reliability > 0.5)])
    try:
        # show expects image-like; convert boolean to 0..255 uint8
        show((relim.astype(np.uint8) * 255), 7)
    except Exception:
        pass

    if filename is not None:
        try:
            cv2.imwrite(filename, (relim.astype(np.uint8) * 255))
        except Exception:
            pass

    return newim, binim, mask, reliability, orientim