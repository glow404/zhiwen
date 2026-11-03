import numpy as np
import cv2

# 从已有的 f_enhance_3420_NIST.py 导入依赖函数
try:
    from .f_enhance_3420_NIST import ridgesegment, ridgeorient, ridgefreq, ridgefilter
except Exception:
    try:
        from f_enhance_3420_NIST import ridgesegment, ridgeorient, ridgefreq, ridgefilter
    except Exception as e:
        raise ImportError("需要 f_enhance_3420_NIST.py 中的 ridgesegment/ridgeorient/ridgefreq/ridgefilter") from e

def FingerNet_testfin_skelonly(im, filename=None):
    """
    等价于 MATLAB FingerNet_testfin_skelonly.m

    参数:
        im       - 输入图像（灰度或二值），ndarray
        filename - 可选，若提供则把 relim 保存为二值图到该路径

    返回:
        newim, binim, mask, reliability
    """
    img = np.asarray(im, dtype=float)

    # 1) Identify ridge-like regions and normalise image
    blksze = 16
    thresh = 0.2  # for skeleton only
    normim, mask = ridgesegment(img, blksze, thresh)

    # 2) Determine ridge orientations
    orientim, reliability = ridgeorient(normim, 1, 5, 5)

    # 3) Determine ridge frequency values across the image
    blksze_freq = 36
    freq, medfreq = ridgefreq(normim, mask, orientim, blksze_freq, 5, 5, 15)

    # 4) Use median frequency across image (MATLAB 行为)
    freq = medfreq * mask

    # 5) Apply filters to enhance the ridge pattern
    newim = ridgefilter(normim, orientim, freq, 0.5, 0.5, 1.0)

    # 6) Binarise, ridge/valley threshold is medfreq
    binim = newim > medfreq

    # 7) relim = binim .* mask .* (reliability>0.5)
    relim = np.logical_and.reduce([binim, mask, reliability > 0.5])

    # 8) 如果指定 filename，则保存 relim（二值 0/255 图像）
    if filename is not None:
        try:
            # 保存为 0/255 二值图，便于后续处理/查看
            cv2.imwrite(filename, (relim.astype(np.uint8) * 255))
        except Exception:
            # 忽略保存错误但不影响返回值
            pass

    return newim, binim, mask, reliability