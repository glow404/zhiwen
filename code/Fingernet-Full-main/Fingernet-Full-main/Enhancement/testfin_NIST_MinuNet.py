"""
等价于 MATLAB testfin_NIST_MinuNet.m 的 Python 实现。

函数:
    newim, binim, mask, reliability = testfin_NIST_MinuNet(im, filename=None)

依赖（同目录下）:
    f_enhance_3420_NIST.py 中的 ridgesegment, ridgeorient, ridgefreq, ridgefilter
"""
import numpy as np

# relative import of helper functions implemented in f_enhance_3420_NIST.py
try:
    from f_enhance_3420_NIST import ridgesegment, ridgeorient, ridgefreq, ridgefilter
except Exception:
    # fallback if run as script
    from f_enhance_3420_NIST import ridgesegment, ridgeorient, ridgefreq, ridgefilter

def testfin_NIST_MinuNet(im, filename=None):
    """
    输入:
        im - 灰度图像 (ndarray, uint8 或 float)
        filename - 可选, 若提供则会把部分中间结果保存为文件名相关文件（可选）
    输出:
        newim       - 增强后的浮点图像
        binim       - 二值图 (bool数组)，threshold: medfreq
        mask        - 掩码 (bool数组)
        reliability - 方向场可靠性 (float数组)
    """
    # 确保为 float 数组
    img = np.asarray(im, dtype=float)

    # 1) Identify ridge-like regions and normalise image
    blksze = 16
    thresh = 0.1
    normim, mask = ridgesegment(img, blksze, thresh)

    # 2) Determine ridge orientations
    # 参数: gradientsigma=1, blocksigma=5, orientsmoothsigma=5 对应 ridgeorient(normim,1,5,5)
    orientim, reliability = ridgeorient(normim, 1, 5, 5)

    # 3) Determine ridge frequency values across the image
    blksze_freq = 36
    freq, medfreq = ridgefreq(normim, mask, orientim, blksze_freq, 5, 15)

    # 4) Use median frequency across image (MATLAB 原版这样处理)
    freq = medfreq * mask

    # 5) Apply filters to enhance the ridge pattern
    newim = ridgefilter(normim, orientim, freq, 0.5, 0.5, 1.0)

    # 6) Binarise, ridge/valley threshold is medfreq
    # newim 可能为任意尺度，使用比较运算生成二值图
    binim = newim > medfreq

    # 可选：生成 relim 用于显示/保存（MATLAB 使用 relim = binim.*mask.*(reliability>0.3) 并取反）
    # relim = np.logical_and.reduce([binim, mask, reliability > 0.3])
    # relim = np.logical_not(relim)
    # if filename is not None:
    #     try:
    #         import cv2
    #         cv2.imwrite(filename, (relim.astype(np.uint8) * 255))
    #     except Exception:
    #         pass

    return newim, binim, mask, reliability