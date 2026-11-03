import numpy as np
import cv2

# 导入依赖（尝试相对导入与顶层导入）
try:
    from .fft_enhance_cubs import fft_enhance_cubs
    from .ridgesegment import ridgesegment
    from .ridgeorient import ridgeorient
    from .ridgefreq import ridgefreq
    from .ridgefilter import ridgefilter
except Exception:
    try:
        from fft_enhance_cubs import fft_enhance_cubs
        from ridgesegment import ridgesegment
        from ridgeorient import ridgeorient
        from ridgefreq import ridgefreq
        from ridgefilter import ridgefilter
    except Exception as e:
        raise ImportError("需要 fft_enhance_cubs, ridgesegment, ridgeorient, ridgefreq, ridgefilter 的 Python 实现") from e

def _ensure_img(arr):
    """确保为灰度 float ndarray"""
    if isinstance(arr, str):
        im = cv2.imread(arr, cv2.IMREAD_UNCHANGED)
    else:
        im = np.asarray(arr)
    if im is None:
        raise FileNotFoundError("无法读取图像")
    if im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 转为 float
    return im.astype(float)

def _call_fft_enhance(img, block):
    """兼容 fft_enhance_cubs 可能返回单值或 (enh, cimg)"""
    out = fft_enhance_cubs(img, block)
    if isinstance(out, tuple) or isinstance(out, list):
        return out[0], out[1] if len(out) > 1 else None
    else:
        return out, None

def testfin_FingerNet_fft(im, filename=None):
    """
    Python 移植的 testfin_FingerNet_fft.m

    参数:
        im       - 输入图像数组或图像路径
        filename - 可选，若提供则把最终二值可靠区域保存到该路径 (0/255)

    返回:
        newim, binim, mask, reliability
    """
    img = _ensure_img(im)

    # 多尺度 FFT 增强
    enhimg, _ = _call_fft_enhance(img, 6)
    enhimg, _ = _call_fft_enhance(enhimg, 12)
    enhimg, cimg2 = _call_fft_enhance(enhimg, 24)

    # segment & normalize
    blksze = 5
    thresh = 0.05
    normim, mask = ridgesegment(enhimg, blksze, thresh)

    # orientation / reliability from normim (first orientation set)
    oimg1, reliability = ridgeorient(normim, 1, 3, 3)

    # another enhancement path from original image (block -1 as in MATLAB)
    enhimg2, cimg1 = _call_fft_enhance(img, -1)
    normim2, mask2 = ridgesegment(enhimg2, blksze, thresh)
    oimg2, reliability1 = ridgeorient(normim2, 1, 3, 3)

    # frequency estimation
    freq, medfreq = ridgefreq(normim2, mask2, oimg2, 32, 5, 5, 15)

    # prepare freq input for ridgefilter: MATLAB used medfreq.*mask
    # medfreq 在 MATLAB 中为标量，这里兼容数组/标量
    if np.isscalar(medfreq):
        freq_for_filter = medfreq * (mask.astype(float))
    else:
        mf = np.asarray(medfreq, dtype=float)
        # 若 medfreq 是与图像同尺寸的数组，则乘以 mask；否则取标量中位数
        if mf.shape == mask.shape:
            freq_for_filter = mf * (mask.astype(float))
        else:
            freq_for_filter = np.median(mf) * (mask.astype(float))

    # oriented filtering: use oimg1 (first orientation set) per MATLAB
    newim = ridgefilter(normim, oimg1, freq_for_filter, 0.5, 0.5, 0)

    # binarize: newim > medfreq (use scalar medfreq when possible)
    if np.isscalar(medfreq):
        binim = newim > medfreq
    else:
        # if medfreq is array use elementwise comparison (mask already applied earlier)
        try:
            binim = newim > np.asarray(medfreq)
        except Exception:
            binim = newim > np.median(np.asarray(medfreq))

    # final reliable region
    reliability_mask = (reliability > 0.5)
    relim = np.logical_and.reduce([binim, mask.astype(bool), reliability_mask])

    # write output file if requested (0/255)
    if filename is not None:
        try:
            out_img = (relim.astype(np.uint8) * 255)
            cv2.imwrite(filename, out_img)
        except Exception:
            pass

    return newim, binim, mask, reliability