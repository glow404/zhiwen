import numpy as np
import cv2

# 导入项目中已移植的函数（按本仓库结构尝试相对/绝对导入）
try:
    from .ridgesegment import ridgesegment
    from .ridgeorient import ridgeorient
    from .ridgefreq import ridgefreq
    from .ridgefilter import ridgefilter
except Exception:
    try:
        from ridgesegment import ridgesegment
        from ridgeorient import ridgeorient
        from ridgefreq import ridgefreq
        from ridgefilter import ridgefilter
    except Exception as e:
        raise ImportError("需要 ridgesegment/ridgeorient/ridgefreq/ridgefilter 的 Python 实现") from e

def testfin_FingerNet_Contact(im, filename=None):
    """
    Python 等价于 MATLAB testfin_FingerNet_Contact.m

    参数:
        im       - 输入图像，可为 numpy ndarray 或 图像文件路径字符串
        filename - 可选输出文件路径，若提供会把最终二值结果写入该文件（0/255）

    返回:
        newim, binim, mask, reliability
        newim      : float ndarray (增强后结果)
        binim      : bool ndarray (二值化结果)
        mask       : bool ndarray (脊线区域掩码)
        reliability: float ndarray (方向可靠度)
    """
    # 读取图像（若为路径）
    if isinstance(im, str):
        img = cv2.imread(im, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {im}")
    else:
        img = np.asarray(im)

    # 灰度化
    if img.ndim == 3 and img.shape[2] == 3:
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # 如果是多通道但非 3 通道，取第一通道；否则保持原状
        if img.ndim == 3:
            im_gray = img[:, :, 0]
        else:
            im_gray = img.copy()

    im_gray = im_gray.astype(float)

    # 1) Identify ridge-like regions and normalise image
    blksze = 16
    thresh = 0.1
    normim, mask = ridgesegment(im_gray, blksze, thresh)

    # 2) Determine ridge orientations
    orientim, reliability = ridgeorient(normim, 1, 5, 5)

    # 3) Determine ridge frequency values across the image
    blksze_freq = 36
    freq, medfreq = ridgefreq(normim, mask, orientim, blksze_freq, 5, 5, 15)

    # Use median frequency across mask (MATLAB 原代码)
    freq = medfreq * mask

    # 4) Apply oriented filters to enhance the ridge pattern
    newim = ridgefilter(normim, orientim, freq, 0.5, 0.5, 1)

    # 5) Binarise: ridge/valley threshold is medfreq
    # medfreq 是标量；若为 0 则 binim 全 False
    if np.isscalar(medfreq):
        binim = newim > medfreq
    else:
        # 兼容 medfreq 可能为数组的情况（但 MATLAB 此处为 scalar）
        binim = newim > (np.asarray(medfreq) * mask)

    # 6) final mask: binim * mask * (reliability > 0.5)
    relim = np.logical_and.reduce([binim, mask.astype(bool), (reliability > 0.5)])

    # 写文件（若提供 filename）
    if filename is not None:
        try:
            out_img = (relim.astype(np.uint8) * 255)
            cv2.imwrite(filename, out_img)
        except Exception:
            # 忽略写入错误但不抛出
            pass

    return newim, binim, mask, reliability