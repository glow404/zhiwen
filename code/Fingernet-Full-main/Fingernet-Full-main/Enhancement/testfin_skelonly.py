"""
Python 版本的 testfin_skelonly.m

调用:
    newim, binim, mask, reliability = testfin_skelonly(im, filename=None)

im: 输入图像（灰度或二值或骨架），ndarray
filename: 可选，若提供则按原 MATLAB 流程保存中间/最终图像
"""
import numpy as np
import cv2

# 导入依赖函数（来自 f_enhance_3420_NIST.py）
try:
    from .f_enhance_3420_NIST import ridgesegment, ridgeorient, ridgefreq, ridgefilter
except Exception:
    try:
        from f_enhance_3420_NIST import ridgesegment, ridgeorient, ridgefreq, ridgefilter
    except Exception as e:
        raise ImportError("需要 f_enhance_3420_NIST.py 中的 ridgesegment/ridgeorient/ridgefreq/ridgefilter") from e

def testfin_skelonly(im, filename=None):
    """
    等价于 MATLAB testfin_skelonly.m
    返回: newim (float array), binim (bool array), mask (bool array), reliability (float array)
    并在 filename 给定时保存与 MATLAB 源码相似的输出文件（先保存取反掩码，再保存映射为 245/75 的图像）。
    """
    img = np.asarray(im, dtype=float)

    # 1) Identify ridge-like regions and normalise image
    blksze = 16
    thresh = 0.1
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

    # 6) Binarise using medfreq
    binim = newim > medfreq

    # 7) relim = binim .* mask .* (reliability>0.5)
    relim = np.logical_and.reduce([binim, mask, reliability > 0.5])
    fin_out = np.logical_not(relim)  # MATLAB 的 fin_out = ~relim

    # 8) 若提供 filename，则按 MATLAB 流程保存并将像素映射后覆盖文件
    if filename is not None:
        # 保存 fin_out 为 0/1 的灰度图（uint8），以便后续按 MATLAB 相同逻辑读取后判断 ==1
        try:
            cv2.imwrite(filename, fin_out.astype(np.uint8))
            # 读取回 L（以灰度方式），值应为 0 或 1
            L = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if L is None:
                raise IOError("写入后读取文件失败")
            # 构造 minu_i: 若 L(i,j)==1 则 245 否则 75
            minu_i = np.where(L == 1, 245, 75).astype(np.uint8)
            cv2.imwrite(filename, minu_i)
        except Exception:
            # 若写入 0/1 失败（某些写入器可能不保留 1），尝试写入 0/255 再转换
            try:
                tmp_path = filename
                cv2.imwrite(tmp_path, (fin_out.astype(np.uint8) * 255))
                L = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
                if L is not None:
                    # 将非零视为 1
                    Lbin = (L != 0).astype(np.uint8)
                    minu_i = np.where(Lbin == 1, 245, 75).astype(np.uint8)
                    cv2.imwrite(filename, minu_i)
            except Exception:
                # 如果最终仍失败，忽略文件写入错误（保持函数可用）
                pass

    return newim, binim, mask, reliability