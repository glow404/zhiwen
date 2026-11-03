import numpy as np
from scipy.ndimage import gaussian_filter

def smoothen_orientation_image(oimg, sigma=1.0):
    """
    Smooth orientation image via vectorial Gaussian filtering.

    Args:
        oimg : 2D ndarray
            Orientation image in radians.
        sigma : float
            Gaussian smoothing sigma (默认与 MATLAB fspecial(5) 相近，推荐 1.0)。

    Returns:
        noimg : 2D ndarray
            Smoothed orientation image in radians (range [0, pi)).
    """
    o = np.asarray(oimg, dtype=float)

    # 双角向量表示
    gx = np.cos(2.0 * o)
    gy = np.sin(2.0 * o)

    # 高斯平滑，mode='reflect' 对应 MATLAB imfilter(...,'symmetric')
    gfx = gaussian_filter(gx, sigma=sigma, mode='reflect')
    gfy = gaussian_filter(gy, sigma=sigma, mode='reflect')

    # 恢复方向并归一到正值，然后除以 2
    noimg = np.arctan2(gfy, gfx)
    # 将负角度映射到 [0, 2*pi)
    noimg[noimg < 0] += 2.0 * np.pi
    noimg = 0.5 * noimg

    # 最终确保范围在 [0, pi)
    noimg = np.mod(noimg, np.pi)

    return noimg