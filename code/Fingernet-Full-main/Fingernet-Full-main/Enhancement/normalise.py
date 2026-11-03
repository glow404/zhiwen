import numpy as np

# 尝试使用 scikit-image 的 rgb<->hsv；若不可用退回到 OpenCV
try:
    from skimage.color import rgb2hsv, hsv2rgb
    _HAS_SKIMAGE = True
except Exception:
    import cv2
    _HAS_SKIMAGE = False

def _to_float01(img):
    """Convert image to float64 in range [0,1]."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0
    # assume already float; clip to [0,1] if seems so, else scale by max
    f = img.astype(np.float64)
    if f.max() > 1.0:
        # scale by max value if reasonable
        mx = float(f.max())
        if mx > 0:
            f = f / mx
    return f

def normalise(im, reqmean=None, reqvar=None):
    """
    Normalise image.

    Usage:
        n = normalise(im)
        n = normalise(im, reqmean, reqvar)

    - 如果只传 im：将灰度图或彩色图（RGB）归一化到 [0,1]（彩色仅对 V 通道归一化）。
    - 如果传 reqmean 和 reqvar：只适用于灰度图像（2D），将图像调整为具有给定均值与方差（返回 float64）。

    返回:
        n - numpy ndarray (dtype float64)
    """
    if not (reqmean is None and reqvar is None) and (reqmean is None or reqvar is None):
        raise ValueError("reqmean and reqvar must be provided together")

    im = np.asarray(im)
    # color image (H x W x 3)
    if im.ndim == 3 and im.shape[2] == 3:
        if reqmean is not None:
            raise ValueError("cannot normalise colour image to desired mean and variance")
        # convert to float in [0,1]
        f = _to_float01(im)
        if _HAS_SKIMAGE:
            hsv = rgb2hsv(f)
            v = hsv[:, :, 2]
            vmin = float(v.min())
            vrange = float(v.max() - vmin)
            if vrange <= 0:
                v[:] = 0.0
            else:
                v = (v - vmin) / vrange
            hsv[:, :, 2] = v
            rgb = hsv2rgb(hsv)
            return rgb.astype(np.float64)
        else:
            # use OpenCV: cv2.cvtColor expects float32 in 0..1 for COLOR_RGB2HSV
            f32 = f.astype(np.float32)
            hsv = cv2.cvtColor(f32, cv2.COLOR_RGB2HSV)  # H:0-179, S:0-255, V:0-255 for uint8; for float32 V in 0..1
            # OpenCV with float32 gives H in [0,360), S,V in [0,1]
            v = hsv[:, :, 2].astype(np.float64)
            vmin = float(v.min())
            vrange = float(v.max() - vmin)
            if vrange <= 0:
                v[:] = 0.0
            else:
                v = (v - vmin) / vrange
            hsv[:, :, 2] = v.astype(hsv.dtype)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return rgb.astype(np.float64)

    # grayscale / single-channel
    if im.ndim == 2:
        if reqmean is None and reqvar is None:
            # normalise to 0-1
            f = im.astype(np.float64)
            mn = float(f.min())
            f = f - mn
            mx = float(f.max())
            if mx <= 0:
                return np.zeros_like(f, dtype=np.float64)
            return (f / mx).astype(np.float64)
        else:
            # normalise to desired mean and variance
            f = im.astype(np.float64)
            mu = float(np.mean(f))
            sigma = float(np.std(f))
            if sigma <= 0:
                # if constant image, produce constant with desired mean
                return np.full_like(f, fill_value=float(reqmean), dtype=np.float64)
            fz = (f - mu) / sigma  # zero mean, unit std
            return (float(reqmean) + fz * np.sqrt(float(reqvar))).astype(np.float64)

    # other dimensions: try to treat as grayscale flatten
    raise ValueError("Unsupported image dimensions: %s" % (im.shape,))
