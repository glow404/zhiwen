import numpy as np
import cv2

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
    except Exception:
        # minimal fallbacks
        def show(*a, **k): pass

def testfin_NIST(im, filename=None):
    # 读取图像如果传入的是路径
    if isinstance(im, str):
        img = cv2.imread(im, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {im}")
    else:
        img = np.asarray(im)

    # 转为灰度 float
    if img.ndim == 3 and img.shape[2] >= 3:
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    else:
        im_gray = img.astype(float)

    # 1) Identify ridge-like regions and normalise image
    blksze = 16
    thresh = 0.1
    seg_result = ridgesegment(im_gray, blksze, thresh)
    if isinstance(seg_result, tuple) and len(seg_result) >= 2:
        normim = seg_result[0]
        mask = seg_result[1]
    else:
        # 错误处理
        normim = seg_result[0] if isinstance(seg_result, tuple) else seg_result
        mask = np.ones_like(im_gray, dtype=bool)  # 默认全掩码

    # 2) Determine ridge orientations
    orient_result = ridgeorient(normim, 1, 5, 5)
    if isinstance(orient_result, tuple) and len(orient_result) >= 2:
        orientim = orient_result[0]
        reliability = orient_result[1]
    else:
        # 错误处理
        orientim = orient_result[0] if isinstance(orient_result, tuple) else orient_result
        reliability = np.ones_like(normim)  # 默认可靠性图

    # 3) Determine ridge frequency values across the image
    blksze_freq = 36
    freq_result = ridgefreq(normim, mask, orientim, blksze_freq, 5, 5, 15)
    if isinstance(freq_result, tuple) and len(freq_result) >= 2:
        freq = freq_result[0]
        medfreq = freq_result[1]
    else:
        # 错误处理
        freq = freq_result[0] if isinstance(freq_result, tuple) else freq_result
        medfreq = np.median(freq[mask])  # 计算中值频率

    # Use median frequency across the whole fingerprint (MATLAB behavior)
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

    # 5) Binarise, threshold with medfreq
    try:
        if np.isscalar(medfreq):
            binim = newim > medfreq
        else:
            binim = newim > np.asarray(medfreq)
    except Exception:
        binim = newim > 0

    # 6) final reliable region: binim .* mask .* (reliability>0.5)
    relim = np.logical_and.reduce([binim, mask.astype(bool), (reliability > .5)])

    # show inverted relim as in MATLAB show(~relim,7)
    try:
        show((~relim).astype(np.uint8) * 255, 7)
    except Exception:
        pass

    # write relim to filename if provided (0/255 image)
    if filename is not None:
        try:
            cv2.imwrite(filename, (relim.astype(np.uint8) * 255))
        except Exception:
            pass

    return newim, binim, mask, reliability