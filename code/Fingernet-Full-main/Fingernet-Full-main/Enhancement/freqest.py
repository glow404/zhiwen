import numpy as np
from scipy.ndimage import rotate, maximum_filter1d

def freqest(im, orientim, windsze, minWaveLength, maxWaveLength):
    """
    Estimate fingerprint ridge frequency within an image block.

    Parameters
    ----------
    im : 2D array
        Image block.
    orientim : scalar or array
        Ridge orientation(s) in radians (within the block).
    windsze : int
        Window length used to identify peaks (odd integer, e.g. 3 or 5).
    minWaveLength, maxWaveLength : float
        Minimum and maximum acceptable ridge wavelengths (pixels).

    Returns
    -------
    freqim : 2D array (same shape as im)
        Estimated spatial frequency per pixel (0 if not found / out of bounds).
    """
    im = np.asarray(im, dtype=float)
    rows, cols = im.shape

    # compute mean orientation robustly using doubled-angle trick
    orient_arr = np.asarray(orientim).ravel()
    # if orientim is a single value, still works
    cos2 = np.cos(2.0 * orient_arr)
    sin2 = np.sin(2.0 * orient_arr)
    mean_cos2 = np.mean(cos2)
    mean_sin2 = np.mean(sin2)
    orient = 0.5 * np.arctan2(mean_sin2, mean_cos2)  # radians

    # rotate block so ridges become vertical
    # MATLAB used orient/pi*180 + 90 degrees
    angle_deg = orient / np.pi * 180.0 + 90.0
    # use nearest interpolation (order=0) and keep same shape (reshape=False)
    rotim = rotate(im, angle_deg, reshape=False, order=0, mode='nearest')

    # crop to central square to avoid edge artifacts: size = floor(rows/sqrt(2))
    cropsze = int(np.fix(rows / np.sqrt(2.0)))
    if cropsze <= 0:
        return np.zeros_like(im)
    offset = int(np.fix((rows - cropsze) / 2.0))
    # ensure indices valid
    y1 = max(0, offset)
    y2 = min(rows, offset + cropsze)
    x1 = max(0, offset)
    x2 = min(cols, offset + cropsze)
    rotim_cropped = rotim[y1:y2, x1:x2]
    if rotim_cropped.size == 0:
        return np.zeros_like(im)

    # projection down columns (sum along rows) -> 1D signal
    proj = np.sum(rotim_cropped, axis=0)

    # find local maxima using 1D dilation / maximum filter
    try:
        dil = maximum_filter1d(proj, size=windsze, mode='reflect')
    except Exception:
        # fallback simple maximum filter
        from scipy.ndimage import maximum_filter
        dil = maximum_filter(proj, size=windsze, mode='reflect')
    maxpts = (dil == proj) & (proj > np.mean(proj))
    maxind = np.nonzero(maxpts)[0]

    # decide frequency
    if maxind.size < 2:
        freqim = np.zeros_like(im, dtype=float)
    else:
        NoOfPeaks = maxind.size
        waveLength = float(maxind[-1] - maxind[0]) / float(NoOfPeaks - 1)
        if (waveLength > minWaveLength) and (waveLength < maxWaveLength):
            freq = 1.0 / waveLength
            freqim = np.ones_like(im, dtype=float) * freq
        else:
            freqim = np.zeros_like(im, dtype=float)

    return freqim