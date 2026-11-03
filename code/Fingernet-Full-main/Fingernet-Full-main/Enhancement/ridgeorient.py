import numpy as np
from scipy.ndimage import gaussian_filter

def ridgeorient(im, gradientsigma, blocksigma, orientsmoothsigma=0):
    """
    Estimate local ridge orientation, reliability and coherence.

    Args:
        im: 2D ndarray (normalized image, float)
        gradientsigma: sigma for derivative-of-Gaussian (gradient)
        blocksigma: sigma for smoothing gradient covariance
        orientsmoothsigma: optional sigma to smooth final doubled-angle field

    Returns:
        orientim: 2D ndarray, orientation in radians (direction along ridges, range ~[0, pi))
        reliability: 2D ndarray, reliability in [0,1]
        coherence: 2D ndarray, coherence measure
    """
    im = np.asarray(im, dtype=float)
    rows, cols = im.shape

    # Small epsilon
    eps = np.finfo(float).eps

    # Compute image gradients by convolving with Gaussian derivatives
    # Using scipy gaussian_filter with order for derivatives:
    # Gx = d/dx (G_sigma * im), Gy = d/dy (G_sigma * im)
    Gx = gaussian_filter(im, sigma=gradientsigma, order=(0,1), mode='reflect')
    Gy = gaussian_filter(im, sigma=gradientsigma, order=(1,0), mode='reflect')

    # Covariance data for gradients
    Gxx = Gx * Gx
    Gxy = Gx * Gy
    Gyy = Gy * Gy

    # Smooth the covariance data with Gaussian (blocksigma)
    Gxx = gaussian_filter(Gxx, sigma=blocksigma, mode='reflect')
    Gxy = 2.0 * gaussian_filter(Gxy, sigma=blocksigma, mode='reflect')  # note factor 2 as in MATLAB
    Gyy = gaussian_filter(Gyy, sigma=blocksigma, mode='reflect')

    # Analytic solution for principal direction via doubled-angle representation
    denom = np.sqrt(Gxy * Gxy + (Gxx - Gyy) * (Gxx - Gyy)) + eps
    sin2theta = Gxy / denom
    cos2theta = (Gxx - Gyy) / denom

    # Optionally smooth doubled-angle components
    if orientsmoothsigma and orientsmoothsigma > 0:
        cos2theta = gaussian_filter(cos2theta, sigma=orientsmoothsigma, mode='reflect')
        sin2theta = gaussian_filter(sin2theta, sigma=orientsmoothsigma, mode='reflect')

    # orientation image (direction along ridges). MATLAB: orientim = pi/2 + atan2(sin2theta,cos2theta)/2;
    orientim = (np.pi / 2.0) + 0.5 * np.arctan2(sin2theta, cos2theta)
    # wrap negative to positive range (optional): keep within [0, pi)
    orientim = np.mod(orientim, np.pi)

    # Compute min/max moments and reliability / coherence
    Imin = (Gyy + Gxx) / 2.0 - (Gxx - Gyy) * cos2theta / 2.0 - Gxy * sin2theta / 2.0
    Imax = (Gyy + Gxx) - Imin

    reliability = 1.0 - Imin / (Imax + 0.001)
    coherence = ((Imax - Imin) / (Imax + Imin + eps)) ** 2

    # Mask reliability where denom is too small (as MATLAB)
    reliability = reliability * (denom > 0.001)

    return orientim, reliability, coherence