"""
RIDGEORIENT - Estimates the local orientation of ridges in a fingerprint

Usage:  [orientim, reliability] = ridgeorientation(im, gradientsigma,
                                            blocksigma, 
                                            orientsmoothsigma)

Arguments:  im                - A normalised input image.
            gradientsigma     - Sigma of the derivative of Gaussian
                                used to compute image gradients.
            blocksigma        - Sigma of the Gaussian weighting used to
                                sum the gradient moments.
            orientsmoothsigma - Sigma of the Gaussian used to smooth
                                the final orientation vector field.

Returns:    orientim          - The orientation image in radians.
                                Orientation values are +ve clockwise
                                and give the direction *along* the
                                ridges.
            reliability       - Measure of the reliability of the
                                orientation measure.  This is a value
                                between 0 and 1. I think a value above
                                about 0.5 can be considered 'reliable'.


With a fingerprint image at a 'standard' resolution of 500dpi suggested
parameter values might be:

   [orientim, reliability] = ridgeorient(im, 1, 3, 3);

See also: RIDGESEGMENT, RIDGEFREQ, RIDGEFILTER

Original version by Raymond Thai,  May 2003
Reworked by Peter Kovesi           January 2005
School of Computer Science & Software Engineering
The University of Western Australia
pk at csse uwa edu au
http://www.csse.uwa.edu.au/~pk
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from typing import Tuple


def ridgeorient(im: np.ndarray, gradientsigma: float, blocksigma: float, 
                orientsmoothsigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the local orientation of ridges in a fingerprint
    
    Args:
        im: A normalised input image
        gradientsigma: Sigma of the derivative of Gaussian used to compute image gradients
        blocksigma: Sigma of the Gaussian weighting used to sum the gradient moments
        orientsmoothsigma: Sigma of the Gaussian used to smooth the final orientation vector field
        
    Returns:
        orientim: The orientation image in radians
        reliability: Measure of the reliability of the orientation measure (0-1)
    """
    rows, cols = im.shape
    
    # Calculate image gradients using Gaussian derivatives
    sze = int(6 * gradientsigma)
    if sze % 2 == 0:
        sze += 1
    
    # Create Gaussian filter and its derivatives
    x = np.arange(-(sze//2), sze//2 + 1)
    y = np.arange(-(sze//2), sze//2 + 1)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian filter
    f = np.exp(-(X**2 + Y**2) / (2 * gradientsigma**2))
    f = f / np.sum(f)
    
    # Derivatives of Gaussian
    fx = -X / (gradientsigma**2) * f
    fy = -Y / (gradientsigma**2) * f
    
    # Compute gradients
    Gx = ndimage.convolve(im, fx, mode='reflect')
    Gy = ndimage.convolve(im, fy, mode='reflect')
    
    # Estimate the local ridge orientation at each point by finding the
    # principal axis of variation in the image gradients
    Gxx = Gx**2       # Covariance data for the image gradients
    Gxy = Gx * Gy
    Gyy = Gy**2
    
    # Now smooth the covariance data to perform a weighted summation
    Gxx = gaussian_filter(Gxx, blocksigma, mode='reflect')
    Gxy = 2 * gaussian_filter(Gxy, blocksigma, mode='reflect')
    Gyy = gaussian_filter(Gyy, blocksigma, mode='reflect')
    
    # Analytic solution of principal direction
    denom = np.sqrt(Gxy**2 + (Gxx - Gyy)**2) + np.finfo(float).eps
    sin2theta = Gxy / denom            # Sine and cosine of doubled angles
    cos2theta = (Gxx - Gyy) / denom
    
    # Smooth the sine and cosine of doubled angles
    cos2theta = gaussian_filter(cos2theta, orientsmoothsigma, mode='reflect')
    sin2theta = gaussian_filter(sin2theta, orientsmoothsigma, mode='reflect')
    
    orientim = np.pi/2 + np.arctan2(sin2theta, cos2theta) / 2
    
    # Calculate 'reliability' of orientation data
    # Calculate the area moment of inertia about the orientation axis found
    # (this will be the minimum inertia) and an axis perpendicular 
    # (which will be the maximum inertia)
    Imin = (Gyy + Gxx) / 2 - (Gxx - Gyy) * cos2theta / 2 - Gxy * sin2theta / 2
    Imax = Gyy + Gxx - Imin
    
    reliability = 1 - Imin / (Imax + 0.001)
    
    # Finally mask reliability to exclude regions where the denominator
    # in the orientation calculation above was small
    reliability = reliability * (denom > 0.001)
    
    return orientim, reliability


if __name__ == "__main__":
    # Example usage
    test_image = np.random.rand(200, 200)
    orientim, reliability = ridgeorient(test_image, 1.0, 3.0, 3.0)
    print(f"Orientation image shape: {orientim.shape}")
    print(f"Reliability image shape: {reliability.shape}")
    print(f"Orientation range: [{orientim.min():.3f}, {orientim.max():.3f}]")
    print(f"Reliability range: [{reliability.min():.3f}, {reliability.max():.3f}]")