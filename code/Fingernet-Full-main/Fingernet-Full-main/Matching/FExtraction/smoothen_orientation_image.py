"""
smoothen_orientation_image
smoothens the orientation image through vectorial gaussian filtering
Usage:
new_oimg = smoothen_orientation_image(oimg)
oimg     - orientation image
new_oimg - filtered orientation image
Contact:
   ssc5@eng.buffalo.edu
   www.eng.buffalo.edu/~ssc5
Reference:
M. Kaas and A. Witkin, "Analyzing oriented patterns", Computer Vision
Graphics Image Processing 37(4), pp 362--385, 1987
"""

import numpy as np
from scipy.ndimage import gaussian_filter

def smoothen_orientation_image(oimg):
    """
    Smoothens the orientation image through vectorial gaussian filtering
    
    Parameters:
    oimg - orientation image
    
    Returns:
    noimg - filtered orientation image
    """
    # ---------------------------
    # smoothen the image
    # ---------------------------
    gx = np.cos(2 * oimg)
    gy = np.sin(2 * oimg)
    
    # Apply Gaussian filter (equivalent to MATLAB's fspecial('gaussian',5) and imfilter)
    sigma = 5 / 6  # Approximate sigma for 5x5 Gaussian kernel
    gfx = gaussian_filter(gx, sigma=sigma, mode='reflect')
    gfy = gaussian_filter(gy, sigma=sigma, mode='reflect')
    
    noimg = np.arctan2(gfy, gfx)
    noimg[noimg < 0] = noimg[noimg < 0] + 2 * np.pi
    noimg = 0.5 * noimg
    
    return noimg


# Example usage
if __name__ == "__main__":
    # Create a sample orientation image
    oimg = np.random.rand(64, 64) * np.pi
    
    # Test the function
    try:
        smoothed_oimg = smoothen_orientation_image(oimg)
        print("Orientation image smoothing completed successfully")
        print(f"Input shape: {oimg.shape}")
        print(f"Output shape: {smoothed_oimg.shape}")
        print(f"Input range: [{np.min(oimg):.3f}, {np.max(oimg):.3f}]")
        print(f"Output range: [{np.min(smoothed_oimg):.3f}, {np.max(smoothed_oimg):.3f}]")
    except Exception as e:
        print(f"Error during orientation image smoothing: {e}")