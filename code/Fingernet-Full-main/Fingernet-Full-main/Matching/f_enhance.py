"""
ENHANCING FINGERPRINT IMAGE

Usage:  [ binim, mask, cimg, cimg2, orient_img, orient_img_m ] =
         ... f_enhance( img );

Argument:   img - FingerPrint Image
              
Returns:    binim   - binary image 
            mask    - binary mask
            cimg1,2 - coherence image
            oimg1,2 - The orientation image in radians.

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import numpy as np
from FExtraction.fft_enhance_cubs import fft_enhance_cubs
from FExtraction.ridgesegment import ridgesegment
from FExtraction.ridgeorient import ridgeorient
from FExtraction.ridgefreq import ridgefreq
from FExtraction.ridgefilter import ridgefilter

def f_enhance(img):
    """
    Enhances fingerprint image using multiple processing stages
    
    Parameters:
    img - FingerPrint Image
    
    Returns:
    binim - binary image 
    mask - binary mask
    cimg1, cimg2 - coherence images
    oimg1, oimg2 - The orientation images in radians
    """
    # Enhance with Blocks 6x6
    enhimg, _, _, _, _, _ = fft_enhance_cubs(img, 6)
    
    # Enhance with Blocks 12x12
    enhimg, _, _, _, _, _ = fft_enhance_cubs(enhimg, 12)
    
    # Enhance with Blocks 24x24
    enhimg, cimg2, _, _, _, _ = fft_enhance_cubs(enhimg, 24)
    
    # FVC2002 DB1 parameters
    blksze = 5
    thresh = 0.085
    
    # Ridge segmentation and orientation
    normim, _, _ = ridgesegment(enhimg, blksze, thresh)
    oimg1, _ = ridgeorient(normim, 1, 3, 3)  # FVC2002 DB1
    
    # Second enhancement pass
    enhimg, cimg1, _, _, _, _ = fft_enhance_cubs(img, -1)
    normim, mask, _ = ridgesegment(enhimg, blksze, thresh)
    oimg2, _ = ridgeorient(normim, 1, 3, 3)
    
    # Ridge frequency and filtering
    freq, medfreq = ridgefreq(normim, mask, oimg2, 32, 5, 5, 15)
    binim = ridgefilter(normim, oimg2, medfreq * mask, 0.5, 0.5, showfilter=False) > 0
    
    return binim, mask, cimg1, cimg2, oimg1, oimg2


# Example usage
if __name__ == "__main__":
    # Create a sample fingerprint image
    img = np.random.rand(256, 256) * 255
    img = img.astype(np.float64)
    
    # Test the function
    try:
        binim, mask, cimg1, cimg2, oimg1, oimg2 = f_enhance(img)
        print("Fingerprint enhancement completed successfully")
        print(f"Binary image shape: {binim.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Coherence image 1 shape: {cimg1.shape}")
        print(f"Coherence image 2 shape: {cimg2.shape}")
        print(f"Orientation image 1 shape: {oimg1.shape}")
        print(f"Orientation image 2 shape: {oimg2.shape}")
    except Exception as e:
        print(f"Error during fingerprint enhancement: {e}")