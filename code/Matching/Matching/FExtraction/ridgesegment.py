"""
RIDGESEGMENT - Normalises fingerprint image and segments ridge region

Function identifies ridge regions of a fingerprint image and returns a
mask identifying this region.  It also normalises the intesity values of
the image so that the ridge regions have zero mean, unit standard
deviation.

This function breaks the image up into blocks of size blksze x blksze and
evaluates the standard deviation in each region.  If the standard
deviation is above the threshold it is deemed part of the fingerprint.
Note that the image is normalised to have zero mean, unit standard
deviation prior to performing this process so that the threshold you
specify is relative to a unit standard deviation.

Usage:   [normim, mask, maskind] = ridgesegment(im, blksze, thresh)

Arguments:   im     - Fingerprint image to be segmented.
             blksze - Block size over which the the standard
                      deviation is determined (try a value of 16).
             thresh - Threshold of standard deviation to decide if a
                      block is a ridge region (Try a value 0.1 - 0.2)

Returns:     normim - Image where the ridge regions are renormalised to
                      have zero mean, unit standard deviation.
             mask   - Mask indicating ridge-like regions of the image, 
                      0 for non ridge regions, 1 for ridge regions.
             maskind - Vector of indices of locations within the mask. 

Suggested values for a 500dpi fingerprint image:

  [normim, mask, maskind] = ridgesegment(im, 16, 0.1)

See also: RIDGEORIENT, RIDGEFREQ, RIDGEFILTER

Peter Kovesi         
School of Computer Science & Software Engineering
The University of Western Australia
pk at csse uwa edu au
http://www.csse.uwa.edu.au/~pk

January 2005
"""

import numpy as np
from typing import Tuple
from FExtraction.normalise import normalise
from skimage.util import view_as_blocks


def ridgesegment(im: np.ndarray, blksze: int, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalise fingerprint image and segment ridge region
    
    Args:
        im: Fingerprint image to be segmented
        blksze: Block size for standard deviation calculation
        thresh: Threshold of standard deviation for ridge regions
        
    Returns:
        Tuple of (normim, mask, maskind)
        - normim: Normalised image with ridge regions having zero mean, unit std
        - mask: Binary mask indicating ridge regions (1) and non-ridge (0)
        - maskind: Indices of ridge region locations
    """
    # Normalise to have zero mean, unit std dev
    im = normalise(im, 0, 1)
    
    # Calculate standard deviation for each block
    # Pad image to make it divisible by block size
    rows, cols = im.shape
    pad_rows = (blksze - rows % blksze) % blksze
    pad_cols = (blksze - cols % blksze) % blksze
    
    if pad_rows > 0 or pad_cols > 0:
        im_padded = np.pad(im, ((0, pad_rows), (0, pad_cols)), mode='edge')
    else:
        im_padded = im
    
    # Create blocks and calculate standard deviation
    blocks = view_as_blocks(im_padded, (blksze, blksze))
    stddev_blocks = np.std(blocks, axis=(2, 3))
    
    # Resize stddev image back to original size
    stddevim = np.repeat(np.repeat(stddev_blocks, blksze, axis=0), blksze, axis=1)
    stddevim = stddevim[:rows, :cols]  # Crop to original size
    
    # Create mask based on threshold
    mask = (stddevim > thresh).astype(np.uint8)
    maskind = np.where(mask.flatten())[0]
    
    # Renormalise image so that the ridge regions have zero mean, unit std dev
    if len(maskind) > 0:
        ridge_pixels = im.flatten()[maskind]
        im_adjusted = im - np.mean(ridge_pixels)
        normim = im_adjusted / np.std(ridge_pixels)
    else:
        normim = im
    
    return normim, mask, maskind


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create a synthetic fingerprint-like image
    rows, cols = 256, 256
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Create ridge pattern
    ridge_freq = 0.1
    ridge_pattern = np.sin(2 * np.pi * ridge_freq * x) + 0.3 * np.random.randn(rows, cols)
    
    # Add noise to some regions
    noise_region = np.random.randn(rows, cols) * 0.1
    test_image = ridge_pattern + noise_region
    
    # Apply ridge segmentation
    normim, mask, maskind = ridgesegment(test_image, 16, 0.1)
    
    print(f"Original image shape: {test_image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Number of ridge pixels: {len(maskind)}")
    print(f"Ridge region percentage: {len(maskind) / (rows * cols) * 100:.1f}%")