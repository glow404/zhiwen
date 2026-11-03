"""
RIDGEFREQ - Calculates a ridge frequency image

Function to estimate the fingerprint ridge frequency across a
fingerprint image. This is done by considering blocks of the image and
determining a ridgecount within each block by a call to FREQEST.

Usage:
 [freqim, medianfreq] =  ridgefreq(im, mask, orientim, blksze, windsze,
                                   minWaveLength, maxWaveLength)

Arguments:
        im       - Image to be processed.
        mask     - Mask defining ridge regions (obtained from RIDGESEGMENT)
        orientim - Ridge orientation image (obtained from RIDGORIENT)
        blksze   - Size of image block to use (say 32) 
        windsze  - Window length used to identify peaks. This should be
                   an odd integer, say 3 or 5.
        minWaveLength,  maxWaveLength - Minimum and maximum ridge
                    wavelengths, in pixels, considered acceptable.

Returns:
        freqim     - An image  the same size as im with  values set to
                     the estimated ridge spatial frequency within each
                     image block.  If a  ridge frequency cannot be
                     found within a block, or cannot be found within the
                     limits set by min and max Wavlength freqim is set
                     to zeros within that block.
        medianfreq - Median frequency value evaluated over all the
                     valid regions of the image.

Suggested parameters for a 500dpi fingerprint image
  [freqim, medianfreq] = ridgefreq(im,orientim, 32, 5, 5, 15);

I seem to find that the median frequency value is more useful as an
input to RIDGEFILTER than the more detailed freqim.  This is possibly
due to deficiencies in FREQEST.

See also: RIDGEORIENT, FREQEST, RIDGESEGMENT

Reference: 
Hong, L., Wan, Y., and Jain, A. K. Fingerprint image enhancement:
Algorithm and performance evaluation. IEEE Transactions on Pattern
Analysis and Machine Intelligence 20, 8 (1998), 777 789.

Peter Kovesi  
School of Computer Science & Software Engineering
The University of Western Australia
pk at csse uwa edu au
http://www.csse.uwa.edu.au/~pk

January 2005
"""

import numpy as np
from typing import Tuple
from FExtraction.freqest import freqest


def ridgefreq(im: np.ndarray, mask: np.ndarray, orient: np.ndarray, 
              blksze: int, windsze: int, minWaveLength: float, 
              maxWaveLength: float) -> Tuple[np.ndarray, float]:
    """
    Calculate ridge frequency image
    
    Args:
        im: Image to be processed
        mask: Mask defining ridge regions
        orient: Ridge orientation image
        blksze: Size of image block to use
        windsze: Window length used to identify peaks (odd integer)
        minWaveLength: Minimum ridge wavelength in pixels
        maxWaveLength: Maximum ridge wavelength in pixels
        
    Returns:
        freq: Ridge frequency image
        medianfreq: Median frequency value over valid regions
    """
    rows, cols = im.shape
    freq = np.zeros_like(im)
    
    # Process image in blocks
    for r in range(0, rows - blksze, blksze):
        for c in range(0, cols - blksze, blksze):
            # Extract blocks
            blkim = im[r:r+blksze, c:c+blksze]
            blkor = orient[r:r+blksze, c:c+blksze]
            
            # Estimate frequency for this block
            freq[r:r+blksze, c:c+blksze] = freqest(blkim, blkor, windsze, 
                                                   minWaveLength, maxWaveLength)
    
    # Mask out frequencies calculated for non ridge regions
    freq = freq * mask
    
    # Find median frequency over all the valid regions of the image
    valid_freq = freq[freq > 0]
    if len(valid_freq) > 0:
        medianfreq = np.median(valid_freq)
    else:
        medianfreq = 0.0
    
    return freq, medianfreq


if __name__ == "__main__":
    # Example usage
    test_im = np.random.rand(200, 200)
    test_mask = np.ones((200, 200))
    test_orient = np.random.rand(200, 200) * np.pi
    
    freq_im, median_freq = ridgefreq(test_im, test_mask, test_orient, 32, 5, 5, 15)
    print(f"Frequency image shape: {freq_im.shape}")
    print(f"Median frequency: {median_freq:.6f}")
    print(f"Frequency range: [{freq_im.min():.6f}, {freq_im.max():.6f}]")