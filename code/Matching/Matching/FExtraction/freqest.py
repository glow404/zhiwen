"""
FREQEST - Estimate fingerprint ridge frequency within image block

Function to estimate the fingerprint ridge frequency within a small block
of a fingerprint image.  This function is used by RIDGEFREQ

Usage:
 freqim =  freqest(im, orientim, windsze, minWaveLength, maxWaveLength)

Arguments:
        im       - Image block to be processed.
        orientim - Ridge orientation image of image block.
        windsze  - Window length used to identify peaks. This should be
                   an odd integer, say 3 or 5.
        minWaveLength,  maxWaveLength - Minimum and maximum ridge
                    wavelengths, in pixels, considered acceptable.

Returns:
        freqim    - An image block the same size as im with all values
                    set to the estimated ridge spatial frequency.  If a
                    ridge frequency cannot be found, or cannot be found
                    within the limits set by min and max Wavlength
                    freqim is set to zeros.

Suggested parameters for a 500dpi fingerprint image
  freqim = freqest(im,orientim, 5, 5, 15);

See also:  RIDGEFREQ, RIDGEORIENT, RIDGESEGMENT

Note I am not entirely satisfied with the output of this function.

Peter Kovesi 
School of Computer Science & Software Engineering
The University of Western Australia
pk at csse uwa edu au
http://www.csse.uwa.edu.au/~pk

January 2005
"""

import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import maximum_filter1d
import matplotlib.pyplot as plt
from typing import Optional


def freqest(im: np.ndarray, orientim: np.ndarray, windsze: int, 
            minWaveLength: float, maxWaveLength: float, debug: bool = False) -> np.ndarray:
    """
    Estimate fingerprint ridge frequency within image block
    
    Args:
        im: Image block to be processed
        orientim: Ridge orientation image of image block
        windsze: Window length used to identify peaks (odd integer)
        minWaveLength: Minimum ridge wavelength in pixels
        maxWaveLength: Maximum ridge wavelength in pixels
        debug: Whether to show debug plots
        
    Returns:
        freqim: Image block with estimated ridge spatial frequency
    """
    rows, cols = im.shape
    
    # Find mean orientation within the block
    # Average the sines and cosines of the doubled angles to avoid wraparound
    orientim_doubled = 2 * orientim.flatten()
    cosorient = np.mean(np.cos(orientim_doubled))
    sinorient = np.mean(np.sin(orientim_doubled))
    orient = np.arctan2(sinorient, cosorient) / 2
    
    # Rotate the image block so that the ridges are vertical
    rotim = rotate(im, orient / np.pi * 180 + 90, reshape=False, order=1)
    
    # Crop the image so that the rotated image doesn't contain invalid regions
    cropsze = int(rows / np.sqrt(2))
    offset = int((rows - cropsze) / 2)
    rotim = rotim[offset:offset+cropsze, offset:offset+cropsze]
    
    # Sum down the columns to get a projection of the grey values down the ridges
    proj = np.sum(rotim, axis=0)
    
    # Find peaks in projected grey values using maximum filter
    # This is equivalent to MATLAB's ordfilt2 with max operation
    dilation = maximum_filter1d(proj, size=windsze, mode='constant')
    maxpts = (dilation == proj) & (proj > np.mean(proj))
    maxind = np.where(maxpts)[0]
    
    # Determine the spatial frequency of the ridges
    if len(maxind) < 2:
        freqim = np.zeros_like(im)
    else:
        NoOfPeaks = len(maxind)
        waveLength = (maxind[-1] - maxind[0]) / (NoOfPeaks - 1)
        
        if minWaveLength < waveLength < maxWaveLength:
            freqim = (1 / waveLength) * np.ones_like(im)
        else:
            freqim = np.zeros_like(im)
    
    if debug:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(rotim, cmap='gray')
        plt.title('Rotated Image')
        
        plt.subplot(1, 2, 2)
        plt.plot(proj)
        plt.axhline(y=np.mean(proj), color='g', linestyle='--', label='Mean')
        
        if len(maxind) >= 2:
            plt.plot(maxind, dilation[maxind], 'r*', markersize=8, label='Peaks')
            waveLength = (maxind[-1] - maxind[0]) / (NoOfPeaks - 1)
            plt.title(f'Projection (Wavelength: {waveLength:.2f})')
        else:
            plt.title('Projection (No peaks found)')
        
        plt.legend()
        plt.show(block=False)
        plt.pause(1.0)  # Display for 1 second
        plt.close()  # Automatically close the figure
    
    return freqim


if __name__ == "__main__":
    # Example usage
    test_im = np.random.rand(32, 32)
    test_orient = np.random.rand(32, 32) * np.pi
    
    freq_result = freqest(test_im, test_orient, 5, 5, 15, debug=True)
    print(f"Frequency estimation result shape: {freq_result.shape}")
    print(f"Frequency range: [{freq_result.min():.6f}, {freq_result.max():.6f}]")