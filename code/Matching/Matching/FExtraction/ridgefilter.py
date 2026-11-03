"""
RIDGEFILTER - enhances fingerprint image via oriented filters

Function to enhance fingerprint image via oriented filters

Usage:
 newim =  ridgefilter(im, orientim, freqim, kx, ky, showfilter)

Arguments:
        im       - Image to be processed.
        orientim - Ridge orientation image, obtained from RIDGEORIENT.
        freqim   - Ridge frequency image, obtained from RIDGEFREQ.
        kx, ky   - Scale factors specifying the filter sigma relative
                   to the wavelength of the filter.  This is done so
                   that the shapes of the filters are invariant to the
                   scale.  kx controls the sigma in the x direction
                   which is along the filter, and hence controls the
                   bandwidth of the filter.  ky controls the sigma
                   across the filter and hence controls the
                   orientational selectivity of the filter. A value of
                   0.5 for both kx and ky is a good starting point.
        showfilter - An optional flag 0/1.  When set an image of the
                     largest scale filter is displayed for inspection.

Returns:
        newim    - The enhanced image

See also: RIDGEORIENT, RIDGEFREQ, RIDGESEGMENT

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
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from typing import Optional


def ridgefilter(im: np.ndarray, orient: np.ndarray, freq: np.ndarray, 
                kx: float, ky: float, showfilter: bool = False) -> np.ndarray:
    """
    Enhance fingerprint image via oriented filters
    
    Args:
        im: Image to be processed
        orient: Ridge orientation image
        freq: Ridge frequency image
        kx: Scale factor for filter sigma in x direction
        ky: Scale factor for filter sigma in y direction
        showfilter: Flag to display largest scale filter
        
    Returns:
        newim: Enhanced image
    """
    angleInc = 3  # Fixed angle increment between filter orientations in degrees
    
    im = im.astype(np.float64)
    rows, cols = im.shape
    newim = np.zeros((rows, cols))
    
    # Find where there is valid frequency data
    validr, validc = np.where(freq > 0)
    
    if len(validr) == 0:
        return newim
    
    # Round frequencies to nearest 0.01 to reduce distinct frequencies
    freq_rounded = freq.copy()
    for i in range(len(validr)):
        r, c = validr[i], validc[i]
        freq_rounded[r, c] = np.round(freq[r, c] * 100) / 100
    
    # Generate array of distinct frequencies
    valid_freq_values = []
    for i in range(len(validr)):
        r, c = validr[i], validc[i]
        valid_freq_values.append(freq_rounded[r, c])
    
    unfreq = np.unique(valid_freq_values)
    
    # Generate frequency index lookup table
    freqindex = np.ones(100, dtype=int)
    for k, freq_val in enumerate(unfreq):
        idx = int(np.round(freq_val * 100))
        if 0 <= idx < 100:
            freqindex[idx] = k
    
    # Generate filters for distinct frequencies and orientations
    num_orientations = 180 // angleInc
    filters = {}
    sze = np.zeros(len(unfreq), dtype=int)
    
    for k, freq_val in enumerate(unfreq):
        sigmax = (1 / freq_val) * kx
        sigmay = (1 / freq_val) * ky
        
        sze[k] = int(np.round(3 * max(sigmax, sigmay)))
        x, y = np.meshgrid(np.arange(-sze[k], sze[k] + 1),
                          np.arange(-sze[k], sze[k] + 1))
        
        # Create reference filter
        reffilter = (np.exp(-(x**2 / sigmax**2 + y**2 / sigmay**2) / 2) * 
                    np.cos(2 * np.pi * freq_val * x))
        
        # Generate rotated versions of the filter
        for o in range(num_orientations):
            angle = -(o * angleInc + 90)  # Note: negative for correct rotation
            rotated_filter = rotate(reffilter, angle, reshape=False, order=1)
            filters[(k, o)] = rotated_filter
    
    if showfilter and len(unfreq) > 0:
        plt.figure(figsize=(8, 6))
        plt.imshow(filters[(0, num_orientations-1)], cmap='gray')
        plt.title('Largest Scale Filter')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(1.0)  # Display for 1 second
        plt.close()  # Automatically close the figure
    
    # Find indices of matrix points greater than maxsze from image boundary
    if len(sze) > 0:
        maxsze = sze[0]
        
        # Filter valid points that are far enough from boundaries
        valid_mask = ((validr > maxsze) & (validr < rows - maxsze) & 
                     (validc > maxsze) & (validc < cols - maxsze))
        finalr = validr[valid_mask]
        finalc = validc[valid_mask]
        
        # Convert orientation matrix values to index values
        maxorientindex = num_orientations
        orientindex = np.round(orient / np.pi * 180 / angleInc).astype(int)
        
        # Handle wraparound
        orientindex[orientindex < 0] += maxorientindex
        orientindex[orientindex >= maxorientindex] -= maxorientindex
        
        # Apply filtering
        for i in range(len(finalr)):
            r, c = finalr[i], finalc[i]
            
            # Find filter corresponding to freq(r,c)
            freq_idx = int(np.round(freq_rounded[r, c] * 100))
            if 0 <= freq_idx < 100:
                filterindex = freqindex[freq_idx]
                
                if filterindex < len(unfreq):
                    s = sze[filterindex]
                    orient_idx = orientindex[r, c]
                    
                    if (filterindex, orient_idx) in filters:
                        # Extract image patch and apply filter
                        patch = im[r-s:r+s+1, c-s:c+s+1]
                        filter_kernel = filters[(filterindex, orient_idx)]
                        
                        if patch.shape == filter_kernel.shape:
                            newim[r, c] = np.sum(patch * filter_kernel)
    
    return newim


if __name__ == "__main__":
    # Example usage
    test_im = np.random.rand(200, 200)
    test_orient = np.random.rand(200, 200) * np.pi
    test_freq = np.random.rand(200, 200) * 0.2 + 0.05
    
    enhanced_im = ridgefilter(test_im, test_orient, test_freq, 0.5, 0.5, showfilter=True)
    print(f"Enhanced image shape: {enhanced_im.shape}")
    print(f"Enhancement range: [{enhanced_im.min():.6f}, {enhanced_im.max():.6f}]")