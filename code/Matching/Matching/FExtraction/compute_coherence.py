"""
------------------------------------------------------------------------
compute_coherence
Computes the coherence image. 
Usage:
[cimg] = compute_coherence(oimg)
oimg - orientation image
cimg - coherence image(0-low coherence,1-high coherence)
Contact:
   ssc5@eng.buffalo.edu
   www.eng.buffalo.edu/~ssc5
Reference:
A. Ravishankar Rao,"A taxonomy of texture description", Springer Verlag
------------------------------------------------------------------------
"""

import numpy as np
from typing import Tuple


def compute_coherence(oimg: np.ndarray) -> np.ndarray:
    """
    Compute the coherence image from orientation image
    
    Args:
        oimg: orientation image
        
    Returns:
        cimg: coherence image (0-low coherence, 1-high coherence)
    """
    h, w = oimg.shape
    cimg = np.zeros((h, w))
    N = 2
    
    # ---------------
    # pad the image
    # ---------------
    # pad the rows
    oimg_padded = np.vstack([
        np.flipud(oimg[:N, :]),
        oimg,
        np.flipud(oimg[h-N:h, :])
    ])
    
    # pad the cols
    oimg_padded = np.hstack([
        np.fliplr(oimg_padded[:, :N]),
        oimg_padded,
        np.fliplr(oimg_padded[:, w-N:w])
    ])
    
    # compute coherence
    for i in range(N, h + N):
        for j in range(N, w + N):
            th = oimg_padded[i, j]
            blk = oimg_padded[i-N:i+N+1, j-N:j+N+1]
            cimg[i-N, j-N] = np.sum(np.abs(np.cos(blk - th))) / ((2*N + 1)**2)
    
    return cimg


if __name__ == "__main__":
    # Example usage
    test_oimg = np.random.rand(100, 100) * np.pi
    coherence_img = compute_coherence(test_oimg)
    print(f"Coherence image shape: {coherence_img.shape}")
    print(f"Coherence range: [{coherence_img.min():.3f}, {coherence_img.max():.3f}]")