"""
smoothen_frequency_image
smoothens the frequency image through a process of diffusion
Usage:
new_oimg = smoothen_frequency_image(fimg,RLOW,RHIGH,diff_cycles)
fimg       - frequency image image
nimg       - filtered frequency image
RLOW       - lowest allowed ridge separation
RHIGH      - highest allowed ridge separation
diff_cyles - number of diffusion cycles
Contact:
   ssc5@eng.buffalo.edu
   www.eng.buffalo.edu/~ssc5
Reference:
S. Chikkerur, C.Wu and V. Govindaraju, "Systematic approach for feature
extraction in Fingerprint Images", ICBA 2004
"""

import numpy as np
from scipy import ndimage

def smoothen_frequency_image(fimg, RLOW, RHIGH, diff_cycles):
    """
    Smoothens the frequency image through a process of diffusion
    
    Parameters:
    fimg - frequency image
    RLOW - lowest allowed ridge separation
    RHIGH - highest allowed ridge separation
    diff_cycles - number of diffusion cycles
    
    Returns:
    nfimg - filtered frequency image
    """
    valid_nbrs = 3  # uses only pixels with more than valid_nbrs for diffusion
    ht, wt = fimg.shape
    nfimg = fimg.copy()
    N = 1
    
    # ---------------------------------
    # perform diffusion
    # ---------------------------------
    # Create Gaussian kernel (equivalent to MATLAB's fspecial('gaussian', 2*N+1))
    kernel_size = 2*N + 1
    sigma = kernel_size / 6  # Approximate sigma
    
    # Create 1D Gaussian kernel manually
    x = np.arange(kernel_size) - kernel_size // 2
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    h = np.outer(kernel_1d, kernel_1d)
    h = h / np.sum(h)  # Normalize
    
    cycles = 0
    invalid_cnt = np.sum((fimg < RLOW) | (fimg > RHIGH))
    
    while (invalid_cnt > 0 and cycles < diff_cycles) or cycles < diff_cycles:
        # ---------------
        # pad the image
        # ---------------
        # Pad the rows
        fimg_padded = np.vstack([
            np.flipud(fimg[:N, :]),
            fimg,
            np.flipud(fimg[ht-N:ht, :])
        ])
        
        # Pad the columns
        fimg_padded = np.hstack([
            np.fliplr(fimg_padded[:, :N]),
            fimg_padded,
            np.fliplr(fimg_padded[:, wt-N:wt])
        ])
        
        # ---------------
        # perform diffusion
        # ---------------
        for i in range(N, ht + N):
            for j in range(N, wt + N):
                blk = fimg_padded[i-N:i+N+1, j-N:j+N+1]
                msk = (blk >= RLOW) & (blk <= RHIGH)
                
                if np.sum(msk) >= valid_nbrs:
                    blk_masked = blk * msk
                    h_masked = h * msk
                    nfimg[i-N, j-N] = np.sum(blk_masked * h) / np.sum(h_masked)
                else:
                    nfimg[i-N, j-N] = -1  # invalid value
        
        # ---------------
        # prepare for next iteration
        # ---------------
        fimg = nfimg.copy()
        invalid_cnt = np.sum((fimg < RLOW) | (fimg > RHIGH))
        cycles += 1
    
    return nfimg


# Example usage
if __name__ == "__main__":
    # Create a sample frequency image
    fimg = np.random.rand(32, 32) * 20 + 5  # Random values between 5 and 25
    RLOW = 3
    RHIGH = 18
    diff_cycles = 5
    
    # Test the function
    try:
        smoothed_fimg = smoothen_frequency_image(fimg, RLOW, RHIGH, diff_cycles)
        print("Frequency image smoothing completed successfully")
        print(f"Input shape: {fimg.shape}")
        print(f"Output shape: {smoothed_fimg.shape}")
        print(f"Input range: [{np.min(fimg):.3f}, {np.max(fimg):.3f}]")
        print(f"Output range: [{np.min(smoothed_fimg):.3f}, {np.max(smoothed_fimg):.3f}]")
        print(f"Valid pixels in output: {np.sum(smoothed_fimg >= 0)}")
    except Exception as e:
        print(f"Error during frequency image smoothing: {e}")