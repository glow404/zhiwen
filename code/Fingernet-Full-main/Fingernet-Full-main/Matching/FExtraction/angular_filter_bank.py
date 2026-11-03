"""
----------------------------------------------------------
angular_filter_bank
precomputes angular filter bank and also generates a C
header file with filter coefficients. The python variable
angf is stored in file 'angular_filters.npz' and the C 
variables are stored in 'angular_fiter.h'.
There are TSTEPS number of raised cosine filter that span the 
interval [0,180]
usage
 angular_filter_bank(BW,fname)
BW    - bandwidth (radians)
fname - file name
Contact:
   ssc5@eng.buffalo.edu
   www.eng.buffalo.edu/~ssc5
References:
B.G.Sherlock and  D.M.Monro and  K.Millard,"Fingerprint Enhancement by
directional Fourier Filtering",IEEE Visual Image Signal Processing,
141(2), pp. 87--94, 1994
----------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def angular_filter_bank(BW: float, fname: str) -> None:
    """
    Generate angular filter bank for fingerprint enhancement
    
    Args:
        BW: bandwidth (radians)
        fname: file name for saving filters
    """
    plt.close('all')
    
    # ---------------
    # parameters
    # ---------------
    FFTN = 32  # 64; 32;
    TSTEPS = 12  # 15 degrees interval
    DELTAT = np.pi / TSTEPS
    
    # ---------------
    # precompute
    # ---------------
    x, y = np.meshgrid(np.arange(-FFTN//2, FFTN//2), 
                       np.arange(-FFTN//2, FFTN//2))
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y, x)
    th[th < 0] = th[th < 0] + 2*np.pi  # unsigned
    filter_bank = []
    
    # -------------------------
    # generate the filters
    # -------------------------
    for i in range(TSTEPS):
        t0 = i * DELTAT
        t1 = t0 + np.pi  # for the other lobe
        
        # -----------------
        # first lobe
        # -----------------
        d = angular_distance(th, t0)
        msk = 1 + np.cos(d * np.pi / BW)
        msk[d > BW] = 0
        rmsk = msk.copy()  # save first lobe
        
        # -----------------
        # second lobe
        # -----------------
        d = angular_distance(th, t1)
        msk = 1 + np.cos(d * np.pi / BW)
        msk[d > BW] = 0
        rmsk = rmsk + msk
        
        # Display filter (optional - comment out for batch processing)
        # plt.imshow(rmsk)
        # plt.pause(0.1)
        
        rmsk = rmsk.T
        filter_bank.append(rmsk.flatten())
    
    # -----------------
    # save the filters
    # -----------------
    angf = np.column_stack(filter_bank)
    np.savez(f'{fname}.npz', angf=angf)
    
    # ----------------------
    # write to a C file
    # ----------------------
    with open(f'{fname}.h', 'w') as fp:
        fp.write('{\n')
        for i in range(angf.shape[1]):
            print(f"Processing filter {i+1}")
            k = 1
            fp.write('{')
            for j in range(angf.shape[0]):
                fp.write(f'{angf[j, i]:.6f},')
                if k == 32:
                    k = 0
                    fp.write('\n')
                k += 1
            fp.write('},\n')
        fp.write('};\n')


def angular_distance(th: np.ndarray, t0: float) -> np.ndarray:
    """
    Compute angular distance - acute angle
    
    Args:
        th: angle array
        t0: reference angle
        
    Returns:
        Angular distance array
    """
    d = np.abs(th - t0)
    d = np.minimum(d, 2*np.pi - d)
    return d


if __name__ == "__main__":
    # Example usage
    BW = np.pi/4  # bandwidth
    fname = "angular_filters_pi_4"
    angular_filter_bank(BW, fname)