"""
fft_enhance_cubs
enhances the fingerprint image
syntax:
[oimg,fimg,bwimg,eimg,enhimg] =  fft_enhance_cubs(img, BLKSZ)
oimg -  [OUT] block orientation image(can be viewed using
        view_orientation_image.m)
fimg  - [OUT] block frequency image(indicates ridge spacing)
bwimg - [OUT] shows angular bandwidth image(filter bandwidth adapts near the
        singular points)
eimg  - [OUT] energy image. Indicates the 'ridgeness' of a block (can be 
        used for fingerprint segmentation)
enhimg- [OUT] enhanced image
img   - [IN]  input fingerprint image (HAS to be of DOUBLE type)
Contact:
   ssc5@eng.buffalo.edu
   www.eng.buffalo.edu/~ssc5
Reference:
S. Chikkerur,C. Wu and Govindaraju, "A Systematic approach for 
feature extraction in fingerprint images",ICBA 2004
"""

import numpy as np
import scipy.io
from scipy.fft import fft2, ifft2
from scipy.ndimage import zoom
import os

# Global variable
NFFT = None

def fft_enhance_cubs(img, BLKSZ):
    """
    Enhances the fingerprint image using FFT-based filtering
    
    Parameters:
    img: input fingerprint image (should be of double type)
    BLKSZ: block size for processing
    
    Returns:
    enhimg: enhanced image
    cimg: coherence image
    oimg: block orientation image
    fimg: block frequency image
    bwimg: angular bandwidth image
    eimg: energy image
    """
    global NFFT
    
    if BLKSZ > 0:
        NFFT = 32        # size of FFT
        OVRLP = 2        # size of overlap
        ALPHA = 0.5      # root filtering
        RMIN = 4         # min allowable ridge spacing
        RMAX = 40        # maximum allowable ridge spacing
        ESTRETCH = 20    # for contrast enhancement
        ETHRESH = 19     # threshold for the energy
    else:
        NFFT = 32        # size of FFT
        BLKSZ = 12       # size of the block
        OVRLP = 6        # size of overlap
        ALPHA = 0.5      # root filtering
        RMIN = 3         # min allowable ridge spacing
        RMAX = 18        # maximum allowable ridge spacing
        ESTRETCH = 20    # for contrast enhancement
        ETHRESH = 6      # threshold for the energy
    
    nHt, nWt = img.shape
    img = img.astype(np.float64)  # convert to DOUBLE
    nBlkHt = int(np.floor((nHt - 2*OVRLP) / BLKSZ))
    nBlkWt = int(np.floor((nWt - 2*OVRLP) / BLKSZ))
    fftSrc = np.zeros((nBlkHt * nBlkWt, NFFT * NFFT))  # stores FFT
    nWndSz = BLKSZ + 2*OVRLP  # size of analysis window
    
    # -------------------------
    # allocate outputs
    # -------------------------
    oimg = np.zeros((nBlkHt, nBlkWt))
    fimg = np.zeros((nBlkHt, nBlkWt))
    bwimg = np.zeros((nBlkHt, nBlkWt))
    eimg = np.zeros((nBlkHt, nBlkWt))
    enhimg = np.zeros((nHt, nWt))
    
    # -------------------------
    # precomputations
    # -------------------------
    x, y = np.meshgrid(np.arange(nWndSz), np.arange(nWndSz))
    dMult = (-1.0) ** (x + y)  # used to center the FFT
    x, y = np.meshgrid(np.arange(-NFFT//2, NFFT//2), np.arange(-NFFT//2, NFFT//2))
    r = np.sqrt(x**2 + y**2) + np.finfo(float).eps
    th = np.arctan2(y, x)
    th[th < 0] = th[th < 0] + np.pi
    w = raised_cosine_window(BLKSZ, OVRLP)  # spectral window
    
    # -------------------------
    # Load filters
    # -------------------------
    # Load angular filters from .mat files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        angf_pi_4_data = scipy.io.loadmat(os.path.join(current_dir, 'angular_filters_pi_4.mat'))
        angf_pi_4 = angf_pi_4_data['angf']
    except:
        # If .mat file not found, create a dummy filter
        angf_pi_4 = np.ones((NFFT*NFFT, 32))
    
    try:
        angf_pi_2_data = scipy.io.loadmat(os.path.join(current_dir, 'angular_filters_pi_2.mat'))
        angf_pi_2 = angf_pi_2_data['angf']
    except:
        # If .mat file not found, create a dummy filter
        angf_pi_2 = np.ones((NFFT*NFFT, 32))
    
    # -------------------------
    # Bandpass filter
    # -------------------------
    FLOW = NFFT / RMAX
    FHIGH = NFFT / RMIN
    
    dRLow = 1.0 / (1 + (r/FHIGH)**4)    # low pass butterworth filter
    dRHigh = 1.0 / (1 + (FLOW/r)**4)    # high pass butterworth filter
    dBPass = dRLow * dRHigh             # bandpass
    
    # -------------------------
    # FFT Analysis
    # -------------------------
    for i in range(nBlkHt):
        nRow = i * BLKSZ + OVRLP
        for j in range(nBlkWt):
            nCol = j * BLKSZ + OVRLP
            # extract local block
            blk = img[nRow-OVRLP:nRow+BLKSZ+OVRLP, nCol-OVRLP:nCol+BLKSZ+OVRLP]
            # remove dc
            dAvg = np.sum(blk) / (nWndSz * nWndSz)
            blk = blk - dAvg  # remove DC content
            blk = blk * w     # multiply by spectral window
            
            # --------------------------
            # do pre filtering
            # --------------------------
            blkfft = fft2(blk * dMult, s=(NFFT, NFFT))
            blkfft = blkfft * dBPass                    # band pass filtering
            dEnergy = np.abs(blkfft)**2
            blkfft = blkfft * np.sqrt(dEnergy)          # root filtering(for diffusion)
            fftSrc[nBlkWt*i + j, :] = np.real(blkfft.flatten())  # Fix ComplexWarning
            dEnergy = np.abs(blkfft)**2                 # ----REDUCE THIS COMPUTATION----
            
            # --------------------------
            # compute statistics
            # --------------------------
            dTotal = np.sum(dEnergy) / (NFFT * NFFT)
            fimg[i, j] = NFFT / (compute_mean_frequency(dEnergy, r) + np.finfo(float).eps)  # ridge separation
            oimg[i, j] = compute_mean_angle(dEnergy, th)                                     # ridge angle
            eimg[i, j] = np.log(dTotal + np.finfo(float).eps)                               # used for segmentation
    
    # -------------------------
    # precomputations for reconstruction
    # -------------------------
    x, y = np.meshgrid(np.arange(-NFFT//2, NFFT//2), np.arange(-NFFT//2, NFFT//2))
    dMult = (-1.0) ** (x + y)  # used to center the FFT
    
    # -------------------------
    # process the resulting maps
    # -------------------------
    # Import required functions
    from FExtraction.smoothen_orientation_image import smoothen_orientation_image
    from FExtraction.smoothen_frequency_image import smoothen_frequency_image
    from FExtraction.compute_coherence import compute_coherence
    
    for i in range(3):
        oimg = smoothen_orientation_image(oimg)            # smoothen orientation image
    
    fimg = smoothen_frequency_image(fimg, RMIN, RMAX, 5)   # diffuse frequency image
    cimg = compute_coherence(oimg)                         # coherence image for bandwidth
    bwimg = get_angular_bw_image(cimg)                     # QUANTIZED bandwidth image
    
    # -------------------------
    # FFT reconstruction
    # -------------------------
    for i in range(nBlkHt):
        for j in range(nBlkWt):
            nRow = i * BLKSZ + OVRLP
            nCol = j * BLKSZ + OVRLP
            
            # --------------------------
            # apply the filters
            # --------------------------
            blkfft = fftSrc[nBlkWt*i + j, :].reshape((NFFT, NFFT))
            
            # --------------------------
            # reconstruction
            # --------------------------
            af = get_angular_filter(oimg[i, j], bwimg[i, j], angf_pi_4, angf_pi_2)
            blkfft = blkfft * af
            blk = np.real(ifft2(blkfft) * dMult)
            enhimg[nRow:nRow+BLKSZ, nCol:nCol+BLKSZ] = blk[OVRLP:OVRLP+BLKSZ, OVRLP:OVRLP+BLKSZ]
    
    # end block processing
    # --------------------------
    # contrast enhancement
    # --------------------------
    enhimg = np.sqrt(np.abs(enhimg)) * np.sign(enhimg)
    mx = np.max(enhimg)
    mn = np.min(enhimg)
    enhimg = ((enhimg - mn) / (mx - mn) * 254 + 1).astype(np.uint8)
    
    # --------------------------
    # clean up the image
    # --------------------------
    emsk = zoom(eimg, (nHt/eimg.shape[0], nWt/eimg.shape[1]), order=1)
    enhimg[emsk < ETHRESH] = 128
    
    return enhimg, cimg, oimg, fimg, bwimg, eimg


def raised_cosine(nBlkSz, nOvrlp):
    """
    returns 1D spectral window
    
    Parameters:
    nBlkSz - the window is constant here
    nOvrlp - the window has transition here
    
    Returns:
    y - 1D raised cosine function
    """
    nWndSz = nBlkSz + 2*nOvrlp
    x = np.abs(np.arange(-nWndSz//2, nWndSz//2))
    y = 0.5 * (np.cos(np.pi * (x - nBlkSz/2) / nOvrlp) + 1)
    y[np.abs(x) < nBlkSz/2] = 1
    return y


def raised_cosine_window(blksz, ovrlp):
    """
    returns 2D spectral window
    
    Parameters:
    blksz - the window is constant here
    ovrlp - the window has transition here
    
    Returns:
    w - 2D raised cosine function
    """
    y = raised_cosine(blksz, ovrlp)
    w = np.outer(y, y)
    return w


def get_angular_filter(t0, bw, angf_pi_4, angf_pi_2):
    """
    generates an angular filter centered around 'th' and with bandwidth 'bw'
    the filters in angf_xx are precomputed using angular_filter_bank.m
    
    Parameters:
    t0 - mean angle (obtained from orientation image)
    bw - angular bandwidth(obtained from bandwidth image)
    angf_pi_4, angf_pi_2 - precomputed filters
    
    Returns:
    r - angular filter of size NFFTxNFFT
    """
    global NFFT
    TSTEPS = angf_pi_4.shape[1]
    DELTAT = np.pi / TSTEPS
    
    # get the closest filter
    i = int(np.floor((t0 + DELTAT/2) / DELTAT))
    i = (i % TSTEPS)
    
    if bw == np.pi/4:
        r = angf_pi_4[:, i].reshape((NFFT, NFFT)).T
    elif bw == np.pi/2:
        r = angf_pi_2[:, i].reshape((NFFT, NFFT)).T
    else:
        r = np.ones((NFFT, NFFT))
    
    return r


def get_angular_bw_image(c):
    """
    the bandwidth allocation is currently based on heuristics
    (domain knowledge :)). 
    
    Parameters:
    c - coherence image
    
    Returns:
    bwimg - bandwidth image
    """
    bwimg = np.zeros(c.shape)
    bwimg[:, :] = np.pi/2                    # med bw
    bwimg[c <= 0.7] = np.pi                  # high bw
    bwimg[c >= 0.9] = np.pi/4                # low bw
    return bwimg


def compute_mean_angle(dEnergy, th):
    """
    Compute mean angle from energy and angle matrices
    
    Parameters:
    dEnergy - energy matrix
    th - angle matrix
    
    Returns:
    mth - mean angle
    """
    global NFFT
    sth = np.sin(2*th)
    cth = np.cos(2*th)
    num = np.sum(dEnergy * sth)
    den = np.sum(dEnergy * cth)
    mth = 0.5 * np.arctan2(num, den)
    if mth < 0:
        mth = mth + np.pi
    return mth


def compute_mean_frequency(dEnergy, r):
    """
    Compute mean frequency from energy and radius matrices
    
    Parameters:
    dEnergy - energy matrix
    r - radius matrix
    
    Returns:
    mr - mean frequency
    """
    global NFFT
    num = np.sum(dEnergy * r)
    den = np.sum(dEnergy)
    mr = num / (den + np.finfo(float).eps)
    return mr


# Example usage
if __name__ == "__main__":
    # Create a sample fingerprint image
    img = np.random.rand(256, 256) * 255
    img = img.astype(np.float64)
    
    # Test the function
    try:
        enhimg, cimg, oimg, fimg, bwimg, eimg = fft_enhance_cubs(img, 16)
        print("FFT enhancement completed successfully")
        print(f"Enhanced image shape: {enhimg.shape}")
        print(f"Orientation image shape: {oimg.shape}")
        print(f"Frequency image shape: {fimg.shape}")
    except Exception as e:
        print(f"Error during FFT enhancement: {e}")