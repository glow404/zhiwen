"""
输入：指纹图像和块大小 BLKSZ。
输出：增强图像 enhimg，连贯性图 cimg，方向图 oimg，频率图 fimg，带宽图 bwimg，能量图 eimg。
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from skimage.transform import resize
from compute_coherence import compute_coherence

def raised_cosine_window(blksz, ovrlp):
    nWndSz = blksz + 2*ovrlp
    x = np.abs(np.arange(-nWndSz/2, nWndSz/2))
    y = 0.5 * (np.cos(np.pi*(x - blksz/2)/ovrlp) + 1.0)
    y[np.abs(x) < blksz/2] = 1.0
    w = np.outer(y, y)
    return w

def compute_mean_angle(dEnergy, th):
    sth = np.sin(2*th)
    cth = np.cos(2*th)
    num = np.sum(dEnergy * sth)
    den = np.sum(dEnergy * cth) + np.finfo(float).eps
    mth = 0.5 * np.arctan2(num, den)
    if mth < 0:
        mth = mth + np.pi
    return mth

def compute_mean_frequency(dEnergy, r):
    num = np.sum(dEnergy * r)
    den = np.sum(dEnergy) + np.finfo(float).eps
    return num / den

def get_angular_bw_image(cimg):
    bwimg = np.full_like(cimg, np.pi/2)
    bwimg[cimg <= 0.7] = np.pi
    bwimg[cimg >= 0.9] = np.pi/4
    return bwimg

def get_angular_filter(t0, bw, angf_pi_4, angf_pi_2, NFFT):
    # angf arrays expected shape (NFFT*NFFT, TSTEPS)
    TSTEPS = angf_pi_4.shape[1] if (angf_pi_4 is not None) else 1
    DELTAT = np.pi / TSTEPS
    i = int(np.floor((t0 + DELTAT/2.0) / DELTAT))
    i = (i % TSTEPS)
    if bw == np.pi/4 and angf_pi_4 is not None:
        r = angf_pi_4[:, i].reshape(NFFT, NFFT).T
    elif bw == np.pi/2 and angf_pi_2 is not None:
        r = angf_pi_2[:, i].reshape(NFFT, NFFT).T
    else:
        r = np.ones((NFFT, NFFT), dtype=float)
    return r

def smoothen_orientation_image(oimg):
    # vector smoothing via double-angle and gaussian blur
    cos2 = np.cos(2*oimg)
    sin2 = np.sin(2*oimg)
    cos2s = gaussian_filter(cos2, sigma=1)
    sin2s = gaussian_filter(sin2, sigma=1)
    o = 0.5 * np.arctan2(sin2s, cos2s)
    o[o < 0] += np.pi
    return o

def smoothen_frequency_image(fimg, RMIN, RMAX, iters=5):
    f = gaussian_filter(fimg, sigma=1)
    # clip to RMIN..RMAX
    f = np.clip(f, RMIN, RMAX)
    return f

def fft_enhance_cubs(img, BLKSZ):
    """
    Python approximate reimplementation of fft_enhance_cubs.m
    Returns: enhimg, cimg, oimg, fimg, bwimg, eimg
    """
    img = np.asarray(img, dtype=np.float64)
    if BLKSZ > 0:
        NFFT = 32
        OVRLP = 2
        ALPHA = 0.5
        RMIN = 4
        RMAX = 40
        ESTRETCH = 20
        ETHRESH = 19
    else:
        NFFT = 32
        BLKSZ = 12
        OVRLP = 6
        ALPHA = 0.5
        RMIN = 3
        RMAX = 18
        ESTRETCH = 20
        ETHRESH = 6

    nHt, nWt = img.shape
    img_d = img.astype(np.float64)
    nBlkHt = int(np.floor((nHt - 2*OVRLP) / BLKSZ))
    nBlkWt = int(np.floor((nWt - 2*OVRLP) / BLKSZ))
    fftSrc = np.zeros((nBlkHt*nBlkWt, NFFT*NFFT), dtype=complex)
    nWndSz = BLKSZ + 2*OVRLP

    oimg = np.zeros((nBlkHt, nBlkWt), dtype=float)
    fimg = np.zeros((nBlkHt, nBlkWt), dtype=float)
    bwimg = np.zeros((nBlkHt, nBlkWt), dtype=float)
    eimg = np.zeros((nBlkHt, nBlkWt), dtype=float)
    enhimg = np.zeros((nHt, nWt), dtype=float)

    # precomputations
    xw = np.arange(0, nWndSz)
    yw = np.arange(0, nWndSz)
    X, Y = np.meshgrid(xw, yw)
    dMult_small = (-1.0) ** (X + Y)

    x, y = np.meshgrid(np.arange(-NFFT/2, NFFT/2), np.arange(-NFFT/2, NFFT/2))
    r = np.sqrt(x.astype(float)**2 + y.astype(float)**2) + np.finfo(float).eps
    th = np.arctan2(y, x)
    th[th < 0] += np.pi

    w = raised_cosine_window(BLKSZ, OVRLP)

    # try load angular filters if saved as .npy
    try:
        angf_pi_4 = np.load('angular_filters_pi_4.npy')
    except Exception:
        angf_pi_4 = None
    try:
        angf_pi_2 = np.load('angular_filters_pi_2.npy')
    except Exception:
        angf_pi_2 = None

    # Bandpass
    FLOW = NFFT / RMAX
    FHIGH = NFFT / RMIN
    dRLow = 1.0 / (1.0 + (r / FHIGH)**4)
    dRHigh = 1.0 / (1.0 + (FLOW / r)**4)
    dBPass = dRLow * dRHigh

    # FFT Analysis
    idx = 0
    for i in range(nBlkHt):
        nRow = i * BLKSZ + OVRLP
        for j in range(nBlkWt):
            nCol = j * BLKSZ + OVRLP
            blk = img_d[nRow-OVRLP:nRow+BLKSZ+OVRLP, nCol-OVRLP:nCol+BLKSZ+OVRLP].copy()
            if blk.shape[0] != nWndSz or blk.shape[1] != nWndSz:
                # pad if necessary
                blk = cv2.copyMakeBorder(blk, 0, nWndSz-blk.shape[0], 0, nWndSz-blk.shape[1], cv2.BORDER_REFLECT)
            dAvg = np.mean(blk)
            blk = blk - dAvg
            blk = blk * w
            # center
            blkc = blk * dMult_small
            blkfft = np.fft.fft2(blkc, s=(NFFT, NFFT))
            blkfft = blkfft * dBPass
            dEnergy = np.abs(blkfft)**2
            # root filtering
            blkfft = blkfft * np.sqrt(dEnergy + np.finfo(float).eps)
            fftSrc[idx, :] = blkfft.T.flatten()
            dTotal = np.sum(dEnergy) / (NFFT*NFFT)
            fimg[i, j] = NFFT / (compute_mean_frequency(dEnergy, r) + np.finfo(float).eps)
            oimg[i, j] = compute_mean_angle(dEnergy, th)
            eimg[i, j] = np.log(dTotal + np.finfo(float).eps)
            idx += 1

    # process maps
    for _ in range(3):
        oimg = smoothen_orientation_image(oimg)
    fimg = smoothen_frequency_image(fimg, RMIN, RMAX, iters=5)
    cimg = compute_coherence(oimg)  # expects orientation image in radians
    bwimg = get_angular_bw_image(cimg)

    # FFT reconstruction
    idx = 0
    for i in range(nBlkHt):
        nRow = i * BLKSZ + OVRLP
        for j in range(nBlkWt):
            nCol = j * BLKSZ + OVRLP
            blkfft = fftSrc[idx, :].reshape(NFFT, NFFT).T
            af = get_angular_filter(oimg[i, j], bwimg[i, j], angf_pi_4, angf_pi_2, NFFT)
            blkfft = blkfft * af
            blk_ifft = np.real(np.fft.ifft2(blkfft)) * ((-1.0) ** (x + y))
            # extract central block
            start = OVRLP
            outblk = blk_ifft[start:start+BLKSZ, start:start+BLKSZ]
            enhimg[nRow:nRow+BLKSZ, nCol:nCol+BLKSZ] = outblk
            idx += 1

    # contrast stretch similar to matlab
    enhimg = np.sign(enhimg) * np.sqrt(np.abs(enhimg))
    mx = np.max(enhimg)
    mn = np.min(enhimg)
    if mx - mn > 1e-8:
        enhimg = ((enhimg - mn) / (mx - mn) * 254.0 + 1.0).astype(np.uint8)
    else:
        enhimg = (enhimg * 0 + 128).astype(np.uint8)

    # clean up using resized eimg
    try:
        emsk = resize(eimg, (nHt, nWt), order=1, preserve_range=True)
    except Exception:
        emsk = cv2.resize(eimg.astype(np.float32), (nWt, nHt), interpolation=cv2.INTER_LINEAR)
    enhimg = enhimg.astype(np.uint8)
    enhimg[emsk < ETHRESH] = 128

    return enhimg, cimg, oimg, fimg, bwimg, eimg