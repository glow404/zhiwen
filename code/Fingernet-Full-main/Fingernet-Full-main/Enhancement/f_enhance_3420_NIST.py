"""
Python 近似实现 f_enhance_3420_NIST.m 的增强管线。

返回:
    binim (H,W) bool
    mask  (H,W) bool
    cimg1, cimg2 - coherence / 可选增强输出 (H,W) float
    oimg1, oimg2 - orientation fields (H,W) radians
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel_h, sobel_v
from skimage.util import view_as_blocks
from compute_coherence import compute_coherence
from dist2 import dist2

def fft_enhance_cubs(img, blocks):
    img = np.asarray(img, dtype=np.uint8)
    if blocks == -1:
        # global unsharp / high-pass enhancement
        img_f = cv2.GaussianBlur(img, (0,0), 3)
        enhanced = cv2.addWeighted(img, 1.5, img_f, -0.5, 0)
        return enhanced.astype(np.uint8), None
    # approximate per-block CLAHE where tileGridSize ~= blocks
    b = max(1, int(blocks))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(b, b))
    enhanced = clahe.apply(img)
    return enhanced.astype(np.uint8), None

def ridgesegment(img, blksze=16, thresh=0.05):
    im = img.astype(np.float32)
    h, w = im.shape
    # normalize: zero mean, unit std per block
    normim = np.zeros_like(im)
    mask = np.zeros((h, w), dtype=bool)
    bs = blksze
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            y1, y2 = y, min(h, y+bs)
            x1, x2 = x, min(w, x+bs)
            block = im[y1:y2, x1:x2]
            if block.size == 0: continue
            m = block.mean()
            s = block.std()
            if s < 1e-6:
                normim[y1:y2, x1:x2] = 0
            else:
                normim[y1:y2, x1:x2] = (block - m) / s
            mask[y1:y2, x1:x2] = (s > thresh*255.0)
    return normim, mask

def ridgeorient(normim, gradientsigma=1, blocksigma=3, orientsmoothsigma=3):
    # gradient-based orientation estimation
    gx = sobel_h(normim)
    gy = sobel_v(normim)
    Gxx = gaussian_filter(gx*gx, gradientsigma)
    Gyy = gaussian_filter(gy*gy, gradientsigma)
    Gxy = gaussian_filter(gx*gy, gradientsigma)
    # orientation: 0..pi (unsigned)
    denom = (Gxx - Gyy)
    orient = 0.5 * np.arctan2(2*Gxy, denom + 1e-12)  # range -pi/2..pi/2
    orient = np.mod(orient + np.pi, np.pi)  # map to [0,pi)
    # smooth orientation via vector smoothing (double-angle)
    cos2 = np.cos(2*orient)
    sin2 = np.sin(2*orient)
    cos2s = gaussian_filter(cos2, orientsmoothsigma)
    sin2s = gaussian_filter(sin2, orientsmoothsigma)
    orient_sm = 0.5 * np.arctan2(sin2s, cos2s)
    orient_sm = np.mod(orient_sm + np.pi, np.pi)
    # produce a simple reliability measure from coherence of gradients
    coherence = (np.sqrt((Gxx - Gyy)**2 + (2*Gxy)**2) / (Gxx + Gyy + 1e-12))
    return orient_sm, coherence

def ridgefreq(normim, mask, orient, block_size=32, minWaveLength=3, maxWaveLength=15):
    h, w = normim.shape
    freq = np.zeros((h,w), dtype=float)
    freqs = []
    bs = block_size
    for y in range(0, h-bs+1, bs):
        for x in range(0, w-bs+1, bs):
            blk = normim[y:y+bs, x:x+bs]
            msk = mask[y:y+bs, x:x+bs]
            if np.sum(msk) < (bs*bs*0.1): continue
            ang = np.median(orient[y:y+bs, x:x+bs])
            # rotate block to make ridges roughly vertical
            M = cv2.getRotationMatrix2D((bs/2, bs/2), -np.degrees(ang), 1.0)
            rot = cv2.warpAffine(blk, M, (bs,bs), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            proj = np.mean(rot, axis=1)  # projection along columns
            # find peaks in proj
            proj = (proj - proj.mean())
            # FFT to find dominant frequency
            N = proj.size
            if N < 4:
                continue
            f = np.abs(np.fft.rfft(proj))
            peak = np.argmax(f[1:]) + 1
            wavelength = N / (peak + 1e-12)
            if minWaveLength <= wavelength <= maxWaveLength:
                freq_val = 1.0 / wavelength
                freq[y:y+bs, x:x+bs] = freq_val
                freqs.append(freq_val)
    if len(freqs) == 0:
        medfreq = 0.1
    else:
        medfreq = float(np.median(freqs))
    return freq, medfreq

def ridgefilter(normim, orient, freq, kx=0.5, ky=0.5, gain=1.0):
    # blockwise Gabor filtering using median frequency (freq may be array or scalar)
    h, w = normim.shape
    filtered = np.zeros_like(normim)
    bs = 16
    medfreq = np.median(freq[freq>0]) if np.any(freq>0) else 0.1
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            y1, y2 = y, min(h, y+bs)
            x1, x2 = x, min(w, x+bs)
            ang = np.median(orient[y1:y2, x1:x2])
            f = medfreq
            if f <= 0:
                f = 0.1
            # build Gabor kernel
            wavelength = 1.0 / f
            sigma_x = kx * wavelength
            sigma_y = ky * wavelength
            theta = ang
            # use OpenCV gabor kernel size
            ksize = int(max(3, np.round(max(3, 6*max(sigma_x, sigma_y)))))
            if ksize % 2 == 0: ksize += 1
            kern = cv2.getGaborKernel((ksize,ksize), sigma_x, theta, wavelength, 0.5, 0, ktype=cv2.CV_32F)
            patch = normim[y1:y2, x1:x2]
            if patch.size == 0: continue
            conv = cv2.filter2D(patch.astype(np.float32), -1, kern)
            filtered[y1:y2, x1:x2] = conv
    return filtered

def kNearestNeighbors(X, Y, k=3):
    # interpret angular lists as 2D unit vectors to handle circular distance
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim == 1: X = X.reshape(-1,1)
    if Y.ndim == 1: Y = Y.reshape(-1,1)
    # map angles to unit vectors if 1D angles
    if X.shape[1] == 1:
        Xv = np.column_stack([np.cos(X[:,0]), np.sin(X[:,0])])
    else:
        Xv = X.copy()
    if Y.shape[1] == 1:
        Yv = np.column_stack([np.cos(Y[:,0]), np.sin(Y[:,0])])
    else:
        Yv = Y.copy()
    D = dist2(Xv, Yv)  # squared euclid
    idx = np.argsort(D, axis=1)[:, :k]
    dists = np.take_along_axis(D, idx, axis=1)
    return idx, np.sqrt(dists)

def f_enhance_3420_NIST(img):
    # img: grayscale uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # multi-scale enhancement approximated by CLAHE/unsharp
    enh1, _ = fft_enhance_cubs(img, 6)
    enh2, _ = fft_enhance_cubs(enh1, 12)
    enh3, _ = fft_enhance_cubs(enh2, 24)
    # coarse segmentation
    blksze = 5
    thresh = 0.05
    normim, mask = ridgesegment(enh3, blksze, thresh)
    oimg1, reliability = ridgeorient(normim, 1, 3, 3)

    # second pass with global enhancement
    enh4, _ = fft_enhance_cubs(img, -1)
    normim2, mask2 = ridgesegment(enh4, blksze, thresh)
    oimg2, reliability1 = ridgeorient(normim2, 1, 3, 3)

    freq, medfreq = ridgefreq(normim2, mask2, oimg2, 32, 5, 15)
    filtered = ridgefilter(normim, oimg1, freq, 0.5, 0.5, 1.0)
    binim = filtered > 0

    # create relim similar to MATLAB: binim * mask * (reliability>0.7)
    relim = np.logical_and.reduce([binim, mask, reliability > 0.7])

    out_im = relim.astype(bool)
    image = out_im
    coords = np.column_stack(np.nonzero(image))  # (y,x)
    if coords.size == 0:
        orient_vec = np.array([])
        orient_vec1 = np.array([])
    else:
        orient_vec = oimg1[image]
        orient_vec1 = oimg2[image]

    # kNN between orientations
    if orient_vec.size:
        neighborIds, neighborDistances = kNearestNeighbors(orient_vec.reshape(-1,1), orient_vec1.reshape(-1,1), 3)
    else:
        neighborIds = np.empty((0,3), dtype=int)
        neighborDistances = np.empty((0,3), dtype=float)

    return binim, mask, None, None, oimg1, oimg2, neighborIds, neighborDistances

# 使用示例:
# import cv2
# img = cv2.imread('finger.png', cv2.IMREAD_GRAYSCALE)
# binim, mask, c1, c2, o1, o2, nbrIdx, nbrDist = f_enhance_3420_NIST(img)