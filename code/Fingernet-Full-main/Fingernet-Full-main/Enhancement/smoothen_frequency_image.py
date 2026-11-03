import numpy as np

def smoothen_frequency_image(fimg, RLOW, RHIGH, diff_cycles):
    """
    Smoothen frequency image via diffusion-like Gaussian weighted averaging.

    Args:
        fimg (2D array): input frequency image (float)
        RLOW, RHIGH (float): allowed ridge separation range (pixels)
        diff_cycles (int): number of diffusion cycles

    Returns:
        nfimg (2D array): smoothed frequency image (float). Invalid locations set to -1.
    """
    valid_nbrs = 3  # uses only pixels with more than this valid neighbours
    fimg = np.asarray(fimg, dtype=float)
    ht, wt = fimg.shape
    nfimg = fimg.copy()
    N = 1  # half-window size (MATLAB code used N=1)
    cycles = 0

    # build Gaussian kernel (fspecial('gaussian',2*N+1) approx)
    size = 2 * N + 1
    sigma = 1.0
    ax = np.arange(-N, N + 1)
    X, Y = np.meshgrid(ax, ax)
    h = np.exp(- (X**2 + Y**2) / (2.0 * sigma * sigma))
    h = h / np.sum(h)

    # initial invalid count
    invalid_cnt = np.sum((fimg < RLOW) | (fimg > RHIGH))

    # perform diffusion up to diff_cycles, break early if no invalid pixels
    while cycles < diff_cycles:
        # pad image by reflection (equivalent to flipud/fliplr pattern)
        padded = np.pad(fimg, pad_width=N, mode='reflect')

        # iterate over pixels
        for i in range(N, N + ht):
            for j in range(N, N + wt):
                blk = padded[i - N:i + N + 1, j - N:j + N + 1]
                msk = (blk >= RLOW) & (blk <= RHIGH)
                if np.sum(msk) >= valid_nbrs:
                    # weighted average using only valid pixels, weights from h
                    num = np.sum(blk * (h * msk))
                    den = np.sum(h * msk)
                    if den != 0:
                        nfimg[i - N, j - N] = num / den
                    else:
                        nfimg[i - N, j - N] = -1.0
                else:
                    nfimg[i - N, j - N] = -1.0

        # prepare next iteration
        fimg = nfimg.copy()
        invalid_cnt = np.sum((fimg < RLOW) | (fimg > RHIGH))
        cycles += 1
        if invalid_cnt == 0:
            break

    return nfimg