import numpy as np

def sc_compute(Bsamp, Tsamp, mean_dist, nbins_theta, nbins_r, r_inner, r_outer, out_vec):
    """
    Compute (r,theta) histograms for points along boundary.

    Parameters
    ----------
    Bsamp : array_like, shape (2, nsamp)
        x and y coordinates.
    Tsamp : array_like, shape (nsamp,) or (1, nsamp)
        tangent angles (radians) for each sample.
    mean_dist : float or None
        mean distance used for length normalization; if None, computed from inliers.
    nbins_theta : int
        number of angular bins.
    nbins_r : int
        number of radial bins.
    r_inner, r_outer : floats
        inner and outer radius (in normalized units) for bin edges (log scale).
    out_vec : array_like, shape (nsamp,)
        0 = inlier, 1 = outlier.

    Returns
    -------
    BH : ndarray, shape (nsamp, nbins_theta*nbins_r)
        per-sample (r,theta) histograms weighted by gauss_dist.
    mean_dist : float
        mean distance used (computed if input was None).
    """
    Bsamp = np.asarray(Bsamp, dtype=float)
    if Bsamp.shape[0] != 2:
        raise ValueError("Bsamp should be shape (2, nsamp)")
    Bx = Bsamp[0, :]
    By = Bsamp[1, :]
    nsamp = Bx.size

    Tsamp = np.asarray(Tsamp).reshape(-1)
    if Tsamp.size != nsamp:
        raise ValueError("Tsamp length must match number of samples")

    out_vec = np.asarray(out_vec).reshape(-1)
    if out_vec.size != nsamp:
        raise ValueError("out_vec length must match number of samples")
    in_vec = (out_vec == 0)

    # pairwise distance matrix r_array (nsamp x nsamp)
    dx = Bx[None, :] - Bx[:, None]  # shape (nsamp, nsamp) ; element [i,j] = x_j - x_i
    dy = By[None, :] - By[:, None]  # element [i,j] = y_j - y_i
    r_array = np.sqrt(dx * dx + dy * dy)

    # theta_array_abs: angle from i -> j : atan2(y_j - y_i, x_j - x_i)
    theta_array_abs = np.arctan2(dy, dx)  # shape (nsamp, nsamp)

    # theta_array = theta_array_abs - Tsamp[:,None]
    theta_array = theta_array_abs - Tsamp[:, None]

    # compute mean_dist if not provided (use only inlier-to-inlier distances)
    if mean_dist is None or (isinstance(mean_dist, (list, tuple, np.ndarray)) and len(mean_dist) == 0):
        if np.any(in_vec):
            sub = r_array[np.ix_(in_vec, in_vec)]
            # exclude self-distances (zeros on diagonal) by taking full matrix mean is OK in MATLAB,
            # but to mimic original behavior include zeros; keep same as MATLAB: mean(tmp(:))
            mean_dist = float(np.mean(sub))
            if mean_dist == 0:
                mean_dist = 1.0
        else:
            mean_dist = 1.0
    mean_dist = float(mean_dist)

    # normalize distances
    r_array_n = r_array / (mean_dist if mean_dist != 0 else 1.0)

    # radial bin edges (log scale)
    r_bin_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)

    # quantize radial bins to produce integers in {0,1,...,nbins_r}
    # Follow MATLAB logic: for m=1:nbins_r r_array_q = r_array_q + (r_array_n < r_bin_edges(m));
    r_array_q = np.zeros_like(r_array, dtype=int)
    for m in range(nbins_r):
        r_array_q += (r_array_n < r_bin_edges[m]).astype(int)
    # r_array_q > 0 flags points inside outer boundary
    fz = r_array_q > 0

    # wrap angles to [0, 2pi)
    theta_array_2 = np.mod(theta_array, 2.0 * np.pi)
    # quantize angular bins: 1 + floor(theta_array_2 / (2*pi/nbins_theta)) -> 1..nbins_theta
    theta_array_q = 1 + np.floor(theta_array_2 / (2.0 * np.pi / nbins_theta)).astype(int)

    # prepare gaussian weighting vector (as in MATLAB)
    gauss_vals = np.exp(-((r_bin_edges - r_outer) ** 2) / 2.5)  # length nbins_r
    # replicate to matrix nbins_r x nbins_theta then sort flattened values
    gauss_matrix = np.tile(gauss_vals[:, None], (1, nbins_theta))  # shape (nbins_r, nbins_theta)
    gauss_dist = np.sort(gauss_matrix.ravel())  # sorted ascending, length nbins_r*nbins_theta

    nbins = nbins_theta * nbins_r
    BH = np.zeros((nsamp, nbins), dtype=float)

    # For each sample n, build sparse histogram Sn (nbins_theta x nbins_r)
    for n in range(nsamp):
        # select valid target indices j: inside outer boundary (for row n) and inlier
        valid_cols = np.where(fz[n, :] & in_vec)[0]
        if valid_cols.size == 0:
            continue
        # theta bins and r bins for these pairs (theta_array_q and r_array_q are 1-based in MATLAB)
        thetabs = theta_array_q[n, valid_cols].astype(int)   # 1..nbins_theta
        rbins = r_array_q[n, valid_cols].astype(int)        # 0..nbins_r (0 means outside)
        # Only keep rbins>0 (should already be ensured by fz)
        valid_mask = rbins > 0
        if not np.any(valid_mask):
            continue
        thetabs = thetabs[valid_mask]
        rbins = rbins[valid_mask]
        # Build counts into matrix Sn (nbins_theta x nbins_r). Note MATLAB used sparse with (theta, r)
        Sn = np.zeros((nbins_theta, nbins_r), dtype=float)
        # Convert to 0-based indices for Python arrays: theta-1, r-1
        theta_idx = thetabs - 1
        r_idx = rbins - 1
        # accumulate counts
        np.add.at(Sn, (theta_idx, r_idx), 1.0)
        # flatten Sn column-major (MATLAB's (:) is column-major)
        Sn_flat = Sn.ravel(order='F')  # Fortran order (column-major)
        # multiply element-wise by gauss_dist and store
        # gauss_dist was sorted in MATLAB; to mimic exactly we multiply with that sorted vector
        # Ensure lengths match
        if Sn_flat.size != gauss_dist.size:
            # fallback: tile gauss matrix flattened in column-major if shapes differ
            gauss_fallback = gauss_matrix.ravel(order='F')
            weights = gauss_fallback
        else:
            weights = gauss_dist
        BH[n, :] = Sn_flat * weights

    return BH, mean_dist