# ...existing code...
import numpy as np
from math import atan2, pi, sqrt
from typing import Tuple

# 依赖：同目录下的 p.py（函数 p(img, x, y, i) 返回 (value, X, Y)，X/Y 为 0-based 列/行）
try:
    from .p import p
except Exception:
    from p import p

def _dist2(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def test_bifurcation(img: np.ndarray, x: int, y: int, o: float, core_x: int, core_y: int) -> Tuple[int,int,int,int,float]:
    """
    Identify branch direction at a bifurcation.

    Parameters:
        img    : 2D array (skeleton/binary) indexable as img[row, col]
        x, y   : center pixel coordinates (column, row) — 0-based
        o      : orientation (radians) — unused for decision here except returned
        core_x, core_y : core coordinates (column, row) — 0-based (may be 0 if unknown)

    Returns:
        res      : int (2 or 3 as in original; here mostly 3 for same/opposite)
        progress : int path length reached (1..path_len)
        sx, sy   : chosen branch pixel coordinates (column, row) — 0-based
        angle    : angle = atan2(y - sy, sx - x) normalized to [0, 2*pi)
    """
    # initialize
    iax = iay = ibx = iby = icx = icy = 0
    progress = 1
    path_len = 4

    pax = pay = pbx = pby = pcx = pcy = 0
    pao = pbo = pco = 0   # 1-8 position (using MATLAB numbering 1..9)

    # first pass: find up to three branch starting neighbors
    for i in range(1, 9):
        ta, xa, ya = p(img, x, y, i)
        tb, xb, yb = p(img, x, y, i + 1)
        if ta > tb:
            if pao == 0:
                pao = (4 + i) if i < 5 else ((4 + i) % 9) + 1
                pax, pay = xa, ya
            else:
                if pbo == 0:
                    pbo = (4 + i) if i < 5 else ((4 + i) % 9) + 1
                    pbx, pby = xa, ya
                else:
                    pco = (4 + i) if i < 5 else ((4 + i) % 9) + 1
                    pcx, pcy = xa, ya
                    break

    xaa = yaa = xbb = ybb = xcc = ycc = 0

    # hist keeps visited pixels as [row, col] to avoid loops
    hist = []
    if pay != 0 or pax != 0:
        hist.append([pay, pax])
    if pby != 0 or pbx != 0:
        hist.append([pby, pbx])
    if pcy != 0 or pcx != 0:
        hist.append([pcy, pcx])
    hist.append([y, x])

    stop = False
    while progress < path_len and not stop:
        progress += 1
        da = db = dc = 0

        # helper to test continuation for a branch
        def _branch_step(pao_flag, px, py):
            nonlocal hist, stop
            if pao_flag == 0 or stop:
                return 0, 0, 0
            # compute cn as in MATLAB: half sum of abs diffs around 8-neighbors
            cn = 0
            for ii in range(1, 9):
                t1, xA, yA = p(img, px, py, ii)
                t2, xB, yB = p(img, px, py, ii + 1)
                cn += abs(t1 - t2)
            cn = cn / 2
            if cn == 1 or cn == 3:
                stop = True
                return 0, 0, 0

            # find next pixel along ridge (first neighbor where ta > tz and not visited)
            i_local = 1
            while i_local < 9:
                ta, xa, ya = p(img, px, py, i_local)
                tz, xz, yz = p(img, px, py, i_local + 1)
                # check visited
                visited = any((h[0] == ya and h[1] == xa) for h in hist)
                if visited:
                    i_local += 1
                    continue
                if ta > tz and not (xa == x and ya == y):
                    hist.append([ya, xa])
                    return 1, xa, ya
                i_local += 1
            return 0, 0, 0

        if pao != 0:
            da, xaa, yaa = _branch_step(pao, pax, pay)
            if da:
                pax, pay = xaa, yaa

        if pbo != 0 and not stop:
            db, xbb, ybb = _branch_step(pbo, pbx, pby)
            if db:
                pbx, pby = xbb, ybb

        if pco != 0 and not stop:
            dc, xcc, ycc = _branch_step(pco, pcx, pcy)
            if dc:
                pcx, pcy = xcc, ycc

    # compute pairwise distances between the three branch endpoints found
    # ensure values defined
    # if any endpoint still zero, set to center to avoid crash
    if xaa == 0 and yaa == 0:
        xaa = pax; yaa = pay
    if xbb == 0 and ybb == 0:
        xbb = pbx; ybb = pby
    if xcc == 0 and ycc == 0:
        xcc = pcx; ycc = pcy

    d1 = sqrt(_dist2([xaa, yaa], [xbb, ybb]))
    d2 = sqrt(_dist2([xaa, yaa], [xcc, ycc]))
    d3 = sqrt(_dist2([xcc, ycc], [xbb, ybb]))

    # choose sx, sy as the "middle" branch (largest two distances >= third)
    sx = sy = 0
    ind = pao
    if d1 >= d3 and d2 >= d3:
        sx, sy, ind = xaa, yaa, pao
    elif d1 >= d2 and d3 >= d2:
        sx, sy, ind = xbb, ybb, pbo
    elif d3 >= d1 and d2 >= d1:
        sx, sy, ind = xcc, ycc, pco
    else:
        sx, sy, ind = xaa, yaa, pao

    # choose farthest from core (qx,qy) but not used in final result here
    qx = qy = 0
    if core_x is None:
        core_x = 0
        core_y = 0
    dd1 = sqrt(_dist2([xaa, yaa], [core_x, core_y]))
    dd2 = sqrt(_dist2([xbb, ybb], [core_x, core_y]))
    dd3 = sqrt(_dist2([xcc, ycc], [core_x, core_y]))
    if dd1 >= dd2 and dd1 >= dd3:
        qx, qy = xaa, yaa
    elif dd2 >= dd3 and dd2 >= dd1:
        qx, qy = xbb, ybb
    else:
        qx, qy = xcc, ycc

    # compute angle same as MATLAB: mod( atan2(y - sy, sx - x), 2*pi )
    angle = atan2(y - sy, sx - x) % (2 * pi)

    # decision: original code returns res=3 in both branches; keep same logic
    if qx == sx and qy == sy:
        res = 3
    else:
        res = 3

    return res, progress, sx, sy, angle
# ...existing code...