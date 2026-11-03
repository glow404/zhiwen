"""
主函数，后面做
从指纹图像中提取 minutiae（终点/分叉/可选核心）。
返回格式与原 MATLAB 类似：N x 6 矩阵，列为 [x, y, type, angle, extra, valid]
 - type: 1 = termination, 3 = bifurcation, 5 = core(optional)
 - angle: 方向（弧度）
注意：此实现使用简单增强与方向估计，作为 MATLAB 版本的可用 Python 替代。
依赖: numpy, scipy, scikit-image, opencv-python
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_local, sobel_h, sobel_v
from skimage.morphology import thin
from math import atan2, pi
from dist2 import dist2  # 如果你已生成 dist2.py，保留；否则可内联实现

def _crossing_number(thinned, x, y):
    # thinned: boolean array, True where ridge pixel
    # p() order matches MATLAB p around 8 neighbors clockwise starting at east (approx)
    # We'll follow common CN ordering (8 neighbors) clockwise
    neighbors = [
        thinned[y, x+1],   # 1 E
        thinned[y-1, x+1], # 2 NE
        thinned[y-1, x],   # 3 N
        thinned[y-1, x-1], # 4 NW
        thinned[y, x-1],   # 5 W
        thinned[y+1, x-1], # 6 SW
        thinned[y+1, x],   # 7 S
        thinned[y+1, x+1], # 8 SE
    ]
    # convert bool to int
    nbr = [1 if v else 0 for v in neighbors]
    s = 0
    for i in range(8):
        s += abs(nbr[i] - nbr[(i+1) % 8])
    CN = s / 2
    return CN

def _local_median_orientation(orient_img_m, x, y):
    # orient_img_m: smoothed orientation in radians [0,2pi)
    patch = orient_img_m[max(0,y-1):y+2, max(0,x-1):x+2].ravel()
    if patch.size == 0:
        return 0.0
    # 平均方向需处理环绕：用向量平均
    vx = np.mean(np.cos(patch))
    vy = np.mean(np.sin(patch))
    ang = np.arctan2(vy, vx)
    if ang < 0:
        ang += 2*pi
    return ang

def compute_orientation_field(img, sigma=3):
    # 简单方向估计：基于 sobel 梯度，平滑后取 atan2
    gx = sobel_h(img.astype(float))
    gy = sobel_v(img.astype(float))
    # 平滑矢量分量以减少噪声
    gx_s = gaussian_filter(gx, sigma=sigma)
    gy_s = gaussian_filter(gy, sigma=sigma)
    orient = np.arctan2(gy_s, gx_s)  # in [-pi,pi]
    orient[orient < 0] += 2*pi       # map to [0,2pi)
    # also produce a smoothed orientation suitable for median using vector smoothing
    cos2 = np.cos(orient * 2.0)
    sin2 = np.sin(orient * 2.0)
    cos2_s = gaussian_filter(cos2, sigma=sigma)
    sin2_s = gaussian_filter(sin2, sigma=sigma)
    orient_m = 0.5 * np.arctan2(sin2_s, cos2_s)
    orient_m[orient_m < 0] += pi*2 if False else 0  # keep in [-pi/2,pi/2], but we'll keep raw in radians
    # normalize orient_m to [0,2pi)
    orient_m = np.mod(orient_m, 2*pi)
    return orient, orient_m

def ext_finger_skeleton(img, display_flag=0, filename=None):
    # 输入 img: 灰度 numpy 数组 uint8 或 float 0-255
    if img.dtype != np.uint8:
        img_u = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_u = img.copy()

    h, w = img_u.shape[:2]

    # 1) 简单裁剪与白边处理（尽量模拟原 MATLAB 的边缘白化）
    # 将上、下、左、右显著空白边缘置白（255）
    #（此处用简单阈值方法）
    # detect rows/cols with few dark pixels and blank them (like原代码)
    for r in range(min(55, h)):
        if np.count_nonzero(img_u[r, :] < 200) < 8:
            img_u[:r+1, :] = 255
    for r in range(h-1, max(h-230,0), -1):
        if np.count_nonzero(img_u[r, :] < 200) < 3:
            img_u[r-17:h, :] = 255
            break
    for c in range(200, w):
        if np.count_nonzero(img_u[:, c] < 200) < 1:
            img_u[:, c:w] = 255
            break
    for c in range(min(75, w)):
        if np.count_nonzero(img_u[:, c] < 200) < 1:
            img_u[:, :c+1] = 255

    # 2) 增强: CLAHE + 中值平滑
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_u)
    enhanced = cv2.medianBlur(enhanced, 3)

    # 3) 二值化（自适应阈值）
    block_size = 35
    local_thresh = threshold_local(enhanced, block_size, offset=10)
    binim = enhanced > local_thresh  # True for bright (背景). 指纹脊通常为暗，故取反
    binim = ~binim  # 现在 binim True 表示 ridge (黑 -> True)

    # 4) mask: 简化的块方差法生成有效区
    # 如果一个块内方差小则判为背景
    blk = 24
    mask = np.ones_like(binim, dtype=np.uint8)
    for y in range(0, h, blk):
        for x in range(0, w, blk):
            sub = enhanced[y:min(h,y+blk), x:min(w,x+blk)]
            if sub.size == 0: continue
            if np.std(sub) < 5:  # 经验阈值
                mask[y:min(h,y+blk), x:min(w,x+blk)] = 0

    # 5) 细化（骨架化）
    thinned = thin(binim)  # boolean array

    # 6) 方向场计算
    orient_img, orient_img_m = compute_orientation_field(enhanced.astype(float)/255.0, sigma=3)

    # 把被 mask 掩盖区域的方向置为大值以便后续跳过（模拟原代码 o_img(mask==0)=1）
    orient_img_m_masked = orient_img_m.copy()
    orient_img_m_masked[mask == 0] = 0.0

    # 7) 寻找 minutiae（Crossing Number）
    minutiae = []
    # 忽略边界 1 像素，确保邻域合法
    for y in range(1, h-1):
        for x in range(1, w-1):
            if not thinned[y, x]:
                continue
            # 跳过 mask 外区域
            if mask[y, x] == 0:
                continue
            CN = _crossing_number(thinned, x, y)
            if CN == 1 or CN == 3:
                angle = _local_median_orientation(orient_img_m_masked, x, y)
                typ = 1 if CN == 1 else 3
                # 保留与原格式兼容的六列 [x,y,type,angle,0,1]
                minutiae.append([x, y, typ, float(angle), 0.0, 1])
    if len(minutiae) == 0:
        return np.empty((0,6))

    minutiae = np.array(minutiae, dtype=float)

    # 8) 过滤靠近 mask=0 的细节点（移除伪细节点）
    kept = []
    for m in minutiae:
        x, y = int(m[0]), int(m[1])
        neighborhood = mask[max(0,y-2):min(h,y+3), max(0,x-2):min(w,x+3)]
        if np.any(neighborhood == 0):
            continue
        kept.append(m)
    if len(kept) == 0:
        return np.empty((0,6))
    minutiae = np.vstack(kept)

    # 9) 距离过滤：移除过近（重复）点，保留第一个
    coords = minutiae[:, :2]
    D = dist2(coords, coords)
    n = len(minutiae)
    reject = np.zeros(n, dtype=bool)
    min_sep2 = 7.0**2  # 半径阈值，和原 MATLAB 近似
    for i in range(n):
        if reject[i]: continue
        for j in range(i+1, n):
            if reject[j]: continue
            if D[i,j] <= min_sep2:
                # 保留类型优先级：保留非-termination (3) 优于 termination(1)；或按任意规则
                if minutiae[i,2] == minutiae[j,2]:
                    reject[j] = True
                else:
                    # 若 i 是 termination 且 j 是 bifurcation，优先保留 bifurcation
                    if minutiae[i,2] == 1 and minutiae[j,2] == 3:
                        reject[i] = True
                    else:
                        reject[j] = True
    minutiae = minutiae[~reject]

    # 10) optionally detect cores (very approximate): find local maxima of orientation variance - 可选，暂不精确实现
    # （保留占位；如果需要可以实现更复杂的 Poincare index / core detection）

    # 11) 可视化与保存（如果需要）
    if display_flag:
        import matplotlib.pyplot as plt
        vis = cv2.cvtColor((~binim).astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
        for m in minutiae:
            x,y,t,ang,_,_ = m
            color = (255,0,0) if t==1 else (0,0,255)
            cv2.circle(vis, (int(x),int(y)), 2, color, -1)
        plt.figure(figsize=(8,8)); plt.imshow(vis[:,:,::-1]); plt.title('minutiae'); plt.axis('off')
        if filename:
            cv2.imwrite(filename, vis)
        plt.show()

    return minutiae  # numpy array N x 6