"""
生成角度滤波器矩阵并保存为 .npy 文件

函数:
    angular_filter_bank(BW, fname, FFTN=32, TSTEPS=12, show=False)

返回:
    angf - (FFTN*FFTN, TSTEPS) 的 numpy 数组
"""
import numpy as np

def angular_distance(th, t0):
    d = np.abs(th - t0)
    return np.minimum(d, 2 * np.pi - d)

def angular_filter_bank(BW, fname, FFTN=32, TSTEPS=12, show=False):
    DELTAT = np.pi / TSTEPS

    xs = np.arange(-FFTN/2, FFTN/2)
    ys = np.arange(-FFTN/2, FFTN/2)
    x, y = np.meshgrid(xs, ys)
    th = np.arctan2(y, x)
    th[th < 0] += 2 * np.pi  # unsigned [0,2pi)

    filters = []
    for t0 in np.arange(0, TSTEPS * DELTAT, DELTAT):
        t1 = t0 + np.pi
        # first lobe
        d = angular_distance(th, t0)
        msk = 1.0 + np.cos(d * np.pi / BW)
        msk[d > BW] = 0.0
        rmsk = msk
        # second lobe
        d = angular_distance(th, t1)
        msk = 1.0 + np.cos(d * np.pi / BW)
        msk[d > BW] = 0.0
        rmsk = rmsk + msk

        if show:
            try:
                import matplotlib.pyplot as plt
                plt.imshow(rmsk, cmap='gray', origin='lower')
                plt.title(f'angle {t0:.3f} rad')
                plt.pause(0.05)
            except Exception:
                pass

        vec = rmsk.flatten(order='C')
        filters.append(vec)

    angf = np.column_stack(filters)  # shape (FFTN*FFTN, TSTEPS)

    # 保存为单一的 numpy 文件，方便后续加载：np.load(fname + '.npy')
    np.save(fname + '.npy', angf)

    return angf

# 简单使用示例（命令行或交互中）：
# angf = angular_filter_bank(BW=0.3, fname='angular_filters', FFTN=32, TSTEPS=12, show=False)