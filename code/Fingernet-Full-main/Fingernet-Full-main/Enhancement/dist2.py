"""
等价于 MATLAB dist2.m 的 Python 实现——计算两组点之间的平方欧氏距离。

函数:
    n2 = dist2(x, c)

输入:
    x : (M, D) array_like
    c : (L, D) array_like

输出:
    n2 : (M, L) numpy array, 第 (i,j) 项为 ||x[i,:] - c[j,:]||^2
"""
import numpy as np

def dist2(x, c):
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    if x.ndim != 2 or c.ndim != 2:
        raise ValueError("x and c must be 2-D arrays")
    if x.shape[1] != c.shape[1]:
        raise ValueError("Data dimension does not match dimension of centres")

    # 使用恒等式 ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
    x_sq = np.sum(x * x, axis=1).reshape(-1, 1)   # (M,1)
    c_sq = np.sum(c * c, axis=1).reshape(1, -1)   # (1,L)
    cross = np.dot(x, c.T)                        # (M,L)

    n2 = x_sq + c_sq - 2.0 * cross

    # 数值误差可能导致极小负值，裁剪为 0
    n2 = np.maximum(n2, 0.0)

    return n2

# 示例:
# import numpy as np
# X = np.random.rand(100, 2)
# C = np.random.rand(5, 2)
# D = dist2(X, C)  # 返回 100x5 的平方距离矩阵