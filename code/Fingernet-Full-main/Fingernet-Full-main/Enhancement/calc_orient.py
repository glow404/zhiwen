"""
等价于 MATLAB calc_orient.m 的 Python 实现。

函数签名:
    similarity, index, r_g = calc_orient(A, rA, B, rB, W=None)

返回:
    similarity - 相似度分值（float），若匹配点 <=1 返回 -1
    index      - 匹配点数量（int）
    r_g        - 匹配的列索引列表（0-based，Python 风格）
"""
import numpy as np

def calc_orient(A, rA, B, rB, W=None):
    A  = np.asarray(A)
    B  = np.asarray(B)
    rA = np.asarray(rA)
    rB = np.asarray(rB)

    # 检查尺寸
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2-D arrays")
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")

    raw_vals = []
    r_g = []

    nrows, ncols = A.shape

    # 遍历所有位置，收集在 rA,rB 都不为 -1 处的差的绝对值
    for i in range(nrows):
        # 保守检查：确保 rA/rB 有相应的行
        if rA.shape[0] <= i or rB.shape[0] <= i:
            continue
        for j in range(ncols):
            # 若 rA/rB 有相应的列，且值不为 -1 则视为有效匹配
            if rA.shape[1] <= j or rB.shape[1] <= j:
                continue
            if rA[i, j] != -1 and rB[i, j] != -1:
                temp = abs(float(A[i, j]) - float(B[i, j]))
                raw_vals.append(temp)
                r_g.append(j)  # Python 风格 0-based 列索引

    count = len(raw_vals)
    index = count if count >= 0 else 0

    if count > 1:
        raw_arr = np.array(raw_vals, dtype=float)
        # 与原 MATLAB 等价的变换：exp(-raw*8) * exp(max(0,70-count)*-1/50)
        scale = np.exp(max(0, 70 - count) * -1.0 / 50.0)
        transformed = np.exp(-raw_arr * 8.0) * scale
        similarity = float(np.mean(transformed))
    else:
        similarity = -1.0

    return similarity, index, r_g

# 简单示例（在交互或测试中使用）：
# import numpy as np
# A = np.array([[0.1, 0.2],[0.3, 0.4]])
# rA = np.array([[0, -1],[0, 0]])
# B = A + 0.05
# rB = rA.copy()
# sim, idx, rg = calc_orient(A, rA, B, rB)