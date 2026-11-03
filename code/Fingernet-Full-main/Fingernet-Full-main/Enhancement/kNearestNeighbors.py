"""
等价于 MATLAB kNearestNeighbors.m 的 Python 实现。

函数:
    neighbor_ids, neighbor_dists = kNearestNeighbors(data_matrix, query_matrix, k)

输入:
    data_matrix  : (N, D) ndarray
    query_matrix : (M, D) ndarray
    k            : int

输出:
    neighbor_ids  : (M, k) int ndarray, 每行为最近邻在 data_matrix 中的索引（0-based）
    neighbor_dists: (M, k) float ndarray, 对应的欧氏距离
"""
import numpy as np

def kNearestNeighbors(data_matrix, query_matrix, k):
    data = np.asarray(data_matrix, dtype=np.float64)
    query = np.asarray(query_matrix, dtype=np.float64)

    if data.ndim != 2 or query.ndim != 2:
        raise ValueError("data_matrix and query_matrix must be 2-D arrays")
    if data.shape[1] != query.shape[1]:
        raise ValueError("Dimensionality of data_matrix and query_matrix must match")

    N = data.shape[0]
    M = query.shape[0]
    if N == 0 or M == 0:
        return np.empty((M, 0), dtype=int), np.empty((M, 0), dtype=float)

    k = int(k)
    if k <= 0:
        return np.empty((M, 0), dtype=int), np.empty((M, 0), dtype=float)
    if k > N:
        k = N

    # 使用向量化公式计算平方距离：||q - d||^2 = ||q||^2 + ||d||^2 - 2 q·d^T
    data_sq = np.sum(data * data, axis=1)    # (N,)
    query_sq = np.sum(query * query, axis=1) # (M,)

    cross = np.dot(query, data.T)            # (M, N)
    d2 = query_sq[:, None] + data_sq[None, :] - 2.0 * cross
    # 数值误差可能导致负数
    d2 = np.maximum(d2, 0.0)

    # argsort 小到大并取前 k
    idx = np.argpartition(d2, kth=k-1, axis=1)[:, :k]  # (M, k) 未排序
    # 将每行索引按实际距离排序
    row_idx = np.arange(M)[:, None]
    unsorted_dists = d2[row_idx, idx]
    order = np.argsort(unsorted_dists, axis=1)
    sorted_idx = idx[row_idx, order]
    sorted_d2 = d2[row_idx, sorted_idx]

    neighbor_ids = sorted_idx.astype(int)
    neighbor_dists = np.sqrt(sorted_d2.astype(float))

    return neighbor_ids, neighbor_dists

# 简单示例:
# if __name__ == "__main__":
#     a = np.array([[1,1],[2,2],[3,2],[4,4],[5,6]])
#     b = np.array([[1,1],[2,1],[6,2]])
#     ids, dists = kNearestNeighbors(a, b, 2)
#     print(ids)    # 每行两个索引（0-based）
#     print(dists)  # 每行两个距离