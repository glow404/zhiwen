import numpy as np

try:
    from .p import p
except Exception:
    from p import p

def _map_pao(i):
    """MATLAB 中 pao/pbo/pco 的映射保持 1..9 范围以便比较"""
    if i < 5:
        return 4 + i
    else:
        return ((4 + i) % 9) + 1

def trace_path(img, m_list, path_len):
    """
    Trace short paths from minutiae points on a thinned binary image.

    参数:
      img      : 2D numpy array (thinned binary image, 0/1 或 False/True)
      m_list   : iterable of minutiae rows, 每行 [x, y, CN] (x=col, y=row)，坐标为 0-based
      path_len : int, 最大追踪步数

    返回:
      img_n : 2D numpy uint8 array，追踪到的位置标记为 1
    """
    img = np.asarray(img)
    h, w = img.shape[:2]
    img_n = np.zeros((h, w), dtype=np.uint8)

    for idx in range(int(np.shape(m_list)[0])):
        # 读取 minutiae（假定 m_list 为 Nx3 数组或列表）
        x = int(m_list[idx][0])
        y = int(m_list[idx][1])
        # CN = m_list[idx][2]  # 未使用，但保留格式一致性

        progress = 1
        # 标记起点
        if 0 <= y < h and 0 <= x < w:
            img_n[y, x] = 1
        else:
            continue

        # 初始化三条可能的分支坐标与方向编码
        pax = pay = pbx = pby = pcx = pcy = 0
        pao = pbo = pco = 0  # MATLAB 中为 1..9 编号

        # 在中心点周围寻找最大 3 条分支（与 MATLAB 相同逻辑）
        for i in range(1, 9):
            ta, xa, ya = p(img, x, y, i)
            tb, xb, yb = p(img, x, y, i + 1)
            if ta > tb:
                if pao == 0:
                    pao = _map_pao(i)
                    pax, pay = xa, ya
                elif pbo == 0:
                    pbo = _map_pao(i)
                    pbx, pby = xa, ya
                else:
                    pco = _map_pao(i)
                    pcx, pcy = xa, ya

        # 记录第一次邻居（可能为 0 表示不存在）
        # 迭代追踪，直到达到 path_len 或无可追踪分支
        while progress < path_len:
            if pax == 0 and pbx == 0 and pcx == 0:
                break

            # 标记现有分支点
            if pax != 0 and 0 <= pay < h and 0 <= pax < w:
                img_n[pay, pax] = 1
            if pbx != 0 and 0 <= pby < h and 0 <= pbx < w:
                img_n[pby, pbx] = 1
            if pcx != 0 and 0 <= pcy < h and 0 <= pcx < w:
                img_n[pcy, pcx] = 1

            progress += 1

            # 对每个分支尝试找到下一像素（跳过回头方向）
            if pao != 0:
                found = False
                for i in range(1, 9):
                    if i == pao:
                        continue
                    ta, xa, ya = p(img, pax, pay, i)
                    # ta == 1 表示是脊点，且未被标记过（避免循环）
                    if ta == 1 and (0 <= ya < h and 0 <= xa < w) and img_n[ya, xa] != 1:
                        pao = _map_pao(i)
                        pax, pay = xa, ya
                        found = True
                        break
                if not found:
                    # 无后继则置 0
                    pao = 0
                    pax = pay = 0

            if pbo != 0:
                found = False
                for i in range(1, 9):
                    if i == pbo:
                        continue
                    ta, xa, ya = p(img, pbx, pby, i)
                    if ta == 1 and (0 <= ya < h and 0 <= xa < w) and img_n[ya, xa] != 1:
                        pbo = _map_pao(i)
                        pbx, pby = xa, ya
                        found = True
                        break
                if not found:
                    pbo = 0
                    pbx = pby = 0

            if pco != 0:
                found = False
                for i in range(1, 9):
                    if i == pco:
                        continue
                    ta, xa, ya = p(img, pcx, pcy, i)
                    if ta == 1 and (0 <= ya < h and 0 <= xa < w) and img_n[ya, xa] != 1:
                        pco = _map_pao(i)
                        pcx, pcy = xa, ya
                        found = True
                        break
                if not found:
                    pco = 0
                    pcx = pcy = 0

    return img_n
