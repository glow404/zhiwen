# ...existing code...
import numpy as np

def tico(img, x, y, oimg, orient_img, o_rel, radius, theta_count, c_flag, start_t, blk_sz):
    """
    Python 移植的 tico.m

    参数:
      img         - 原始图像 (未直接读取像素值，只用于尺寸检查)
      x, y        - 中心点坐标 (0-based: x = col, y = row)
      oimg        - （未使用的输入，占位）
      orient_img  - 方向图 (2D ndarray, 弧度)
      o_rel       - 方向可靠度图 (2D ndarray)
      radius      - 1D iterable: 待采样的半径列表
      theta_count - 1D iterable: 每个半径对应的角度步数
      c_flag      - 标志 (若为1 则使用 start_t 作为起始角)
      start_t     - 起始角 (弧度)
      blk_sz      - 块大小 (未使用)

    返回:
      orients (list of float) - 每次采样的方向差值归一化(-1 表示无效)
      radii  (list of float) - 对应的半径或 -1（无效）
      minX, minY (int,int)   - 使差值最小的采样点坐标 (0-based col,row)
      minT   (float)         - 最小归一化差值
    """
    h, w = orient_img.shape[:2]
    orients = []
    radii_out = []

    minX = 0
    minY = 0
    minT = 2.0 * np.pi

    # 起始角
    if c_flag == 1:
        start_theta = start_t
    else:
        # 使用 orient_img[y,x]（假定索引合法）
        start_theta = float(orient_img[y, x])

    theta = start_theta

    radius = np.asarray(radius).ravel()
    theta_count = np.asarray(theta_count).ravel()

    o_index = 0
    r_index = 0

    # 主循环：对每个 radius value
    while r_index < radius.size and o_index < theta_count.size:
        tc = float(theta_count[o_index])
        o_i = 0

        # MATLAB 条件: ((o_i < theta_count(o_index)) && ~c_flag) || (c_flag && o_i < theta_count(o_index)/2)
        while ((o_i < tc) and (not bool(c_flag))) or (bool(c_flag) and (o_i < tc / 2.0)):
            Rt = float(radius[r_index])
            Xt = int(round(x + Rt * np.cos(theta)))
            Yt = int(round(y - Rt * np.sin(theta)))

            valid = True
            # 边界与安全范围检查（与原 MATLAB 相同阈值）
            if Xt <= 20 or Xt >= (w - 20) or Yt <= 20 or Yt >= (h - 20):
                valid = False

            # 检查方向可靠度 (o_rel) 小于阈值视为无效
            try:
                relv = o_rel[Yt, Xt]
                if not np.isnan(relv) and relv < 0.5:
                    valid = False
            except Exception:
                # 若索引越界或 o_rel 不可用，则标记无效
                valid = False

            if not valid:
                radii_out.append(-1)
                orients.append(-1)
            else:
                # 在 3x3 邻域内收集 orient_img 的值并取中值
                vals = []
                for ee in range(Yt - 1, Yt + 2):
                    if ee < 0 or ee >= h:
                        continue
                    for ff in range(Xt - 1, Xt + 2):
                        if ff < 0 or ff >= w:
                            continue
                        v = orient_img[ee, ff]
                        if not np.isnan(v):
                            vals.append(float(v))
                if len(vals) == 0:
                    radii_out.append(-1)
                    orients.append(-1)
                else:
                    ta = float(np.median(vals))
                    tb = start_theta
                    radii_out.append(Rt)
                    diff = abs(tb - ta)
                    norm_val = min(diff, abs(np.pi - diff)) * 2.0 / np.pi
                    orients.append(norm_val)
                    if norm_val < minT:
                        minT = norm_val
                        minX = Xt
                        minY = Yt

            o_i += 1
            # 更新角度（保留 MATLAB 的递增公式）
            if tc != 0:
                theta = theta + o_i * 2.0 * np.pi / tc

        # 进入下一半径 / theta_count 项
        r_index += 1
        theta = start_theta
        o_index += 1

    return orients, radii_out, minX, minY, minT
# ...existing code...