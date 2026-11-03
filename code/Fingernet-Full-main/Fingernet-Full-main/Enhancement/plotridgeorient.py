# ...existing code...
import numpy as np
import matplotlib.pyplot as plt

def plotridgeorient(orient, spacing, im=None, figno=None):
    """
    在图像（可选）上绘制方向场（orient，以弧度为单位）。
    参数:
        orient (2D ndarray) : 方向图（弧度，通常范围 [0, pi)）
        spacing (int)       : 子采样间隔（整数）
        im (2D ndarray)     : 可选的背景指纹图像（灰度）
        figno (int or None) : 可选的 matplotlib figure id（整数）
    返回:
        ax : matplotlib.axes.Axes 对象
    """
    if not float(spacing).is_integer():
        raise ValueError("spacing must be an integer")
    spacing = int(spacing)

    orient = np.asarray(orient)
    if orient.ndim != 2:
        raise ValueError("orient must be a 2D array")

    rows, cols = orient.shape
    if rows <= 2*spacing or cols <= 2*spacing:
        # 无足够点可绘制，仍显示背景图（若提供）
        if im is not None:
            if figno is not None:
                plt.figure(figno)
            fig, ax = plt.subplots()
            ax.imshow(im, cmap='gray', origin='upper')
            ax.set_axis_off()
            return ax
        else:
            return None

    # 子采样索引（兼容 MATLAB 的 spacing:spacing:rows-spacing）
    y_idx = np.arange(spacing, rows - spacing, spacing)
    x_idx = np.arange(spacing, cols - spacing, spacing)

    # 子采样方向
    s_orient = orient[np.ix_(y_idx, x_idx)]

    # 向量可视化长度
    length = 0.8 * spacing

    # 计算偏移（MATLAB 中 xoff = len/2*cos(s); yoff = len/2*sin(s)）
    xoff = (length / 2.0) * np.cos(s_orient)
    yoff = (length / 2.0) * np.sin(s_orient)

    # 网格坐标（x: 列坐标, y: 行坐标）
    X, Y = np.meshgrid(x_idx, y_idx)

    # 起点调整（MATLAB 用 x = x - xoff; y = y - yoff）
    X0 = X - xoff
    Y0 = Y - yoff

    # 向量分量（MATLAB 用 u=xoff*2; v=yoff*2）
    U = xoff * 2.0
    V = yoff * 2.0

    # 建立 figure / axis
    if figno is not None:
        plt.figure(figno)
    fig, ax = plt.subplots() if figno is None else (plt.gcf(), plt.gca())

    # 显示背景图（若提供）
    if im is not None:
        ax.imshow(im, cmap='gray', origin='upper')

    # quiver: origin='upper' + imshow(..., origin='upper') 对应 MATLAB 的 axis ij
    # 使用 scale=1, scale_units='xy' 以像素为单位绘制真实长度向量
    ax.quiver(X0, Y0, U, V, angles='xy', scale_units='xy', scale=1.0,
              color='r', width=0.003, headwidth=0, headlength=0)

    ax.set_aspect('equal')
    ax.invert_yaxis()  # 保证与 MATLAB axis ij 行为一致
    ax.set_axis_off()
    plt.tight_layout()
    return ax
# ...existing code...