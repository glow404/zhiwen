# ...existing code...
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def _read_image(im):
    """如果 im 是路径字符串则读取图像，否则返回原数组。"""
    if isinstance(im, str):
        try:
            img = mpimg.imread(im)
            # matplotlib may return floats in 0..1 or uint8
            return img
        except Exception:
            # fallback: try opencv if available
            try:
                import cv2
                img = cv2.imread(im, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise IOError("cannot read image")
                # convert BGR->RGB if 3 channels
                if img.ndim == 3 and img.shape[2] == 3:
                    img = img[:, :, ::-1]
                return img
            except Exception:
                raise
    else:
        return np.asarray(im)

def show(im, param2=None, param3=None):
    """
    显示图像（等价于 MATLAB show）。

    用法:
        h = show(im)
        h = show(im, figNo)
        h = show(im, title)
        h = show(im, figNo, title)

    参数:
        im      - 图像数组或图像文件路径（字符串）
        param2  - 可选：若为 int 则视作 figNo；若为 str 则视作标题
        param3  - 可选：当 param2 为 figNo 时的标题字符串

    返回:
        fig - matplotlib.figure.Figure 对象（类似 MATLAB 的 figure handle）
    """
    # 解析输入
    Title = None
    figNo = -1  # -1 表示创建新 figure（默认）
    if param2 is not None and param3 is None:
        if isinstance(param2, str):
            Title = param2
        elif isinstance(param2, (int, np.integer)):
            figNo = int(param2)
        else:
            raise ValueError("第二个参数必须为整数(图号)或字符串(标题)")
    elif param2 is not None and param3 is not None:
        if not isinstance(param3, str):
            raise ValueError("第三个参数必须为字符串(标题)")
        if not isinstance(param2, (int, np.integer)):
            raise ValueError("第二个参数必须为整数(图号)")
        figNo = int(param2)
        Title = param3

    img = _read_image(im)

    # 选择或创建 figure
    if figNo > 0:
        fig = plt.figure(num=figNo)
        # 使用整个窗口
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax = fig.add_subplot(1,1,1)
    elif figNo == 0:
        # 使用当前坐标系 / subplot，不创建新窗口
        fig = plt.gcf()
        ax = plt.gca()
    else:
        # 新建 figure
        fig = plt.figure()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax = fig.add_subplot(1,1,1)

    # 显示灰度或彩色图像
    if img.ndim == 2:
        ax.imshow(img, cmap='gray', origin='upper')
    elif img.ndim == 3:
        # 如果是 RGBA 或 RGB，matplotlib 正确处理
        ax.imshow(img, origin='upper')
    else:
        raise ValueError("Unsupported image dimensions: %s" % (img.shape,))

    # 设置标题或图窗名称
    if figNo != 0:
        if Title is not None:
            # 尝试设置窗口标题（不同后端不同方法）
            try:
                fig.canvas.manager.set_window_title(Title)
            except Exception:
                # 如果不能设置窗口标题则在图像上显示标题
                ax.set_title(Title)
    else:
        if Title is not None:
            ax.set_title(Title)

    # 保持像素纵横比并隐藏坐标轴
    ax.set_aspect('equal')
    ax.axis('off')

    # 尽量实现 "truesize"：使 figure 大小对应图像像素（基于当前 DPI）
    try:
        dpi = fig.get_dpi()
        if img.ndim == 2:
            h, w = img.shape
        else:
            h, w = img.shape[0], img.shape[1]
        # 设置画布大小（英寸）
        fig.set_size_inches(max(0.5, w / dpi), max(0.5, h / dpi))
    except Exception:
        pass

    # 强制刷新显示（交互模式下会生效）
    fig.canvas.draw_idle()

    return fig
# ...existing code...