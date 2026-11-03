"""
灰度转换，增强对比度
二值化处理
提取骨架
"""
import os
import glob
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import thin
import matplotlib.pyplot as plt

def imadjust_simple(img, in_range=(0.3, 0.7)):
    lo, hi = in_range
    lo_v = lo * 255.0
    hi_v = hi * 255.0
    imgf = img.astype(np.float32)
    out = (imgf - lo_v) * (255.0 / max(1e-8, (hi_v - lo_v)))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def process_folder(in_folder, out_folder=None, show=False):
    if out_folder is None:
        out_folder = os.path.join(in_folder, 'output')
    os.makedirs(out_folder, exist_ok=True)

    pattern = os.path.join(in_folder, '*.tif')
    files = glob.glob(pattern)
    if not files:
        print("No .tif files found in", in_folder)
        return

    for fullpath in files:
        base = os.path.basename(fullpath)
        name, _ = os.path.splitext(base)
        im = cv2.imread(fullpath, cv2.IMREAD_UNCHANGED)
        if im is None:
            print("Failed to read", fullpath); continue
        if im.ndim == 3:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im.copy()

        adjusted = imadjust_simple(gray, (0.3, 0.7))

        # 二值化（Otsu），然后取反再细化（与 MATLAB 中 bwmorph(~J,'thin','inf') 对应）
        try:
            th = threshold_otsu(adjusted)
            binary = adjusted > th
        except Exception:
            # fallback 固定阈值
            binary = adjusted > 128
        inverted = ~binary  # True 表示前景（将要细化）
        skeleton = thin(inverted)  # boolean array
        result_display = ~skeleton  # MATLAB 显示的是 ~K

        # 保存：调整后图、二值图、细化图、显示图
        cv2.imwrite(os.path.join(out_folder, f'{name}_adj.png'), adjusted)
        cv2.imwrite(os.path.join(out_folder, f'{name}_binary.png'), (binary.astype(np.uint8)*255))
        cv2.imwrite(os.path.join(out_folder, f'{name}_skeleton.png'), (skeleton.astype(np.uint8)*255))
        cv2.imwrite(os.path.join(out_folder, f'{name}_display.png'), (result_display.astype(np.uint8)*255))

        if show:
            plt.figure(figsize=(10,4))
            plt.subplot(1,4,1); plt.title('adjusted'); plt.imshow(adjusted, cmap='gray'); plt.axis('off')
            plt.subplot(1,4,2); plt.title('binary'); plt.imshow(binary, cmap='gray'); plt.axis('off')
            plt.subplot(1,4,3); plt.title('skeleton'); plt.imshow(skeleton, cmap='gray'); plt.axis('off')
            plt.subplot(1,4,4); plt.title('display (~K)'); plt.imshow(result_display, cmap='gray'); plt.axis('off')
            plt.suptitle(name)
            plt.show()

        print("Processed:", base)

if __name__ == '__main__':
    # 修改为你的目录
    myFolder = '/home/tom/fingerprint-deepleaning/End-to-End-Automated-Latent-Fingerprint-Identification/Enhancement/temp'  # tif 图像所在目录
    process_folder(myFolder, show=False)