import os
import glob
import cv2
import numpy as np
from skimage.morphology import thin

def imadjust_simple(img, in_range=(0.3, 0.7)):
    """将输入强度区间映射到 [0,255]（类似 MATLAB imadjust）"""
    lo, hi = in_range
    lo_v = lo * 255.0
    hi_v = hi * 255.0
    imgf = img.astype(np.float32)
    out = (imgf - lo_v) * (255.0 / max(1e-8, (hi_v - lo_v)))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def process_folder(myFolder):
    if not os.path.isdir(myFolder):
        raise FileNotFoundError(f"Folder not found: {myFolder}")

    pattern = os.path.join(myFolder, '*.bmp')
    files = glob.glob(pattern)
    if not files:
        print("No .bmp files found in", myFolder)
        return

    # 尝试导入 Python 版的处理函数（若存在）
    testfin = None
    for modname in ('testfin_NIST', 'testfin_NIST_MinuNet'):
        try:
            mod = __import__(modname)
            testfin = getattr(mod, modname)
            break
        except Exception:
            # 也尝试相对导入（当作为包使用时）
            try:
                from . import testfin_NIST as mod
                testfin = getattr(mod, 'testfin_NIST')
                break
            except Exception:
                pass

    for fullFileName in files:
        baseFileName = os.path.basename(fullFileName)
        name, _ = os.path.splitext(baseFileName)
        print("Processing:", baseFileName)

        im = cv2.imread(fullFileName, cv2.IMREAD_UNCHANGED)
        if im is None:
            print("  failed to read:", fullFileName)
            continue

        # 灰度
        if im.ndim == 3:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im.copy()

        # imadjust([0.3,0.7])
        J = imadjust_simple(gray, (0.3, 0.7))
        out_adj = os.path.join(myFolder, f"{name}_adj.png")
        cv2.imwrite(out_adj, J)

        # invert and thin: MATLAB used ~J then bwmorph('thin','inf')
        inverted = ~(J.astype(bool))      # True 表示 ridge foreground after invert
        skeleton = thin(inverted)         # boolean array (thin)
        # MATLAB displays ~K, keep same display variant as file if needed
        display_img = (~skeleton).astype(np.uint8) * 255

        out_skel = os.path.join(myFolder, f"{name}_skeleton.png")
        out_display = os.path.join(myFolder, f"{name}_display.png")
        cv2.imwrite(out_skel, (skeleton.astype(np.uint8) * 255))
        cv2.imwrite(out_display, display_img)

        # 调用 testfin_NIST 或 testfin_NIST_MinuNet（如果可用）
        if testfin is not None:
            try:
                # MATLAB 传入的是二值/细化图；这里传入与原等价的 uint8 图像（0/255）
                rval = testfin((~skeleton).astype(np.uint8)*255, baseFileName)
                # 若返回多项，尝试保存常见输出（newim, binim, mask, reliability）
                if isinstance(rval, tuple):
                    if len(rval) >= 2 and isinstance(rval[1], np.ndarray):
                        binim = rval[1]
                        cv2.imwrite(os.path.join(myFolder, f"{name}_bin.png"),
                                    (binim.astype(np.uint8)*255) if binim.dtype==bool else binim)
                    if len(rval) >= 3 and isinstance(rval[2], np.ndarray):
                        mask = rval[2]
                        cv2.imwrite(os.path.join(myFolder, f"{name}_mask.png"),
                                    (mask.astype(np.uint8)*255) if mask.dtype==bool else mask)
                print("  called testfin, results saved if returned")
            except Exception as e:
                print("  testfin call failed:", e)
        else:
            print("  testfin_NIST / testfin_NIST_MinuNet not found, skipped")

if __name__ == '__main__':
    # 修改为实际目录
    myFolder = '/path/to/your/folder'  # <-- 替换为实际路径
    process_folder(myFolder)