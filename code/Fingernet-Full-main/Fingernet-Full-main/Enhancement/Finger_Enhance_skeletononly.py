import os
import glob
import cv2
import numpy as np
from skimage.morphology import thin

def imadjust_simple(img, in_range=(0.3, 0.7)):
    lo, hi = in_range
    lo_v = lo * 255.0
    hi_v = hi * 255.0
    imgf = img.astype(np.float32)
    out = (imgf - lo_v) * (255.0 / max(1e-8, (hi_v - lo_v)))
    return np.clip(out, 0, 255).astype(np.uint8)

def process_folder(myFolder):
    if not os.path.isdir(myFolder):
        raise FileNotFoundError(f"folder not found: {myFolder}")
    pattern = os.path.join(myFolder, '*.bmp')
    files = glob.glob(pattern)
    if not files:
        print("No .bmp files found in", myFolder); return

    # 尝试导入 Python 版的 testfin_skelonly（若存在）
    testfin = None
    try:
        from testfin_skelonly import testfin_skelonly
        testfin = testfin_skelonly
    except Exception:
        # 相对导入尝试
        try:
            from .testfin_skelonly import testfin_skelonly
            testfin = testfin_skelonly
        except Exception:
            testfin = None

    for fullFileName in files:
        baseFileName = os.path.basename(fullFileName)
        name, _ = os.path.splitext(baseFileName)
        print("Processing", baseFileName)
        im = cv2.imread(fullFileName, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(" failed to read:", fullFileName); continue
        if im.ndim == 3:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im.copy()

        J = imadjust_simple(gray, (0.3, 0.7))
        out_adj = os.path.join(myFolder, f"{name}_adj.png")
        cv2.imwrite(out_adj, J)

        inverted = ~(J.astype(bool))   # MATLAB 的 ~J
        skeleton = thin(inverted)      # boolean 骨架
        skel_img = (skeleton.astype(np.uint8) * 255)
        out_skel = os.path.join(myFolder, f"{name}_skeleton.png")
        cv2.imwrite(out_skel, skel_img)

        if testfin is not None:
            try:
                # 传入与 MATLAB 类似的二值/骨架图（uint8 0/255）
                result = testfin((~skeleton).astype(np.uint8)*255, baseFileName)
                # 若返回数组，尝试保存常见输出
                if isinstance(result, tuple):
                    # newim, binim, mask, reliability 常见顺序
                    if len(result) >= 2 and isinstance(result[1], np.ndarray):
                        cv2.imwrite(os.path.join(myFolder, f"{name}_bin.png"),
                                    (result[1].astype(np.uint8)*255) if result[1].dtype==bool else result[1])
                    if len(result) >= 3 and isinstance(result[2], np.ndarray):
                        cv2.imwrite(os.path.join(myFolder, f"{name}_mask.png"),
                                    (result[2].astype(np.uint8)*255) if result[2].dtype==bool else result[2])
                print(" testfin_skelonly called")
            except Exception as e:
                print(" testfin_skelonly error:", e)
        else:
            print(" testfin_skelonly not found, skipped")

if __name__ == '__main__':
    # 修改为实际目录后运行
    myFolder = '/path/to/your/try/folder'  # <-- 替换为实际路径
    process_folder(myFolder)