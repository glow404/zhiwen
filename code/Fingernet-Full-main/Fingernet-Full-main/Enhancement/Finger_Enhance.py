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
    return np.clip(out, 0, 255).astype(np.uint8)

def find_testfin_callable():
    """按优先顺序尝试导入可能存在的 Python 版本处理函数"""
    candidates = [
        ('testfin', 'testfin'),
        ('testfin_NIST', 'testfin_NIST'),
        ('testfin_NIST_MinuNet', 'testfin_NIST_MinuNet'),
        ('testfin_skelonly', 'testfin_skelonly'),
        ('testfin_NIST_MinuNet', 'testfin_NIST_MinuNet'),  # duplicate safe
    ]
    for modname, funcname in candidates:
        try:
            mod = __import__(modname)
            fn = getattr(mod, funcname)
            return fn
        except Exception:
            # 尝试相对导入（当作为包运行时）
            try:
                package_mod = __import__(f'.{modname}', fromlist=[funcname])
                fn = getattr(package_mod, funcname)
                return fn
            except Exception:
                continue
    return None

def process_folder(myFolder):
    if not os.path.isdir(myFolder):
        raise FileNotFoundError(f"Folder not found: {myFolder}")

    pattern = os.path.join(myFolder, '*.bmp')
    files = glob.glob(pattern)
    if not files:
        print("No .bmp files found in", myFolder)
        return

    # 只处理前 5 个文件（MATLAB 源中 for kk = 1:5）
    files = files[:5]

    testfin_fn = find_testfin_callable()
    if testfin_fn is None:
        print("No testfin_* function found; processing will still save pre/ post images.")

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
        inverted = ~(J.astype(bool))   # boolean
        skeleton = thin(inverted)      # boolean skeleton
        skel_img = (skeleton.astype(np.uint8) * 255)
        out_skel = os.path.join(myFolder, f"{name}_skeleton.png")
        cv2.imwrite(out_skel, skel_img)

        # MATLAB displays ~K; 保存以便比对
        display_img = (~skeleton).astype(np.uint8) * 255
        out_display = os.path.join(myFolder, f"{name}_display.png")
        cv2.imwrite(out_display, display_img)

        # 尝试调用 testfin 函数（若存在）
        if testfin_fn is not None:
            try:
                # MATLAB 传入的是 K（~J 然后 thin），它显示 ~K 作为输入 - 这里传回类似 uint8 图像
                # 传入 (~skeleton).astype(np.uint8)*255 与 MATLAB 行为相近
                inp = (~skeleton).astype(np.uint8) * 255
                result = testfin_fn(inp, baseFileName)
                # 若返回多项，尝试保存常见输出（newim, binim, mask, reliability）
                if isinstance(result, tuple) or isinstance(result, list):
                    # newim
                    if len(result) >= 1 and isinstance(result[0], np.ndarray):
                        try:
                            newim = result[0]
                            # 归一化并保存可视化
                            nm = newim.copy()
                            mn, mx = nm.min(), nm.max()
                            if mx - mn > 1e-8:
                                vis = ((nm - mn) / (mx - mn) * 255).astype(np.uint8)
                            else:
                                vis = (nm * 0 + 128).astype(np.uint8)
                            cv2.imwrite(os.path.join(myFolder, f"{name}_newim.png"), vis)
                        except Exception:
                            pass
                    # binim
                    if len(result) >= 2 and isinstance(result[1], np.ndarray):
                        binim = result[1]
                        cv2.imwrite(os.path.join(myFolder, f"{name}_bin.png"),
                                    (binim.astype(np.uint8) * 255) if binim.dtype==bool else binim)
                    # mask
                    if len(result) >= 3 and isinstance(result[2], np.ndarray):
                        mask = result[2]
                        cv2.imwrite(os.path.join(myFolder, f"{name}_mask.png"),
                                    (mask.astype(np.uint8) * 255) if mask.dtype==bool else mask)
                    # reliability (save as float image visualization)
                    if len(result) >= 4 and isinstance(result[3], np.ndarray):
                        rel = result[3]
                        # normalize for visualization
                        rr = rel.copy()
                        rr = rr - np.nanmin(rr)
                        if np.nanmax(rr) > 0:
                            rr = (rr / np.nanmax(rr) * 255).astype(np.uint8)
                        else:
                            rr = (rr * 0 + 0).astype(np.uint8)
                        cv2.imwrite(os.path.join(myFolder, f"{name}_reliability.png"), rr)
                else:
                    # 单值返回，尝试保存如可能
                    if isinstance(result, np.ndarray):
                        cv2.imwrite(os.path.join(myFolder, f"{name}_result.png"), result)
                print("  testfin called and outputs saved where available")
            except Exception as e:
                print("  testfin call failed:", repr(e))
        else:
            print("  no testfin function available; saved pre/post images")

if __name__ == '__main__':
    # 修改为你的目录路径后运行
    myFolder = '/path/to/your/temp/folder'  # <-- 替换为实际路径
    process_folder(myFolder)