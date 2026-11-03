import os
import glob
import cv2
import numpy as np
from skimage.morphology import thin

def imadjust_simple(img, in_range=(0.3, 0.7)):
    """类似 MATLAB imadjust，将 in_range 映射到 [0,255]"""
    lo, hi = in_range
    lo_v = lo * 255.0
    hi_v = hi * 255.0
    imgf = img.astype(np.float32)
    out = (imgf - lo_v) * (255.0 / max(1e-8, (hi_v - lo_v)))
    return np.clip(out, 0, 255).astype(np.uint8)

def find_processing_callable():
    """按优先顺序尝试导入项目中可能存在的处理函数"""
    candidates = [
        ('testfin_FingerNet_fft', 'testfin_FingerNet_fft'),
        ('FingerNet_testfin_skelonly', 'FingerNet_testfin_skelonly'),
        ('testfin_FVC', 'testfin_FVC'),
        ('testfin_NIST_MinuNet', 'testfin_NIST_MinuNet'),
        ('testfin_NIST', 'testfin_NIST'),
    ]
    for modname, funcname in candidates:
        try:
            mod = __import__(modname)
            fn = getattr(mod, funcname)
            return fn
        except Exception:
            # 尝试相对导入（当作为包使用时）
            try:
                pkgmod = __import__(f'.{modname}', fromlist=[funcname])
                fn = getattr(pkgmod, funcname)
                return fn
            except Exception:
                continue
    # 退回到 f_enhance_3420_NIST.f_enhance_3420_NIST 如果可用（作为后备增强）
    try:
        mod = __import__('f_enhance_3420_NIST')
        fn = getattr(mod, 'f_enhance_3420_NIST')
        return fn
    except Exception:
        try:
            pkg = __import__('.f_enhance_3420_NIST', fromlist=['f_enhance_3420_NIST'])
            return getattr(pkg, 'f_enhance_3420_NIST')
        except Exception:
            return None

def process_folder(myFolder):
    if not os.path.isdir(myFolder):
        raise FileNotFoundError(f'Folder not found: {myFolder}')

    pattern = os.path.join(myFolder, '*.bmp')
    files = glob.glob(pattern)
    if not files:
        print("No .bmp files found in", myFolder)
        return

    proc_fn = find_processing_callable()
    if proc_fn is None:
        print("No processing function (testfin_*) found. Will still save pre/post images.")

    for fullpath in files:
        base = os.path.basename(fullpath)
        name, _ = os.path.splitext(base)
        print("Processing:", base)

        im = cv2.imread(fullpath, cv2.IMREAD_UNCHANGED)
        if im is None:
            print("  failed to read:", fullpath)
            continue
        # 灰度
        if im.ndim == 3:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im.copy()

        # imadjust([0.3,0.7])
        J = imadjust_simple(gray, (0.3, 0.7))
        cv2.imwrite(os.path.join(myFolder, f'{name}_adj.png'), J)

        # invert and thin (MATLAB used ~J then bwmorph('thin','inf'))
        inverted = ~(J.astype(bool))
        skeleton = thin(inverted)
        skel_img = (skeleton.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(myFolder, f'{name}_skeleton.png'), skel_img)

        # display variant ~K saved for comparison
        display_img = (~skeleton).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(myFolder, f'{name}_display.png'), display_img)

        # 调用处理函数（MATLAB 中传入原始图像 im）
        if proc_fn is not None:
            try:
                # 传入灰度原图以与 MATLAB 行为一致
                res = proc_fn(gray, base)
            except TypeError:
                # 退回传入调整后图像（某些移植的函数可能期望增强图）
                try:
                    res = proc_fn(J, base)
                except Exception as e:
                    print("  processing call failed:", e)
                    res = None
            except Exception as e:
                print("  processing call failed:", e)
                res = None

            # 保存常见返回值：newim, binim, mask, reliability
            if isinstance(res, (tuple, list)):
                try:
                    if len(res) >= 1 and isinstance(res[0], np.ndarray):
                        newim = res[0]
                        # 归一化可视化
                        mn, mx = float(np.nanmin(newim)), float(np.nanmax(newim))
                        if mx - mn > 1e-8:
                            vis = ((newim - mn) / (mx - mn) * 255).astype(np.uint8)
                        else:
                            vis = (newim * 0 + 128).astype(np.uint8)
                        cv2.imwrite(os.path.join(myFolder, f'{name}_newim.png'), vis)
                    if len(res) >= 2 and isinstance(res[1], np.ndarray):
                        binim = res[1]
                        cv2.imwrite(os.path.join(myFolder, f'{name}_bin.png'),
                                    (binim.astype(np.uint8) * 255) if binim.dtype==bool else binim)
                    if len(res) >= 3 and isinstance(res[2], np.ndarray):
                        mask = res[2]
                        cv2.imwrite(os.path.join(myFolder, f'{name}_mask.png'),
                                    (mask.astype(np.uint8) * 255) if mask.dtype==bool else mask)
                    if len(res) >= 4 and isinstance(res[3], np.ndarray):
                        rel = res[3]
                        rr = rel.copy()
                        rr = rr - np.nanmin(rr)
                        if np.nanmax(rr) > 0:
                            rr = (rr / np.nanmax(rr) * 255).astype(np.uint8)
                        else:
                            rr = (rr * 0).astype(np.uint8)
                        cv2.imwrite(os.path.join(myFolder, f'{name}_reliability.png'), rr)
                    print("  processing outputs saved where available")
                except Exception as e:
                    print("  failed saving outputs:", e)
            elif isinstance(res, np.ndarray):
                cv2.imwrite(os.path.join(myFolder, f'{name}_result.png'), res)
            else:
                # no structured return
                pass
        else:
            print("  no processing function available; saved pre/post images only")

if __name__ == '__main__':
    # 修改为你的目录
    myFolder = '/path/to/your/try/folder'  # <-- 替换为实际路径
    process_folder(myFolder)