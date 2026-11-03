import os
import glob
import cv2
import numpy as np
import sys

# 确保可以导入同目录下的模块
this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

try:
    from f_enhance_3420_NIST import f_enhance_3420_NIST
except Exception:
    try:
        from Enhancement.f_enhance_3420_NIST import f_enhance_3420_NIST
    except Exception:
        f_enhance_3420_NIST = None

def imadjust_simple(img, in_range=(0.3, 0.7)):
    """模拟 MATLAB imadjust: 将输入强度区间映射到 [0,255]"""
    lo, hi = in_range
    lo_v = lo * 255.0
    hi_v = hi * 255.0
    imgf = img.astype(np.float32)
    out = (imgf - lo_v) * (255.0 / max(1e-8, (hi_v - lo_v)))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def process_folder(in_folder, out_folder=None, use_function='f_enhance'):
    """
    遍历 in_folder 下的 .bmp 文件，处理并保存结果到 out_folder（若为 None 则在 in_folder/output）
    use_function: 'f_enhance' 调用 f_enhance_3420_NIST；否则只保存预处理结果
    """
    if out_folder is None:
        out_folder = os.path.join(in_folder, 'output')
    os.makedirs(out_folder, exist_ok=True)

    pattern = os.path.join(in_folder, '*.bmp')
    files = glob.glob(pattern)
    if not files:
        print("No .bmp files found in", in_folder)
        return

    for fullpath in files:
        base = os.path.basename(fullpath)
        name, _ = os.path.splitext(base)
        img = cv2.imread(fullpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Failed to read", fullpath)
            continue
        # 灰度
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 模拟 MATLAB 的 imadjust(...,[0.3,0.7],[])
        adjusted = imadjust_simple(gray, (0.3, 0.7))

        # 调用增强/细化管线（若可用）
        if use_function == 'f_enhance' and f_enhance_3420_NIST is not None:
            try:
                # 注意：f_enhance_3420_NIST 的返回签名在不同实现中可能不同
                # 这里兼容之前提供的 Python 版本： returns binim, mask, cimg1, cimg2, oimg1, oimg2, ...
                res = f_enhance_3420_NIST(adjusted)
                # 解析常见返回：若返回 binim, mask...
                if isinstance(res, tuple) and len(res) >= 2:
                    binim = res[0]
                    mask = res[1]
                else:
                    binim = None
                    mask = None
                # 保存二值结果和掩码（若有）
                if binim is not None:
                    out_bin = (binim.astype(np.uint8) * 255) if binim.dtype != np.uint8 else binim
                    cv2.imwrite(os.path.join(out_folder, f'{name}_bin.png'), out_bin)
                if mask is not None:
                    out_mask = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
                    cv2.imwrite(os.path.join(out_folder, f'{name}_mask.png'), out_mask)
                # 保存调整后图像
                cv2.imwrite(os.path.join(out_folder, f'{name}_adj.png'), adjusted)
                print("Processed:", base)
            except Exception as e:
                print("Error processing", base, ":", e)
                # 仍保存预处理图像
                cv2.imwrite(os.path.join(out_folder, f'{name}_adj.png'), adjusted)
        else:
            # 仅保存预处理结果
            cv2.imwrite(os.path.join(out_folder, f'{name}_adj.png'), adjusted)
            print("Preprocessed (no enhance):", base)

if __name__ == '__main__':
    # 修改为你的输入目录
    myFolder = '/home/tom/fingerprint-deepleaning/End-to-End-Automated-Latent-Fingerprint-Identification/Enhancement/temp'  # bmp 图像所在目录
    process_folder(myFolder)