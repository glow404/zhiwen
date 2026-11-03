"""
灰度转换，增强对比度
二值化处理
提取骨架
"""
import os
import glob
import cv2
import numpy as np
from skimage.morphology import thin

def imadjust_simple(img, in_range=(0.4, 0.6)):
    """模拟 MATLAB imadjust 将 in_range 映射到 [0,255]"""
    lo, hi = in_range
    lo_v = lo * 255.0
    hi_v = hi * 255.0
    imgf = img.astype(np.float32)
    out = (imgf - lo_v) * (255.0 / max(1e-8, (hi_v - lo_v)))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def process_folder(myFolder):
    if not os.path.isdir(myFolder):
        raise FileNotFoundError(f"folder not found: {myFolder}")
    pattern = os.path.join(myFolder, '*.bmp')
    files = glob.glob(pattern)
    if not files:
        print("No .bmp files found in", myFolder)
        return
    # try import processing function if exists
    try:
        from testfin_NIST_MinuNet import testfin_NIST_MinuNet
    except Exception:
        testfin_NIST_MinuNet = None

    for fullFileName in files:
        baseFileName = os.path.basename(fullFileName)
        print("Processing", baseFileName)
        im = cv2.imread(fullFileName, cv2.IMREAD_UNCHANGED)
        if im is None:
            print("  failed to read:", fullFileName); continue
        if im.ndim == 3:
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            im_gray = im.copy()
        # imadjust([0.4,0.6])
        J = imadjust_simple(im_gray, (0.4, 0.6))
        # write adjusted image (overwrites original name like MATLAB does)
        out_adjust_path = os.path.join(myFolder, baseFileName.replace('.bmp','_adj.png'))
        cv2.imwrite(out_adjust_path, J)
        # invert and thin: MATLAB used ~J then bwmorph('thin','inf')
        inverted = ~ (J.astype(bool))
        skeleton = thin(inverted)  # boolean array
        skeleton_img = (skeleton.astype(np.uint8) * 255)
        out_skel_path = os.path.join(myFolder, baseFileName.replace('.bmp','_skel.png'))
        cv2.imwrite(out_skel_path, skeleton_img)
        # call testfin_NIST_MinuNet if available (signature may vary)
        if testfin_NIST_MinuNet is not None:
            try:
                # many MATLAB functions expect binary/skeleton images; pass skeleton (bool -> uint8)
                ret = testfin_NIST_MinuNet((~skeleton).astype(np.uint8)*255, baseFileName)
                # save or log returned results if any
                # if ret is tuple, try save first two outputs if they look like images
                if isinstance(ret, tuple) and len(ret) >= 1:
                    # attempt to save common outputs (binim/mask) if returned
                    for i, item in enumerate(ret[:3]):
                        if isinstance(item, (np.ndarray,)):
                            p = os.path.join(myFolder, f"{baseFileName.replace('.bmp','')}_ret{i}.png")
                            cv2.imwrite(p, (item.astype(np.uint8) * 255) if item.dtype==bool else item)
                print("  testfin_NIST_MinuNet called")
            except Exception as e:
                print("  testfin_NIST_MinuNet error:", e)
        else:
            print("  testfin_NIST_MinuNet not found, skipped")

if __name__ == '__main__':
    # 修改为你的输入目录
    myFolder = '/home/tom/fingerprint-deepleaning/End-to-End-Automated-Latent-Fingerprint-Identification/Enhancement/temp'  # bmp 图像所在目录
    process_folder(myFolder)