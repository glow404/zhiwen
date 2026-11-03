"""
读取文件夹中的指纹图像，进行增强、分割、方向计算、频率估计和滤波处理，并保存处理后的结果图像。
"""
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from f_enhance_3420_NIST import ridgesegment, ridgeorient, ridgefreq, ridgefilter 
from f_enhance_3420_NIST import fft_enhance_cubs

def imadjust_simple(img, in_range=(0.3, 0.7)):
    """类似 MATLAB imadjust，将输入强度区间映射到 [0, 255]"""
    lo, hi = in_range
    lo_v = lo * 255.0
    hi_v = hi * 255.0
    imgf = img.astype(np.float32)
    out = (imgf - lo_v) * (255.0 / max(1e-8, (hi_v - lo_v)))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def process_fingerprint_images(input_folder, output_folder):
    """处理文件夹中的所有指纹图像"""
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有 .bmp 图像文件
    pattern = os.path.join(input_folder, '*.bmp')
    files = glob.glob(pattern)
    if not files:
        print("No .bmp files found in", input_folder)
        return

    for file in files:
        base = os.path.basename(file)
        name, _ = os.path.splitext(base)

        # 读取指纹图像
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Failed to read", file)
            continue
        
        # 灰度化
        if img.ndim == 3 and img.shape[2] == 3:
            im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            im_gray = img.astype(float)

        # Step 1: Enhance image using FFT (multi-scale enhancement)
        # 增强图像
        enhanced_img, _ = fft_enhance_cubs(im_gray, 6)
        enhanced_img, _ = fft_enhance_cubs(enhanced_img, 12)
        enhanced_img, cimg2 = fft_enhance_cubs(enhanced_img, 24)

        # Step 2: Segment the image and normalize it
        blksze = 16
        thresh = 0.1
        normim, mask = ridgesegment(enhanced_img, blksze, thresh)

        # Step 3: Compute the ridge orientation and reliability
        orientim, reliability = ridgeorient(normim, 1, 5, 5)

        # Step 4: Estimate the ridge frequency
        freq, medfreq = ridgefreq(normim, mask, orientim, 36, 5, 15)

        # Step 5: Apply ridge filter to enhance the ridge pattern
        freq_for_filter = medfreq * mask.astype(float)
        enhanced_ridges = ridgefilter(normim, orientim, freq_for_filter, 0.5, 0.5, 1)

        # Step 6: Binarize the enhanced image based on median frequency
        binim = enhanced_ridges > medfreq

        # Step 7: Create reliability mask based on reliability threshold
        relim = np.logical_and.reduce([binim, mask.astype(bool), (reliability > 0.5)])

        # Save results
        cv2.imwrite(os.path.join(output_folder, f"{name}_enhanced.png"), enhanced_ridges)
        cv2.imwrite(os.path.join(output_folder, f"{name}_binary.png"), (binim.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(output_folder, f"{name}_mask.png"), (mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(output_folder, f"{name}_reliability.png"), (reliability * 255).astype(np.uint8))

        print(f"Processed {base} and saved results to {output_folder}")


if __name__ == '__main__':
    # 输入和输出文件夹路径
    input_folder = '/home/tom/fingerprint-deepleaning/End-to-End-Automated-Latent-Fingerprint-Identification/Enhancement/temp'
    output_folder = '/home/tom/fingerprint-deepleaning/fingerprint_deeplearning_processed3'

    # 处理图像
    process_fingerprint_images(input_folder, output_folder)
