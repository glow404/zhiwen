import cv2
import numpy as np
import matplotlib.pyplot as plt
from testfin_NIST import testfin_NIST  # 假设你的代码保存在 testfin_NIST.py 中

# 1. 加载指纹图像
image_path = "/home/tom/fingerprint-deepleaning/End-to-End-Automated-Latent-Fingerprint-Identification/Enhancement/temp/101_1.bmp"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. 运行处理流程
newim, binim, mask, reliability = testfin_NIST(img, "output.png")

# 3. 可视化结果
plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('原始指纹')

plt.subplot(232)
plt.imshow(newim, cmap='gray')
plt.title('增强图像')

plt.subplot(233)
plt.imshow(binim, cmap='gray')
plt.title('二值图像')

plt.subplot(234)
plt.imshow(mask, cmap='gray')
plt.title('指纹区域掩码')

plt.subplot(235)
plt.imshow(reliability, cmap='hot')
plt.title('方向可靠性')

plt.subplot(236)
output_img = cv2.imread("output.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(output_img, cmap='gray')
plt.title('可靠区域')

plt.tight_layout()
plt.savefig("all_results.png", dpi=300)
plt.show()