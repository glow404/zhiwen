"""预处理子包。

这个子包负责指纹图像进入特征提取前的全部准备工作，包括：
- 灰度增强
- ROI 提取
- 背景抑制
- debug 中间图保存
"""

from src.preprocess.pipeline import preprocess_fingerprint_image, preprocess_fingerprint_path

__all__ = ["preprocess_fingerprint_image", "preprocess_fingerprint_path"]
