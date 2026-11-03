"""
指纹二值化模块
OTSU二值化算法
"""

import numpy as np
import cv2
from typing import Optional


class FingerprintBinarization:
    """
    指纹二值化类
    使用OTSU算法提取指纹纹路的二值化0-1图
    """
    
    def __init__(self, method: str = "otsu", adaptive: bool = False):
        """
        初始化二值化器
        
        Args:
            method: 二值化方法（"otsu"）
            adaptive: 是否使用自适应阈值
        """
        self.method = method
        self.adaptive = adaptive
    
    def otsu_binarization(self, image: np.ndarray) -> np.ndarray:
        """
        OTSU二值化
        
        Args:
            image: 输入图像（灰度图）
            
        Returns:
            np.ndarray: 二值图（0或1）
        """
        # 确保是uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        # OTSU阈值
        if self.adaptive:
            # 自适应OTSU（局部阈值）
            binary = cv2.adaptiveThreshold(
                img_uint8,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
        else:
            # 全局OTSU
            _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 转换为0-1
        binary = (binary > 127).astype(np.float32)
        
        return binary
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """
        二值化主函数
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 二值图（0或1）
        """
        # 确保是灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8),
                cv2.COLOR_BGR2GRAY
            )
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
        
        if self.method == "otsu":
            return self.otsu_binarization(image)
        else:
            raise ValueError(f"Unknown binarization method: {self.method}")
    
    def post_process(self, binary: np.ndarray) -> np.ndarray:
        """
        后处理：去除小噪点，填充小孔洞
        
        Args:
            binary: 二值图
            
        Returns:
            np.ndarray: 处理后的二值图
        """
        # 转换为uint8
        binary_uint8 = (binary * 255).astype(np.uint8)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        
        # 开运算：去除小噪点
        binary_uint8 = cv2.morphologyEx(binary_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 闭运算：填充小孔洞
        binary_uint8 = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 转换回0-1
        binary = (binary_uint8 > 127).astype(np.float32)
        
        return binary

