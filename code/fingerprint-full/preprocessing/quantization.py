"""
指纹量化模块
全局量化和局部量化，提升局部对比度
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class FingerprintQuantization:
    """
    指纹量化类
    支持全局量化和局部量化
    """
    
    def __init__(
        self,
        global_quantiles: list = [0.1, 0.9],
        local_window_size: int = 15,
        local_template: str = "adaptive"
    ):
        """
        初始化量化器
        
        Args:
            global_quantiles: 全局分位点列表
            local_window_size: 局部量化窗口大小
            local_template: 局部量化模板类型（"adaptive"或"fixed"）
        """
        self.global_quantiles = global_quantiles
        self.local_window_size = local_window_size
        self.local_template = local_template
    
    def global_quantization(
        self,
        image: np.ndarray,
        quantiles: Optional[list] = None
    ) -> np.ndarray:
        """
        全局量化（基于直方图分位点）
        
        Args:
            image: 输入图像
            quantiles: 分位点列表，如果None则使用self.global_quantiles
            
        Returns:
            np.ndarray: 量化后的图像
        """
        if quantiles is None:
            quantiles = self.global_quantiles
        
        # 确保是1D数组用于计算分位点
        flat_image = image.flatten()
        
        # 计算分位点值
        min_val = np.quantile(flat_image, quantiles[0])
        max_val = np.quantile(flat_image, quantiles[1])
        
        # 线性拉伸到0-1范围
        quantized = np.clip((image - min_val) / (max_val - min_val + 1e-10), 0.0, 1.0)
        
        return quantized
    
    def local_quantization(
        self,
        image: np.ndarray,
        window_size: Optional[int] = None,
        template: Optional[str] = None
    ) -> np.ndarray:
        """
        局部量化（基于局部最大最小值）
        
        Args:
            image: 输入图像
            window_size: 窗口大小，如果None则使用self.local_window_size
            template: 模板类型，如果None则使用self.local_template
            
        Returns:
            np.ndarray: 量化后的图像
        """
        if window_size is None:
            window_size = self.local_window_size
        if template is None:
            template = self.local_template
        
        h, w = image.shape
        
        # 创建输出图像
        quantized = np.zeros_like(image)
        
        # 计算局部最小值和最大值
        if template == "adaptive":
            # 自适应窗口
            kernel = np.ones((window_size, window_size), np.float32)
            local_min = cv2.erode(image, kernel, iterations=1)
            local_max = cv2.dilate(image, kernel, iterations=1)
        else:
            # 固定窗口
            kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
            local_mean = cv2.filter2D(image, -1, kernel)
            local_std = cv2.filter2D((image - local_mean) ** 2, -1, kernel)
            local_std = np.sqrt(local_std)
            local_min = local_mean - 2 * local_std
            local_max = local_mean + 2 * local_std
        
        # 局部线性拉伸
        local_range = local_max - local_min
        mask = local_range > 1e-10
        
        quantized[mask] = (image[mask] - local_min[mask]) / local_range[mask]
        quantized[~mask] = image[~mask]  # 保持原值如果范围太小
        
        # 确保在0-1范围
        quantized = np.clip(quantized, 0.0, 1.0)
        
        return quantized
    
    def quantize(
        self,
        image: np.ndarray,
        method: str = "both",
        global_quantiles: Optional[list] = None,
        local_window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        量化主函数
        
        Args:
            image: 输入图像
            method: 量化方法（"global", "local", "both"）
            global_quantiles: 全局分位点
            local_window_size: 局部窗口大小
            
        Returns:
            np.ndarray: 量化后的图像
        """
        if method == "global":
            return self.global_quantization(image, global_quantiles)
        elif method == "local":
            return self.local_quantization(image, local_window_size)
        elif method == "both":
            # 先全局量化，再局部量化
            global_quantized = self.global_quantization(image, global_quantiles)
            local_quantized = self.local_quantization(global_quantized, local_window_size)
            return local_quantized
        else:
            raise ValueError(f"Unknown quantization method: {method}")
    
    def adaptive_quantization(
        self,
        image: np.ndarray,
        quality_score: float
    ) -> np.ndarray:
        """
        自适应量化（根据质量分数调整参数）
        
        Args:
            image: 输入图像
            quality_score: 质量分数（0-100）
            
        Returns:
            np.ndarray: 量化后的图像
        """
        # 质量越低，使用更强的量化
        if quality_score < 30:
            # 低质量：使用较大的分位点范围和较小的局部窗口
            quantiles = [0.05, 0.95]
            window_size = 20
        elif quality_score < 60:
            # 中等质量：使用标准参数
            quantiles = [0.1, 0.9]
            window_size = 15
        else:
            # 高质量：使用较小的分位点范围和较大的局部窗口
            quantiles = [0.15, 0.85]
            window_size = 10
        
        return self.quantize(image, method="both", global_quantiles=quantiles, local_window_size=window_size)

