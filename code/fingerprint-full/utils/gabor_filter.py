"""
Gabor滤波器模块
用于指纹纹理增强
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class GaborFilterBank:
    """
    Gabor滤波器组
    用于指纹纹理增强和特征提取
    """
    
    def __init__(
        self,
        frequencies: List[float] = [0.1, 0.2, 0.3],
        orientations: List[float] = [0, 45, 90, 135],
        kernel_size: int = 21,
        sigma: float = 5.0,
        gamma: float = 0.5
    ):
        """
        初始化Gabor滤波器组
        
        Args:
            frequencies: 频率列表（归一化频率，相对于图像尺寸）
            orientations: 方向列表（度）
            kernel_size: 滤波器核大小
            sigma: 高斯包络的标准差
            gamma: 空间纵横比
        """
        self.frequencies = frequencies
        self.orientations = [np.deg2rad(o) for o in orientations]
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gamma = gamma
        self.filters = self._create_filters()
    
    def _create_filters(self) -> List[np.ndarray]:
        """创建Gabor滤波器组"""
        filters = []
        for freq in self.frequencies:
            for theta in self.orientations:
                kernel = cv2.getGaborKernel(
                    (self.kernel_size, self.kernel_size),
                    self.sigma,
                    theta,
                    2 * np.pi * freq,
                    self.gamma,
                    0,
                    ktype=cv2.CV_32F
                )
                filters.append((kernel, freq, theta))
        return filters
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        应用Gabor滤波器组
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 滤波后的图像（取最大响应）
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 确保是float32
        if image.dtype != np.float32:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
        
        responses = []
        for kernel, freq, theta in self.filters:
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(filtered)
        
        # 取最大响应
        response = np.maximum.reduce(responses)
        
        return response
    
    def apply_with_orientation(
        self,
        image: np.ndarray,
        orientation_field: np.ndarray
    ) -> np.ndarray:
        """
        根据方向场应用Gabor滤波器
        
        Args:
            image: 输入图像
            orientation_field: 方向场（弧度）
            
        Returns:
            np.ndarray: 滤波后的图像
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if image.dtype != np.float32:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
        
        h, w = image.shape
        output = np.zeros_like(image)
        
        # 为每个像素选择最匹配方向的滤波器
        block_size = 8  # 块大小
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # 获取块的方向
                block_orient = orientation_field[i:min(i+block_size, h), 
                                                j:min(j+block_size, w)]
                avg_orient = np.mean(block_orient)
                
                # 选择最接近的方向
                best_theta_idx = np.argmin([abs(avg_orient - theta) for _, _, theta in self.filters])
                kernel, freq, _ = self.filters[best_theta_idx]
                
                # 应用滤波器
                block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
                filtered_block = cv2.filter2D(block, cv2.CV_32F, kernel)
                output[i:min(i+block_size, h), j:min(j+block_size, w)] = filtered_block
        
        return output
    
    def get_filter_responses(self, image: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """
        获取所有滤波器的响应
        
        Args:
            image: 输入图像
            
        Returns:
            List[Tuple]: [(响应图像, 频率, 方向), ...]
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if image.dtype != np.float32:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
        
        responses = []
        for kernel, freq, theta in self.filters:
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append((filtered, freq, theta))
        
        return responses


def create_gabor_filter(
    frequency: float,
    orientation: float,
    kernel_size: int = 21,
    sigma: float = 5.0,
    gamma: float = 0.5
) -> np.ndarray:
    """
    创建单个Gabor滤波器
    
    Args:
        frequency: 频率（归一化）
        orientation: 方向（弧度）
        kernel_size: 核大小
        sigma: 高斯标准差
        gamma: 空间纵横比
        
    Returns:
        np.ndarray: Gabor滤波器核
    """
    return cv2.getGaborKernel(
        (kernel_size, kernel_size),
        sigma,
        orientation,
        2 * np.pi * frequency,
        gamma,
        0,
        ktype=cv2.CV_32F
    )

