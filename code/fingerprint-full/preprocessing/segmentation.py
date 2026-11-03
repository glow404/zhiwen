"""
指纹分割模块
确定指纹按压区域，准确率>=95%
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from utils.orientation_field import compute_orientation_field


class FingerprintSegmentation:
    """
    指纹分割类
    基于方向场和局部梯度确定指纹按压区域
    """
    
    def __init__(
        self,
        orientation_block_size: int = 16,
        gradient_threshold: float = 0.1,
        morphological_kernel_size: int = 5,
        min_area_ratio: float = 0.1
    ):
        """
        初始化分割器
        
        Args:
            orientation_block_size: 方向场计算块大小
            gradient_threshold: 梯度阈值
            morphological_kernel_size: 形态学操作核大小
            min_area_ratio: 最小区域面积比例
        """
        self.orientation_block_size = orientation_block_size
        self.gradient_threshold = gradient_threshold
        self.morphological_kernel_size = morphological_kernel_size
        self.min_area_ratio = min_area_ratio
    
    def compute_gradient_mask(self, image: np.ndarray) -> np.ndarray:
        """
        基于局部梯度计算掩码
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 梯度掩码（二值图）
        """
        # 计算梯度
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        
        # 归一化
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        # 阈值化
        mask = (gradient_magnitude > self.gradient_threshold).astype(np.uint8)
        
        return mask
    
    def compute_orientation_mask(
        self,
        image: np.ndarray,
        orientation_field: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        基于方向场计算掩码
        
        Args:
            image: 输入图像
            orientation_field: 方向场，如果None则计算
            
        Returns:
            np.ndarray: 方向场掩码（二值图）
        """
        if orientation_field is None:
            orientation_field, coherence = compute_orientation_field(
                image, block_size=self.orientation_block_size
            )
        else:
            _, coherence = compute_orientation_field(
                image, block_size=self.orientation_block_size
            )
        
        # 使用方向场一致性作为掩码
        # 指纹区域应该有较高的方向场一致性
        coherence_threshold = 0.3
        mask = (coherence > coherence_threshold).astype(np.uint8)
        
        return mask
    
    def morphological_optimization(self, mask: np.ndarray) -> np.ndarray:
        """
        形态学优化掩码
        
        Args:
            mask: 输入掩码
            
        Returns:
            np.ndarray: 优化后的掩码
        """
        kernel = np.ones(
            (self.morphological_kernel_size, self.morphological_kernel_size),
            np.uint8
        )
        
        # 闭运算：填充小孔洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 开运算：去除小噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return mask
    
    def filter_by_area(self, mask: np.ndarray) -> np.ndarray:
        """
        根据面积过滤区域
        
        Args:
            mask: 输入掩码
            
        Returns:
            np.ndarray: 过滤后的掩码
        """
        h, w = mask.shape
        total_area = h * w
        min_area = total_area * self.min_area_ratio
        
        # 查找连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # 创建过滤后的掩码
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # 跳过背景（标签0）
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == i] = 255
        
        return filtered_mask
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割指纹按压区域
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (分割掩码, 方向场)
        """
        # 确保是灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8),
                cv2.COLOR_BGR2GRAY
            )
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
        elif image.dtype != np.float32:
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
        
        # 计算方向场
        orientation_field, coherence = compute_orientation_field(
            image, block_size=self.orientation_block_size
        )
        
        # 基于梯度的掩码
        gradient_mask = self.compute_gradient_mask(image)
        
        # 基于方向场的掩码
        orientation_mask = self.compute_orientation_mask(image, orientation_field)
        
        # 合并掩码
        combined_mask = cv2.bitwise_and(gradient_mask, orientation_mask)
        
        # 形态学优化
        optimized_mask = self.morphological_optimization(combined_mask)
        
        # 面积过滤
        final_mask = self.filter_by_area(optimized_mask)
        
        # 转换为0-1范围
        final_mask = (final_mask > 0).astype(np.float32)
        
        return final_mask, orientation_field
    
    def apply_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        应用掩码到图像
        
        Args:
            image: 输入图像
            mask: 分割掩码
            
        Returns:
            np.ndarray: 掩码后的图像
        """
        if len(image.shape) == 2:
            masked = image * mask
        else:
            masked = image * mask[:, :, np.newaxis]
        
        return masked

