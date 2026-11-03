"""
指纹增强模块
Gabor滤波增强指纹纹理，保护细节点
"""

import numpy as np
import cv2
from typing import Optional
from utils.gabor_filter import GaborFilterBank
from utils.orientation_field import compute_orientation_field


class FingerprintEnhancement:
    """
    指纹增强类
    使用Gabor滤波增强指纹纹理
    """
    
    def __init__(
        self,
        gabor_frequencies: list = [0.1, 0.2, 0.3],
        gabor_orientations: list = [0, 45, 90, 135],
        local_confidence_weight: float = 0.8,
        preserve_keypoints: bool = True
    ):
        """
        初始化增强器
        
        Args:
            gabor_frequencies: Gabor频率列表
            gabor_orientations: Gabor方向列表（度）
            local_confidence_weight: 局部可信性权重
            preserve_keypoints: 是否保护细节点
        """
        self.gabor_frequencies = gabor_frequencies
        self.gabor_orientations = gabor_orientations
        self.local_confidence_weight = local_confidence_weight
        self.preserve_keypoints = preserve_keypoints
        self.gabor_bank = GaborFilterBank(
            frequencies=gabor_frequencies,
            orientations=gabor_orientations
        )
    
    def compute_local_confidence(
        self,
        image: np.ndarray,
        orientation_field: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算局部可信性
        
        Args:
            image: 输入图像
            orientation_field: 方向场，如果None则计算
            
        Returns:
            np.ndarray: 局部可信性图（0-1）
        """
        if orientation_field is None:
            _, coherence = compute_orientation_field(image)
        else:
            _, coherence = compute_orientation_field(image)
        
        # 方向场一致性作为可信性
        confidence = coherence
        
        # 还可以考虑局部对比度
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
        else:
            img_uint8 = image
        
        # 计算局部标准差
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(img_uint8.astype(np.float32), -1, kernel)
        local_std = cv2.filter2D((img_uint8.astype(np.float32) - local_mean) ** 2, -1, kernel)
        local_std = np.sqrt(local_std)
        local_std_norm = local_std / (local_std.max() + 1e-10)
        
        # 结合方向场一致性和局部对比度
        confidence = 0.7 * confidence + 0.3 * local_std_norm
        
        return confidence
    
    def detect_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        检测细节点位置（用于保护）
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 细节点掩码（二值图）
        """
        # 转换为uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
        else:
            img_uint8 = image
        
        # 使用Harris角点检测
        corners = cv2.cornerHarris(img_uint8, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # 阈值化
        threshold = 0.01 * corners.max()
        keypoint_mask = (corners > threshold).astype(np.float32)
        
        # 膨胀以保护细节点周围区域
        kernel = np.ones((3, 3), np.uint8)
        keypoint_mask = cv2.dilate(keypoint_mask, kernel, iterations=1)
        
        return keypoint_mask
    
    def enhance(
        self,
        image: np.ndarray,
        orientation_field: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        增强指纹纹理
        
        Args:
            image: 输入图像
            orientation_field: 方向场，如果None则计算
            
        Returns:
            np.ndarray: 增强后的图像
        """
        # 确保是灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8),
                cv2.COLOR_BGR2GRAY
            )
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
        
        # 计算方向场
        if orientation_field is None:
            orientation_field, _ = compute_orientation_field(image)
        
        # 使用方向场应用Gabor滤波
        enhanced = self.gabor_bank.apply_with_orientation(image, orientation_field)
        
        # 归一化
        if enhanced.max() > 0:
            enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-10)
        
        # 计算局部可信性
        confidence = self.compute_local_confidence(image, orientation_field)
        
        # 根据可信性加权混合
        # 低可信性区域使用更多原图，高可信性区域使用更多增强图
        weight = confidence * self.local_confidence_weight
        enhanced = enhanced * weight + image * (1 - weight)
        
        # 保护细节点
        if self.preserve_keypoints:
            keypoint_mask = self.detect_keypoints(image)
            # 细节点区域保留更多原图
            enhanced = enhanced * (1 - keypoint_mask * 0.3) + image * (keypoint_mask * 0.3)
        
        # 确保输出范围
        enhanced = np.clip(enhanced, 0.0, 1.0)
        
        return enhanced
    
    def enhance_ridge_continuity(
        self,
        image: np.ndarray,
        orientation_field: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        优化脊线/谷线的连续性
        
        Args:
            image: 输入图像
            orientation_field: 方向场
            
        Returns:
            np.ndarray: 增强后的图像
        """
        # 基本增强
        enhanced = self.enhance(image, orientation_field)
        
        # 使用形态学操作增强连续性
        if enhanced.dtype != np.uint8:
            enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        else:
            enhanced_uint8 = enhanced
        
        # 使用细长的结构元素沿脊线方向
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 闭运算：连接断开的脊线
        enhanced_uint8 = cv2.morphologyEx(enhanced_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 转换回float32
        enhanced = enhanced_uint8.astype(np.float32) / 255.0
        
        return enhanced

