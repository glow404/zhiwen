"""
指纹相似度计算模块
基于变换矩阵的二值图相似度
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class FingerprintSimilarity:
    """
    指纹相似度计算类
    基于变换矩阵对二值图进行Warp，计算重叠区域相似度
    """
    
    def __init__(self, overlap_threshold: float = 0.5):
        """
        初始化相似度计算器
        
        Args:
            overlap_threshold: 最小重叠区域比例
        """
        self.overlap_threshold = overlap_threshold
    
    def warp_binary_image(
        self,
        binary_image: np.ndarray,
        transform_matrix: np.ndarray,
        output_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        对二值图进行Warp变换
        
        Args:
            binary_image: 输入二值图
            transform_matrix: 变换矩阵（3x3单应性矩阵）
            output_shape: 输出形状，如果None则使用输入形状
            
        Returns:
            np.ndarray: Warp后的二值图
        """
        if output_shape is None:
            output_shape = binary_image.shape
        
        # 转换为uint8（OpenCV要求）
        if binary_image.dtype != np.uint8:
            binary_uint8 = (binary_image * 255).astype(np.uint8)
        else:
            binary_uint8 = binary_image
        
        # 执行透视变换
        warped = cv2.warpPerspective(
            binary_uint8,
            transform_matrix,
            (output_shape[1], output_shape[0]),
            flags=cv2.INTER_NEAREST,  # 最近邻插值保持二值特性
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 转换回0-1
        warped = (warped > 127).astype(np.float32)
        
        return warped
    
    def compute_overlap_region(
        self,
        binary1: np.ndarray,
        binary2: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        计算重叠区域
        
        Args:
            binary1: 第一幅二值图
            binary2: 第二幅二值图
            
        Returns:
            Tuple[np.ndarray, float]: (重叠区域掩码, 重叠比例)
        """
        # 确保形状一致
        if binary1.shape != binary2.shape:
            h, w = min(binary1.shape[0], binary2.shape[0]), min(binary1.shape[1], binary2.shape[1])
            binary1 = binary1[:h, :w]
            binary2 = binary2[:h, :w]
        
        # 计算重叠区域（两个图像都有有效像素的区域）
        valid_region1 = binary1 > 0.5
        valid_region2 = binary2 > 0.5
        overlap_mask = valid_region1 & valid_region2
        
        # 计算重叠比例
        total_valid = (valid_region1 | valid_region2).sum()
        overlap_ratio = overlap_mask.sum() / (total_valid + 1e-10)
        
        return overlap_mask, overlap_ratio
    
    def compute_similarity(
        self,
        binary1: np.ndarray,
        binary2: np.ndarray,
        transform_matrix: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        计算相似度
        
        Args:
            binary1: 第一幅二值图（模板）
            binary2: 第二幅二值图（样本）
            transform_matrix: 变换矩阵，如果None则直接比较
            
        Returns:
            Tuple[float, float]: (相似度分数0-1, 重叠比例)
        """
        # 如果提供了变换矩阵，对binary1进行Warp
        if transform_matrix is not None:
            binary1_warped = self.warp_binary_image(binary1, transform_matrix, binary2.shape)
        else:
            binary1_warped = binary1.copy()
            # 调整大小以匹配binary2
            if binary1_warped.shape != binary2.shape:
                binary1_warped = cv2.resize(
                    binary1_warped,
                    (binary2.shape[1], binary2.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
        
        # 计算重叠区域
        overlap_mask, overlap_ratio = self.compute_overlap_region(binary1_warped, binary2)
        
        # 检查重叠是否足够
        if overlap_ratio < self.overlap_threshold:
            return 0.0, overlap_ratio
        
        # 在重叠区域计算相似度
        # 相似度 = (同0同1的像素个数) / (重叠区域总像素数)
        overlap_binary1 = binary1_warped[overlap_mask]
        overlap_binary2 = binary2[overlap_mask]
        
        # 同0同1的像素
        same_pixels = (overlap_binary1 == overlap_binary2).sum()
        total_overlap_pixels = len(overlap_binary1)
        
        if total_overlap_pixels == 0:
            return 0.0, overlap_ratio
        
        similarity = same_pixels / total_overlap_pixels
        
        return similarity, overlap_ratio
    
    def compute_hamming_distance(
        self,
        binary1: np.ndarray,
        binary2: np.ndarray,
        transform_matrix: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        计算汉明距离（用于二值描述子）
        
        Args:
            binary1: 第一幅二值图或描述子
            binary2: 第二幅二值图或描述子
            transform_matrix: 变换矩阵（仅用于图像）
            
        Returns:
            Tuple[float, float]: (汉明距离, 重叠比例)
        """
        if len(binary1.shape) == 1 and len(binary2.shape) == 1:
            # 描述子
            if len(binary1) != len(binary2):
                return 1.0, 0.0
            hamming_dist = np.sum(binary1 != binary2) / len(binary1)
            return hamming_dist, 1.0
        else:
            # 图像
            similarity, overlap_ratio = self.compute_similarity(binary1, binary2, transform_matrix)
            hamming_dist = 1.0 - similarity
            return hamming_dist, overlap_ratio

