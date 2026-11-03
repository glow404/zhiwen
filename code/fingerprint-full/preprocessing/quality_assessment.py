"""
指纹质量评估模块
实现NFIQv2质量评估算法
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from utils.orientation_field import compute_orientation_field
from utils.gabor_filter import GaborFilterBank


class QualityAssessment:
    """
    指纹质量评估类
    基于NFIQv2标准计算质量分数
    """
    
    def __init__(
        self,
        nfiq2_enabled: bool = True,
        gabor_frequencies: list = [0.1, 0.2, 0.3],
        quality_threshold: float = 30.0
    ):
        """
        初始化质量评估器
        
        Args:
            nfiq2_enabled: 是否启用NFIQ2评估
            gabor_frequencies: Gabor滤波器频率列表
            quality_threshold: 质量阈值（用于注册卡控）
        """
        self.nfiq2_enabled = nfiq2_enabled
        self.gabor_frequencies = gabor_frequencies
        self.quality_threshold = quality_threshold
        self.gabor_bank = GaborFilterBank(frequencies=gabor_frequencies)
    
    def compute_orientation_consistency(
        self,
        image: np.ndarray,
        orientation_field: Optional[np.ndarray] = None
    ) -> float:
        """
        计算方向场一致性
        
        Args:
            image: 输入图像
            orientation_field: 方向场，如果None则计算
            
        Returns:
            float: 方向场一致性分数（0-100）
        """
        if orientation_field is None:
            _, coherence = compute_orientation_field(image)
        else:
            _, coherence = compute_orientation_field(image)
        
        # 计算平均一致性
        mean_coherence = np.mean(coherence)
        
        # 转换为0-100分数
        score = mean_coherence * 100.0
        
        return score
    
    def compute_gabor_response(self, image: np.ndarray) -> float:
        """
        计算Gabor响应
        
        Args:
            image: 输入图像
            
        Returns:
            float: Gabor响应分数（0-100）
        """
        # 应用Gabor滤波器组
        responses = self.gabor_bank.get_filter_responses(image)
        
        # 计算最大响应
        max_responses = [np.max(resp[0]) for resp in responses]
        mean_max_response = np.mean(max_responses)
        
        # 归一化到0-100
        score = np.clip(mean_max_response * 100.0, 0, 100)
        
        return score
    
    def compute_local_contrast(self, image: np.ndarray) -> float:
        """
        计算局部对比度
        
        Args:
            image: 输入图像
            
        Returns:
            float: 局部对比度分数（0-100）
        """
        # 转换为uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
        else:
            img_uint8 = image
        
        # 计算局部对比度（使用标准差）
        kernel = np.ones((15, 15), np.float32) / 225
        local_mean = cv2.filter2D(img_uint8.astype(np.float32), -1, kernel)
        local_std = cv2.filter2D((img_uint8.astype(np.float32) - local_mean) ** 2, -1, kernel)
        local_std = np.sqrt(local_std)
        
        # 计算平均局部对比度
        mean_contrast = np.mean(local_std)
        
        # 归一化到0-100
        score = np.clip(mean_contrast / 50.0 * 100.0, 0, 100)
        
        return score
    
    def compute_noise_level(self, image: np.ndarray) -> float:
        """
        计算噪声水平
        
        Args:
            image: 输入图像
            
        Returns:
            float: 噪声水平分数（0-100，越低越好）
        """
        # 使用拉普拉斯算子的方差估计噪声
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        noise_std = np.std(laplacian)
        
        # 归一化（噪声越低，分数越高）
        # 假设噪声标准差在0-50范围内
        score = np.clip(100.0 - (noise_std / 50.0 * 100.0), 0, 100)
        
        return score
    
    def compute_variance(self, image: np.ndarray) -> float:
        """
        计算图像方差
        
        Args:
            image: 输入图像
            
        Returns:
            float: 方差分数（0-100）
        """
        # 计算全局方差
        variance = np.var(image)
        
        # 归一化到0-100
        # 假设方差在0-0.1范围内（对于归一化图像）
        if image.max() <= 1.0:
            score = np.clip(variance / 0.1 * 100.0, 0, 100)
        else:
            score = np.clip(variance / 2500.0 * 100.0, 0, 100)
        
        return score
    
    def compute_nfiq2_score(self, image: np.ndarray) -> Tuple[float, dict]:
        """
        计算NFIQ2质量分数
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[float, dict]: (质量分数0-100, 各子分数字典)
        """
        # 确保是灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8),
                cv2.COLOR_BGR2GRAY
            )
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
        
        # 计算各子指标
        orientation_score = self.compute_orientation_consistency(image)
        gabor_score = self.compute_gabor_response(image)
        contrast_score = self.compute_local_contrast(image)
        noise_score = self.compute_noise_level(image)
        variance_score = self.compute_variance(image)
        
        # 加权平均得到最终分数
        # 权重可以根据实际需求调整
        weights = {
            'orientation': 0.25,
            'gabor': 0.25,
            'contrast': 0.20,
            'noise': 0.15,
            'variance': 0.15
        }
        
        total_score = (
            weights['orientation'] * orientation_score +
            weights['gabor'] * gabor_score +
            weights['contrast'] * contrast_score +
            weights['noise'] * noise_score +
            weights['variance'] * variance_score
        )
        
        sub_scores = {
            'orientation': orientation_score,
            'gabor': gabor_score,
            'contrast': contrast_score,
            'noise': noise_score,
            'variance': variance_score
        }
        
        return total_score, sub_scores
    
    def assess(self, image: np.ndarray) -> Tuple[float, dict, bool]:
        """
        评估图像质量
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[float, dict, bool]: (质量分数, 子分数字典, 是否通过阈值)
        """
        if self.nfiq2_enabled:
            score, sub_scores = self.compute_nfiq2_score(image)
        else:
            # 简化评估
            score = self.compute_local_contrast(image)
            sub_scores = {'contrast': score}
        
        # 判断是否通过阈值
        passed = score >= self.quality_threshold
        
        return score, sub_scores, passed
    
    def is_registerable(self, image: np.ndarray) -> bool:
        """
        判断图像是否可用于注册
        
        Args:
            image: 输入图像
            
        Returns:
            bool: 是否可用于注册
        """
        _, _, passed = self.assess(image)
        return passed

