"""
指纹降噪模块
使用AI降噪（Unet）抑制相干斑噪声与传感器噪声
"""

import numpy as np
import cv2
from typing import Optional, Tuple
import tensorflow as tf
from models.unet_denoising import UNetDenoising


class FingerprintDenoising:
    """
    指纹降噪类
    支持AI降噪和噪声估计
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        noise_estimation: bool = True,
        adaptive_strength: bool = True,
        preserve_edges: bool = True
    ):
        """
        初始化降噪器
        
        Args:
            model_path: Unet模型路径
            noise_estimation: 是否启用噪声估计
            adaptive_strength: 是否自适应降噪力度
            preserve_edges: 是否保护边缘
        """
        self.model_path = model_path
        self.noise_estimation = noise_estimation
        self.adaptive_strength = adaptive_strength
        self.preserve_edges = preserve_edges
        self.model = None
        
        # 加载模型
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """加载Unet模型"""
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Warning: Failed to load model from {model_path}: {e}")
            print("Using default model initialization")
            self.model = UNetDenoising().build_model()
    
    def estimate_noise(self, image: np.ndarray) -> float:
        """
        估计图像噪声水平
        
        Args:
            image: 输入图像
            
        Returns:
            float: 噪声水平（0-1）
        """
        # 使用拉普拉斯算子的方差估计噪声
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        noise_level = np.std(laplacian)
        
        # 归一化到0-1范围
        noise_level = np.clip(noise_level / 100.0, 0.0, 1.0)
        
        return noise_level
    
    def compute_edge_map(self, image: np.ndarray) -> np.ndarray:
        """
        计算边缘图，用于保护边缘
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 边缘图（0-1）
        """
        # 使用Canny边缘检测
        if image.dtype != np.uint8:
            img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        edges = cv2.Canny(img_uint8, 50, 150)
        edge_map = (edges > 0).astype(np.float32)
        
        # 膨胀边缘区域以保护更多边缘像素
        kernel = np.ones((3, 3), np.uint8)
        edge_map = cv2.dilate(edge_map, kernel, iterations=1)
        
        return edge_map
    
    def denoise_ai(
        self,
        image: np.ndarray,
        model: Optional[tf.keras.Model] = None
    ) -> np.ndarray:
        """
        使用AI模型降噪
        
        Args:
            image: 输入图像
            model: 降噪模型，如果None则使用self.model
            
        Returns:
            np.ndarray: 降噪后的图像
        """
        if model is None:
            model = self.model
        
        if model is None:
            # 如果没有模型，使用传统方法
            return self.denoise_traditional(image)
        
        # 预处理图像
        if len(image.shape) == 2:
            img_input = np.expand_dims(image, axis=-1)
        else:
            img_input = image
        
        # 归一化到0-1
        if img_input.max() > 1.0:
            img_input = img_input / 255.0
        
        # 添加batch维度
        img_batch = np.expand_dims(img_input, axis=0)
        
        # 预测
        denoised = model.predict(img_batch, verbose=0)
        
        # 移除batch和channel维度
        if len(denoised.shape) == 4:
            denoised = denoised[0, :, :, 0]
        elif len(denoised.shape) == 3:
            denoised = denoised[0]
        
        return denoised
    
    def denoise_traditional(self, image: np.ndarray) -> np.ndarray:
        """
        传统降噪方法（作为AI模型的备用）
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 降噪后的图像
        """
        # 转换为uint8
        if image.dtype != np.uint8:
            img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        # 使用非局部均值去噪
        denoised = cv2.fastNlMeansDenoising(img_uint8, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 转换回float32
        denoised = denoised.astype(np.float32) / 255.0
        
        return denoised
    
    def denoise(
        self,
        image: np.ndarray,
        strength: Optional[float] = None
    ) -> np.ndarray:
        """
        降噪主函数
        
        Args:
            image: 输入图像
            strength: 降噪强度（0-1），如果None则自适应
            
        Returns:
            np.ndarray: 降噪后的图像
        """
        # 噪声估计
        noise_level = 0.5  # 默认值
        if self.noise_estimation:
            noise_level = self.estimate_noise(image)
        
        # 自适应降噪强度
        if strength is None and self.adaptive_strength:
            # 噪声水平越高，降噪强度越大
            strength = noise_level
        elif strength is None:
            strength = 0.5
        
        strength = np.clip(strength, 0.0, 1.0)
        
        # AI降噪
        if self.model is not None:
            denoised = self.denoise_ai(image)
        else:
            denoised = self.denoise_traditional(image)
        
        # 保护边缘
        if self.preserve_edges:
            edge_map = self.compute_edge_map(image)
            # 混合原图和降噪图，边缘区域保留更多原图
            denoised = denoised * (1 - edge_map * 0.3) + image * (edge_map * 0.3)
        
        # 根据强度混合原图和降噪图
        if strength < 1.0:
            denoised = denoised * strength + image * (1 - strength)
        
        # 确保输出范围
        denoised = np.clip(denoised, 0.0, 1.0)
        
        return denoised
    
    def enhance_contrast(
        self,
        image: np.ndarray,
        condition: str = "normal"
    ) -> np.ndarray:
        """
        自适应增强不同采集条件下的对比度
        
        Args:
            image: 输入图像
            condition: 采集条件（"normal", "dry", "wet"）
            
        Returns:
            np.ndarray: 增强后的图像
        """
        # 直方图均衡化
        if image.dtype != np.uint8:
            img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        # 根据条件调整
        if condition == "wet":
            # 湿手：增强对比度
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_uint8)
        elif condition == "dry":
            # 干手：适度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_uint8)
        else:
            # 正常情况
            enhanced = cv2.equalizeHist(img_uint8)
        
        # 转换回float32
        enhanced = enhanced.astype(np.float32) / 255.0
        
        return enhanced

