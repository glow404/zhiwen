"""
特征点提取与描述模块
SIFT特征点 + AI二值化描述子
"""

import numpy as np
import cv2
import tensorflow as tf
from typing import List, Tuple, Optional
from models.feature_descriptor import FeatureDescriptor


class FeaturePointExtractor:
    """
    特征点提取与描述类
    提取SIFT特征点并生成AI描述子
    """
    
    def __init__(
        self,
        n_features: int = 0,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        descriptor_model_path: Optional[str] = None,
        descriptor_bits: int = 512,
        min_features: int = 50
    ):
        """
        初始化特征提取器
        
        Args:
            n_features: SIFT特征点数量（0表示提取所有）
            n_octave_layers: SIFT八度层数
            contrast_threshold: SIFT对比度阈值
            descriptor_model_path: 描述子模型路径
            descriptor_bits: 描述子位数
            min_features: 最小特征点数
        """
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.descriptor_model_path = descriptor_model_path
        self.descriptor_bits = descriptor_bits
        self.min_features = min_features
        
        # 初始化SIFT
        self.sift = cv2.SIFT_create(
            nfeatures=n_features if n_features > 0 else 0,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold
        )
        
        # 加载描述子模型
        self.descriptor_model = None
        if descriptor_model_path:
            self.load_descriptor_model(descriptor_model_path)
    
    def load_descriptor_model(self, model_path: str):
        """加载描述子模型"""
        try:
            self.descriptor_model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Warning: Failed to load descriptor model from {model_path}: {e}")
            print("Using default model initialization")
            self.descriptor_model = FeatureDescriptor(descriptor_bits=self.descriptor_bits).build_model()
    
    def extract_sift_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        提取SIFT特征点
        
        Args:
            image: 输入图像
            mask: 掩码（可选）
            
        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray]: (关键点列表, SIFT描述子)
        """
        # 确保是uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        # 提取特征点和描述子
        keypoints, descriptors = self.sift.detectAndCompute(img_uint8, mask)
        
        # 如果特征点太少，尝试降低阈值
        if len(keypoints) < self.min_features and self.n_features == 0:
            # 降低对比度阈值
            sift_low_threshold = cv2.SIFT_create(
                nfeatures=0,
                nOctaveLayers=self.n_octave_layers,
                contrastThreshold=self.contrast_threshold * 0.5
            )
            keypoints, descriptors = sift_low_threshold.detectAndCompute(img_uint8, mask)
        
        return keypoints, descriptors
    
    def extract_ai_descriptor(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint],
        patch_size: int = 32
    ) -> np.ndarray:
        """
        使用AI模型提取二值化描述子
        
        Args:
            image: 输入图像
            keypoints: 关键点列表
            patch_size: 提取的patch大小
            
        Returns:
            np.ndarray: 二值化描述子数组 (N, descriptor_bits)
        """
        if self.descriptor_model is None:
            # 如果没有模型，使用SIFT描述子并二值化
            _, sift_descriptors = self.extract_sift_features(image)
            if sift_descriptors is None or len(sift_descriptors) == 0:
                return np.array([])
            
            # 截断到descriptor_bits长度
            if sift_descriptors.shape[1] > self.descriptor_bits:
                sift_descriptors = sift_descriptors[:, :self.descriptor_bits]
            elif sift_descriptors.shape[1] < self.descriptor_bits:
                # 填充
                padding = np.zeros((sift_descriptors.shape[0], self.descriptor_bits - sift_descriptors.shape[1]))
                sift_descriptors = np.concatenate([sift_descriptors, padding], axis=1)
            
            # 二值化（取中位数作为阈值）
            threshold = np.median(sift_descriptors)
            binary_descriptors = (sift_descriptors > threshold).astype(np.uint8)
            
            return binary_descriptors
        
        # 确保是float32
        if image.dtype != np.float32:
            if image.max() > 1.0:
                img_float = image.astype(np.float32) / 255.0
            else:
                img_float = image.astype(np.float32)
        else:
            img_float = image
        
        # 提取每个关键点周围的patch
        patches = []
        valid_indices = []
        
        h, w = image.shape[:2]
        half_patch = patch_size // 2
        
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # 检查边界
            if x < half_patch or x >= w - half_patch or y < half_patch or y >= h - half_patch:
                continue
            
            # 提取patch
            patch = img_float[y - half_patch:y + half_patch, x - half_patch:x + half_patch]
            
            # 调整大小到模型输入尺寸
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
            
            patches.append(patch)
            valid_indices.append(i)
        
        if len(patches) == 0:
            return np.array([])
        
        # 转换为数组
        patches_array = np.array(patches)
        if len(patches_array.shape) == 3:
            patches_array = np.expand_dims(patches_array, axis=-1)
        
        # 预测描述子
        descriptors = self.descriptor_model.predict(patches_array, verbose=0)
        
        # 二值化描述子
        binary_descriptors = (descriptors > 0.5).astype(np.uint8)
        
        return binary_descriptors
    
    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        use_ai_descriptor: bool = True
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray, Optional[np.ndarray]]:
        """
        提取特征点和描述子
        
        Args:
            image: 输入图像
            mask: 掩码（可选）
            use_ai_descriptor: 是否使用AI描述子
            
        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray, Optional[np.ndarray]]: 
                (关键点列表, SIFT描述子, AI描述子)
        """
        # 提取SIFT特征点
        keypoints, sift_descriptors = self.extract_sift_features(image, mask)
        
        # 提取AI描述子
        ai_descriptors = None
        if use_ai_descriptor:
            ai_descriptors = self.extract_ai_descriptor(image, keypoints)
        
        return keypoints, sift_descriptors, ai_descriptors
    
    def match_features(
        self,
        keypoints1: List[cv2.KeyPoint],
        descriptors1: np.ndarray,
        keypoints2: List[cv2.KeyPoint],
        descriptors2: np.ndarray,
        ratio_threshold: float = 0.75
    ) -> List[cv2.DMatch]:
        """
        匹配特征点
        
        Args:
            keypoints1: 第一幅图像的关键点
            descriptors1: 第一幅图像的描述子
            keypoints2: 第二幅图像的关键点
            descriptors2: 第二幅图像的描述子
            ratio_threshold: 比率阈值（Lowe's ratio test）
            
        Returns:
            List[cv2.DMatch]: 匹配点对
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
        
        # 使用FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 匹配
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches

