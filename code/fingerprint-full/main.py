"""
主程序入口
支持注册和识别流程
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pickle
from typing import Optional, Dict, Any

# 导入各个模块
from utils.image_io import load_image, save_image
from preprocessing.diffraction_correction import DiffractionCorrection
from preprocessing.denoising import FingerprintDenoising
from preprocessing.segmentation import FingerprintSegmentation
from preprocessing.quality_assessment import QualityAssessment
from preprocessing.quantization import FingerprintQuantization
from preprocessing.enhancement import FingerprintEnhancement
from feature_extraction.feature_point_extractor import FeaturePointExtractor
from feature_extraction.binarization import FingerprintBinarization
from matching.matcher import FingerprintMatcher
from matching.similarity import FingerprintSimilarity
from authentication.classifier import AuthenticationClassifier


class FingerprintRecognitionSystem:
    """
    指纹识别系统主类
    整合所有模块
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化各个模块
        self.diffraction_correction = DiffractionCorrection(**self.config['diffraction_correction'])
        self.denoising = FingerprintDenoising(**self.config['denoising'])
        self.segmentation = FingerprintSegmentation(**self.config['segmentation'])
        self.quality_assessment = QualityAssessment(**self.config['quality_assessment'])
        self.quantization = FingerprintQuantization(**self.config['quantization'])
        self.enhancement = FingerprintEnhancement(**self.config['enhancement'])
        self.feature_extractor = FeaturePointExtractor(**self.config['feature_extraction'])
        self.binarization = FingerprintBinarization(**self.config['binarization'])
        self.matcher = FingerprintMatcher(**self.config['matching'])
        self.similarity = FingerprintSimilarity(**self.config['similarity'])
        self.classifier = AuthenticationClassifier(**self.config['authentication'])
        
        # 模板存储目录
        self.template_dir = "data/templates"
        os.makedirs(self.template_dir, exist_ok=True)
    
    def preprocess(self, image: np.ndarray, condition: str = "normal") -> Dict[str, Any]:
        """
        预处理流程
        
        Args:
            image: 输入图像
            condition: 采集条件（"normal", "dry", "wet"）
            
        Returns:
            Dict: 预处理结果
        """
        # 1. 衍射校正（如果有多相位数据）
        # 这里假设输入是单帧图像
        corrected = self.diffraction_correction.process([image])
        
        # 2. 降噪
        denoised = self.denoising.denoise(corrected)
        
        # 3. 分割
        mask, orientation_field = self.segmentation.segment(denoised)
        segmented = self.segmentation.apply_mask(denoised, mask)
        
        # 4. 质量评估
        quality_score, quality_subscores, quality_passed = self.quality_assessment.assess(segmented)
        
        # 5. 量化
        quantized = self.quantization.quantize(segmented)
        
        # 6. 增强
        enhanced = self.enhancement.enhance(quantized, orientation_field)
        
        return {
            'image': enhanced,
            'mask': mask,
            'orientation_field': orientation_field,
            'quality_score': quality_score,
            'quality_passed': quality_passed
        }
    
    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        特征提取流程
        
        Args:
            image: 预处理后的图像
            mask: 分割掩码
            
        Returns:
            Dict: 特征提取结果
        """
        # 1. 特征点提取
        keypoints, sift_descriptors, ai_descriptors = self.feature_extractor.extract(
            image, mask, use_ai_descriptor=True
        )
        
        # 2. 二值化
        binary = self.binarization.binarize(image)
        binary = self.binarization.post_process(binary)
        
        return {
            'keypoints': keypoints,
            'sift_descriptors': sift_descriptors,
            'ai_descriptors': ai_descriptors,
            'binary': binary
        }
    
    def register(self, image_path: str, user_id: str) -> bool:
        """
        注册流程
        
        Args:
            image_path: 指纹图像路径
            user_id: 用户ID
            
        Returns:
            bool: 是否注册成功
        """
        print(f"Registering fingerprint for user {user_id}...")
        
        # 1. 加载图像
        image = load_image(image_path, grayscale=True)
        if image.max() > 1.0:
            image = image / 255.0
        
        # 2. 预处理
        preprocessed = self.preprocess(image)
        
        # 3. 质量检查
        if not preprocessed['quality_passed']:
            print(f"Quality check failed: {preprocessed['quality_score']:.2f}")
            return False
        
        # 4. 特征提取
        features = self.extract_features(preprocessed['image'], preprocessed['mask'])
        
        # 5. 检查特征点数量
        if len(features['keypoints']) < self.config['feature_extraction']['min_features']:
            print(f"Not enough features: {len(features['keypoints'])}")
            return False
        
        # 6. 保存模板
        template = {
            'user_id': user_id,
            'keypoints': features['keypoints'],
            'sift_descriptors': features['sift_descriptors'],
            'ai_descriptors': features['ai_descriptors'],
            'binary': features['binary'],
            'quality_score': preprocessed['quality_score']
        }
        
        template_path = os.path.join(self.template_dir, f"{user_id}.pkl")
        with open(template_path, 'wb') as f:
            pickle.dump(template, f)
        
        print(f"Registration successful. Template saved to {template_path}")
        return True
    
    def identify(self, image_path: str, condition: str = "normal") -> Optional[Dict[str, Any]]:
        """
        识别流程
        
        Args:
            image_path: 指纹图像路径
            condition: 采集条件
            
        Returns:
            Optional[Dict]: 识别结果，如果失败返回None
        """
        print(f"Identifying fingerprint...")
        
        # 1. 加载图像
        image = load_image(image_path, grayscale=True)
        if image.max() > 1.0:
            image = image / 255.0
        
        # 2. 预处理
        preprocessed = self.preprocess(image, condition)
        
        # 3. 特征提取
        features = self.extract_features(preprocessed['image'], preprocessed['mask'])
        
        # 4. 与所有模板匹配
        best_match = None
        best_similarity = 0.0
        
        for template_file in os.listdir(self.template_dir):
            if not template_file.endswith('.pkl'):
                continue
            
            template_path = os.path.join(self.template_dir, template_file)
            with open(template_path, 'rb') as f:
                template = pickle.load(f)
            
            # 匹配
            transform_matrix, inliers, n_matches = self.matcher.match(
                features['keypoints'],
                features['sift_descriptors'],
                template['keypoints'],
                template['sift_descriptors']
            )
            
            if transform_matrix is None or n_matches < 10:
                continue
            
            # 计算相似度
            similarity, overlap_ratio = self.similarity.compute_similarity(
                template['binary'],
                features['binary'],
                transform_matrix
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'user_id': template['user_id'],
                    'similarity': similarity,
                    'overlap_ratio': overlap_ratio,
                    'n_matches': n_matches,
                    'transform_matrix': transform_matrix
                }
        
        if best_match is None:
            print("No match found")
            return None
        
        # 5. 鉴权分类
        passed, confidence = self.classifier.authenticate(
            best_match['similarity'],
            condition
        )
        
        best_match['passed'] = passed
        best_match['confidence'] = confidence
        
        print(f"Match found: {best_match['user_id']}, Similarity: {best_match['similarity']:.4f}, Passed: {passed}")
        
        return best_match


def main():
    parser = argparse.ArgumentParser(description="Fingerprint Recognition System")
    parser.add_argument("--mode", type=str, choices=["register", "identify"], required=True,
                       help="Mode: register or identify")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--user_id", type=str, default=None, help="User ID (for registration)")
    parser.add_argument("--condition", type=str, default="normal", choices=["normal", "special"],
                       help="Collection condition")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # 初始化系统
    system = FingerprintRecognitionSystem(args.config)
    
    if args.mode == "register":
        if args.user_id is None:
            print("Error: user_id is required for registration")
            return
        
        success = system.register(args.input, args.user_id)
        if success:
            print("Registration successful!")
        else:
            print("Registration failed!")
    
    elif args.mode == "identify":
        result = system.identify(args.input, args.condition)
        if result and result['passed']:
            print(f"Authentication successful! User: {result['user_id']}")
        else:
            print("Authentication failed!")


if __name__ == "__main__":
    main()

