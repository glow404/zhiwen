"""
测试和评估脚本
各模块单元测试和端到端评估（FRR/FAR等指标）
"""

import os
import sys
import numpy as np
import cv2
import argparse
from typing import List, Tuple, Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.image_io import load_image
from utils.metrics import (
    compute_psnr, compute_ssim, compute_feature_repeatability,
    compute_segmentation_accuracy, compute_frr_far, compute_roc_curve, compute_auc
)
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
from main import FingerprintRecognitionSystem


def test_diffraction_correction():
    """测试衍射校正模块"""
    print("Testing diffraction correction...")
    
    # 创建测试图像
    test_image = np.random.rand(256, 256).astype(np.float32)
    
    correction = DiffractionCorrection()
    corrected = correction.process([test_image])
    
    assert corrected.shape == test_image.shape, "Shape mismatch"
    print("✓ Diffraction correction test passed")


def test_denoising():
    """测试降噪模块"""
    print("Testing denoising...")
    
    # 创建测试图像（带噪声）
    clean_image = np.random.rand(256, 256).astype(np.float32)
    noisy_image = clean_image + np.random.randn(256, 256).astype(np.float32) * 0.1
    noisy_image = np.clip(noisy_image, 0, 1)
    
    denoising = FingerprintDenoising()
    denoised = denoising.denoise(noisy_image)
    
    # 计算PSNR和SSIM
    psnr = compute_psnr(clean_image, denoised)
    ssim = compute_ssim(clean_image, denoised)
    
    print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    assert psnr > 0, "PSNR should be positive"
    print("✓ Denoising test passed")


def test_segmentation():
    """测试分割模块"""
    print("Testing segmentation...")
    
    # 创建测试图像（模拟指纹）
    test_image = np.random.rand(256, 256).astype(np.float32)
    
    segmentation = FingerprintSegmentation()
    mask, orientation_field = segmentation.segment(test_image)
    
    assert mask.shape == test_image.shape, "Mask shape mismatch"
    assert orientation_field.shape == test_image.shape, "Orientation field shape mismatch"
    print("✓ Segmentation test passed")


def test_quality_assessment():
    """测试质量评估模块"""
    print("Testing quality assessment...")
    
    test_image = np.random.rand(256, 256).astype(np.float32)
    
    quality_assessment = QualityAssessment()
    score, subscores, passed = quality_assessment.assess(test_image)
    
    assert 0 <= score <= 100, "Quality score out of range"
    print(f"  Quality score: {score:.2f}")
    print("✓ Quality assessment test passed")


def test_quantization():
    """测试量化模块"""
    print("Testing quantization...")
    
    test_image = np.random.rand(256, 256).astype(np.float32)
    
    quantization = FingerprintQuantization()
    quantized = quantization.quantize(test_image, method="both")
    
    # 检查SSIM
    ssim = compute_ssim(test_image, quantized)
    assert ssim > 0.5, "SSIM too low after quantization"
    print(f"  SSIM: {ssim:.4f}")
    print("✓ Quantization test passed")


def test_enhancement():
    """测试增强模块"""
    print("Testing enhancement...")
    
    test_image = np.random.rand(256, 256).astype(np.float32)
    
    enhancement = FingerprintEnhancement()
    enhanced = enhancement.enhance(test_image)
    
    assert enhanced.shape == test_image.shape, "Shape mismatch"
    print("✓ Enhancement test passed")


def test_feature_extraction():
    """测试特征提取模块"""
    print("Testing feature extraction...")
    
    test_image = np.random.rand(256, 256).astype(np.float32)
    if test_image.max() <= 1.0:
        test_image = (test_image * 255).astype(np.uint8)
    
    extractor = FeaturePointExtractor()
    keypoints, sift_descriptors, ai_descriptors = extractor.extract(test_image)
    
    assert len(keypoints) > 0, "No keypoints detected"
    print(f"  Keypoints: {len(keypoints)}")
    print("✓ Feature extraction test passed")


def test_binarization():
    """测试二值化模块"""
    print("Testing binarization...")
    
    test_image = np.random.rand(256, 256).astype(np.float32)
    
    binarization = FingerprintBinarization()
    binary = binarization.binarize(test_image)
    
    assert binary.shape == test_image.shape, "Shape mismatch"
    assert np.all((binary == 0) | (binary == 1)), "Binary image should contain only 0 and 1"
    print("✓ Binarization test passed")


def test_matching():
    """测试匹配模块"""
    print("Testing matching...")
    
    # 创建两个测试图像
    image1 = np.random.rand(256, 256).astype(np.float32)
    image2 = np.random.rand(256, 256).astype(np.float32)
    
    if image1.max() <= 1.0:
        image1 = (image1 * 255).astype(np.uint8)
    if image2.max() <= 1.0:
        image2 = (image2 * 255).astype(np.uint8)
    
    extractor = FeaturePointExtractor()
    kp1, desc1, _ = extractor.extract(image1)
    kp2, desc2, _ = extractor.extract(image2)
    
    if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
        matcher = FingerprintMatcher()
        transform_matrix, inliers, n_matches = matcher.match(kp1, desc1, kp2, desc2)
        
        print(f"  Matches: {n_matches}")
        print("✓ Matching test passed")
    else:
        print("  No descriptors, skipping matching test")


def test_similarity():
    """测试相似度计算模块"""
    print("Testing similarity...")
    
    binary1 = np.random.randint(0, 2, (256, 256)).astype(np.float32)
    binary2 = np.random.randint(0, 2, (256, 256)).astype(np.float32)
    
    similarity = FingerprintSimilarity()
    sim_score, overlap_ratio = similarity.compute_similarity(binary1, binary2)
    
    assert 0 <= sim_score <= 1, "Similarity score out of range"
    print(f"  Similarity: {sim_score:.4f}, Overlap: {overlap_ratio:.4f}")
    print("✓ Similarity test passed")


def evaluate_end_to_end(
    genuine_pairs: List[Tuple[str, str]],
    impostor_pairs: List[Tuple[str, str]],
    condition: str = "normal"
) -> Dict[str, float]:
    """
    端到端评估
    
    Args:
        genuine_pairs: 同一人的图像对列表 [(image1, image2), ...]
        impostor_pairs: 不同人的图像对列表
        condition: 采集条件
        
    Returns:
        Dict: 评估指标
    """
    print("Evaluating end-to-end system...")
    
    system = FingerprintRecognitionSystem()
    
    similarity_scores = []
    labels = []
    
    # 处理同一人的图像对
    print(f"Processing {len(genuine_pairs)} genuine pairs...")
    for img1_path, img2_path in genuine_pairs:
        try:
            # 预处理和特征提取
            img1 = load_image(img1_path, grayscale=True)
            if img1.max() > 1.0:
                img1 = img1 / 255.0
            
            img2 = load_image(img2_path, grayscale=True)
            if img2.max() > 1.0:
                img2 = img2 / 255.0
            
            preprocessed1 = system.preprocess(img1, condition)
            preprocessed2 = system.preprocess(img2, condition)
            
            features1 = system.extract_features(preprocessed1['image'], preprocessed1['mask'])
            features2 = system.extract_features(preprocessed2['image'], preprocessed2['mask'])
            
            # 匹配和相似度计算
            if features1['sift_descriptors'] is not None and features2['sift_descriptors'] is not None:
                transform_matrix, inliers, n_matches = system.matcher.match(
                    features1['keypoints'], features1['sift_descriptors'],
                    features2['keypoints'], features2['sift_descriptors']
                )
                
                if transform_matrix is not None:
                    similarity, _ = system.similarity.compute_similarity(
                        features1['binary'], features2['binary'], transform_matrix
                    )
                    similarity_scores.append(similarity)
                    labels.append(True)
        except Exception as e:
            print(f"Error processing genuine pair ({img1_path}, {img2_path}): {e}")
    
    # 处理不同人的图像对
    print(f"Processing {len(impostor_pairs)} impostor pairs...")
    for img1_path, img2_path in impostor_pairs:
        try:
            img1 = load_image(img1_path, grayscale=True)
            if img1.max() > 1.0:
                img1 = img1 / 255.0
            
            img2 = load_image(img2_path, grayscale=True)
            if img2.max() > 1.0:
                img2 = img2 / 255.0
            
            preprocessed1 = system.preprocess(img1, condition)
            preprocessed2 = system.preprocess(img2, condition)
            
            features1 = system.extract_features(preprocessed1['image'], preprocessed1['mask'])
            features2 = system.extract_features(preprocessed2['image'], preprocessed2['mask'])
            
            if features1['sift_descriptors'] is not None and features2['sift_descriptors'] is not None:
                transform_matrix, inliers, n_matches = system.matcher.match(
                    features1['keypoints'], features1['sift_descriptors'],
                    features2['keypoints'], features2['sift_descriptors']
                )
                
                if transform_matrix is not None:
                    similarity, _ = system.similarity.compute_similarity(
                        features1['binary'], features2['binary'], transform_matrix
                    )
                    similarity_scores.append(similarity)
                    labels.append(False)
        except Exception as e:
            print(f"Error processing impostor pair ({img1_path}, {img2_path}): {e}")
    
    if len(similarity_scores) == 0:
        print("No valid similarity scores computed")
        return {}
    
    # 计算FRR和FAR
    threshold = 0.5
    frr, far = compute_frr_far(similarity_scores, labels, threshold)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = compute_roc_curve(similarity_scores, labels)
    auc = compute_auc(fpr, tpr)
    
    # 找到最优阈值（平衡FRR和FAR）
    optimal_threshold_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    optimal_threshold = thresholds[optimal_threshold_idx]
    optimal_frr, optimal_far = compute_frr_far(similarity_scores, labels, optimal_threshold)
    
    results = {
        'FRR': frr,
        'FAR': far,
        'AUC': auc,
        'optimal_threshold': optimal_threshold,
        'optimal_FRR': optimal_frr,
        'optimal_FAR': optimal_far,
        'n_genuine': sum(labels),
        'n_impostor': sum(~np.array(labels))
    }
    
    print(f"\nEvaluation Results:")
    print(f"  FRR (threshold={threshold}): {frr:.4f}")
    print(f"  FAR (threshold={threshold}): {far:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Optimal FRR: {optimal_frr:.4f}")
    print(f"  Optimal FAR: {optimal_far:.4f}")
    
    return results


def run_unit_tests():
    """运行所有单元测试"""
    print("=" * 50)
    print("Running unit tests...")
    print("=" * 50)
    
    try:
        test_diffraction_correction()
        test_denoising()
        test_segmentation()
        test_quality_assessment()
        test_quantization()
        test_enhancement()
        test_feature_extraction()
        test_binarization()
        test_matching()
        test_similarity()
        
        print("=" * 50)
        print("All unit tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"Unit test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and evaluation script")
    parser.add_argument("--mode", type=str, choices=["unit", "evaluate", "both"], default="both",
                       help="Test mode")
    parser.add_argument("--genuine_pairs", type=str, default=None,
                       help="File containing genuine pairs (one pair per line: img1 img2)")
    parser.add_argument("--impostor_pairs", type=str, default=None,
                       help="File containing impostor pairs (one pair per line: img1 img2)")
    parser.add_argument("--condition", type=str, default="normal", choices=["normal", "special"],
                       help="Collection condition")
    
    args = parser.parse_args()
    
    if args.mode in ["unit", "both"]:
        run_unit_tests()
    
    if args.mode in ["evaluate", "both"]:
        if args.genuine_pairs and args.impostor_pairs:
            # 加载图像对
            genuine_pairs = []
            with open(args.genuine_pairs, 'r') as f:
                for line in f:
                    img1, img2 = line.strip().split()
                    genuine_pairs.append((img1, img2))
            
            impostor_pairs = []
            with open(args.impostor_pairs, 'r') as f:
                for line in f:
                    img1, img2 = line.strip().split()
                    impostor_pairs.append((img1, img2))
            
            results = evaluate_end_to_end(genuine_pairs, impostor_pairs, args.condition)
        else:
            print("Warning: genuine_pairs and impostor_pairs files are required for evaluation")

