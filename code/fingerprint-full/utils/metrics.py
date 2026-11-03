"""
评估指标模块
包含PSNR、SSIM、特征点复现率等指标
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import List, Tuple, Optional


def compute_psnr(image1: np.ndarray, image2: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    计算峰值信噪比（PSNR）
    
    Args:
        image1: 第一张图像
        image2: 第二张图像
        data_range: 数据范围，如果None则自动计算
        
    Returns:
        float: PSNR值（dB）
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes must match: {image1.shape} vs {image2.shape}")
    
    # 确保数据类型一致
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    
    # 计算数据范围
    if data_range is None:
        if img1.max() > 1.0 or img2.max() > 1.0:
            data_range = 255.0
        else:
            data_range = 1.0
    
    # 使用skimage的PSNR函数
    return psnr(img1, img2, data_range=data_range)


def compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    data_range: Optional[float] = None,
    win_size: Optional[int] = None
) -> float:
    """
    计算结构相似性指数（SSIM）
    
    Args:
        image1: 第一张图像
        image2: 第二张图像
        data_range: 数据范围
        win_size: 窗口大小，如果None则自动计算
        
    Returns:
        float: SSIM值（0-1之间）
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes must match: {image1.shape} vs {image2.shape}")
    
    # 确保是2D图像
    if len(image1.shape) == 3:
        # 如果是彩色图，转换为灰度
        if image1.shape[2] == 3:
            image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            image1 = image1[:, :, 0]
            image2 = image2[:, :, 0]
    
    # 确保数据类型一致
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    
    # 计算数据范围
    if data_range is None:
        if img1.max() > 1.0 or img2.max() > 1.0:
            data_range = 255.0
        else:
            data_range = 1.0
    
    # 使用skimage的SSIM函数
    return ssim(img1, img2, data_range=data_range, win_size=win_size)


def compute_feature_repeatability(
    keypoints1: List[cv2.KeyPoint],
    keypoints2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    threshold: float = 3.0
) -> Tuple[float, int, int]:
    """
    计算特征点复现率
    
    Args:
        keypoints1: 第一幅图像的关键点
        keypoints2: 第二幅图像的关键点
        matches: 匹配点对
        threshold: 重复检测阈值（像素）
        
    Returns:
        Tuple[float, int, int]: (复现率, 匹配点数, 总特征点数)
    """
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return 0.0, 0, max(len(keypoints1), len(keypoints2))
    
    # 统计匹配的特征点
    matched_count = len(matches)
    total_count = max(len(keypoints1), len(keypoints2))
    
    if total_count == 0:
        return 0.0, 0, 0
    
    repeatability = matched_count / total_count
    
    return repeatability, matched_count, total_count


def compute_segmentation_accuracy(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray
) -> float:
    """
    计算分割准确率
    
    Args:
        mask_pred: 预测的mask（二值图）
        mask_gt: 真实的mask（二值图）
        
    Returns:
        float: 准确率（0-1之间）
    """
    if mask_pred.shape != mask_gt.shape:
        raise ValueError(f"Mask shapes must match: {mask_pred.shape} vs {mask_gt.shape}")
    
    # 确保是二值图
    mask_pred = (mask_pred > 0.5).astype(np.uint8)
    mask_gt = (mask_gt > 0.5).astype(np.uint8)
    
    # 计算准确率
    correct = np.sum(mask_pred == mask_gt)
    total = mask_pred.size
    
    return correct / total if total > 0 else 0.0


def compute_frr_far(
    similarity_scores: List[float],
    labels: List[bool],
    threshold: float
) -> Tuple[float, float]:
    """
    计算FRR（False Rejection Rate）和FAR（False Acceptance Rate）
    
    Args:
        similarity_scores: 相似度分数列表
        labels: 标签列表（True表示同一人，False表示不同人）
        threshold: 阈值
        
    Returns:
        Tuple[float, float]: (FRR, FAR)
    """
    if len(similarity_scores) != len(labels):
        raise ValueError("Length of scores and labels must match")
    
    scores = np.array(similarity_scores)
    labels = np.array(labels)
    
    # 同一人（positive）的分数
    positive_scores = scores[labels]
    # 不同人（negative）的分数
    negative_scores = scores[~labels]
    
    # FRR: False Rejection Rate - 同一人被拒绝的比例
    if len(positive_scores) > 0:
        frr = np.sum(positive_scores < threshold) / len(positive_scores)
    else:
        frr = 0.0
    
    # FAR: False Acceptance Rate - 不同人被接受的比例
    if len(negative_scores) > 0:
        far = np.sum(negative_scores >= threshold) / len(negative_scores)
    else:
        far = 0.0
    
    return frr, far


def compute_roc_curve(
    similarity_scores: List[float],
    labels: List[bool],
    num_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算ROC曲线
    
    Args:
        similarity_scores: 相似度分数列表
        labels: 标签列表
        num_thresholds: 阈值数量
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (FPR, TPR, thresholds)
    """
    scores = np.array(similarity_scores)
    labels = np.array(labels)
    
    # 生成阈值
    min_score = scores.min()
    max_score = scores.max()
    thresholds = np.linspace(min_score, max_score, num_thresholds)
    
    tpr = []  # True Positive Rate
    fpr = []  # False Positive Rate
    
    for threshold in thresholds:
        # True Positive: 同一人且接受
        tp = np.sum((scores >= threshold) & labels)
        # False Positive: 不同人但接受
        fp = np.sum((scores >= threshold) & (~labels))
        # True Negative: 不同人且拒绝
        tn = np.sum((scores < threshold) & (~labels))
        # False Negative: 同一人但拒绝
        fn = np.sum((scores < threshold) & labels)
        
        # TPR = TP / (TP + FN)
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # FPR = FP / (FP + TN)
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    
    return np.array(fpr), np.array(tpr), thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    计算AUC（Area Under Curve）
    
    Args:
        fpr: False Positive Rate数组
        tpr: True Positive Rate数组
        
    Returns:
        float: AUC值
    """
    # 使用梯形法则计算积分
    auc = np.trapz(tpr, fpr)
    return auc

