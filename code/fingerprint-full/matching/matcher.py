"""
指纹匹配模块
粗匹配和细匹配，计算变换矩阵
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist


class FingerprintMatcher:
    """
    指纹匹配类
    使用RANSAC进行粗匹配和细匹配
    """
    
    def __init__(
        self,
        ransac_threshold: float = 3.0,
        ransac_max_iter: int = 2000,
        min_inliers: int = 10,
        coarse_match_ratio: float = 0.8
    ):
        """
        初始化匹配器
        
        Args:
            ransac_threshold: RANSAC阈值（像素）
            ransac_max_iter: RANSAC最大迭代次数
            min_inliers: 最小内点数
            coarse_match_ratio: 粗匹配比率阈值
        """
        self.ransac_threshold = ransac_threshold
        self.ransac_max_iter = ransac_max_iter
        self.min_inliers = min_inliers
        self.coarse_match_ratio = coarse_match_ratio
    
    def coarse_match(
        self,
        keypoints1: List[cv2.KeyPoint],
        descriptors1: np.ndarray,
        keypoints2: List[cv2.KeyPoint],
        descriptors2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        粗匹配：计算粗匹配点对
        
        Args:
            keypoints1: 第一幅图像的关键点
            descriptors1: 第一幅图像的描述子
            keypoints2: 第二幅图像的关键点
            descriptors2: 第二幅图像的描述子
            
        Returns:
            List[cv2.DMatch]: 粗匹配点对
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
        
        # KNN匹配
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.coarse_match_ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def compute_transformation_matrix(
        self,
        keypoints1: List[cv2.KeyPoint],
        keypoints2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        计算变换矩阵（使用RANSAC）
        
        Args:
            keypoints1: 第一幅图像的关键点
            keypoints2: 第二幅图像的关键点
            matches: 匹配点对
            
        Returns:
            Tuple[Optional[np.ndarray], List[int]]: (变换矩阵, 内点索引列表)
        """
        if len(matches) < 4:
            return None, []
        
        # 提取匹配点坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC计算单应性矩阵
        transform_matrix, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            self.ransac_threshold,
            maxIters=self.ransac_max_iter
        )
        
        if transform_matrix is None:
            return None, []
        
        # 提取内点索引
        inlier_indices = [i for i, m in enumerate(mask) if m[0] == 1]
        
        # 检查内点数
        if len(inlier_indices) < self.min_inliers:
            return None, []
        
        return transform_matrix, inlier_indices
    
    def fine_match(
        self,
        keypoints1: List[cv2.KeyPoint],
        keypoints2: List[cv2.KeyPoint],
        coarse_matches: List[cv2.DMatch],
        transform_matrix: np.ndarray
    ) -> Tuple[List[cv2.DMatch], List[int]]:
        """
        细匹配：计算满足匹配矩阵的匹配点
        
        Args:
            keypoints1: 第一幅图像的关键点
            keypoints2: 第二幅图像的关键点
            coarse_matches: 粗匹配点对
            transform_matrix: 变换矩阵
            
        Returns:
            Tuple[List[cv2.DMatch], List[int]]: (细匹配点对, 内点索引)
        """
        if len(coarse_matches) == 0:
            return [], []
        
        # 提取匹配点坐标
        src_pts = np.array([keypoints1[m.queryIdx].pt for m in coarse_matches])
        dst_pts = np.array([keypoints2[m.trainIdx].pt for m in coarse_matches])
        
        # 使用变换矩阵变换源点
        src_pts_homogeneous = np.column_stack([src_pts, np.ones(len(src_pts))])
        transformed_pts = transform_matrix @ src_pts_homogeneous.T
        transformed_pts = (transformed_pts[:2] / transformed_pts[2]).T
        
        # 计算变换后的点与目标点的距离
        distances = np.sqrt(np.sum((transformed_pts - dst_pts) ** 2, axis=1))
        
        # 内点是距离小于阈值的点
        inlier_mask = distances < self.ransac_threshold
        inlier_indices = np.where(inlier_mask)[0].tolist()
        
        fine_matches = [coarse_matches[i] for i in inlier_indices]
        
        return fine_matches, inlier_indices
    
    def match(
        self,
        keypoints1: List[cv2.KeyPoint],
        descriptors1: np.ndarray,
        keypoints2: List[cv2.KeyPoint],
        descriptors2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], List[int], int]:
        """
        完整的匹配流程
        
        Args:
            keypoints1: 第一幅图像的关键点
            descriptors1: 第一幅图像的描述子
            keypoints2: 第二幅图像的关键点
            descriptors2: 第二幅图像的描述子
            
        Returns:
            Tuple[Optional[np.ndarray], List[int], int]: 
                (变换矩阵, 内点索引列表, 匹配点数)
        """
        # 粗匹配
        coarse_matches = self.coarse_match(
            keypoints1, descriptors1, keypoints2, descriptors2
        )
        
        if len(coarse_matches) < self.min_inliers:
            return None, [], 0
        
        # 计算变换矩阵
        transform_matrix, ransac_inliers = self.compute_transformation_matrix(
            keypoints1, keypoints2, coarse_matches
        )
        
        if transform_matrix is None:
            return None, [], 0
        
        # 细匹配
        fine_matches, fine_inliers = self.fine_match(
            keypoints1, keypoints2, coarse_matches, transform_matrix
        )
        
        return transform_matrix, fine_inliers, len(fine_matches)

