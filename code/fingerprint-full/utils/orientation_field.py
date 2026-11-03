"""
方向场计算模块
基于梯度计算指纹方向场
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, Optional


def compute_orientation_field(
    image: np.ndarray,
    block_size: int = 16,
    smooth: bool = True,
    gaussian_kernel_size: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算指纹图像的方向场
    
    Args:
        image: 输入指纹图像（灰度图）
        block_size: 块大小，默认16
        smooth: 是否平滑方向场
        gaussian_kernel_size: 高斯平滑核大小
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (方向场角度, 方向场一致性)
                                       角度范围: [-π/2, π/2]
                                       一致性范围: [0, 1]
    """
    h, w = image.shape
    
    # 计算梯度
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算块方向
    block_h = h // block_size
    block_w = w // block_size
    
    orientation_field = np.zeros((block_h, block_w))
    coherence = np.zeros((block_h, block_w))
    
    for i in range(block_h):
        for j in range(block_w):
            y_start = i * block_size
            y_end = min((i + 1) * block_size, h)
            x_start = j * block_size
            x_end = min((j + 1) * block_size, w)
            
            # 提取块内的梯度
            block_gx = gx[y_start:y_end, x_start:x_end]
            block_gy = gy[y_start:y_end, x_start:x_end]
            
            # 计算方向场
            # 使用梯度平方和
            gxx = np.sum(block_gx * block_gx)
            gyy = np.sum(block_gy * block_gy)
            gxy = np.sum(block_gx * block_gy)
            
            # 计算方向角度（垂直于脊线方向）
            if gxx + gyy != 0:
                # 主方向（垂直于脊线）
                theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
                
                # 计算一致性（coherence）
                # 一致性 = (λ1 - λ2) / (λ1 + λ2)
                # 其中λ1和λ2是结构张量的特征值
                trace = gxx + gyy
                det = gxx * gyy - gxy * gxy
                
                if trace > 0:
                    # 特征值
                    lambda1 = 0.5 * (trace + np.sqrt(trace * trace - 4 * det))
                    lambda2 = 0.5 * (trace - np.sqrt(trace * trace - 4 * det))
                    coherence[i, j] = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
                else:
                    coherence[i, j] = 0.0
            else:
                theta = 0.0
                coherence[i, j] = 0.0
            
            orientation_field[i, j] = theta
    
    # 平滑方向场
    if smooth:
        # 使用高斯平滑
        orientation_field = ndimage.gaussian_filter(orientation_field, sigma=1.0)
        
        # 方向场平滑（考虑周期性）
        # 将角度转换为复数进行平滑
        complex_field = np.exp(1j * 2 * orientation_field)
        smoothed_complex = ndimage.gaussian_filter(complex_field, sigma=1.0)
        orientation_field = np.angle(smoothed_complex) / 2.0
    
    # 扩展到原图尺寸
    orientation_field_full = cv2.resize(
        orientation_field, (w, h), interpolation=cv2.INTER_LINEAR
    )
    coherence_full = cv2.resize(
        coherence, (w, h), interpolation=cv2.INTER_LINEAR
    )
    
    return orientation_field_full, coherence_full


def get_orientation_at_point(
    orientation_field: np.ndarray,
    x: int,
    y: int
) -> float:
    """
    获取指定点的方向角度
    
    Args:
        orientation_field: 方向场
        x: x坐标
        y: y坐标
        
    Returns:
        float: 方向角度（弧度）
    """
    h, w = orientation_field.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return orientation_field[y, x]


def visualize_orientation_field(
    image: np.ndarray,
    orientation_field: np.ndarray,
    step: int = 10
) -> np.ndarray:
    """
    可视化方向场
    
    Args:
        image: 原始图像
        orientation_field: 方向场
        step: 采样步长
        
    Returns:
        np.ndarray: 可视化图像
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    h, w = orientation_field.shape
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            angle = orientation_field[y, x]
            # 方向向量
            dx = np.cos(angle) * step / 2
            dy = np.sin(angle) * step / 2
            
            pt1 = (int(x - dx), int(y - dy))
            pt2 = (int(x + dx), int(y + dy))
            
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
    
    return vis

