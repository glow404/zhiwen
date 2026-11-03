"""
图像IO模块
支持tif、bmp、png多种格式
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Union, Optional, Tuple

SUPPORTED_FORMATS = ['tif', 'tiff', 'bmp', 'png', 'jpg', 'jpeg']


def supported_formats() -> list:
    """返回支持的图像格式列表"""
    return SUPPORTED_FORMATS.copy()


def load_image(image_path: str, grayscale: bool = True) -> np.ndarray:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
        grayscale: 是否转换为灰度图，默认True
        
    Returns:
        numpy.ndarray: 图像数组，shape为(H, W)或(H, W, 3)
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的格式
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # 获取文件扩展名
    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {ext}. Supported formats: {SUPPORTED_FORMATS}")
    
    # 使用OpenCV加载图像
    if grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        # 如果OpenCV失败，尝试使用PIL
        try:
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
            if grayscale and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    if img is None:
        raise ValueError(f"Failed to load image from: {image_path}")
    
    return img.astype(np.float32)


def save_image(image: np.ndarray, output_path: str, normalize: bool = True) -> bool:
    """
    保存图像文件
    
    Args:
        image: 图像数组
        output_path: 输出文件路径
        normalize: 是否归一化到0-255范围
        
    Returns:
        bool: 是否保存成功
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # 处理图像数据
        img = image.copy()
        
        # 转换为uint8
        if img.dtype != np.uint8:
            if normalize:
                img = img - img.min()
                if img.max() > 0:
                    img = (img / img.max() * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 保存图像
        return cv2.imwrite(output_path, img)
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def load_multi_phase_data(data_paths: list) -> list:
    """
    加载多相位数据
    
    Args:
        data_paths: 多个图像文件路径列表
        
    Returns:
        list: 图像数组列表
    """
    images = []
    for path in data_paths:
        img = load_image(path, grayscale=True)
        images.append(img)
    return images


def ensure_image_format(image: np.ndarray, target_dtype: type = np.float32) -> np.ndarray:
    """
    确保图像格式正确
    
    Args:
        image: 输入图像
        target_dtype: 目标数据类型
        
    Returns:
        numpy.ndarray: 格式化后的图像
    """
    img = image.copy()
    
    # 确保是2D或3D数组
    if len(img.shape) > 3:
        raise ValueError(f"Invalid image shape: {img.shape}")
    
    # 转换为目标数据类型
    if img.dtype != target_dtype:
        if target_dtype == np.float32 or target_dtype == np.float64:
            if img.dtype == np.uint8:
                img = img.astype(target_dtype) / 255.0
            else:
                img = img.astype(target_dtype)
        else:
            img = img.astype(target_dtype)
    
    return img

