"""
AI模型定义模块
包含Unet降噪模型和特征描述子模型
"""

from .unet_denoising import UNetDenoising
from .feature_descriptor import FeatureDescriptor

__all__ = [
    'UNetDenoising',
    'FeatureDescriptor',
]

