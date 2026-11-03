"""
特征描述子模型
二值化描述子网络（类似DeepBit/HashNet）
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


class FeatureDescriptor:
    """
    特征描述子模型类
    输出紧凑二进制描述符
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (32, 32),
        descriptor_bits: int = 512
    ):
        """
        初始化描述子模型
        
        Args:
            input_size: 输入patch尺寸 (H, W)
            descriptor_bits: 描述子位数
        """
        self.input_size = input_size
        self.descriptor_bits = descriptor_bits
        self.model = None
    
    def build_model(self) -> keras.Model:
        """
        构建描述子模型
        
        Returns:
            keras.Model: 描述子模型
        """
        inputs = keras.Input(shape=(*self.input_size, 1))
        
        # 特征提取网络
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # 展平
        x = layers.Flatten()(x)
        
        # 全连接层
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # 输出层（描述子）
        # 使用tanh激活，然后二值化
        descriptors = layers.Dense(self.descriptor_bits, activation='tanh')(x)
        
        model = keras.Model(inputs=inputs, outputs=descriptors)
        self.model = model
        
        return model
    
    def compile_model(
        self,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        loss: Optional[str] = None
    ):
        """
        编译模型
        
        Args:
            optimizer: 优化器
            loss: 损失函数
        """
        if self.model is None:
            self.build_model()
        
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        if loss is None:
            # 三元组损失（简化版，实际训练时需要实现三元组损失）
            def triplet_loss(y_true, y_pred):
                # 这里简化处理，实际应该使用三元组损失
                # 使用均方误差作为占位
                return keras.losses.mean_squared_error(y_true, y_pred)
            
            loss = triplet_loss
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

