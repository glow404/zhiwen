"""
Unet降噪模型
TensorFlow实现
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


class UNetDenoising:
    """
    Unet降噪模型类
    编码器-解码器结构，带跳跃连接
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        n_filters: int = 64
    ):
        """
        初始化Unet模型
        
        Args:
            input_size: 输入图像尺寸 (H, W)
            n_filters: 初始滤波器数量
        """
        self.input_size = input_size
        self.n_filters = n_filters
        self.model = None
    
    def build_model(self) -> keras.Model:
        """
        构建Unet模型
        
        Returns:
            keras.Model: Unet模型
        """
        inputs = keras.Input(shape=(*self.input_size, 1))
        
        # 编码器（下采样）
        # Block 1
        c1 = layers.Conv2D(self.n_filters, 3, activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(self.n_filters, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D(2)(c1)
        
        # Block 2
        c2 = layers.Conv2D(self.n_filters * 2, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(self.n_filters * 2, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D(2)(c2)
        
        # Block 3
        c3 = layers.Conv2D(self.n_filters * 4, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(self.n_filters * 4, 3, activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D(2)(c3)
        
        # Block 4
        c4 = layers.Conv2D(self.n_filters * 8, 3, activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(self.n_filters * 8, 3, activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D(2)(c4)
        
        # 瓶颈层
        c5 = layers.Conv2D(self.n_filters * 16, 3, activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(self.n_filters * 16, 3, activation='relu', padding='same')(c5)
        
        # 解码器（上采样）
        # Block 6
        u6 = layers.UpSampling2D(2)(c5)
        u6 = layers.Concatenate()([u6, c4])
        c6 = layers.Conv2D(self.n_filters * 8, 3, activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(self.n_filters * 8, 3, activation='relu', padding='same')(c6)
        
        # Block 7
        u7 = layers.UpSampling2D(2)(c6)
        u7 = layers.Concatenate()([u7, c3])
        c7 = layers.Conv2D(self.n_filters * 4, 3, activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(self.n_filters * 4, 3, activation='relu', padding='same')(c7)
        
        # Block 8
        u8 = layers.UpSampling2D(2)(c7)
        u8 = layers.Concatenate()([u8, c2])
        c8 = layers.Conv2D(self.n_filters * 2, 3, activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(self.n_filters * 2, 3, activation='relu', padding='same')(c8)
        
        # Block 9
        u9 = layers.UpSampling2D(2)(c8)
        u9 = layers.Concatenate()([u9, c1])
        c9 = layers.Conv2D(self.n_filters, 3, activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(self.n_filters, 3, activation='relu', padding='same')(c9)
        
        # 输出层
        outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(c9)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
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
            # 组合损失：L1 + L2 + 感知损失（简化版）
            def combined_loss(y_true, y_pred):
                l1 = keras.losses.mean_absolute_error(y_true, y_pred)
                l2 = keras.losses.mean_squared_error(y_true, y_pred)
                return l1 + 0.5 * l2
        
            loss = combined_loss
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae'])

