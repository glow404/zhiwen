"""
衍射校正模块
支持多相位数据衍射校正时域处理和空域衍射校正
"""

import numpy as np
import cv2
from scipy import signal
from scipy.ndimage import convolve
from typing import List, Optional, Union


class DiffractionCorrection:
    """
    衍射校正类
    支持时域和空域衍射校正
    """
    
    def __init__(
        self,
        enable_temporal: bool = True,
        enable_spatial: bool = True,
        deconv_kernel_size: int = 5,
        iq_recovery_mode: str = "max_amplitude"
    ):
        """
        初始化衍射校正
        
        Args:
            enable_temporal: 是否启用时域处理
            enable_spatial: 是否启用空域处理
            deconv_kernel_size: 反卷积核大小
            iq_recovery_mode: IQ恢复模式（"max_amplitude"或"average"）
        """
        self.enable_temporal = enable_temporal
        self.enable_spatial = enable_spatial
        self.deconv_kernel_size = deconv_kernel_size
        self.iq_recovery_mode = iq_recovery_mode
    
    def correct_temporal(
        self,
        rawdata_frames: List[np.ndarray],
        kernel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        时域衍射校正（反卷积）
        
        Args:
            rawdata_frames: 多相位rawdata帧列表
            kernel: 反卷积核，如果None则使用默认核
            
        Returns:
            np.ndarray: 校正后的单帧图像
        """
        if len(rawdata_frames) == 0:
            raise ValueError("Empty rawdata frames")
        
        # 如果只有一帧，直接返回
        if len(rawdata_frames) == 1:
            return rawdata_frames[0].copy()
        
        # 融合多相位数据
        # 使用平均或加权平均
        fused = np.mean(rawdata_frames, axis=0)
        
        # 反卷积
        if kernel is None:
            # 使用简单的反卷积核（高斯核的逆）
            kernel = self._create_deconv_kernel(self.deconv_kernel_size)
        
        # 使用维纳滤波进行反卷积
        corrected = self._wiener_deconv(fused, kernel)
        
        return corrected
    
    def correct_spatial(
        self,
        image: np.ndarray,
        kernel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        空域衍射校正
        
        Args:
            image: 输入图像（可以是融合后的diff或rawdata）
            kernel: 反卷积核
            
        Returns:
            np.ndarray: 校正后的图像
        """
        if kernel is None:
            kernel = self._create_deconv_kernel(self.deconv_kernel_size)
        
        # 空域反卷积
        corrected = self._wiener_deconv(image, kernel)
        
        return corrected
    
    def iq_recovery(
        self,
        i_data: np.ndarray,
        q_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        IQ恢复，获取最大幅值
        
        Args:
            i_data: I通道数据
            q_data: Q通道数据，如果None则只使用I数据
            
        Returns:
            np.ndarray: 恢复后的幅值
        """
        if q_data is None:
            # 如果没有Q数据，假设输入已经是幅值或只有I通道
            if self.iq_recovery_mode == "max_amplitude":
                # 取绝对值作为幅值
                amplitude = np.abs(i_data)
            else:
                amplitude = i_data
        else:
            # I和Q通道恢复
            if i_data.shape != q_data.shape:
                raise ValueError("I and Q data must have the same shape")
            
            # 计算幅值：sqrt(I^2 + Q^2)
            amplitude = np.sqrt(i_data ** 2 + q_data ** 2)
            
            if self.iq_recovery_mode == "max_amplitude":
                # 取最大幅值
                amplitude = np.maximum.reduce([amplitude, np.abs(i_data), np.abs(q_data)])
        
        return amplitude
    
    def process(
        self,
        rawdata: Union[np.ndarray, List[np.ndarray]],
        q_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        完整的衍射校正流程
        
        Args:
            rawdata: 输入数据（单帧或多帧列表）
            q_data: Q通道数据（可选）
            
        Returns:
            np.ndarray: 校正后的图像
        """
        # 处理输入
        if isinstance(rawdata, list):
            frames = rawdata
        else:
            frames = [rawdata]
        
        # 时域处理
        if self.enable_temporal and len(frames) > 1:
            corrected = self.correct_temporal(frames)
        else:
            corrected = frames[0].copy()
        
        # IQ恢复
        corrected = self.iq_recovery(corrected, q_data)
        
        # 空域处理
        if self.enable_spatial:
            corrected = self.correct_spatial(corrected)
        
        # 归一化到0-1范围
        if corrected.max() > 0:
            corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min() + 1e-10)
        
        return corrected
    
    def _create_deconv_kernel(self, size: int) -> np.ndarray:
        """
        创建反卷积核
        
        Args:
            size: 核大小
            
        Returns:
            np.ndarray: 反卷积核
        """
        # 创建高斯核作为点扩散函数（PSF）
        kernel = np.zeros((size, size))
        center = size // 2
        sigma = size / 3.0
        
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                kernel[i, j] = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        
        kernel = kernel / kernel.sum()
        return kernel
    
    def _wiener_deconv(
        self,
        image: np.ndarray,
        kernel: np.ndarray,
        noise_var: float = 0.01
    ) -> np.ndarray:
        """
        维纳滤波反卷积
        
        Args:
            image: 输入图像
            kernel: 卷积核（PSF）
            noise_var: 噪声方差
            
        Returns:
            np.ndarray: 反卷积后的图像
        """
        # 使用频域维纳滤波
        # H(u,v) = G(u,v) * H*(u,v) / (|H(u,v)|^2 + K)
        # 其中K = noise_var / signal_var
        
        # 转换到频域
        img_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel, s=image.shape)
        
        # 计算信号方差
        signal_var = np.var(image)
        K = noise_var / (signal_var + 1e-10)
        
        # 维纳滤波器
        kernel_conj = np.conj(kernel_fft)
        kernel_mag_sq = np.abs(kernel_fft) ** 2
        wiener_filter = kernel_conj / (kernel_mag_sq + K)
        
        # 应用滤波器
        deconv_fft = img_fft * wiener_filter
        
        # 转换回空域
        deconv = np.real(np.fft.ifft2(deconv_fft))
        
        return deconv

