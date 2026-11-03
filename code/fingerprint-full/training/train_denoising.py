"""
训练降噪模型
支持监督和非监督训练
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import argparse
import yaml
import cv2
from typing import Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_denoising import UNetDenoising
from utils.image_io import load_image
from utils.metrics import compute_psnr, compute_ssim


def load_dataset(noisy_dir: str, clean_dir: str = None, max_samples: int = None):
    """
    加载数据集
    
    Args:
        noisy_dir: 噪声图像目录
        clean_dir: 干净图像目录（监督训练），如果None则为非监督训练
        max_samples: 最大样本数
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (噪声图像, 干净图像)
    """
    noisy_images = []
    clean_images = []
    
    # 获取图像文件列表
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.bmp']
    noisy_files = [f for f in os.listdir(noisy_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if max_samples:
        noisy_files = noisy_files[:max_samples]
    
    print(f"Loading {len(noisy_files)} images...")
    
    for filename in tqdm(noisy_files):
        noisy_path = os.path.join(noisy_dir, filename)
        try:
            noisy_img = load_image(noisy_path, grayscale=True)
            
            # 归一化到0-1
            if noisy_img.max() > 1.0:
                noisy_img = noisy_img / 255.0
            
            # 调整大小
            noisy_img = cv2.resize(noisy_img, (256, 256))
            
            noisy_images.append(noisy_img)
            
            # 如果有干净图像，加载它
            if clean_dir:
                clean_path = os.path.join(clean_dir, filename)
                if os.path.exists(clean_path):
                    clean_img = load_image(clean_path, grayscale=True)
                    if clean_img.max() > 1.0:
                        clean_img = clean_img / 255.0
                    clean_img = cv2.resize(clean_img, (256, 256))
                    clean_images.append(clean_img)
                else:
                    # 如果没有对应的干净图像，使用噪声图像（非监督）
                    clean_images.append(noisy_img.copy())
            else:
                # 非监督训练：使用噪声图像本身
                clean_images.append(noisy_img.copy())
        
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    return np.array(noisy_images), np.array(clean_images)


def create_data_generator(noisy_images, clean_images, batch_size=16):
    """
    创建数据生成器
    
    Args:
        noisy_images: 噪声图像数组
        clean_images: 干净图像数组
        batch_size: 批次大小
        
    Yields:
        Tuple[np.ndarray, np.ndarray]: (批次噪声图像, 批次干净图像)
    """
    n_samples = len(noisy_images)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_noisy = noisy_images[batch_indices]
        batch_clean = clean_images[batch_indices]
        
        # 添加通道维度
        batch_noisy = np.expand_dims(batch_noisy, axis=-1)
        batch_clean = np.expand_dims(batch_clean, axis=-1)
        
        yield batch_noisy, batch_clean


def train(
    noisy_dir: str,
    clean_dir: str = None,
    model_save_path: str = "data/models/unet_denoising.h5",
    batch_size: int = 16,
    epochs: int = 100,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    input_size: Tuple[int, int] = (256, 256)
):
    """
    训练降噪模型
    
    Args:
        noisy_dir: 噪声图像目录
        clean_dir: 干净图像目录（None则为非监督）
        model_save_path: 模型保存路径
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        validation_split: 验证集比例
        input_size: 输入图像尺寸
    """
    print("Loading dataset...")
    noisy_images, clean_images = load_dataset(noisy_dir, clean_dir)
    
    if len(noisy_images) == 0:
        raise ValueError("No images loaded!")
    
    print(f"Loaded {len(noisy_images)} image pairs")
    
    # 划分训练集和验证集
    n_train = int(len(noisy_images) * (1 - validation_split))
    train_noisy = noisy_images[:n_train]
    train_clean = clean_images[:n_train]
    val_noisy = noisy_images[n_train:]
    val_clean = clean_images[n_train:]
    
    print(f"Training samples: {len(train_noisy)}, Validation samples: {len(val_noisy)}")
    
    # 构建模型
    print("Building model...")
    unet = UNetDenoising(input_size=input_size)
    model = unet.build_model()
    unet.compile_model(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    
    # 准备数据
    train_noisy = np.expand_dims(train_noisy, axis=-1)
    train_clean = np.expand_dims(train_clean, axis=-1)
    val_noisy = np.expand_dims(val_noisy, axis=-1)
    val_clean = np.expand_dims(val_clean, axis=-1)
    
    # 回调函数
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1
        )
    ]
    
    # 训练
    print("Training model...")
    history = model.fit(
        train_noisy, train_clean,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_noisy, val_clean),
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估
    print("Evaluating model...")
    val_pred = model.predict(val_noisy, verbose=0)
    
    # 计算PSNR和SSIM
    psnr_scores = []
    ssim_scores = []
    
    for i in range(len(val_clean)):
        pred = val_pred[i, :, :, 0]
        gt = val_clean[i, :, :, 0]
        
        psnr = compute_psnr(gt, pred)
        ssim = compute_ssim(gt, pred)
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
    
    print(f"Validation PSNR: {np.mean(psnr_scores):.2f} dB")
    print(f"Validation SSIM: {np.mean(ssim_scores):.4f}")
    
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train denoising model")
    parser.add_argument("--noisy_dir", type=str, required=True, help="Directory of noisy images")
    parser.add_argument("--clean_dir", type=str, default=None, help="Directory of clean images (for supervised training)")
    parser.add_argument("--model_path", type=str, default="data/models/unet_denoising.h5", help="Model save path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    
    args = parser.parse_args()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    train(
        args.noisy_dir,
        args.clean_dir,
        args.model_path,
        args.batch_size,
        args.epochs,
        args.lr,
        args.val_split
    )

