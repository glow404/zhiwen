# 手机指纹识别系统

完整的指纹识别系统，用于手机解锁功能。

## 功能模块

### 预处理层
- **衍射校正**：支持多相位数据衍射校正时域处理和空域衍射校正
- **指纹降噪**：AI降噪（Unet），抑制相干斑噪声与传感器噪声
- **指纹分割**：确定指纹按压区域，准确率>=95%
- **指纹质量评估**：NFIQv2质量评估，用于注册卡控和识别参数自适应
- **指纹量化**：全局和局部量化，提升局部对比度
- **指纹增强**：Gabor滤波增强指纹纹理，保护细节点

### 特征提取层
- **特征点提取与描述**：SIFT特征点 + AI二值化描述子
- **指纹二值化**：OTSU二值化算法

### 匹配层
- **指纹匹配**：粗匹配和细匹配，计算变换矩阵
- **指纹相似度计算**：基于变换矩阵的二值图相似度

### 鉴权层
- **鉴权分类**：弱分类器级联，支持不同安全等级

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 注册指纹
```python
python main.py --mode register --input path/to/fingerprint.tif
```

### 识别指纹
```python
python main.py --mode identify --input path/to/fingerprint.tif
```

### 训练模型
```bash
python training/train_denoising.py
python training/train_descriptor.py
```

## 项目结构

- `preprocessing/`: 预处理模块
- `feature_extraction/`: 特征提取模块
- `matching/`: 匹配模块
- `authentication/`: 鉴权模块
- `models/`: AI模型定义
- `training/`: 训练脚本
- `utils/`: 工具函数

## 评估指标

- **降噪**: PSNR/SSIM, 特征点复现率
- **分割**: 分割准确率>=95%
- **质量评估**: NFIQv2标准
- **量化**: SSIM指标
- **特征提取**: 特征点>=50个，描述子<512bit，复现率>50%
- **鉴权**: normal场景 FRR<3%, FAR<1/5W；特殊场景 FRR<5%

