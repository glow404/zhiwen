### 项目概览
- **定位**: 基于 TensorFlow 2 的指纹比对模型（验证两张指纹是否来自同一手指），属于二分类相似度判断，而非多类识别。
- **核心思路**: 双分支卷积特征提取后做元素乘法融合，接全连接层输出相似度（sigmoid），训练时用“匹配/不匹配”二元标签。
- **输入尺寸**: 默认 128×128（单通道），允许最大不超过 192×192。

### 目录结构与职责
- `source/make_dataset.py`: 从原始指纹图生成训练/验证/测试三份 `.npy` 数据。
- `source/prepare.py`: 核心预处理与数据增强（裁剪、旋转、自适应对比度、噪声）。
- `source/model_net.py`: 模型骨干，双输入特征网络 + 融合判别头。
- `source/train.py`: 训练主脚本，含在线配对的 `DataGenerator`（动态采样正/负对）。
- `source/eval.py`: 评估脚本，用保存的 checkpoint 对测试集做穷举比对统计错误率。
- `dataset/README.md`: 数据组织规范与示例，含数据下载链接。
- `user/tflite_micro/01_convert`: 量化/转 C 数组脚本，便于部署到 TFLite Micro。
- TFLite Micro 是TensorFlow生态系统中最轻量级的推理框架，专门设计用于：微控制器：Arduino、ESP32、STM32等  
• 嵌入式系统：ARM Cortex-M系列、RISC-V等  
• 资源受限设备：内存通常只有几KB到几MB
- `requirements.txt`: 依赖集合（TF2、OpenCV、Pillow、scikit-* 等）。

### 数据与预处理要点
- **文件命名**: `00000_00.bmp`（5位手指ID + 2位索引），8bit 灰度，≥160×160。
- **裁剪逻辑**: Otsu 阈值 + 形态学闭运算估计指纹区域，扩展到 `230×230` 再中心裁切到目标尺寸，确保旋转增广后仍能裁到完整指纹。
- **增强策略**:
  - 原图裁剪
  - 随机旋转（3次）
  - 自适应对比度 + 旋转（3次）
  - 加噪 + 旋转（3次）
- **标签**: 由文件名解析出的手指ID（`uint16`），训练/评估阶段用于构造正/负对。

### 模型结构（简述）
- 两个共享权重的特征提取网络：多层 `Conv2D + BatchNorm + MaxPool`。
- 特征融合：`Multiply`（逐元素乘法）代替传统的距离度量层。
- 判别头：`Conv2D(256)` → `Flatten` → `Dense(1024)` → `Dense(1, sigmoid)`。
- 损失与优化：`binary_crossentropy` + `RMSprop(1e-4)`，指标 `acc`。

### 训练与评估流程
- **数据生成**: 运行 `source/make_dataset.py`，将 `dataset/original/` 转成 `dataset/` 下的 `img_*.npy`、`label_*.npy`。
- **在线配对**: `train.py` 的 `DataGenerator` 在每个 batch 内为图1动态采样匹配或不匹配的图2，构造正负样本各半，且双顺序翻转（提高数据多样性）。
- **Checkpoint/断点续训**: `../model/checkpoint/cp_{W}_{H}.ckpt`；每轮后保存权重与完整模型 `../model/result/fp{W}_{H}.h5`。
- **评估**:
  - `train.py` 每轮后内置一个简化评估：对 `eval_num` 张测试图做两两比对（默认阈值 0.8）。
  - `eval.py` 可对完整测试集进行全量穷举比对，统计误拒/误受与总体错误率。

### 运行方法（Windows 示例）
- 安装依赖（建议 Python 3.7 + TF2 环境）:
```bash
pip install -r requirements.txt
```
- 准备数据（确保 `dataset/original/` 已放入按规范命名的 bmp 图）:
```bash
python source/make_dataset.py --image_width 128 --image_height 128 --dataset_path ./dataset/original/ --out_path ./dataset/
```
- 训练（会自动从 `dataset/` 读取 npy 数据并保存 checkpoint 与模型）:
```bash
python source/train.py --image_width 128 --image_height 128 --dataset_path ./dataset/ --check_point ./model/checkpoint/ --save_model_path ./model/result/ --train_epoch 20 --eval_num 100
```
- 仅从 checkpoint 导出 `.h5`（不训练）:
```bash
python source/train.py --image_width 128 --image_height 128 --check_point ./model/checkpoint/ --save_model 1 --save_model_path ./model/result/
```
- 评估（基于已存在的 checkpoint）:
```bash
python source/eval.py --image_width 128 --image_height 128 --dataset_path ./dataset/ --check_point ./model/checkpoint/ --eval_count 100
```

### 可调参数与实践建议
- **图像尺寸**: 128×128 是速度与精度的折中；允许 ≤192，但需与数据生成一致。
- **阈值 `error_rage`**: 0.8 为经验值，建议在验证集上绘制 ROC/DET 曲线调参，确定最佳阈值或设定 EER。
- **batch 与学习率**: 可按显存与收敛情况调整；优化器亦可尝试 Adam。
- **数据均衡**: 生成器已平衡正/负对；如果类别分布极不均衡，可监控验证集 FAR/FRR。
- **性能注意**: 穷举评估为 O(N^2)，大测试集需限制 `eval_num` 或改用索引/检索评估策略。
- **导出/部署**: 模型为 Keras `.h5`，可进一步用 `user/tflite_micro/01_convert/01_01_quantization.py` 量化并转 C 数组以适配 TFLite Micro。

### 可能的坑
- 路径默认使用相对路径（如 `../dataset/`），在不同工作目录运行需改成项目内相对或绝对路径。
- OpenCV/Pillow/TF 在 Windows 的版本兼容性，建议按 `requirements.txt` 安装并固定 TF2.x 版本。
- `prepare.py` 的 Otsu + 形态学裁剪依赖指纹前景显著性，若原图背景复杂或对比度低，建议在 `EXPAND_*` 和对比度参数上做小幅调优。
- 生成器返回的两个顺序（img1,img2 与 img2,img1）设计用于平衡顺序偏差，修改时需保持标签一致性。

- 我已通读训练、评估、预处理与数据说明文件，并总结了结构与使用方法。若你计划在本机跑通，我可以按你的数据路径帮你写好一条可直接执行的命令行并检查环境。