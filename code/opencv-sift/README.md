# OpenCV SIFT 指纹特征提取项目

基于 Python 与 OpenCV 的指纹特征提取实验项目。

当前项目的目标是先搭建一个清晰、可维护、可调参的本地实验骨架，用于完成：

- 指纹灰度图读取与合法性检查
- 预处理
- ROI 提取与背景抑制
- SIFT 关键点检测与描述子提取
- 关键点可视化
- 指纹模板保存与加载

## 当前已实现功能

目前已经完成的模块包括：

- 单张指纹图读取
- 目录递归扫描与批处理入口
- 灰度归一化
- 可选 CLAHE 对比度增强
- 可选去噪
- 基于局部方差的 ROI 提取
- 背景抑制
- SIFT 特征提取
- 关键点结构化保存
- 模板保存与加载
- Debug 中间图保存
- 文件日志与批处理 summary 输出

## 当前项目还建议补充的部分

虽然核心单图流程已经成型，但要把项目继续补强到“更完整可研究/可扩展”的程度，建议后续优先补这些内容：

### 1. ROI 稳定性增强

当前 ROI 已有基础版本，但真实数据上仍建议继续强化：

- 更稳健的阈值策略
- 多 ROI 方法切换
- 失败样本标记与回退策略统计

### 2. 匹配模块

模板已经具备直接对接匹配的条件，后续建议补：

- BFMatcher / FLANN
- ratio test
- 几何一致性过滤
- 匹配得分定义

### 3. 评估与实验脚本

为了真正做实验，还建议补：

- 批量提取脚本
- 模板匹配实验脚本
- 指标统计脚本

### 4. 测试覆盖继续补齐

目前已有基础测试，但还建议继续增加：

- 单图端到端 smoke test
- 模板与匹配衔接测试
- 不同配置组合测试

### 5. 移动端迁移准备

当前模板结构已经比较清晰，但以后如需上 Android，建议继续做：

- 模板字段精简
- 描述子量化
- 二进制协议优化

## 项目目录说明

当前主要目录如下：

```text
opencv-sift/
  configs/                  配置文件
  docs/                     文档目录
  outputs/                  输出目录
  scripts/                  运行脚本
  src/
    config/                 配置数据结构与加载器
    features/               SIFT 特征提取
    io/                     输入输出、模板存储
    pipeline/               主流程编排
    preprocess/             指纹预处理
    utils/                  路径等工具
    visualization/          可视化
  tests/                    测试
```

## 环境要求

- Python 3.10 及以上
- OpenCV Contrib 版本的 Python 包

项目依赖写在：

- `requirements.txt`
- `pyproject.toml`

## 安装依赖

建议先创建虚拟环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

如果你使用的是本地开发安装方式，也可以使用：

```bash
pip install -e .
```

## 配置文件说明

当前常用配置文件：

- `configs/default.yaml`
  - 通用默认配置
- `configs/mvp_single.yaml`
  - 单图最小可运行示例配置

当前重点配置分为几类：

- `preprocess`
  - 控制归一化、增强、去噪、背景抑制、debug 输出
- `roi`
  - 控制 ROI 提取方法与阈值
- `sift`
  - 控制关键点与描述子提取参数
- `output`
  - 控制模板、可视化图和中间结果的输出目录
- `logging`
  - 控制控制台日志、文件日志和日志文件名

## 如何运行

当前最推荐的入口是：

- `scripts/run_mvp.py`
- `scripts/run_batch_extract.py`

分别用于：

- `scripts/run_mvp.py`
  - 处理单张指纹图
- `scripts/run_batch_extract.py`
  - 递归处理一个或多个输入目录中的全部指纹图

### 方式一：直接使用配置文件中的图片路径

先在 `configs/mvp_single.yaml` 中设置：

```yaml
input:
  single_image_path: "你的指纹图路径"
```

然后运行：

```bash
python scripts/run_mvp.py --config configs/mvp_single.yaml
```

### 方式二：命令行直接传入图片路径

```bash
python scripts/run_mvp.py --config configs/mvp_single.yaml --image "你的指纹图路径"
```

### 批处理方式一：使用配置文件中的输入目录

先在配置文件中设置：

```yaml
input:
  input_roots:
    - "你的指纹目录路径"
  recursive: true
```

然后运行：

```bash
python scripts/run_batch_extract.py --config configs/default.yaml
```

### 批处理方式二：命令行直接传入输入目录

```bash
python scripts/run_batch_extract.py --config configs/default.yaml --input "目录1" "目录2"
```

### 批处理方式三：同时覆盖输出目录

```bash
python scripts/run_batch_extract.py --config configs/default.yaml --input "你的指纹目录路径" --output "outputs/batch_run"
```

## 运行后会输出什么

默认情况下会在配置中的 `output.output_root` 下生成结果。

例如 `configs/mvp_single.yaml` 默认会输出到：

```text
outputs/mvp_single/
```

典型输出包括：

- `templates/`
  - 保存模板文件
- `visualizations/`
  - 保存关键点可视化图
- `intermediate/`
  - 保存预处理后图像
- `reports/`
  - 保存单图 summary 和批处理 `batch_summary.json`
- `logs/`
  - 保存日志文件，例如 `pipeline.log`

如果开启 debug，还会额外输出：

```text
outputs/debug/<image_id>/
```

其中会保存：

- 原图
- resize 结果
- normalize 结果
- enhanced 结果
- denoised 结果
- variance map
- 原始 ROI mask
- refined ROI mask
- background suppressed 图
- ROI crop

## 模板格式说明

当前模板模块支持两种格式：

### 1. `npz + json`

推荐默认使用。

优点：

- 数值数组紧凑
- 元信息可读
- 很适合 Python 实验和后续匹配模块接入

### 2. 目录格式

目录下包含：

- `descriptors.npy`
- `keypoints.npz`
- `metadata.json`

优点：

- 结构直观
- 便于单独检查 descriptors 和 metadata

## 当前推荐的默认参数

在没有真实数据统计前，可以先用以下基线参数：

### ROI

- `roi.method: block_variance`
- `roi.block_size: 16`
- `roi.variance_threshold: 12.0`
- `roi.morph_kernel: 3`

### SIFT

- `sift.nfeatures: 0`
- `sift.n_octave_layers: 3`
- `sift.contrast_threshold: 0.04`
- `sift.edge_threshold: 10.0`
- `sift.sigma: 1.6`
- `sift.descriptor_type: default`

### 预处理

- `preprocess.normalize.enabled: true`
- `preprocess.clahe.enabled: true`
- `preprocess.denoise.method: gaussian`
- `preprocess.background_suppression.enabled: true`

## 真实数据上最值得优先调的参数

如果你开始跑真实指纹数据，建议优先调这些：

- `roi.variance_threshold`
- `roi.block_size`
- `roi.morph_kernel`
- `preprocess.clahe.clip_limit`
- `preprocess.denoise.kernel_size`
- `sift.contrast_threshold`
- `sift.edge_threshold`
- `sift.nfeatures`

## 已知限制

当前版本有这些限制：

- 匹配模块还没有实现
- ROI 仍然是基础工程版，不是最终高鲁棒方案
- 日志系统目前以基础文件日志为主，还没有处理耗时统计和更细粒度分级
- 还没有移动端专用模板压缩方案

## 后续建议开发顺序

建议下一步按这个顺序继续补：

1. 匹配模块
2. 评估脚本
3. ROI 稳定性增强
4. 更细的日志与统计
5. 移动端压缩与迁移适配
