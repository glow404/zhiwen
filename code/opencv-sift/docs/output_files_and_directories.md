# 输出文件与目录说明

本文专门整理项目运行后常见输出的含义，重点说明：

- `templates/*.npz` 保存了什么
- `templates/*.mnt` 的格式是什么
- `outputs/default_run/` 下各目录分别表示什么

适用场景：

- 想知道后续分析应该优先看哪个文件
- 想区分 `.npz`、`.json`、`.mnt` 的角色
- 想理解 `visualizations/`、`reports/`、`intermediate/`、`logs/` 的用途

## 1. 输出目录整体结构

默认输出根目录通常是：

- `outputs/default_run/`

该目录下常见子目录有：

- `templates/`：模板主数据目录
- `visualizations/`：关键点可视化图
- `intermediate/`：预处理中间结果图
- `reports/`：单张图摘要和批处理汇总
- `logs/`：运行日志

## 2. `templates/` 目录

`templates/` 是最核心的输出目录。每张成功处理的图，通常会在这里生成：

- 一个 `.npz`
- 一个同名 `.json`
- 一个同名 `.mnt`

例如：

- `templates/sample.npz`
- `templates/sample.json`
- `templates/sample.mnt`

### 2.1 `templates/*.npz`

`.npz` 是模板本体，保存真正参与后续分析和匹配的数值数组。

当前默认会保存这些字段：

- `image_shape`
- `roi_bbox`
- `keypoints_x`
- `keypoints_y`
- `keypoints_size`
- `keypoints_angle`
- `keypoints_response`
- `keypoints_octave`
- `keypoints_class_id`
- `descriptors`
- `has_descriptors`

这些字段可以这样理解：

- `image_shape`：图像尺寸，通常是 `(H, W)`
- `roi_bbox`：ROI 外接矩形，格式为 `(x0, y0, x1, y1)`
- `keypoints_x`：关键点横坐标
- `keypoints_y`：关键点纵坐标
- `keypoints_size`：关键点尺度
- `keypoints_angle`：关键点主方向，单位是角度
- `keypoints_response`：关键点响应值，可用于排序或 NMS 分数
- `keypoints_octave`：关键点所在金字塔层级
- `keypoints_class_id`：兼容 OpenCV `KeyPoint` 结构的附加字段
- `descriptors`：SIFT 描述子矩阵，通常形状为 `N x 128`
- `has_descriptors`：是否存在描述子

如果你要做：

- 关键点数量统计
- 响应值分析
- 描述子匹配
- 基于响应值做 NMS

优先看 `.npz`。

### 2.2 `templates/*.json`

这个文件不是模板本体，而是模板说明书，主要记录：

- 原图路径
- 图像尺寸
- ROI 边界框
- 预处理摘要
- 本次运行的配置快照
- 模板存储格式

如果你要确认：

- 这张图从哪里来的
- 本次用的是什么参数
- ROI 是多少
- 是否启用了某个配置

优先看同名 `.json`。

### 2.3 `templates/*.mnt`

`.mnt` 是文本格式导出，适合和只接受细节点文本的工具对接，也常用于后处理脚本。

格式如下：

```text
图像名
N H W
x y o
x y o
...
```

含义如下：

- 第 1 行：图像名，即 `image_id`
- 第 2 行：`N H W`
- 第 3 行开始：每行一个点，格式为 `x y o`

其中：

- `N`：点数量
- `H`：图像高
- `W`：图像宽
- `x`：列坐标
- `y`：行坐标
- `o`：方向，单位是弧度

它和 `.npz` 的关系是：

- `x` 对应 `.npz` 里的 `keypoints_x`
- `y` 对应 `.npz` 里的 `keypoints_y`
- `o` 对应 `.npz` 里的 `keypoints_angle`，但会从角度转换为弧度

需要注意：

- `.mnt` 中坐标通常已做四舍五入，变成整数像素
- `.mnt` 不保存 `response`
- `.mnt` 不保存 `size`
- `.mnt` 不保存 `descriptors`

所以 `.mnt` 更适合做轻量文本对接，不适合替代 `.npz`。

## 3. `visualizations/` 目录

这个目录保存关键点可视化图，例如：

- `visualizations/sample_keypoints.png`

它不是原图本身，而是：

- 预处理后的灰度图
- 叠加关键点绘制结果
- 可选叠加 ROI 蓝色边框

主要用途是人工检查，例如：

- 关键点是否太密
- 关键点是否靠近边缘
- 关键点是否主要落在有效脊线区域
- ROI 是否合理

## 4. `intermediate/` 目录

这个目录保存预处理后的中间图，当前常见的是：

- `*_preprocessed.png`

它反映的是送入特征提取阶段之前的图像结果，通常已经做过：

- 归一化
- 对比度增强
- 去噪
- ROI 背景抑制

如果你怀疑关键点过多、过少、过于靠边，先看这里通常最有帮助，因为它能反映 SIFT 实际看到的输入。

## 5. `reports/` 目录

这个目录保存结果摘要，不保存大数组。

常见文件有：

- `*_summary.json`：单张图处理摘要
- `batch_summary.json`：整批任务汇总

### 5.1 `reports/*_summary.json`

用于快速确认单张图处理是否成功，常见字段包括：

- `status`
- `image_path`
- `foreground_ratio`
- `keypoint_count`
- `descriptor_dim`
- `template_path`
- `mnt_path`
- `visualization_path`
- `error_message`

如果你只想知道：

- 这张图有没有跑成功
- 提取了多少关键点
- 结果文件保存到了哪里

优先看这个文件。

### 5.2 `reports/batch_summary.json`

这是批处理级别的总汇总，适合查看：

- 一共处理了多少张图
- 成功多少、失败多少
- 每张图对应的关键点数量
- 每张图的输出路径

适合做批量统计和排查异常样本。

## 6. `logs/` 目录

这个目录保存运行日志，例如：

- `logs/pipeline.log`

主要用于排查：

- 某张图处理失败的原因
- 某次批处理是否中途报错
- 每张图最终提取了多少关键点

## 7. 默认输出格式与可选格式

当前默认模板格式是：

- `npz_json`

即：

- 数值数组保存在 `.npz`
- 元信息保存在同名 `.json`
- 同时额外导出一个 `.mnt`

代码里还支持一种目录格式：

- `directory`

该格式会保存为一个目录，内部通常包含：

- `descriptors.npy`
- `keypoints.npz`
- `metadata.json`

但当前默认批处理流程主要使用的仍然是 `.npz + .json + .mnt`。

## 8. 什么时候看哪个文件

- 想拿到真正的关键点和描述子：看 `templates/*.npz`
- 想做人可读的文本格式对接：看 `templates/*.mnt`
- 想确认来源、ROI 和参数：看 `templates/*.json`
- 想直观看关键点分布：看 `visualizations/*.png`
- 想看预处理后送入 SIFT 的输入：看 `intermediate/*.png`
- 想快速确认单张图成功与否：看 `reports/*_summary.json`
- 想看整批结果统计：看 `reports/batch_summary.json`
- 想排查报错和运行过程：看 `logs/*.log`
