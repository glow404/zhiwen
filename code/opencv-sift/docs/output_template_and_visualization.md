# 输出模板与可视化说明

本文说明当前项目运行后生成的模板文件结构，以及 `visualizations/*.png` 应该如何理解。

适用输出目录：

- `outputs/default_run/templates/`
- `outputs/default_run/reports/`
- `outputs/default_run/visualizations/`

示例样本：

- 模板：`outputs/default_run/templates/wi-Cycle=2Freq=41Rgd=1237-0-20260318171224.303782_pair0.npz`
- 元信息：`outputs/default_run/templates/wi-Cycle=2Freq=41Rgd=1237-0-20260318171224.303782_pair0.json`
- 摘要：`outputs/default_run/reports/wi-Cycle=2Freq=41Rgd=1237-0-20260318171224.303782_pair0_summary.json`
- 可视化：`outputs/default_run/visualizations/wi-Cycle=2Freq=41Rgd=1237-0-20260318171224.303782_pair0_keypoints.png`

## 1. 输出文件各自保存什么

一次成功处理通常会生成三类结果：

当前版本下，如果启用了模板保存，还会额外生成一个与 `.npz` 同名的 `.mnt` 文本文件。

### 1.1 `templates/*.npz`

保存可用于后续匹配和分析的数值数组，是最核心的模板数据。

当前默认保存的字段如下：

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

其中：

- `keypoints_x` 和 `keypoints_y` 表示每个关键点的坐标
- `keypoints_size` 表示关键点尺度
- `keypoints_angle` 表示关键点主方向
- `keypoints_response` 表示关键点响应强度
- `keypoints_octave` 表示金字塔层级信息
- `keypoints_class_id` 当前通常没有特殊业务含义，主要用于兼容 OpenCV 关键点结构
- `descriptors` 是 SIFT 描述子矩阵，形状通常为 `N x 128`

这里的 `N` 就是关键点数量。

以本示例样本为例，摘要文件显示：

- `keypoint_count = 491`
- `descriptor_dim = 128`

因此该样本的描述子矩阵通常可以理解为：

```text
descriptors.shape = (491, 128)
```

### 1.2 `templates/*.json`

保存模板的元信息，不保存大数组本体，主要用于：

- 记录图片来源
- 记录图像尺寸和 ROI
- 记录预处理摘要
- 记录本次运行使用的配置快照
- 说明模板采用什么存储格式

示例结构如下：

```json
{
  "template_version": "v0.1",
  "image_id": "wi-Cycle=2Freq=41Rgd=1237-0-20260318171224.303782_pair0",
  "image_path": "C:/.../pair0.bmp",
  "image_shape": [110, 100],
  "roi_bbox": [0, 0, 100, 110],
  "preprocess_metadata": {
    "foreground_ratio": 1.0,
    "roi_bbox": [0, 0, 100, 110],
    "debug_output_dir": null,
    "debug_image_paths": {},
    "artifacts": {
      "input_shape": [110, 100],
      "resized_shape": [110, 100],
      "normalized_shape": [110, 100],
      "enhanced_shape": [110, 100],
      "denoised_shape": [110, 100],
      "has_variance_map": true,
      "has_roi_mask_raw": true,
      "has_roi_mask_refined": true,
      "has_background_suppressed_image": true
    }
  },
  "timestamp": "2026-04-08T15:21:41.507296+00:00",
  "config_snapshot": {
    "...": "本次运行的完整配置"
  },
  "descriptor_dtype": "float32",
  "has_descriptors": true,
  "storage_format": "npz_json"
}
```

可以把这个文件理解为“模板说明书”，而不是模板本体。

### 1.3 `templates/*.mnt`

保存与关键点一一对应的文本格式导出，用于和只接受文本细节点格式的工具对接。

格式如下：

```text
图像名
N H W
x y o
x y o
...
```

字段含义：

- 第 1 行：图像名，即 `image_id`
- 第 2 行：`N H W`
- 第 3 行开始：每行一个关键点，格式为 `x y o`

其中：

- `x` 是列坐标，对应 `.npz` 中的 `keypoints_x`
- `y` 是行坐标，对应 `.npz` 中的 `keypoints_y`
- `o` 是方向，单位为弧度，对应 `.npz` 中 `keypoints_angle` 从角度转为弧度后的值

注意：

- `.npz` 中的关键点坐标是浮点数，`.mnt` 中会四舍五入到整数像素坐标
- `.mnt` 中的方向单位是弧度，不是 OpenCV 原始的角度值

### 1.4 `reports/*_summary.json`

保存单张图的处理结果摘要，方便快速确认这张图是否成功处理。

常见字段包括：

- `status`
- `image_path`
- `foreground_ratio`
- `keypoint_count`
- `descriptor_dim`
- `template_path`
- `visualization_path`
- `error_message`

如果你只是想快速看某张图是否提取成功、提取了多少关键点，优先看这个文件即可。

## 2. `.npz` 里的关键点如何理解

项目内部会先把 OpenCV 的 `KeyPoint` 列表转换为结构化数组，再写入模板。

因此，`.npz` 里的第 `i` 个关键点可以这样对应理解：

- 坐标：`(keypoints_x[i], keypoints_y[i])`
- 尺度：`keypoints_size[i]`
- 方向：`keypoints_angle[i]`
- 响应值：`keypoints_response[i]`
- 层级：`keypoints_octave[i]`
- 描述子：`descriptors[i]`

也就是说：

- 一行描述子对应一个关键点
- 每个关键点对应一个 128 维 SIFT 向量

## 4. `visualizations/*.png` 是什么

`visualizations/*.png` 不是原图本身，而是“预处理后的图像 + 关键点绘制结果 + 可选 ROI 框”的总览图。

当前绘制逻辑是：

1. 在预处理后的灰度图上绘制 SIFT 关键点
2. 如果配置允许，再叠加 ROI 矩形框

其中：

- 绿色圆圈表示检测到的 SIFT 关键点
- 圆圈中心表示关键点位置
- 圆圈大小大致反映关键点尺度
- 圆圈上的方向标记反映关键点主方向
- 蓝色矩形框表示 ROI 边界框

## 5. 如何理解你这张示例可视化图

对应文件：

- `outputs/default_run/visualizations/wi-Cycle=2Freq=41Rgd=1237-0-20260318171224.303782_pair0_keypoints.png`

这张图可以这样理解：

- 背景灰度图不是原始输入图，而是经过归一化、增强、去噪、ROI 处理后的结果
- 图上的绿色标记表示 SIFT 认为“局部纹理足够稳定、适合描述和匹配”的位置
- 如果某一区域绿色标记更多，通常说明该区域局部纹理更丰富
- 如果某一区域几乎没有关键点，常见原因是该处过平滑、纹理弱、对比度低，或被背景抑制

结合本样本摘要：

- `keypoint_count = 491`
- `foreground_ratio = 1.0`
- `roi_bbox = [0, 0, 100, 110]`

这说明：

- 当前这张图提取到了 491 个关键点
- ROI 覆盖了整张图
- 因此蓝色 ROI 框基本贴着整张图边界，更多是在说明“整张图都被当作前景区域参与了特征提取”

## 6. 看可视化图时要注意什么

- 关键点多，不一定代表一定更好，还要看是否落在有效脊线纹理上
- 关键点过少，后续匹配通常会不稳定
- 关键点如果大量集中在噪声、污点、边缘伪纹理上，匹配质量也可能不好
- `visualizations/*.png` 更适合做人工质检，不适合替代模板数据本身
- 真正用于匹配和后续分析的核心数据仍然是 `templates/*.npz`

## 7. 什么时候看哪个文件

- 想快速看是否成功：看 `reports/*_summary.json`
- 想看关键点画在图上的效果：看 `visualizations/*.png`
- 想拿到关键点和描述子做分析或匹配：看 `templates/*.npz`
- 想和只接收文本关键点格式的工具对接：看 `templates/*.mnt`
- 想确认本次样本使用了什么配置和 ROI：看 `templates/*.json`
