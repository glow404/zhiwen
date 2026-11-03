# 项目运行命令

这份文档只整理实际会用到的运行命令，按从“能跑起来”到“常用处理流程”的顺序写。

## 1. 进入项目目录

先进入项目根目录：

```powershell
cd C:\Users\ZYH\Desktop\research\zhiwen\code\todo\opencv-sift
```

## 2. 创建并激活虚拟环境

推荐先创建虚拟环境，再安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. 安装依赖

先安装项目依赖：

```powershell
pip install -r requirements.txt
```

再执行一次开发安装：

```powershell
pip install -e .
```

说明：

- `pip install -r requirements.txt` 用来安装第三方依赖。
- `pip install -e .` 用来让当前环境正确识别 `src` 包，避免出现 `ModuleNotFoundError: No module named 'src'`。

## 4. 运行单张图片

### 方式一：使用配置文件里的图片路径

先在 `configs/mvp_single.yaml` 中设置：

```yaml
input:
  single_image_path: "你的图片路径"
```

然后运行：

```powershell
python scripts/run_mvp.py --config configs/mvp_single.yaml
```

### 方式二：命令行直接传图片路径

```powershell
python scripts/run_mvp.py --config configs/mvp_single.yaml --image "你的图片路径"
```

适用场景：

- 只想快速验证一张图能不能提取特征。
- 想检查单张图的模板、可视化和 summary 输出。

## 5. 运行批处理

### 方式一：直接使用默认配置

`configs/default.yaml` 里已经写好了输入目录时，直接运行：

```powershell
python scripts/run_batch_extract.py --config configs/default.yaml
```

### 方式二：命令行覆盖输入目录

```powershell
python scripts/run_batch_extract.py --config configs/default.yaml --input "目录1" "目录2"
```

### 方式三：命令行同时覆盖输入目录和输出目录

```powershell
python scripts/run_batch_extract.py --config configs/default.yaml --input "你的指纹目录路径" --output "outputs/batch_run"
```

适用场景：

- 需要递归处理整个数据目录。
- 不想反复改配置文件，直接在命令行切换输入和输出路径。

## 6. 对 `.npz` 模板关键点做 NMS 筛选

这个脚本会读取现有 `templates` 目录里的 `.npz` 模板，按极大值抑制筛选后重新输出筛选后的 `.npz`，并自动派生同名 `.mnt` 和可视化图。

### 处理单个 `.npz`

```powershell
python scripts/filter_mnt_nms.py --input "outputs/default_run/templates/xxx.npz"
```

### 处理整个模板目录

```powershell
python scripts/filter_mnt_nms.py --input "outputs/default_run/templates"
```

### 指定输出目录

```powershell
python scripts/filter_mnt_nms.py --input "outputs/default_run/templates" --output "outputs/default_run/templates_nms"
```

### 调整 NMS 半径和方向阈值

```powershell
python scripts/filter_mnt_nms.py --input "outputs/default_run/templates" --radius 10 --angle-threshold-deg 15
```

## 7. 查看输出结果

默认输出目录通常是：

- 单图：`outputs/mvp_single/`
- 批处理：`outputs/default_run/`

常见结果文件：

- `templates/`：模板文件，包含 `.npz`、`.json`、`.mnt`
- `visualizations/`：关键点可视化图
- `intermediate/`：预处理后的图像
- `reports/`：单图 summary 或批处理 `batch_summary.json`
- `logs/`：日志文件

## 8. 常见问题

### 报错：`ModuleNotFoundError: No module named 'src'`

先执行：

```powershell
pip install -e .
```

