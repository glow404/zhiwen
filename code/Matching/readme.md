to do
1.将工具函数整理成包

## 核心功能模块

### 1. 主程序入口
- **`main_total.py`** - 完整匹配系统入口，实现数据库构建和批量匹配
- **`main_single.py`** - 单个指纹匹配和批量匹配功能
- **`main_calc_fmr.py`** - 计算误匹配率(FMR)和误拒率(FNMR)

### 2. 数据库管理
- **`build_db.py`** - 构建FVC2002指纹数据库，构建指纹细节点模板数据库
- **`build_extended_db.py`** - 构建扩展数据库（包含自定义图像）
- **`rebuild_database.py`** - 重新构建数据库工具

### 3. 核心算法
- **`ext_finger.py`** - 指纹特征提取（细节点提取）
- **`match.py`** - 细节点匹配算法
- **`f_enhance.py`** - 指纹图像增强
- **`transform.py`** - 坐标变换算法

调用关系
main_total.py
├── build_db.py (构建数据库)
├── ext_finger.py (特征提取)
└── match.py (核心匹配)

main_single.py
├── ext_finger.py (特征提取)
└── match.py (核心匹配)

### 4. 特征提取模块 (FExtraction/)
这个目录包含了完整的指纹图像处理算法链：
- **图像预处理**: `normalise.py` (归一化)
- **方向场计算**: `ridgeorient.py`, `smoothen_orientation_image.py`
- **频率估计**: `freqest.py`, `ridgefreq.py`, `smoothen_frequency_image.py`
- **脊线增强**: `ridgefilter.py`, `fft_enhance_cubs.py`
- **脊线分割**: `ridgesegment.py`
- **辅助工具**: `dist2.py`, `angular_filter_bank.py`, `compute_coherence.py`

## 数据结构

### 1. 数据库文件
- **`db.pkl`** - Python pickle格式的指纹模板数据库
- **`fmr.mat`**, **`fnmr.mat`** - 性能评估数据

### 2. 自定义图像目录
- **`temp/`** - 用于存放用户自定义指纹图像



## 1. `build_db.py` - 指纹模板数据库构建

### 主要功能
**构建指纹细节点模板数据库**，将原始指纹图像转换为可用于匹配的细节点特征数据。

### 输出结果
- **文件**: `db.pkl` (Python pickle格式)
- **格式**: `{模板ID: 细节点数组}`
- **示例**: `{1: array([[x1,y1,θ1,type1], [x2,y2,θ2,type2], ...])}`

## 2. `build_fmr.py` - 性能评估数据生成

### 主要功能
**生成FMR（False Match Rate）和FNMR（False Non-Match Rate）数据**，用于评估指纹识别系统的性能。

3. **性能指标计算**
   - **真阳性**: 同一指纹的不同图像间的匹配
   - **假阳性**: 不同指纹间的错误匹配
   - **FMR**: 假阳性率 = 假匹配数 / 总匹配数
   - **FNMR**: 假阴性率 = 未匹配的真匹配数 / 真匹配总数

### 输出结果
- **FMR数据**: `fmr.npy`, `fmr.pkl` (100×72数组，100个阈值×72个模板)
- **FNMR数据**: `fnmr.npy`, `fnmr.pkl`
- **用途**: 用于绘制ROC曲线，评估系统性能

