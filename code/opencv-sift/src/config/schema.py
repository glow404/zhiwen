"""项目配置数据结构定义。

这个文件把配置文件中的各个分区映射为 dataclass，便于代码中以强类型方式访问配置。

输入：
- 由配置加载器解析出的字典数据

输出：
- `AppConfig` 及其子配置对象
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ResizeConfig:
    """控制预处理前是否对图像进行缩放。"""

    enabled: bool = False
    max_size: int = 1024


@dataclass(slots=True)
class NormalizeConfig:
    """控制灰度归一化参数。"""

    enabled: bool = True
    mean_target: float = 128.0
    std_target: float = 50.0


@dataclass(slots=True)
class CLAHEConfig:
    """控制 CLAHE 局部直方图均衡参数。"""

    enabled: bool = True
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)


@dataclass(slots=True)
class DenoiseConfig:
    """控制去噪方式与对应参数。"""

    method: str = "gaussian"
    kernel_size: int = 3
    sigma: float = 0.0


@dataclass(slots=True)
class BackgroundSuppressionConfig:
    """控制背景抑制行为。"""

    enabled: bool = True
    fill_value: int = 0
    blur_kernel: int = 5


@dataclass(slots=True)
class DebugConfig:
    """控制预处理阶段的调试输出行为。"""

    enabled: bool = False
    save_dir: Path = Path("outputs/debug")
    save_intermediate: bool = True


@dataclass(slots=True)
class PreprocessConfig:
    """聚合预处理阶段的全部参数。"""

    resize: ResizeConfig = field(default_factory=ResizeConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    clahe: CLAHEConfig = field(default_factory=CLAHEConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    background_suppression: BackgroundSuppressionConfig = field(
        default_factory=BackgroundSuppressionConfig
    )
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass(slots=True)
class ROIConfig:
    """控制 ROI 提取阶段的参数。"""

    enabled: bool = True
    method: str = "block_variance"
    margin: int = 0
    block_size: int = 16
    variance_threshold: float = 12.0
    morph_kernel: int = 3
    close_iters: int = 1
    open_iters: int = 1
    min_area_ratio: float = 0.05
    use_largest_component: bool = True


@dataclass(slots=True)
class SIFTConfig:
    """控制 SIFT 关键点检测与描述子提取参数。"""

    nfeatures: int = 0
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6
    descriptor_type: str = "default"


@dataclass(slots=True)
class VisualizationConfig:
    """控制关键点可视化输出行为。"""

    enabled: bool = True
    draw_rich_keypoints: bool = True
    draw_roi_bbox: bool = True


@dataclass(slots=True)
class OutputConfig:
    """控制输出目录结构和结果保存开关。"""

    output_root: Path = Path("outputs/default_run")
    template_dirname: str = "templates"
    visualization_dirname: str = "visualizations"
    intermediate_dirname: str = "intermediate"
    report_dirname: str = "reports"
    log_dirname: str = "logs"
    overwrite: bool = True
    save_template: bool = True
    save_visualization: bool = True
    save_intermediate: bool = True


@dataclass(slots=True)
class InputConfig:
    """控制单图输入和目录输入相关参数。"""

    single_image_path: Path | None = None
    input_roots: list[Path] = field(default_factory=list)
    recursive: bool = True
    patterns: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LoggingConfig:
    """控制日志等级等运行时日志参数。"""

    level: str = "INFO"
    console_enabled: bool = True
    file_enabled: bool = True
    filename: str = "pipeline.log"
    logger_name: str = "fingerprint_pipeline"


@dataclass(slots=True)
class AppMetaConfig:
    """保存项目级元信息，例如项目名和模板版本。"""

    name: str = "opencv_sift_fingerprint"
    template_version: str = "v0.1"


@dataclass(slots=True)
class AppConfig:
    """顶层应用配置对象。

    输出字段：
    - `app`：项目元信息
    - `input`：输入参数
    - `output`：输出参数
    - `preprocess`：预处理参数
    - `roi`：ROI 参数
    - `sift`：SIFT 参数
    - `visualization`：可视化参数
    - `logging`：日志参数
    """

    app: AppMetaConfig = field(default_factory=AppMetaConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    sift: SIFTConfig = field(default_factory=SIFTConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
