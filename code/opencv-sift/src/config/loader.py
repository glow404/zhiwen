"""配置文件加载与校验工具。

这个文件负责把 YAML 或 JSON 配置文件读取为 `AppConfig`，
并在进入主流程前做基础参数校验。

输入：
- 配置文件路径

输出：
- 强类型配置对象 `AppConfig`
- 可序列化的配置快照字典
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import yaml

from src.config.schema import (
    AppConfig,
    AppMetaConfig,
    BackgroundSuppressionConfig,
    CLAHEConfig,
    DebugConfig,
    DenoiseConfig,
    InputConfig,
    LoggingConfig,
    NormalizeConfig,
    OutputConfig,
    PreprocessConfig,
    ResizeConfig,
    ROIConfig,
    SIFTConfig,
    VisualizationConfig,
)
from src.exceptions import ConfigError


def load_config(config_path: Path) -> AppConfig:
    """从 YAML 或 JSON 文件加载应用配置。

    输入：
    - `config_path`：配置文件路径

    输出：
    - `AppConfig` 配置对象
    """

    path = config_path.expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    raw_data = _read_config_data(path)
    config = _build_config(raw_data)
    validate_config(config)
    return config


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """把配置对象转换为可写入 JSON 的普通字典。

    输入：
    - `config`：应用配置对象

    输出：
    - 可序列化字典，其中 `Path` 已转换为字符串
    """

    serialized = asdict(config)
    return _convert_paths_to_strings(serialized)


def validate_config(config: AppConfig) -> None:
    """对配置对象执行基础合法性校验。

    输入：
    - `config`：待校验配置

    输出：
    - 无返回值
    - 校验失败时抛出 `ConfigError`
    """

    if config.preprocess.resize.max_size <= 0:
        raise ConfigError("preprocess.resize.max_size must be positive.")
    if config.preprocess.denoise.kernel_size <= 0:
        raise ConfigError("preprocess.denoise.kernel_size must be positive.")
    if config.preprocess.denoise.kernel_size % 2 == 0:
        raise ConfigError("preprocess.denoise.kernel_size must be odd.")
    if config.preprocess.background_suppression.blur_kernel <= 0:
        raise ConfigError("preprocess.background_suppression.blur_kernel must be positive.")
    if config.preprocess.background_suppression.blur_kernel % 2 == 0:
        raise ConfigError("preprocess.background_suppression.blur_kernel must be odd.")
    if config.roi.margin < 0:
        raise ConfigError("roi.margin must be non-negative.")
    if config.roi.block_size <= 0:
        raise ConfigError("roi.block_size must be positive.")
    if config.roi.morph_kernel <= 0:
        raise ConfigError("roi.morph_kernel must be positive.")
    if config.roi.morph_kernel % 2 == 0:
        raise ConfigError("roi.morph_kernel must be odd.")
    if not 0.0 <= config.roi.min_area_ratio <= 1.0:
        raise ConfigError("roi.min_area_ratio must be between 0 and 1.")
    if config.sift.n_octave_layers <= 0:
        raise ConfigError("sift.n_octave_layers must be positive.")
    if config.sift.descriptor_type not in {"default", "float32", "uint8"}:
        raise ConfigError("sift.descriptor_type must be one of: default, float32, uint8.")
    if not config.logging.filename.strip():
        raise ConfigError("logging.filename must not be empty.")


def _read_config_data(path: Path) -> dict[str, Any]:
    """读取配置文件原始内容并解析为字典。"""

    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            with path.open("r", encoding="utf-8") as file:
                return yaml.safe_load(file) or {}
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Failed to parse config file: {path}") from exc

    raise ConfigError(f"Unsupported config format: {path.suffix}")


def _build_config(raw_data: dict[str, Any]) -> AppConfig:
    """根据原始字典构建强类型配置对象。"""

    app_data = raw_data.get("app", {})
    input_data = raw_data.get("input", {})
    output_data = raw_data.get("output", {})
    preprocess_data = raw_data.get("preprocess", {})
    roi_data = raw_data.get("roi", {})
    sift_data = raw_data.get("sift", {})
    visualization_data = raw_data.get("visualization", {})
    logging_data = raw_data.get("logging", {})

    resize_data = preprocess_data.get("resize", {})
    normalize_data = preprocess_data.get("normalize", {})
    clahe_data = preprocess_data.get("clahe", {})
    denoise_data = preprocess_data.get("denoise", {})
    background_suppression_data = preprocess_data.get("background_suppression", {})
    debug_data = preprocess_data.get("debug", {})

    return AppConfig(
        app=AppMetaConfig(**app_data),
        input=InputConfig(
            single_image_path=_to_optional_path(input_data.get("single_image_path")),
            input_roots=[Path(item) for item in input_data.get("input_roots", [])],
            recursive=input_data.get("recursive", True),
            patterns=input_data.get("patterns", []),
        ),
        output=OutputConfig(
            output_root=Path(output_data.get("output_root", "outputs/default_run")),
            template_dirname=output_data.get("template_dirname", "templates"),
            visualization_dirname=output_data.get("visualization_dirname", "visualizations"),
            intermediate_dirname=output_data.get("intermediate_dirname", "intermediate"),
            report_dirname=output_data.get("report_dirname", "reports"),
            log_dirname=output_data.get("log_dirname", "logs"),
            overwrite=output_data.get("overwrite", True),
            save_template=output_data.get("save_template", True),
            save_visualization=output_data.get("save_visualization", True),
            save_intermediate=output_data.get("save_intermediate", True),
        ),
        preprocess=PreprocessConfig(
            resize=ResizeConfig(**resize_data),
            normalize=NormalizeConfig(**normalize_data),
            clahe=CLAHEConfig(
                enabled=clahe_data.get("enabled", True),
                clip_limit=clahe_data.get("clip_limit", 2.0),
                tile_grid_size=tuple(clahe_data.get("tile_grid_size", [8, 8])),
            ),
            denoise=DenoiseConfig(**denoise_data),
            background_suppression=BackgroundSuppressionConfig(
                enabled=background_suppression_data.get("enabled", True),
                fill_value=background_suppression_data.get("fill_value", 0),
                blur_kernel=background_suppression_data.get("blur_kernel", 5),
            ),
            debug=DebugConfig(
                enabled=debug_data.get("enabled", False),
                save_dir=Path(debug_data.get("save_dir", "outputs/debug")),
                save_intermediate=debug_data.get("save_intermediate", True),
            ),
        ),
        roi=ROIConfig(**roi_data),
        sift=SIFTConfig(**sift_data),
        visualization=VisualizationConfig(**visualization_data),
        logging=LoggingConfig(
            level=logging_data.get("level", "INFO"),
            console_enabled=logging_data.get("console_enabled", True),
            file_enabled=logging_data.get("file_enabled", True),
            filename=logging_data.get("filename", "pipeline.log"),
            logger_name=logging_data.get("logger_name", "fingerprint_pipeline"),
        ),
    )


def _to_optional_path(value: Any) -> Path | None:
    """把可空值转换为 `Path`，空值则返回 `None`。"""

    if value in {None, ""}:
        return None
    return Path(value)


def _convert_paths_to_strings(data: Any) -> Any:
    """递归地把配置中的 `Path` 转换为字符串。"""

    if isinstance(data, Path):
        return data.as_posix()
    if isinstance(data, dict):
        return {key: _convert_paths_to_strings(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_convert_paths_to_strings(item) for item in data]
    return data
