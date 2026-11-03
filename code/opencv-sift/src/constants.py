"""项目全局常量定义。

这个文件集中放置跨模块共用、且语义稳定的常量，避免把默认值散落在各处。

输入：
- 无直接运行输入

输出：
- 供其他模块导入使用的常量，例如支持的图片后缀、默认配置路径、默认模板版本
"""

from pathlib import Path


SUPPORTED_IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_TEMPLATE_VERSION = "v0.1"
