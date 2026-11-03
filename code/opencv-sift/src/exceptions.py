"""项目异常类型定义。

这个文件统一管理项目中的业务异常，目的是让不同模块抛出的错误有明确语义，
便于上层流程做分级处理、日志记录和错误定位。

输入：
- 无直接输入

输出：
- 一组可在各模块中复用的异常类
"""


class AppError(Exception):
    """项目内所有自定义异常的基类。"""


class ConfigError(AppError):
    """配置加载、解析或校验失败时抛出。"""


class ImageLoadError(AppError):
    """图像读取失败或图像格式非法时抛出。"""


class PreprocessError(AppError):
    """图像预处理阶段失败时抛出。"""


class ROIExtractionError(AppError):
    """ROI 提取失败时抛出。"""


class FeatureExtractionError(AppError):
    """SIFT 等特征提取失败时抛出。"""


class TemplateStoreError(AppError):
    """模板序列化或反序列化失败时抛出。"""


class OutputWriteError(AppError):
    """输出文件写入失败时抛出。"""
