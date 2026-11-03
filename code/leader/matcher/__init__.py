"""`matcher` 包的统一导出入口。

这个文件的作用是把核心数据结构和常用函数集中导出，
这样外部脚本只需从 `matcher` 包导入即可，不必关心内部文件拆分细节。
"""

from matcher.core import MatchParameters, MatchResult, match_minutiae
from matcher.eval import SearchOptions, evaluate_identity_splits, run_full_evaluation
from matcher.fusion import FusionConfig, fuse_scores, score_template_set, select_templates
from matcher.io import IdentitySplit, Minutia, SampleRecord, build_dataset_index, build_identity_splits

# 统一暴露给包外使用的公共符号。
__all__ = [
    "FusionConfig",
    "IdentitySplit",
    "MatchParameters",
    "MatchResult",
    "Minutia",
    "SampleRecord",
    "SearchOptions",
    "build_dataset_index",
    "build_identity_splits",
    "evaluate_identity_splits",
    "fuse_scores",
    "match_minutiae",
    "run_full_evaluation",
    "score_template_set",
    "select_templates",
]
