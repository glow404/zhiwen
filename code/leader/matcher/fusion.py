"""`matcher.fusion` 负责多模板融合。

这个文件不关心底层如何做细节点配对，而是专注于两个问题：
1. 从注册样本池中挑选若干张模板。
2. 把多张模板分别得到的匹配分数聚合成一个最终分数。

多模板的作用是降低单次采集波动带来的影响，提高验证的稳定性。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

from matcher.core import MatchParameters, match_minutiae
from matcher.io import SampleRecord


@dataclass(frozen=True)
class FusionConfig:
    """多模板融合配置。"""

    template_count: int
    fusion_method: str


def select_templates(enrollment_pool: tuple[SampleRecord, ...], template_count: int) -> tuple[SampleRecord, ...]:
    """从注册池中选出模板。

    当前采用质量代理分数排序，优先选更稳定、更完整的样本。
    """

    if template_count <= 0:
        raise ValueError("模板数必须大于 0")
    ordered = sorted(
        enrollment_pool,
        key=lambda item: (-item.quality_proxy, item.sample_index, str(item.path)),
    )
    return tuple(ordered[: min(template_count, len(ordered))])


def fuse_scores(scores: list[float], method: str) -> float:
    """将多模板分数融合为一个总分。

    不同融合方式代表不同的风险偏好：
    - `max`：激进，容易放大某一次高分。
    - `mean` / `median`：较稳健。
    - `harmonic_mean`：更保守，低分模板会明显拉低结果。
    """

    if not scores:
        return 0.0
    method_normalized = method.lower()
    if method_normalized == "max":
        return max(scores)
    if method_normalized == "mean":
        return statistics.fmean(scores)
    if method_normalized == "median":
        return statistics.median(scores)
    if method_normalized == "trimmed_mean":
        # 去掉极端高低分，减弱偶发异常模板的影响。
        ordered = sorted(scores)
        if len(ordered) <= 2:
            return statistics.fmean(ordered)
        return statistics.fmean(ordered[1:-1])
    if method_normalized == "harmonic_mean":
        # 调和平均更强调“短板”，适合低 FAR 取向的保守融合。
        adjusted = [max(score, 1e-6) for score in scores]
        return len(adjusted) / sum(1.0 / score for score in adjusted)
    raise ValueError(f"未知融合方式: {method}")


def score_template_set(
    templates: tuple[SampleRecord, ...],
    probe: SampleRecord,
    matcher_params: MatchParameters,
    fusion_method: str,
    score_provider: Callable[[SampleRecord, SampleRecord, MatchParameters], float] | None = None,
) -> tuple[float, list[float]]:
    """计算“多模板 vs 单个探测样本”的融合得分。

    返回值包含：
    - 融合后的总分。
    - 每个模板单独的匹配分数，便于后续分析。
    """

    individual_scores: list[float] = []
    for template in templates:
        # `score_provider` 允许上层传入缓存版本，避免重复计算。
        if score_provider is None:
            score = match_minutiae(template.minutiae, probe.minutiae, matcher_params).score
        else:
            score = score_provider(template, probe, matcher_params)
        individual_scores.append(score)
    return fuse_scores(individual_scores, fusion_method), individual_scores
