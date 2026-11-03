"""`matcher.core` 负责单模板细节点匹配。

这个文件实现的是整套算法的核心打分过程，主要包括：
1. 从模板和待测样本中选出更稳定的锚点候选。
2. 基于锚点估计旋转和平移，把待测样本对齐到模板坐标系。
3. 在对齐后做一对一细节点配对，并统计几何一致性。
4. 把匹配数、覆盖率、距离误差和方向误差融合成最终相似度分数。

这里不负责多模板融合与阈值搜索，只负责“单个模板 vs 单个探测样本”的匹配。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from matcher.io import Minutia, angle_distance, wrap_angle


@dataclass(frozen=True)
class MatchParameters:
    """单模板匹配参数。

    这些参数会在开发集上被自动搜索，用来平衡匹配严格度与容错能力。
    """

    distance_tolerance: float = 12.0
    angle_tolerance: float = 0.50
    min_matches: int = 6
    local_radius: float = 36.0
    anchor_pool_size: int = 12
    max_anchor_pairs: int = 96

    def as_key(self) -> tuple[float, float, int, float, int, int]:
        """返回适合做缓存键的参数元组。"""

        return (
            self.distance_tolerance,
            self.angle_tolerance,
            self.min_matches,
            self.local_radius,
            self.anchor_pool_size,
            self.max_anchor_pairs,
        )


@dataclass(frozen=True)
class MatchedPair:
    """一对成功匹配的细节点及其误差。"""

    template_index: int
    probe_index: int
    distance_error: float
    angle_error: float


@dataclass(frozen=True)
class MatchResult:
    """单模板匹配输出结果。"""

    score: float
    matched_count: int
    coverage: float
    mean_distance_error: float
    mean_angle_error: float
    anchor_template_index: int
    anchor_probe_index: int
    pairs: tuple[MatchedPair, ...]


def match_minutiae(
    template_minutiae: Iterable[Minutia],
    probe_minutiae: Iterable[Minutia],
    params: MatchParameters,
) -> MatchResult:
    """执行单模板细节点匹配。

    流程概括：
    - 先从模板和探测样本中各选一批锚点候选。
    - 遍历锚点对，尝试做旋转对齐。
    - 对每种对齐结果执行一对一配对。
    - 取分数最高的对齐结果作为最终匹配结果。
    """

    template = tuple(template_minutiae)
    probe = tuple(probe_minutiae)
    if len(template) < params.min_matches or len(probe) < params.min_matches:
        return _empty_result()

    # 局部支持度越高的点，通常越适合作为对齐锚点。
    template_strength = _local_strengths(template, params.local_radius)
    probe_strength = _local_strengths(probe, params.local_radius)
    template_anchors = _select_anchor_indices(template, template_strength, params.anchor_pool_size)
    probe_anchors = _select_anchor_indices(probe, probe_strength, params.anchor_pool_size)

    best_result = _empty_result()
    anchor_pairs_tested = 0
    for template_anchor in template_anchors:
        template_kind = template[template_anchor].kind
        for probe_anchor in probe_anchors:
            # 细节点类型不一致时，通常没有继续尝试对齐的必要。
            if probe[probe_anchor].kind != template_kind:
                continue
            anchor_pairs_tested += 1
            result = _evaluate_anchor_alignment(
                template,
                probe,
                template_anchor,
                probe_anchor,
                params,
            )
            if result.score > best_result.score:
                best_result = result
            if anchor_pairs_tested >= params.max_anchor_pairs:
                return best_result

    return best_result


def _empty_result() -> MatchResult:
    """返回统一的空匹配结果。"""

    return MatchResult(
        score=0.0,
        matched_count=0,
        coverage=0.0,
        mean_distance_error=math.inf,
        mean_angle_error=math.inf,
        anchor_template_index=-1,
        anchor_probe_index=-1,
        pairs=(),
    )


def _local_strengths(minutiae: tuple[Minutia, ...], radius: float) -> list[float]:
    """估计每个细节点的局部支撑强度。

    思路是统计某个点在给定半径内拥有多少邻域支持。
    邻域越丰富、同类型点越多，这个点作为锚点通常越稳。
    """

    strengths: list[float] = []
    for index, source in enumerate(minutiae):
        support = 0.0
        for neighbor_index, neighbor in enumerate(minutiae):
            if index == neighbor_index:
                continue
            distance = math.hypot(source.x - neighbor.x, source.y - neighbor.y)
            if distance > radius:
                continue
            weight = 1.0 - min(distance / radius, 1.0)
            if source.kind == neighbor.kind:
                weight += 0.25
            support += weight
        strengths.append(support)
    return strengths


def _select_anchor_indices(
    minutiae: tuple[Minutia, ...],
    strengths: list[float],
    pool_size: int,
) -> list[int]:
    """选择一小批锚点候选。

    先按局部支撑强度排序，再用与中心距离作次级排序，
    尽量优先取“稳定且不过分偏边缘”的细节点。
    """

    if not minutiae:
        return []
    center_x = sum(item.x for item in minutiae) / len(minutiae)
    center_y = sum(item.y for item in minutiae) / len(minutiae)
    ranked = sorted(
        range(len(minutiae)),
        key=lambda idx: (
            -strengths[idx],
            math.hypot(minutiae[idx].x - center_x, minutiae[idx].y - center_y),
            idx,
        ),
    )
    return ranked[: min(pool_size, len(ranked))]


def _evaluate_anchor_alignment(
    template: tuple[Minutia, ...],
    probe: tuple[Minutia, ...],
    template_anchor_index: int,
    probe_anchor_index: int,
    params: MatchParameters,
) -> MatchResult:
    """评估某一组锚点假设下的对齐质量。

    当前策略：
    - 用锚点方向差估计旋转量。
    - 先做一次粗对齐并配对。
    - 再根据已匹配对估计平均平移，做一次细化。
    - 使用细化后的最佳结果计算最终分数。
    """

    template_anchor = template[template_anchor_index]
    probe_anchor = probe[probe_anchor_index]
    rotation = wrap_angle(template_anchor.angle - probe_anchor.angle)

    transformed_probe = _transform_probe(probe, probe_anchor, template_anchor, rotation)
    pairs = _build_greedy_pairs(template, transformed_probe, params)
    if len(pairs) < params.min_matches:
        return _empty_result()

    refined_probe = _refine_translation(template, transformed_probe, pairs)
    refined_pairs = _build_greedy_pairs(template, refined_probe, params)
    if len(refined_pairs) > len(pairs):
        pairs = refined_pairs
        transformed_probe = refined_probe

    if len(pairs) < params.min_matches:
        return _empty_result()

    # 分数同时考虑匹配数量、误差大小和匹配覆盖率。
    distances = [pair.distance_error for pair in pairs]
    angles = [pair.angle_error for pair in pairs]
    coverage = len(pairs) / max(1, min(len(template), len(probe)))
    distance_quality = math.exp(-sum(distances) / (len(distances) * max(params.distance_tolerance, 1e-6)))
    angle_quality = math.exp(-sum(angles) / (len(angles) * max(params.angle_tolerance, 1e-6)))
    support_quality = len(pairs) / max(len(pairs) + 2.0, params.min_matches)
    score = coverage * distance_quality * angle_quality * support_quality

    return MatchResult(
        score=score,
        matched_count=len(pairs),
        coverage=coverage,
        mean_distance_error=sum(distances) / len(distances),
        mean_angle_error=sum(angles) / len(angles),
        anchor_template_index=template_anchor_index,
        anchor_probe_index=probe_anchor_index,
        pairs=tuple(pairs),
    )


def _transform_probe(
    probe: tuple[Minutia, ...],
    probe_anchor: Minutia,
    template_anchor: Minutia,
    rotation: float,
) -> list[Minutia]:
    """把探测样本绕锚点旋转并平移到模板坐标系。"""

    cos_value = math.cos(rotation)
    sin_value = math.sin(rotation)
    transformed: list[Minutia] = []
    for minutia in probe:
        # 先将坐标平移到探测锚点局部坐标系，再做旋转，最后对齐到模板锚点。
        offset_x = minutia.x - probe_anchor.x
        offset_y = minutia.y - probe_anchor.y
        rotated_x = offset_x * cos_value - offset_y * sin_value
        rotated_y = offset_x * sin_value + offset_y * cos_value
        transformed.append(
            Minutia(
                x=template_anchor.x + rotated_x,
                y=template_anchor.y + rotated_y,
                angle=wrap_angle(minutia.angle + rotation),
                kind=minutia.kind,
            )
        )
    return transformed


def _refine_translation(
    template: tuple[Minutia, ...],
    transformed_probe: list[Minutia],
    pairs: list[MatchedPair],
) -> list[Minutia]:
    """根据已匹配对估计一个更精确的平均平移量。"""

    if not pairs:
        return transformed_probe
    dx = sum(template[pair.template_index].x - transformed_probe[pair.probe_index].x for pair in pairs) / len(pairs)
    dy = sum(template[pair.template_index].y - transformed_probe[pair.probe_index].y for pair in pairs) / len(pairs)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return transformed_probe
    return [
        Minutia(x=item.x + dx, y=item.y + dy, angle=item.angle, kind=item.kind)
        for item in transformed_probe
    ]


def _build_greedy_pairs(
    template: tuple[Minutia, ...],
    transformed_probe: list[Minutia],
    params: MatchParameters,
) -> list[MatchedPair]:
    """构建一对一贪心匹配。

    对所有满足约束的候选对先计算代价，再按代价升序选择，
    每个模板点和探测点都只允许参与一次匹配。
    """

    candidates: list[MatchedPair] = []
    for template_index, template_minutia in enumerate(template):
        for probe_index, probe_minutia in enumerate(transformed_probe):
            if template_minutia.kind != probe_minutia.kind:
                continue
            distance_error = math.hypot(template_minutia.x - probe_minutia.x, template_minutia.y - probe_minutia.y)
            if distance_error > params.distance_tolerance:
                continue
            # 部分细节点方向存在 180 度对称歧义，因此同时比较原方向与翻转方向。
            angle_error = angle_distance(template_minutia.angle, probe_minutia.angle)
            flipped_angle_error = angle_distance(
                template_minutia.angle,
                wrap_angle(probe_minutia.angle + math.pi),
            )
            effective_angle_error = min(angle_error, flipped_angle_error)
            if effective_angle_error > params.angle_tolerance:
                continue
            candidates.append(
                MatchedPair(
                    template_index=template_index,
                    probe_index=probe_index,
                    distance_error=distance_error,
                    angle_error=effective_angle_error,
                )
            )

    # 距离误差优先，方向误差作为次要但仍重要的约束。
    candidates.sort(
        key=lambda item: (
            item.distance_error + 2.5 * item.angle_error,
            item.distance_error,
            item.angle_error,
        )
    )

    used_template: set[int] = set()
    used_probe: set[int] = set()
    pairs: list[MatchedPair] = []
    for candidate in candidates:
        if candidate.template_index in used_template or candidate.probe_index in used_probe:
            continue
        used_template.add(candidate.template_index)
        used_probe.add(candidate.probe_index)
        pairs.append(candidate)
    return pairs
