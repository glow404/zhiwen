"""`matcher.eval` 负责参数搜索与验证评测。

这个文件把前面的数据层、单模板匹配和多模板融合串成一条完整流水线，
核心工作包括：
1. 在开发集上搜索模板数、匹配参数、融合方式和阈值。
2. 以“低 FAR 优先”为目标，选择更保守的工作点。
3. 在测试集上复用开发集选出的配置，输出 FAR / FRR / EER 等指标。
4. 生成可落盘的结果摘要，便于后续分析和画图。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from matcher.core import MatchParameters, match_minutiae
from matcher.fusion import score_template_set, select_templates
from matcher.io import IdentitySplit, SampleRecord, build_dataset_index, build_identity_splits


@dataclass(frozen=True)
class SearchOptions:
    """开发集搜索空间与风险偏好配置。"""

    template_count_candidates: tuple[int, ...] = (1, 3, 5, 8)
    distance_tolerance_candidates: tuple[float, ...] = (10.0, 12.0, 14.0)
    angle_tolerance_candidates: tuple[float, ...] = (0.35, 0.50, 0.65)
    min_match_candidates: tuple[int, ...] = (5, 6, 7)
    fusion_methods: tuple[str, ...] = ("mean", "median", "harmonic_mean")
    weight_far: float = 12.0
    low_far_targets: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01)
    anchor_pool_size: int = 12
    max_anchor_pairs: int = 96
    local_radius: float = 36.0


@dataclass(frozen=True)
class OperatingPoint:
    """某个阈值下对应的 FAR / FRR 工作点。"""

    threshold: float
    far: float
    frr: float


@dataclass(frozen=True)
class EvaluationBundle:
    """某一阶段（开发集或测试集）的完整评测结果。"""

    genuine_scores: tuple[float, ...]
    impostor_scores: tuple[float, ...]
    points: tuple[OperatingPoint, ...]
    chosen_point: OperatingPoint
    eer: float
    low_far_points: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class SearchResult:
    """开发集搜索得到的最佳配置。"""

    match_parameters: MatchParameters
    fusion_method: str
    template_count: int
    threshold: float
    dev_bundle: EvaluationBundle
    objective: float


class ScoreCache:
    """模板-探测样本分数缓存。

    参数搜索会重复请求大量相同的模板/探测样本组合。
    通过缓存底层匹配分数，可以显著减少重复计算。
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str, tuple[float, float, int, float, int, int]], float] = {}

    def get_score(
        self,
        template: SampleRecord,
        probe: SampleRecord,
        matcher_params: MatchParameters,
    ) -> float:
        """获取单模板分数，若缓存命中则直接返回。"""

        key = (template.sample_id, probe.sample_id, matcher_params.as_key())
        score = self._store.get(key)
        if score is None:
            score = match_minutiae(template.minutiae, probe.minutiae, matcher_params).score
            self._store[key] = score
        return score


def run_full_evaluation(
    root_dir: str | Path,
    options: SearchOptions | None = None,
) -> dict:
    """从数据根目录出发，执行完整评测流程。"""

    search_options = options or SearchOptions()
    records = build_dataset_index(root_dir)
    splits = build_identity_splits(records)
    return evaluate_identity_splits(splits, search_options, root_dir)


def evaluate_identity_splits(
    splits: list[IdentitySplit],
    options: SearchOptions | None = None,
    root_dir: str | Path | None = None,
) -> dict:
    """对已经划分好的身份数据执行完整评测。

    这一步会先在开发集上选出最佳配置，再固定该配置去测试集评估。
    """

    search_options = options or SearchOptions()
    cache = ScoreCache()
    best_result = _search_best_configuration(splits, search_options, cache)
    templates_by_identity = _build_templates_by_identity(splits, best_result.template_count)
    test_bundle = _collect_stage_bundle(
        splits=splits,
        templates_by_identity=templates_by_identity,
        matcher_params=best_result.match_parameters,
        fusion_method=best_result.fusion_method,
        threshold=best_result.threshold,
        stage_name="test",
        cache=cache,
        low_far_targets=search_options.low_far_targets,
        weight_far=search_options.weight_far,
    )

    # 返回结果尽量使用纯 Python 基础类型，方便直接转成 JSON。
    summary = {
        "dataset_root": str(Path(root_dir)) if root_dir is not None else None,
        "identity_count": len(splits),
        "enrollment_size_min": min(len(split.enrollment_pool) for split in splits),
        "enrollment_size_max": max(len(split.enrollment_pool) for split in splits),
        "dev_probe_count": sum(len(split.dev_probes) for split in splits),
        "test_probe_count": sum(len(split.test_probes) for split in splits),
        "selected_template_count": best_result.template_count,
        "selected_fusion_method": best_result.fusion_method,
        "selected_threshold": best_result.threshold,
        "selected_match_parameters": {
            "distance_tolerance": best_result.match_parameters.distance_tolerance,
            "angle_tolerance": best_result.match_parameters.angle_tolerance,
            "min_matches": best_result.match_parameters.min_matches,
            "local_radius": best_result.match_parameters.local_radius,
            "anchor_pool_size": best_result.match_parameters.anchor_pool_size,
            "max_anchor_pairs": best_result.match_parameters.max_anchor_pairs,
        },
        "dev_metrics": _bundle_to_dict(best_result.dev_bundle, include_points=True),
        "test_metrics": _bundle_to_dict(test_bundle, include_points=True),
    }
    return summary


def _search_best_configuration(
    splits: list[IdentitySplit],
    options: SearchOptions,
    cache: ScoreCache,
) -> SearchResult:
    """遍历搜索空间，找到开发集上最优的保守配置。"""

    min_enrollment_size = min(len(split.enrollment_pool) for split in splits)
    candidate_template_counts = tuple(
        count for count in options.template_count_candidates if count <= min_enrollment_size
    )
    if not candidate_template_counts:
        candidate_template_counts = (min_enrollment_size,)

    best_result: SearchResult | None = None
    template_cache: dict[int, dict[str, tuple[SampleRecord, ...]]] = {}

    for template_count in candidate_template_counts:
        templates_by_identity = template_cache.setdefault(
            template_count,
            _build_templates_by_identity(splits, template_count),
        )
        for fusion_method, distance_tolerance, angle_tolerance, min_matches in product(
            options.fusion_methods,
            options.distance_tolerance_candidates,
            options.angle_tolerance_candidates,
            options.min_match_candidates,
        ):
            matcher_params = MatchParameters(
                distance_tolerance=distance_tolerance,
                angle_tolerance=angle_tolerance,
                min_matches=min_matches,
                local_radius=options.local_radius,
                anchor_pool_size=options.anchor_pool_size,
                max_anchor_pairs=options.max_anchor_pairs,
            )
            dev_bundle = _collect_stage_bundle(
                splits=splits,
                templates_by_identity=templates_by_identity,
                matcher_params=matcher_params,
                fusion_method=fusion_method,
                threshold=None,
                stage_name="dev",
                cache=cache,
                low_far_targets=options.low_far_targets,
                weight_far=options.weight_far,
            )
            # 目标函数对 FAR 给予更高权重，体现“低 FAR 优先”的偏好。
            chosen_point = dev_bundle.chosen_point
            objective = options.weight_far * chosen_point.far + chosen_point.frr
            candidate = SearchResult(
                match_parameters=matcher_params,
                fusion_method=fusion_method,
                template_count=template_count,
                threshold=chosen_point.threshold,
                dev_bundle=dev_bundle,
                objective=objective,
            )
            if _is_better_candidate(candidate, best_result):
                best_result = candidate

    if best_result is None:
        raise ValueError("没有生成任何可用配置。")
    return best_result


def _is_better_candidate(candidate: SearchResult, incumbent: SearchResult | None) -> bool:
    """比较两个候选配置谁更优。

    排序优先级：
    1. 目标函数更小。
    2. FAR 更低。
    3. FRR 更低。
    4. 在前面接近时，优先更多模板和更严格的最小匹配数。
    """

    if incumbent is None:
        return True
    candidate_point = candidate.dev_bundle.chosen_point
    incumbent_point = incumbent.dev_bundle.chosen_point
    candidate_key = (
        candidate.objective,
        candidate_point.far,
        candidate_point.frr,
        -candidate.template_count,
        -candidate.match_parameters.min_matches,
        -candidate.threshold,
    )
    incumbent_key = (
        incumbent.objective,
        incumbent_point.far,
        incumbent_point.frr,
        -incumbent.template_count,
        -incumbent.match_parameters.min_matches,
        -incumbent.threshold,
    )
    return candidate_key < incumbent_key


def _build_templates_by_identity(
    splits: list[IdentitySplit],
    template_count: int,
) -> dict[str, tuple[SampleRecord, ...]]:
    """为每个身份提前选好固定数量的模板。"""

    return {
        split.identity: select_templates(split.enrollment_pool, template_count)
        for split in splits
    }


def _collect_stage_bundle(
    splits: list[IdentitySplit],
    templates_by_identity: dict[str, tuple[SampleRecord, ...]],
    matcher_params: MatchParameters,
    fusion_method: str,
    threshold: float | None,
    stage_name: str,
    cache: ScoreCache,
    low_far_targets: tuple[float, ...],
    weight_far: float,
) -> EvaluationBundle:
    """收集某一阶段的 genuine / impostor 分数并计算指标。"""

    genuine_scores, impostor_scores = _collect_scores(
        splits=splits,
        templates_by_identity=templates_by_identity,
        matcher_params=matcher_params,
        fusion_method=fusion_method,
        stage_name=stage_name,
        cache=cache,
    )
    points = _compute_operating_points(genuine_scores, impostor_scores)
    chosen_point = _choose_operating_point(points, threshold, weight_far, genuine_scores, impostor_scores)
    eer = _estimate_eer(points)
    low_far_points = _summarize_low_far_points(points, low_far_targets)
    return EvaluationBundle(
        genuine_scores=tuple(genuine_scores),
        impostor_scores=tuple(impostor_scores),
        points=tuple(points),
        chosen_point=chosen_point,
        eer=eer,
        low_far_points=tuple(low_far_points),
    )


def _collect_scores(
    splits: list[IdentitySplit],
    templates_by_identity: dict[str, tuple[SampleRecord, ...]],
    matcher_params: MatchParameters,
    fusion_method: str,
    stage_name: str,
    cache: ScoreCache,
) -> tuple[list[float], list[float]]:
    """收集 genuine 与 impostor 两类分数。

    - genuine：模板身份与探测身份相同。
    - impostor：模板身份与探测身份不同。
    """

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    probes_attr = "dev_probes" if stage_name == "dev" else "test_probes"
    all_split_map = {split.identity: split for split in splits}

    for claimant_split in splits:
        claimant_templates = templates_by_identity[claimant_split.identity]
        claimant_probes = getattr(claimant_split, probes_attr)

        for probe in claimant_probes:
            genuine_score, _ = score_template_set(
                claimant_templates,
                probe,
                matcher_params,
                fusion_method,
                score_provider=cache.get_score,
            )
            genuine_scores.append(genuine_score)

        # 逐个把其他身份的探测样本拿来与当前身份模板比较，构造 impostor 分数。
        for impostor_identity, impostor_split in all_split_map.items():
            if impostor_identity == claimant_split.identity:
                continue
            for probe in getattr(impostor_split, probes_attr):
                impostor_score, _ = score_template_set(
                    claimant_templates,
                    probe,
                    matcher_params,
                    fusion_method,
                    score_provider=cache.get_score,
                )
                impostor_scores.append(impostor_score)

    return genuine_scores, impostor_scores


def _compute_operating_points(
    genuine_scores: list[float],
    impostor_scores: list[float],
) -> list[OperatingPoint]:
    """遍历候选阈值，生成 FAR / FRR 曲线。"""

    if not genuine_scores or not impostor_scores:
        raise ValueError("genuine 或 impostor 分数为空，无法计算 FAR/FRR。")

    # 阈值集合来自所有已观测分数，再额外补一个略大阈值作为“全拒绝”端点。
    thresholds = sorted(
        {0.0, 1.0, *genuine_scores, *impostor_scores, max(genuine_scores + impostor_scores) + 1e-9}
    )
    points: list[OperatingPoint] = []
    genuine_total = len(genuine_scores)
    impostor_total = len(impostor_scores)

    for threshold in thresholds:
        false_rejects = sum(score < threshold for score in genuine_scores)
        false_accepts = sum(score >= threshold for score in impostor_scores)
        points.append(
            OperatingPoint(
                threshold=threshold,
                far=false_accepts / impostor_total,
                frr=false_rejects / genuine_total,
            )
        )
    return points


def _choose_operating_point(
    points: list[OperatingPoint],
    threshold: float | None,
    weight_far: float,
    genuine_scores: list[float],
    impostor_scores: list[float],
) -> OperatingPoint:
    """选择最终工作点。

    - 若阈值已固定，则直接在该阈值处重新计算点。
    - 若阈值未固定，则按“低 FAR 优先”的目标函数选点。
    """

    if threshold is not None:
        return _compute_point_at_threshold(genuine_scores, impostor_scores, threshold)

    return min(
        points,
        key=lambda point: (
            weight_far * point.far + point.frr,
            point.far,
            point.frr,
            -point.threshold,
        ),
    )


def _compute_point_at_threshold(
    genuine_scores: list[float],
    impostor_scores: list[float],
    threshold: float,
) -> OperatingPoint:
    """在给定阈值处直接计算 FAR / FRR。"""

    genuine_total = len(genuine_scores)
    impostor_total = len(impostor_scores)
    false_rejects = sum(score < threshold for score in genuine_scores)
    false_accepts = sum(score >= threshold for score in impostor_scores)
    return OperatingPoint(
        threshold=threshold,
        far=false_accepts / impostor_total,
        frr=false_rejects / genuine_total,
    )


def _estimate_eer(points: list[OperatingPoint]) -> float:
    """用 FAR 与 FRR 最接近的点近似 EER。"""

    closest = min(points, key=lambda point: abs(point.far - point.frr))
    return 0.5 * (closest.far + closest.frr)


def _summarize_low_far_points(
    points: list[OperatingPoint],
    low_far_targets: tuple[float, ...],
) -> list[dict[str, float]]:
    """汇总若干低 FAR 目标下的最佳可行工作点。"""

    summaries: list[dict[str, float]] = []
    for target in low_far_targets:
        feasible = [point for point in points if point.far <= target + 1e-12]
        if not feasible:
            # 如果达不到目标 FAR，就退而求其次取 FAR 最低的点。
            chosen = min(points, key=lambda point: point.far)
        else:
            chosen = min(feasible, key=lambda point: (point.frr, -point.threshold))
        summaries.append(
            {
                "target_far": target,
                "threshold": chosen.threshold,
                "far": chosen.far,
                "frr": chosen.frr,
            }
        )
    return summaries


def _bundle_to_dict(bundle: EvaluationBundle, include_points: bool) -> dict:
    """把评测对象转换成适合序列化的字典。"""

    chosen = bundle.chosen_point
    result = {
        "genuine_count": len(bundle.genuine_scores),
        "impostor_count": len(bundle.impostor_scores),
        "chosen_threshold": chosen.threshold,
        "far": chosen.far,
        "frr": chosen.frr,
        "eer": bundle.eer,
        "genuine_score_mean": sum(bundle.genuine_scores) / len(bundle.genuine_scores),
        "impostor_score_mean": sum(bundle.impostor_scores) / len(bundle.impostor_scores),
        "low_far_points": list(bundle.low_far_points),
    }
    if include_points:
        result["det_curve"] = [
            asdict(point) for point in _downsample_points(bundle.points, max_points=256)
        ]
    return result


def _downsample_points(points: tuple[OperatingPoint, ...], max_points: int) -> tuple[OperatingPoint, ...]:
    """对曲线点做均匀下采样，防止 JSON 过大。"""

    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    selected: list[OperatingPoint] = []
    cursor = 0.0
    while int(cursor) < len(points):
        selected.append(points[int(cursor)])
        cursor += step
    if selected[-1] != points[-1]:
        selected.append(points[-1])
    return tuple(selected)
