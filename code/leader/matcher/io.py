"""`matcher.io` 负责数据输入层。

这个文件主要解决三类问题：
1. 读取 `.mnt` 细节点文件，并把文本行解析成结构化细节点对象。
2. 根据当前数据目录和文件命名规则，自动推断身份、轮次、组别和样本编号。
3. 把原始样本进一步整理为注册集、开发集、测试集，供后续匹配与评测模块统一使用。

由于当前数据集没有单独提供元数据表，本文件默认通过目录结构自动推断标签，
因此这里的路径解析逻辑是整套算法的基础。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# 目录名形如 `zyh_R2`，表示某个主体在第 2 轮采集。
SESSION_PATTERN = re.compile(r"^(?P<subject>.+)_R(?P<repeat>\d+)$")

# 文件名形如 `zyh_R2_pair99.mnt`，其中 `pair99` 作为样本编号使用。
FILE_PATTERN = re.compile(r"^(?P<subject>.+)_R(?P<repeat>\d+)_pair(?P<sample>\d+)\.mnt$")


@dataclass(frozen=True)
class Minutia:
    """单个细节点。

    字段说明：
    - `x`, `y`：细节点坐标。
    - `angle`：方向角，统一规整到 `[-pi, pi]`。
    - `kind`：细节点类型，当前仅保留 `B` / `E` 两种。
    """

    x: float
    y: float
    angle: float
    kind: str


@dataclass(frozen=True)
class SampleRecord:
    """单个 `.mnt` 文件对应的一条完整样本记录。

    这个对象除了保存细节点本身，还额外缓存了从路径解析出的身份信息、
    采集轮次信息，以及用于模板挑选的质量代理分数。
    """

    path: Path
    identity: str
    subject: str
    session_label: str
    session_index: int
    group_id: str
    sample_index: int
    minutiae: tuple[Minutia, ...]
    minutia_count: int
    quality_proxy: float

    @property
    def sample_id(self) -> str:
        """返回全局唯一的样本字符串 ID。

        这个 ID 会被评测阶段用于缓存模板-探测样本之间的匹配分数，
        从而避免重复计算。
        """

        return f"{self.identity}:{self.session_label}:pair{self.sample_index}"


@dataclass(frozen=True)
class IdentitySplit:
    """某个身份在验证任务中的三段式划分结果。

    - `enrollment_pool`：注册候选样本池，用于挑选模板。
    - `dev_probes`：开发集探测样本，用于自动调参和阈值搜索。
    - `test_probes`：测试集探测样本，用于最终报告指标。
    """

    identity: str
    subject: str
    group_id: str
    enrollment_pool: tuple[SampleRecord, ...]
    dev_probes: tuple[SampleRecord, ...]
    test_probes: tuple[SampleRecord, ...]


def wrap_angle(angle: float) -> float:
    """把角度规整到 `[-pi, pi]`。

    指纹细节点方向是周期变量，直接比较原始浮点值容易出现
    `pi` 与 `-pi` 边界不连续的问题，所以这里先做标准化。
    """

    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi:
        return math.pi
    return wrapped


def angle_distance(angle_a: float, angle_b: float) -> float:
    """计算两个方向角的最小圆周距离。"""

    return abs(wrap_angle(angle_a - angle_b))


def parse_mnt_file(path: str | Path) -> tuple[Minutia, ...]:
    """读取并解析一个 `.mnt` 文件。

    当前数据格式假定为：
    - 第一行：细节点数量。
    - 后续每行：`x y angle type`。

    解析时会顺带做三类容错：
    - 跳过空行和不完整行。
    - 跳过数值非法或类型非法的记录。
    - 对重复细节点进行去重，降低噪声影响。
    """

    file_path = Path(path)
    raw_lines = file_path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    if not lines:
        raise ValueError(f"空的细节点文件: {file_path}")

    minutiae: list[Minutia] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        x_text, y_text, angle_text, kind_text = parts[:4]
        try:
            x_value = float(x_text)
            y_value = float(y_text)
            angle_value = wrap_angle(float(angle_text))
        except ValueError:
            continue
        kind = kind_text.upper()
        if kind not in {"B", "E"}:
            continue
        minutiae.append(Minutia(x=x_value, y=y_value, angle=angle_value, kind=kind))

    if not minutiae:
        raise ValueError(f"未解析到有效细节点: {file_path}")

    # 首行数量只作为软约束使用。
    # 如果正文比首行更短，不强制报错；如果更长，则截断到首行声明的数量。
    expected_count = None
    try:
        expected_count = int(lines[0])
    except ValueError:
        expected_count = None
    if expected_count is not None and expected_count > 0 and len(minutiae) > expected_count:
        minutiae = minutiae[:expected_count]

    return tuple(_deduplicate_minutiae(minutiae))


def _deduplicate_minutiae(minutiae: Iterable[Minutia]) -> list[Minutia]:
    """对细节点做轻量去重。

    这里按“近似整数坐标 + 毫弧度量级角度 + 类型”构造键，
    目的是抑制同一位置重复记录的噪声，而不是做严格聚类。
    """

    seen: set[tuple[int, int, int, str]] = set()
    unique: list[Minutia] = []
    for minutia in minutiae:
        key = (
            int(round(minutia.x)),
            int(round(minutia.y)),
            int(round(minutia.angle * 1000.0)),
            minutia.kind,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(minutia)
    return unique


def infer_sample_record(path: str | Path, root_dir: str | Path) -> SampleRecord:
    """根据路径和文件内容构建 `SampleRecord`。

    当前默认目录层级为：
    `root / session_dir / group_id / filename`

    例如：
    `minutiae/zyh_R2/Rgd1237/zyh_R2_pair99.mnt`
    """

    file_path = Path(path)
    root_path = Path(root_dir)
    relative_parts = file_path.relative_to(root_path).parts
    if len(relative_parts) != 3:
        raise ValueError(f"无法解析路径层级: {file_path}")

    # 第一层目录编码“主体 + 轮次”，第二层目录编码组别或批次。
    session_dir, group_id, filename = relative_parts
    session_match = SESSION_PATTERN.match(session_dir)
    file_match = FILE_PATTERN.match(filename)
    if session_match is None or file_match is None:
        raise ValueError(f"无法解析文件命名: {file_path}")

    session_subject = session_match.group("subject")
    file_subject = file_match.group("subject")
    session_repeat = int(session_match.group("repeat"))
    file_repeat = int(file_match.group("repeat"))
    # 父目录和文件名里的标签必须一致，否则说明命名存在异常。
    if session_subject != file_subject or session_repeat != file_repeat:
        raise ValueError(f"父目录与文件名标签不一致: {file_path}")

    minutiae = parse_mnt_file(file_path)
    quality_proxy = _estimate_quality(minutiae)
    session_label = f"R{session_repeat}"

    # identity 这里用 `group_id:subject` 组合，避免不同组下同名主体冲突。
    identity = f"{group_id}:{session_subject}"
    return SampleRecord(
        path=file_path,
        identity=identity,
        subject=session_subject,
        session_label=session_label,
        session_index=session_repeat,
        group_id=group_id,
        sample_index=int(file_match.group("sample")),
        minutiae=minutiae,
        minutia_count=len(minutiae),
        quality_proxy=quality_proxy,
    )


def _estimate_quality(minutiae: tuple[Minutia, ...]) -> float:
    """计算一个简单的模板质量代理分数。

    当前没有图像质量分数，因此用可从细节点直接估计的指标近似：
    - 细节点数量越多通常越稳定。
    - 覆盖范围越大，说明采集区域更充分。
    - 空间分布越分散，通常比高度聚集的样本更有区分性。

    这个分数仅用于模板优先级排序，不代表真实质量标签。
    """

    if not minutiae:
        return 0.0
    xs = [item.x for item in minutiae]
    ys = [item.y for item in minutiae]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    footprint = math.sqrt(max(span_x, 1.0) * max(span_y, 1.0))
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    radial_dispersion = sum(
        math.hypot(item.x - center_x, item.y - center_y) for item in minutiae
    ) / len(minutiae)
    return len(minutiae) + 0.015 * footprint + 0.01 * radial_dispersion


def build_dataset_index(root_dir: str | Path) -> list[SampleRecord]:
    """扫描数据根目录，建立全部样本索引。

    这里会自动跳过没有有效细节点的样本，例如：
    - 文件为空。
    - 首行数量为 0，且正文没有细节点记录。
    - 正文存在，但所有记录都因格式问题被过滤掉。

    这样可以避免少量坏样本直接中断整套评测流程。
    但如果是路径层级错误、命名错误等结构性问题，仍然保留报错。
    """

    root_path = Path(root_dir)
    sample_paths = sorted(root_path.glob("*/*/*.mnt"))
    records: list[SampleRecord] = []
    for path in sample_paths:
        try:
            records.append(infer_sample_record(path, root_path))
        except ValueError as exc:
            error_text = str(exc)
            if "空的细节点文件" in error_text or "未解析到有效细节点" in error_text:
                continue
            raise
    if not records:
        raise ValueError(f"数据目录下没有找到 .mnt 文件: {root_path}")
    return records


def group_records_by_identity(records: Iterable[SampleRecord]) -> dict[str, list[SampleRecord]]:
    """按身份聚合样本，并保持稳定排序。"""

    grouped: dict[str, list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.identity, []).append(record)
    for identity_records in grouped.values():
        identity_records.sort(key=lambda item: (item.session_index, item.sample_index, str(item.path)))
    return grouped


def build_identity_splits(records: Iterable[SampleRecord]) -> list[IdentitySplit]:
    """把原始样本整理为每个身份的注册/开发/测试三段结构。

    优先策略：
    - 如果某个身份存在至少 3 个采集轮次，则按轮次顺序划分：
      `R0 -> enrollment`，`R1 -> dev`，`R2 -> test`。
    - 如果轮次数不足 3，则退化为按时间顺序比例切分。

    这样做的目的，是尽可能让开发集和测试集来自不同轮次，
    从而更接近真实验证场景。
    """

    grouped = group_records_by_identity(records)
    splits: list[IdentitySplit] = []

    for identity, identity_records in sorted(grouped.items()):
        # 先按轮次聚合，为“跨轮次验证”做准备。
        sessions: dict[int, list[SampleRecord]] = {}
        for record in identity_records:
            sessions.setdefault(record.session_index, []).append(record)
        session_keys = sorted(sessions)

        if len(session_keys) >= 3:
            enrollment_pool = tuple(_sort_pool(sessions[session_keys[0]]))
            dev_probes = tuple(_sort_pool(sessions[session_keys[1]]))
            test_probes = tuple(_sort_pool(sessions[session_keys[2]]))
        else:
            enrollment_pool, dev_probes, test_probes = _fallback_split(identity_records)

        if not enrollment_pool or not dev_probes or not test_probes:
            continue

        seed_record = enrollment_pool[0]
        splits.append(
            IdentitySplit(
                identity=identity,
                subject=seed_record.subject,
                group_id=seed_record.group_id,
                enrollment_pool=enrollment_pool,
                dev_probes=dev_probes,
                test_probes=test_probes,
            )
        )

    if len(splits) < 2:
        raise ValueError("有效身份数不足，无法做验证评测。")
    return splits


def _sort_pool(records: Iterable[SampleRecord]) -> list[SampleRecord]:
    """对同一池中的样本排序。

    排序原则优先质量代理分数，其次是样本编号和路径，
    这样后续挑选模板时会优先拿到更稳定的样本。
    """

    return sorted(records, key=lambda item: (-item.quality_proxy, item.sample_index, str(item.path)))


def _fallback_split(
    identity_records: list[SampleRecord],
) -> tuple[tuple[SampleRecord, ...], tuple[SampleRecord, ...], tuple[SampleRecord, ...]]:
    """在轮次信息不足时，使用保守的比例切分。

    默认比例接近 50% / 25% / 25%，保证三部分都非空。
    这个逻辑是兜底方案，优先级低于“按轮次划分”。
    """

    ordered = sorted(
        identity_records,
        key=lambda item: (item.session_index, item.sample_index, str(item.path)),
    )
    total = len(ordered)
    enroll_end = max(1, int(total * 0.5))
    dev_end = max(enroll_end + 1, int(total * 0.75))
    enrollment_pool = tuple(_sort_pool(ordered[:enroll_end]))
    dev_probes = tuple(_sort_pool(ordered[enroll_end:dev_end]))
    test_probes = tuple(_sort_pool(ordered[dev_end:]))
    return enrollment_pool, dev_probes, test_probes
