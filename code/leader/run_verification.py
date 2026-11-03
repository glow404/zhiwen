"""命令行入口文件。

这个文件的作用是把前面各模块封装成一个可直接运行的脚本，
用户只需要给出数据目录和输出路径，就可以完成：
1. 数据索引构建。
2. 多模板参数搜索。
3. 开发集阈值选择。
4. 测试集 FAR / FRR / EER 评估。
5. 结果 JSON 报告输出。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matcher.eval import SearchOptions, run_full_evaluation


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    """把逗号分隔的整数列表解析成元组。"""

    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    """把逗号分隔的浮点数列表解析成元组。"""

    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="多模板细节点 1:1 指纹验证")
    parser.add_argument(
        "--root",
        default="leader-minutiae/minutiae",
        help="`.mnt` 数据根目录",
    )
    parser.add_argument(
        "--output",
        default="verification_report.json",
        help="输出 JSON 报告路径",
    )
    parser.add_argument(
        "--template-counts",
        default="5",
        help="候选模板数，逗号分隔",
    )
    parser.add_argument(
        "--distance-tolerances",
        default="10,12,14",
        help="空间容差候选，逗号分隔",
    )
    parser.add_argument(
        "--angle-tolerances",
        default="0.35,0.5,0.65",
        help="角度容差候选，单位弧度",
    )
    parser.add_argument(
        "--min-matches",
        default="5,6,7",
        help="最小匹配数候选，逗号分隔",
    )
    parser.add_argument(
        "--fusion-methods",
        default="mean,median,harmonic_mean",
        help="融合方式候选，逗号分隔",
    )
    parser.add_argument(
        "--weight-far",
        type=float,
        default=12.0,
        help="开发集上 FAR 的惩罚权重，越大越保守",
    )
    return parser


def main() -> None:
    """脚本主入口。"""

    parser = build_parser()
    args = parser.parse_args()

    # 用户输入的是字符串形式的候选集合，这里统一转换为搜索配置。
    options = SearchOptions(
        template_count_candidates=_parse_int_tuple(args.template_counts),
        distance_tolerance_candidates=_parse_float_tuple(args.distance_tolerances),
        angle_tolerance_candidates=_parse_float_tuple(args.angle_tolerances),
        min_match_candidates=_parse_int_tuple(args.min_matches),
        fusion_methods=tuple(item.strip() for item in args.fusion_methods.split(",") if item.strip()),
        weight_far=args.weight_far,
    )

    result = run_full_evaluation(args.root, options)
    output_path = Path(args.output)

    # 如果输出目录不存在，则自动创建，方便直接落盘报告。
    if output_path.parent and output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 控制台输出只给出最关键的摘要，详细信息都保存在 JSON 中。
    print(f"数据集: {result['dataset_root']}")
    print(f"身份数: {result['identity_count']}")
    print(
        "最佳配置: "
        f"templates={result['selected_template_count']}, "
        f"fusion={result['selected_fusion_method']}, "
        f"threshold={result['selected_threshold']:.6f}"
    )
    print(
        "开发集: "
        f"FAR={result['dev_metrics']['far']:.6f}, "
        f"FRR={result['dev_metrics']['frr']:.6f}, "
        f"EER={result['dev_metrics']['eer']:.6f}"
    )
    print(
        "测试集: "
        f"FAR={result['test_metrics']['far']:.6f}, "
        f"FRR={result['test_metrics']['frr']:.6f}, "
        f"EER={result['test_metrics']['eer']:.6f}"
    )
    print(f"报告已写入: {output_path}")


if __name__ == "__main__":
    main()
