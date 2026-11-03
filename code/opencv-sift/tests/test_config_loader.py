"""配置加载器的最小测试用例。

这个文件验证配置加载模块是否能把基础 YAML 正确解析为强类型配置对象。

输入：
- 临时生成的 YAML 配置文件

输出：
- 通过断言验证关键字段是否正确映射
"""

from __future__ import annotations

from pathlib import Path

from src.config.loader import load_config


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    """验证加载器能正确读取简单 YAML 并映射到配置对象。"""

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "app:",
                "  name: test_app",
                "  template_version: test_v1",
                "input:",
                "  single_image_path: sample.png",
                "output:",
                "  output_root: outputs/test_run",
                "sift:",
                "  nfeatures: 32",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.app.name == "test_app"
    assert config.app.template_version == "test_v1"
    assert config.input.single_image_path == Path("sample.png")
    assert config.output.output_root == Path("outputs/test_run")
    assert config.sift.nfeatures == 32
