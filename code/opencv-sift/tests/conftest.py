"""测试启动辅助文件。

这个文件负责在运行测试时把项目根目录加入 `sys.path`，
从而保证测试文件能够稳定导入 `src` 下的模块。

输入：
- 无显式输入

输出：
- 调整后的 Python 导入路径环境
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
