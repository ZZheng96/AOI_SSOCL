"""参数状态工具：从元数据构建各瑕疵的默认配置。"""

from __future__ import annotations

from typing import Any, Dict

from ..metadata import DEFECT_METAS


def make_default_state() -> Dict[str, Dict[str, Any]]:
    return {m.code: m.default_config() for m in DEFECT_METAS}
