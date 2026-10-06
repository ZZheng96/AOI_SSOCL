"""标准图配置持久化：按 ``template_id`` 存取 JSON"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..core.models import TemplateConfig
from .. import config as app_config

_SAFE_CHARS = re.compile(r"[^0-9A-Za-z_\-.\u4e00-\u9fff]+")


def template_id_for_path(path: str) -> str:
    """由标准图文件路径派生一个稳定、适合做文件名的 template_id（默认取文件名去扩展名）。"""
    stem = Path(path).stem or "template"
    safe = _SAFE_CHARS.sub("_", stem).strip("_")
    return safe or "template"


class TemplateStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else app_config.DEFAULT_TEMPLATES_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, template_id: str) -> Path:
        return self.root / f"{template_id}.json"

    def exists(self, template_id: str) -> bool:
        return self.path_for(template_id).is_file()

    def load(self, template_id: str) -> Optional[TemplateConfig]:
        p = self.path_for(template_id)
        if not p.is_file():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TemplateConfig.from_json_dict(data)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def save(self, cfg: TemplateConfig) -> Path:
        p = self.path_for(cfg.template_id)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg.to_json_dict(), f, ensure_ascii=False, indent=2)
        return p

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))
