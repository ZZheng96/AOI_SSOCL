from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import TEMPLATE_CALIB_DIR

_SAFE_RE = re.compile(r'[\\/:*?"<>|]')


def _path_for(template_name: str | None) -> Path:
    name = (template_name or "").strip() or "_default"
    safe = _SAFE_RE.sub("_", name)
    return TEMPLATE_CALIB_DIR / f"{safe}.json"


def load_calib(template_name: str | None) -> dict:
    path = _path_for(template_name)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_calib(template_name: str | None, data: dict) -> None:
    path = _path_for(template_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_calib(template_name: str | None, **patch) -> dict:
    data = load_calib(template_name)
    data.update(patch)
    save_calib(template_name, data)
    return data
