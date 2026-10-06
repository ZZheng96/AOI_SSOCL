from __future__ import annotations

from pathlib import Path


def parse_component_filename(path: str | Path) -> dict[str, str | None]:
    """Best-effort parse. v1.1: only category is useful; part_no optional legacy."""
    stem = Path(path).stem
    parts = [p for p in stem.split("_") if p != ""]
    result = {
        "category": None,
        "part_no": None,
        "refdes": None,
        "index": None,
        "template_name": stem,
    }
    if len(parts) >= 1:
        result["category"] = parts[0]
    if len(parts) >= 2:
        result["part_no"] = parts[1]
    if len(parts) >= 3:
        result["refdes"] = parts[2]
    if len(parts) >= 4:
        result["index"] = parts[3]
    return result
