"""CAD 导入 / 拼板复制 / FOV 划分（P2：借鉴 Java AOI CAD流程.md + 编程设计器）。

- parse_cad_file：解析 CAD 导出文件（csv/txt），字段映射（位号/物料号/封装/坐标 mm/角度）
- duplicate_panels：拼板复制（目标坐标 = 基准 + 拼板偏移；角度 = 基准 + 旋转）
- compute_fov_grid：FOV 网格划分（长宽 mm + 视场 mm + 重叠率）
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

# 常见 CAD 导出列名别名
_ALIASES = {
    "reference": "position_on", "refdes": "position_on", "位号": "position_on",
    "designator": "position_on", "ref": "position_on",
    "material": "material_code", "物料号": "material_code", "part_no": "material_code",
    "mpn": "material_code", "料号": "material_code",
    "package": "pack", "封装": "pack", "footprint": "pack",
    "x": "x_mm", "x_mm": "x_mm", "坐标x": "x_mm", "center_x": "x_mm",
    "y": "y_mm", "y_mm": "y_mm", "坐标y": "y_mm", "center_y": "y_mm",
    "rotation": "angle", "angle": "angle", "角度": "angle", "rot": "angle",
}


def _norm_key(k: str) -> str:
    return str(k).strip().lower().replace(" ", "_")


def parse_cad_file(path: str, field_mapping: Optional[dict] = None) -> list[dict]:
    """解析 CAD 导出文件（支持 .csv / .txt），返回器件表。

    field_mapping 可覆盖列名别名（用户字段映射）。无坐标/位号的行丢弃。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"CAD 文件不存在: {path}")
    mapping = {_norm_key(k): v for k, v in (_ALIASES | (field_mapping or {})).items()}
    items: list[dict] = []
    encodings = ("utf-8-sig", "utf-8", "gbk")
    rows: list[list] = []
    header: list[str] = []
    for enc in encodings:
        try:
            text = p.read_text(encoding=enc)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(text.splitlines()[0] if text.splitlines() else ",")
            reader = list(csv.reader(text.splitlines(), dialect))
            header = [h.strip() for h in reader[0]]
            rows = reader[1:]
            break
        except (UnicodeDecodeError, csv.Error):
            continue
    if not header:
        raise ValueError(f"无法解析 CAD 文件: {path}")
    for r in rows:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        rec = {}
        for i, h in enumerate(header):
            field = mapping.get(_norm_key(h), _norm_key(h))
            rec[field] = r[i].strip()
        pos = rec.get("position_on")
        x = _to_float(rec.get("x_mm"))
        y = _to_float(rec.get("y_mm"))
        if not pos or x is None or y is None:
            continue
        items.append({
            "position_on": pos,
            "material_code": rec.get("material_code", ""),
            "pack": rec.get("pack", ""),
            "x_mm": x, "y_mm": y,
            "angle": _to_float(rec.get("angle")) or 0.0,
        })
    return items


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def duplicate_panels(items: list[dict], panel_dx_mm: float, panel_dy_mm: float,
                     cols: int = 1, rows: int = 1,
                     rotation: float = 0.0) -> list[dict]:
    """拼板复制（CAD流程.md：目标坐标 = 基准 + 拼板偏移；目标角度 = 基准 + 旋转）。

    返回含 panel_col/panel_row 的器件列表（含基准板自身 col=0,row=0）。
    """
    out: list[dict] = []
    import math
    rad = math.radians(rotation)
    for col in range(cols):
        for row in range(rows):
            dx = col * panel_dx_mm
            dy = row * panel_dy_mm
            for it in items:
                copy = dict(it)
                # 基准坐标经旋转再平移（绕原点旋转）
                rx = it["x_mm"] * math.cos(rad) - it["y_mm"] * math.sin(rad)
                ry = it["x_mm"] * math.sin(rad) + it["y_mm"] * math.cos(rad)
                copy["x_mm"] = round(rx + dx, 3)
                copy["y_mm"] = round(ry + dy, 3)
                copy["angle"] = round((it.get("angle", 0.0) or 0.0) + rotation, 1)
                copy["panel_col"] = col
                copy["panel_row"] = row
                out.append(copy)
    return out


def compute_fov_grid(board_w_mm: float, board_h_mm: float,
                     fov_w_mm: float, fov_h_mm: float,
                     overlap: float = 0.1) -> list[dict]:
    """FOV 网格划分：覆盖整板所需视场网格（含重叠率）。

    返回 [{row, col, x_mm, y_mm, w_mm, h_mm}, ...]（左上角坐标）。
    """
    if fov_w_mm <= 0 or fov_h_mm <= 0:
        raise ValueError("FOV 尺寸必须为正")
    step_x = fov_w_mm * (1 - overlap)
    step_y = fov_h_mm * (1 - overlap)
    cols = max(1, int((board_w_mm - fov_w_mm) / step_x) + 2)
    rows = max(1, int((board_h_mm - fov_h_mm) / step_y) + 2)
    grid = []
    for r in range(rows):
        for c in range(cols):
            x = min(c * step_x, board_w_mm - fov_w_mm)
            y = min(r * step_y, board_h_mm - fov_h_mm)
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            grid.append({"row": r, "col": c,
                         "x_mm": round(x, 3), "y_mm": round(y, 3),
                         "w_mm": fov_w_mm, "h_mm": fov_h_mm})
    return grid
