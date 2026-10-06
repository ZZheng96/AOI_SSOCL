"""双坐标转换（P2：借鉴 Java AOI 坐标系设计.md + 相机标定值.md）。

- 算法/图像内存 = 像素坐标（px）；数据库/报表/机械/MES = 物理坐标（mm）
- pxPerMMX/Y 分离存储（镜头畸变/倾斜导致 X/Y 当量不一致）
- 物理原点 = PCB 工艺基准角（Mark 原点）
- 公式：Board_mm = Origin_mm + (Img_px - ImgOrigin_px) / pxPerMM
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Calibration:
    """单套标定参数。"""
    px_per_mm_x: float = 1.0
    px_per_mm_y: float = 1.0
    origin_x_mm: float = 0.0      # 物理原点（Mark 基准，mm）
    origin_y_mm: float = 0.0
    img_origin_x: float = 0.0     # 图像上物理原点所在像素
    img_origin_y: float = 0.0
    device_code: str = ""         # 所属设备（跨设备通用靠标定）

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Calibration":
        if not d:
            return cls()
        return cls(
            px_per_mm_x=float(d.get("px_per_mm_x", 1.0)),
            px_per_mm_y=float(d.get("px_per_mm_y", 1.0)),
            origin_x_mm=float(d.get("origin_x_mm", 0.0)),
            origin_y_mm=float(d.get("origin_y_mm", 0.0)),
            img_origin_x=float(d.get("img_origin_x", 0.0)),
            img_origin_y=float(d.get("img_origin_y", 0.0)),
            device_code=str(d.get("device_code", "")),
        )

    def to_dict(self) -> dict:
        return {"px_per_mm_x": self.px_per_mm_x, "px_per_mm_y": self.px_per_mm_y,
                "origin_x_mm": self.origin_x_mm, "origin_y_mm": self.origin_y_mm,
                "img_origin_x": self.img_origin_x, "img_origin_y": self.img_origin_y,
                "device_code": self.device_code}

    def px_to_board(self, x_px: float, y_px: float) -> tuple[float, float]:
        """像素 → 物理（mm）。"""
        bx = self.origin_x_mm + (x_px - self.img_origin_x) / self.px_per_mm_x
        by = self.origin_y_mm + (y_px - self.img_origin_y) / self.px_per_mm_y
        return round(bx, 3), round(by, 3)

    def board_to_px(self, bx_mm: float, by_mm: float) -> tuple[float, float]:
        """物理（mm）→ 像素。"""
        x = self.img_origin_x + (bx_mm - self.origin_x_mm) * self.px_per_mm_x
        y = self.img_origin_y + (by_mm - self.origin_y_mm) * self.px_per_mm_y
        return round(x, 2), round(y, 2)


def get_calibration(template_id: Optional[str] = None) -> Calibration:
    """取标定：模板级优先（template.px_calibration），回退全局配置。"""
    if template_id:
        try:
            from app.template.store import TemplateStore
            tpl = TemplateStore().load(template_id)
            if tpl is not None:
                px = getattr(tpl, "px_calibration", None)
                if isinstance(px, dict) and px.get("px_per_mm_x"):
                    return Calibration.from_dict(px)
        except Exception:  # noqa: BLE001
            pass
    from app.config import get_settings
    return Calibration.from_dict(get_settings().section("coordinate"))


def save_template_calibration(template_id: str, cal: Calibration) -> None:
    """把标定写入模板 JSON（px_calibration 段）。"""
    from app.template.store import TemplateStore
    tpl = TemplateStore().load(template_id)
    if tpl is None:
        raise ValueError(f"模板不存在: {template_id}")
    tpl.px_calibration = cal.to_dict()
    TemplateStore().save(tpl)
