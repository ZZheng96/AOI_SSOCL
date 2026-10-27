"""AC-001/002：品类分辨率档案 + 视图模式选择（整图 / 整图+局部 / 先排除域差）。

档案字段均来自品类档案（cfg tiling.profile），缺失字段不编默认值：
min_defect_px 缺失 → profile_missing，保持整图（AC-002 整图专家始终独立保留）。
contrast / miss_req 仅记录，选择规则未冻结（标注 unfrozen）。
"""
from dataclasses import dataclass, asdict, field
from typing import Optional

from .tiling import tile_size_bounds


@dataclass
class ResolutionProfile:
    category: str = ""
    min_defect_px: Optional[float] = None    # 原图像素下最小需检缺陷尺寸
    max_defect_px: Optional[float] = None
    contrast: Optional[str] = None           # low/mid/high，仅记录（未冻结）
    miss_req: Optional[str] = None           # 漏检要求，仅记录（未冻结）
    domain_range: Optional[str] = None       # 域差范围描述
    domain_shift: bool = False               # 已知存在未排除的域差
    roi: Optional[list] = None               # (y1,x1,y2,x2)
    context_px: int = 0
    min_patches: float = 2.0                 # 最小缺陷在视图中应占 patch 数（候选值）
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, cfg):
        p = ((cfg or {}).get("tiling", {}) or {}).get("profile") or {}
        keys = set(cls.__dataclass_fields__) - {"extra"}
        kw = {k: p[k] for k in keys if k in p}
        kw["extra"] = {k: v for k, v in p.items() if k not in keys}
        return cls(**kw)

    def to_dict(self):
        return asdict(self)


def select_view_mode(profile, image_size, grid=32, train_auroc=None,
                     domain_auroc_thresh=0.85, model_side=448):
    """返回 {mode, reason, bounds, patches_global, profile, unfrozen}。

    顺序（AC-001）：
      1) 域差优先：profile.domain_shift 或 机制 C（train_auroc < 阈值）→ domain_first
         （先排除域差再谈分辨率，局部分支不启用）；
      2) min_defect_px 缺失 → profile_missing（整图）；
      3) 整图视图下最小缺陷占 patch 数 ≥ min_patches → single；
      4) 否则 global_local，附块尺寸上下界；冲突 → global_only（交给全局分支）。
    """
    H, W = image_size
    out = {"profile": profile.to_dict(), "bounds": None, "patches_global": None,
           "unfrozen": ["contrast", "miss_req", "min_patches"]}
    if profile.domain_shift or (train_auroc is not None
                                and float(train_auroc) < float(domain_auroc_thresh)):
        out.update(mode="domain_first",
                   reason="存在未排除的域差（档案或机制C train_auroc 低），先排除域差")
        return out
    if not profile.min_defect_px or profile.min_defect_px <= 0:
        out.update(mode="profile_missing", reason="min_defect_px 未提供，保持整图")
        return out
    pg = float(profile.min_defect_px) * grid / float(max(H, W))
    out["patches_global"] = round(pg, 3)
    if pg >= profile.min_patches:
        out.update(mode="single",
                   reason=f"整图视图最小缺陷占 {pg:.2f} patch ≥ {profile.min_patches}")
        return out
    b = tile_size_bounds(profile.min_defect_px, grid=grid, model_side=model_side,
                         min_patches=profile.min_patches, context_px=profile.context_px,
                         max_defect_px=profile.max_defect_px)
    out["bounds"] = b
    if b["status"] != "ok":
        out.update(mode="global_only", reason=b["reason"])
        return out
    out.update(mode="global_local",
               reason=f"整图视图最小缺陷仅 {pg:.2f} patch，需局部视图；块尺寸 ∈ [{b['lo']},{b['hi']}]")
    return out
