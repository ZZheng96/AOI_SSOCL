"""统一特征注册表（AC-010，项目文档 §3.2.3）

生产 trad 与评价共用同一提取实现（algo/vendor/traditional.py）；这里只登记
15 组 200 维的组级元数据（ID/版本/函数/参数/维度切片/视图/启用/档案引用），
7 类（color/geometry/texture/freq/blob/logic/material）仅为语义分类，不参与切片。
DINO-patch 主干单独登记，不进入 trad 切片。组启用通过维度掩码实现，
不靠 set_dim 截断（截断仅保留给 172 维旧快照兼容）。
"""
import hashlib
import json

REGISTRY_VERSION = 1
EXTRACT_FN = "algo.vendor.traditional.TraditionalFeatureExtractor.extract"
PREPROCESS = {"max_side": 256, "min_side": 32, "color": "RGB", "nan_policy": "zero",
              "selfsim_rng": "default_rng(42)", "global_seed": 0}
CATEGORIES = ("color", "geometry", "texture", "freq", "blob", "logic", "material")

# (key, 名称, 语义类, 维度, 参数)
_SPEC = [
    ("glcm", "GLCM 纹理", "texture", 4, {"distances": [1], "angles": 4, "levels": 256}),
    ("selfsim", "区域自相似", "texture", 5, {"scales": [4, 8, 16], "max_blocks": 100}),
    ("edge", "边缘梯度+LBP", "texture", 28, {"canny_sigma": [1, 2, 3], "ang_bins": 16, "lbp": [8, 1, 6]}),
    ("hog", "HOG", "geometry", 11, {"orient": 9, "cell": 16, "block": 2, "keep": 11}),
    ("shape", "形状/Hu/轮廓", "geometry", 11, {"hu": 7, "contour": 4, "thresh": "otsu"}),
    ("gray", "灰度统计", "material", 12, {}),
    ("gloss", "材质反光/高光", "material", 4, {"k_sigma": 2, "min_obj": 50}),
    ("color", "HSV 直方图+色彩矩+RGB 相关", "color", 36, {"bins": 8}),
    ("fft", "FFT 频域", "freq", 13, {"low_radius": "min/4", "radial_bins": 10}),
    ("wavelet", "Haar 小波", "freq", 40, {"wavelet": "haar", "level": 3}),
    ("logic", "逻辑关系代理(连通域/对称)", "logic", 6, {"thresh": "otsu(shape)"}),
    ("gabor", "Gabor", "texture", 8, {"thetas": 4, "freqs": [0.1, 0.3], "ksize": 21}),
    ("log", "LoG 斑点", "blob", 5, {"sigmas": [1, 2, 4, 8, 16]}),
    ("pc", "相位一致性", "blob", 3, {"nscale": 4, "norient": 4}),
    ("surf", "SURF 多尺度 Hessian 斑点", "blob", 14, {"octaves": 3, "scales": 3}),
]


def _build():
    groups, start = [], 0
    for key, name, cat, dim, params in _SPEC:
        groups.append({"key": key, "name": name, "category": cat, "dim": dim,
                       "slice": [start, start + dim], "params": params, "version": 1,
                       "fn": EXTRACT_FN, "views": ["global", "tile"], "enabled": True,
                       "profile": None})
        start += dim
    return groups


TRAD_GROUPS = _build()
TRAD_DIM = TRAD_GROUPS[-1]["slice"][1]
GROUP_KEYS = tuple(g["key"] for g in TRAD_GROUPS)
BACKBONE = {"key": "dino_patch", "name": "DINO patch token", "category": "backbone",
            "dim": 384, "fn": "algo.backbone.dino", "views": ["tile"], "version": 1,
            "enabled": True, "note": "主干单独登记，不进入 trad 切片"}


def group(key):
    for g in TRAD_GROUPS:
        if g["key"] == key:
            return g
    raise KeyError(f"未知特征组: {key}")


def by_category():
    out = {c: [] for c in CATEGORIES}
    for g in TRAD_GROUPS:
        out[g["category"]].append(g["key"])
    return out


def resolve_groups(spec):
    """spec: None/'all' → 全部；list/tuple → 指定组（保持注册表顺序）；校验未知组。"""
    if spec in (None, "all", "legacy"):
        return list(GROUP_KEYS)
    keys = [spec] if isinstance(spec, str) else list(spec)
    unknown = [k for k in keys if k not in GROUP_KEYS]
    if unknown:
        raise ValueError(f"未知特征组: {unknown}")
    if not keys:
        raise ValueError("至少启用一个特征组")
    return [k for k in GROUP_KEYS if k in keys]


def dim_index(keys, dim=TRAD_DIM):
    """启用组 → 维度索引（dim<200 时截掉超界维，兼容 172 维旧快照）。"""
    idx = []
    for k in keys:
        a, b = group(k)["slice"]
        idx.extend(i for i in range(a, b) if i < dim)
    return idx


def check_trace(trace):
    """对照提取器分组边界标记 [(key, end, ms)] 与注册表切片；返回差异列表。"""
    errs = []
    if [t[0] for t in trace] != list(GROUP_KEYS):
        errs.append(f"组顺序不一致: {[t[0] for t in trace]}")
        return errs
    for (key, end, _ms), g in zip(trace, TRAD_GROUPS):
        if end != g["slice"][1]:
            errs.append(f"{key}: 提取边界 {end} != 注册表 {g['slice'][1]}")
    return errs


def manifest(enabled=None):
    keys = resolve_groups(enabled)
    groups = [dict(g, enabled=g["key"] in keys) for g in TRAD_GROUPS]
    return {"registry_version": REGISTRY_VERSION, "extractor": EXTRACT_FN,
            "preprocess": PREPROCESS, "dimension": TRAD_DIM, "groups": groups,
            "categories": by_category(), "backbone": BACKBONE}


def registry_hash():
    body = {"v": REGISTRY_VERSION, "pre": PREPROCESS,
            "groups": [{k: g[k] for k in ("key", "dim", "slice", "params", "version", "fn")}
                       for g in TRAD_GROUPS]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()
