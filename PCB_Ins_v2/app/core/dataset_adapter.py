# -*- coding: utf-8 -*-
"""数据集适配模块（前端反馈 2026-08-31：非标准数据集识别 + 用户确认后导入）。

设计动机：总不能要求每个数据集都是标准形式（MVTec 目录）。data_origin（mvtec/
MPDD/BTAD/GYU-DET）、data_local（自采 4 品类）、test_images（solder/shift/jump，
_OK 后缀、templ_ 前缀、ok/NG/良品/误报/漏检 命名约定）各有各的组织方式。
本模块只读扫描目录 -> 识别格式与内容构成 -> 产出确认报告（probe_dataset），
用户在前端确认/修正映射后按映射批量入库（import_adapted）。

支持格式：
  mvtec_like   {根}/{品类}/train|test/{good|ok|缺陷类型}（mvtec/MPDD/BTAD/data_local；
               根本身即单品类时也识别，如 data_local/component）
  yolo_split   {根}/{train,valid,test}/{images,labels} + YOLO txt（GYU-DET；
               label 文件存在且非空 -> 缺陷，缺失/空 -> 正常）
  dir_rules    目录/文件名命名规则识别（test_images_* 全系）：
               目录标记：模板/模板小图、ok/ok2/good/良品/误报(=假报警，实为正常)、
               NG/miss/漏检/不良(=缺陷)；文件名标记：templ_ 前缀 -> 模板图，
               _OK 后缀 -> 正常图，_Shift_/_MissPart_ 等缺陷关键词 -> 缺陷图
  plain        未识别出任何结构/标记（全部待用户确认）

识别优先级（dir_rules，逐条命中即止）：
  1 目录含模板标记            -> template（L3 模板库）
  2 目录含缺陷标记            -> anomaly
  3 文件名模板标记(templ_/_tpl/_template) -> template
  4 文件名缺陷关键词(shift/misspart/…)   -> anomaly（缺陷类型=关键词）
  5 目录含正常标记            -> normal（误报目录=正常，AOI 假报警）
  6 文件名 _OK 后缀           -> normal
  7 其余 -> 未识别：默认按「缺陷」入检测组，由确认弹窗按分组修正
  （目录标记优先于文件名：NG 目录里的 templ_*.jpg 是缺陷样本，不是模板）

性能口径：probe 只做目录遍历 + 少量 txt 读取（GYU-DET ~1100 个），不解码图像，
10k+ 文件秒级完成；导入复用 ingest._register_batch（快速指纹，逐图进度）。
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.core.datasets import create_dataset, refresh_dataset_count, update_dataset_params
from app.core.ingest import IMG_EXTS, _register_batch

# ── 格式元数据 ────────────────────────────────────────────────
FORMATS: Dict[str, dict] = {
    "mvtec_like": {
        "cn": "标准目录结构（MVTec/BTAD/MPDD/data_local）",
        "desc": "{品类}/train/good 正常 + test/{good|缺陷类型}；"
                "_template/_tpl 命名与 template 目录识别为模板图，"
                "ground_truth/*_mask 掩码计数不入库。",
    },
    "yolo_split": {
        "cn": "YOLO 标注数据集（train/valid/test + labels）",
        "desc": "images + 同名 txt：txt 非空 -> 缺陷，缺失/空 -> 正常；"
                "目录名 train/valid/test 直接映射 split。",
    },
    "dir_rules": {
        "cn": "目录/命名规则数据集（OK/NG/模板/良品/误报 等约定）",
        "desc": "目录标记（模板、ok/良品/误报、NG/漏检/不良）+ 文件名标记"
                "（templ_ 模板、_OK 正常、_Shift_/_MissPart_ 缺陷）逐图识别；"
                "未识别部分按分组默认缺陷导入，可在确认弹窗修正。",
    },
    "plain": {
        "cn": "未识别出结构（全部待确认）",
        "desc": "未发现任何已知目录/命名标记，全部图片标记为未识别，"
                "请在确认弹窗中逐分组指定处理方式。",
    },
}

IMG_EXTS = IMG_EXTS or {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 探测文件数上限（防止误选巨大根目录拖死服务；超出截断并警告）
PROBE_MAX_IMAGES = 60000

# ── 目录标记（英文精确匹配，中文子串匹配）────────────────────
_DIR_TEMPLATE_EXACT = {"template", "templates", "tpl", "gold_sample", "样板"}
_DIR_NORMAL_EXACT = {"ok", "ok2", "ok3", "good", "good2", "normal", "norm",
                     "pass", "clean", "false-err", "false_err", "falseerr"}
_DIR_ANOMALY_EXACT = {"ng", "ng2", "ng3", "ko", "bad", "defect", "defects",
                      "fail", "miss", "miss2", "missing", "anomaly", "anomalous"}
# 中文子串（目录名含即命中）：模板小图 含「模板」、OK小图/误报小图 含「误报」等
_DIR_SUBSTR = {"模板": "template", "良品": "normal", "误报": "normal",
               "OK小图": "normal", "漏检": "anomaly", "缺陷": "anomaly",
               "不良": "anomaly", "坏品": "anomaly"}

# ── 文件名标记 ────────────────────────────────────────────────
_FN_TPL_PREFIX = ("templ_", "templ-")
_FN_TPL_SUFFIX = ("_tpl", "_template")
# 缺陷关键词：子串匹配（多为多字符复合词，误报率低）
_FN_DEFECT_SUB = ("shift", "misspart", "missing", "tombstone", "less_tin",
                  "over_tin", "broken", "damaged", "cold_solder",
                  "insufficient", "excess", "defect")
# 缺陷关键词：词元匹配（短词，避免 ng 之类子串误伤 PixPin 等文件名）
_FN_DEFECT_TOKEN = {"miss", "ng", "bad", "ko"}

_SPLIT_DIRS = {"train": "train", "test": "test", "val": "val",
               "valid": "val", "validation": "val"}
# 无语义目录（找缺陷类型时跳过）：纯数字/日期/pairN/groupN/params 等
_SKIP_DIR_RE = re.compile(r"^\d+$|^\d{6,}|^pair\d*$|^group\d*$|^params?$",
                          re.IGNORECASE)
_MASK_SUFFIX = "_mask"


def _norm_dir_marker(name: str) -> Optional[str]:
    """目录名 -> 标记类型（template/normal/anomaly），无标记返回 None。"""
    low = name.strip().lower()
    if low in _DIR_TEMPLATE_EXACT:
        return "template"
    if low in _DIR_NORMAL_EXACT:
        return "normal"
    if low in _DIR_ANOMALY_EXACT:
        return "anomaly"
    for key, mark in _DIR_SUBSTR.items():
        if key in name:
            return mark
    return None


def _fn_template(stem: str) -> bool:
    low = stem.lower()
    return low.startswith(_FN_TPL_PREFIX) or low.endswith(_FN_TPL_SUFFIX)


def _fn_ok(stem: str) -> bool:
    """_OK / _OK_ 后缀（SMT 命名：X.png 缺陷 + X_OK.png 正常）。"""
    return low_ok(stem)


def low_ok(stem: str) -> bool:
    low = stem.lower().rstrip("_")
    return low.endswith("_ok") or low == "ok"


def _fn_defect(stem: str) -> Optional[str]:
    """文件名缺陷关键词 -> 缺陷类型名；无返回 None。"""
    low = stem.lower()
    for kw in _FN_DEFECT_SUB:
        if kw in low:
            return kw
    tokens = set(re.split(r"[^0-9a-zA-Z]+", low))
    for kw in _FN_DEFECT_TOKEN:
        if kw in tokens:
            return kw
    return None


def _split_hint(parts: tuple) -> Optional[str]:
    for p in parts:
        sp = _SPLIT_DIRS.get(str(p).strip().lower())
        if sp:
            return sp
    return None


def _defect_type_from_parts(parts: tuple, group: str) -> Optional[str]:
    """从最深往上找第一个「有语义」目录名作缺陷类型（跳过标记/序号目录）。

    分组名去 _pair 后缀（test_images_solder：insufficient_pair -> insufficient）。
    """
    for p in reversed(parts):
        if _norm_dir_marker(p) or _SKIP_DIR_RE.match(p):
            continue
        if p in ("images", "labels"):
            continue
        return str(p).removesuffix("_pair") or str(p)
    if group:
        return str(group).removesuffix("_pair") or None
    return None


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


# ── 格式判定 ──────────────────────────────────────────────────
def _detect_format(root: Path) -> tuple[str, List[Path]]:
    """返回 (format, mvtec 品类目录列表)。"""
    if ((root / "train" / "images").is_dir()
            and (root / "train" / "labels").is_dir()):
        return "yolo_split", []
    subdirs = [d for d in root.iterdir()
               if d.is_dir() and not d.name.startswith(".")]
    cats = [d for d in subdirs
            if (d / "train").is_dir() or (d / "test").is_dir()]
    if cats:
        return "mvtec_like", cats
    # 根本身即单品类（如 data_local/component）：train/{good|ok|normal}
    for split in ("train", "test"):
        d = root / split
        if d.is_dir() and any(
                (d / m).is_dir() for m in ("good", "ok", "normal")):
            return "mvtec_like", [root]
    # 单一包装目录（如 BTAD/BTech_Dataset_transformed/{01,02,03}）：下钻一层
    if len(subdirs) == 1:
        inner = subdirs[0]
        inner_cats = [d for d in inner.iterdir()
                      if d.is_dir() and not d.name.startswith(".")
                      and ((d / "train").is_dir() or (d / "test").is_dir())]
        if inner_cats:
            return "mvtec_like", inner_cats
    return "dir_rules", []


# ── 逐图分类引擎（probe 与 import 共用，保证口径一致）────────
def _classify_dir_rules(rel: Path) -> dict:
    """dir_rules/plain 逐图分类。rel 相对 root 的路径。

    返回 {label: normal/anomaly/template/unknown, split_hint, defect_type}。
    split/label 的最终确定在导入时结合 ok_role 与分组 override。
    """
    parts = tuple(rel.parts[:-1])
    stem = rel.stem
    group = rel.parts[0] if len(rel.parts) > 1 else ""
    # 1-3 目录标记（模板 > 缺陷 > 正常）。目录是用户组织语义，优先于文件名
    # （误报/ok 目录里带 ng/miss 字样的文件是假报警，不是缺陷；
    #  NG 目录里的 templ_*.jpg 是缺陷样本，不是模板）。
    # 多层都有标记时取最深层（离文件最近、语义最具体）。
    for p in reversed(parts):
        m = _norm_dir_marker(p)
        if m == "template":
            return {"label": "template", "split_hint": None,
                    "defect_type": None}
        if m == "anomaly":
            return {"label": "anomaly", "split_hint": _split_hint(parts),
                    "defect_type": _defect_type_from_parts(parts, group)}
        if m == "normal":
            return {"label": "normal", "split_hint": _split_hint(parts),
                    "defect_type": None}
    # 4 文件名模板标记（templ_ 前缀 / _tpl / _template 后缀）
    if _fn_template(stem):
        return {"label": "template", "split_hint": None, "defect_type": None}
    # 5 文件名缺陷关键词（_Shift_ / _MissPart_ / less_tin 等）
    kw = _fn_defect(stem)
    if kw:
        return {"label": "anomaly", "split_hint": _split_hint(parts),
                "defect_type": kw}
    # 6 文件名 _OK 后缀（X.png 缺陷 + X_OK.png 正常的成对命名）
    if _fn_ok(stem):
        return {"label": "normal", "split_hint": _split_hint(parts),
                "defect_type": None}
    # 7 未识别
    return {"label": "unknown", "split_hint": _split_hint(parts),
            "defect_type": _defect_type_from_parts(parts, group)}


def _classify_mvtec(rel: Path) -> dict:
    """mvtec_like 单品类内分类。rel 相对品类目录。

    {train|test|val}/{good|ok|normal|缺陷类型}/file；模板命名/目录优先识别；
    _mask 文件与 ground_truth 目录按掩码计数（不入库）。
    """
    parts = tuple(rel.parts[:-1])
    stem = rel.stem
    if stem.lower().endswith(_MASK_SUFFIX):
        return {"label": "mask", "split_hint": None, "defect_type": None}
    if any(str(p).lower() in ("ground_truth", "ground-truth") for p in parts):
        return {"label": "mask", "split_hint": None, "defect_type": None}
    split = _split_hint(parts)
    # 模板：目录标记或文件名标记
    if any(_norm_dir_marker(p) == "template" for p in parts) or _fn_template(stem):
        return {"label": "template", "split_hint": None, "defect_type": None}
    if not parts:
        return {"label": "unknown", "split_hint": split, "defect_type": None}
    sub = str(parts[1]) if len(parts) > 1 else str(parts[0])
    m = _norm_dir_marker(sub)
    if m == "normal":
        return {"label": "normal", "split_hint": split, "defect_type": None}
    if m == "anomaly" or (m is None and sub not in ("train", "test")):
        # 缺陷类型目录（missing/shift/bent/…；BTAD ko 也在此列）
        if len(parts) > 1:
            return {"label": "anomaly", "split_hint": split,
                    "defect_type": sub}
        return {"label": "unknown", "split_hint": split, "defect_type": None}
    return {"label": "unknown", "split_hint": split, "defect_type": None}


def _yolo_label(img: Path) -> Optional[bool]:
    """YOLO 标注：True 缺陷 / False 正常。

    查找顺序：{split}/labels/{stem}.txt（GYU-DET images/labels 分离结构）->
    同目录 {stem}.txt（sidecar 约定）。无 txt 按正常。
    """
    candidates = [img.parent.parent / "labels" / (img.stem + ".txt"),
                  img.with_suffix(".txt")]
    for lbl in candidates:
        if lbl.is_file():
            try:
                with open(lbl, encoding="utf-8", errors="ignore") as fh:
                    return any(line.strip() for line in fh)
            except OSError:
                return False
    return False


# ── probe：只读扫描，产出确认报告 ─────────────────────────────
def probe_dataset(root: str | Path) -> dict:
    """识别数据集内容（不写库）。返回格式/品类/分组/计数/告警，供确认弹窗展示。"""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"目录不存在: {root}")
    fmt, cats = _detect_format(root)
    meta = FORMATS[fmt]
    report = {
        "root": str(root),
        "format": fmt,
        "format_cn": meta["cn"],
        "format_desc": meta["desc"],
        "category_hint": root.name,
        "n_images": 0,
        "n_masks": 0,
        "n_other": 0,
        "categories": [],
        "warnings": [],
    }
    truncated = False
    n_other = 0

    if fmt == "mvtec_like":
        cat_items = []
        for cat_dir in cats:
            counts = {"train_normal": 0, "test_normal": 0, "test_anomaly": 0,
                      "val_normal": 0, "val_anomaly": 0, "template": 0,
                      "unknown": 0}
            n_masks = 0
            samples: List[str] = []
            for dp, _dn, fn in os.walk(cat_dir):
                for f in fn:
                    p = Path(dp) / f
                    if _is_image(p):
                        if report["n_images"] >= PROBE_MAX_IMAGES:
                            truncated = True
                            continue
                        rel = p.relative_to(cat_dir)
                        c = _classify_mvtec(rel)
                        if c["label"] == "mask":
                            n_masks += 1
                            continue
                        report["n_images"] += 1
                        lbl = c["label"]
                        if lbl == "template":
                            counts["template"] += 1
                        elif lbl == "unknown":
                            counts["unknown"] += 1
                        else:
                            sp = c["split_hint"] or "train"
                            counts[f"{sp}_{lbl}"] = \
                                counts.get(f"{sp}_{lbl}", 0) + 1
                        if len(samples) < 3:
                            samples.append(str(rel))
                    else:
                        n_other += 1
            cat_items.append({"name": cat_dir.name, "dir": str(cat_dir),
                              "counts": counts, "n_masks": n_masks,
                              "n_images": sum(counts.values()),
                              "samples": samples})
            if not cat_items[-1]["n_images"]:
                report["warnings"].append(
                    f"品类 {cat_dir.name} 无图片（空目录），默认不导入")
        report["categories"] = cat_items
        if not any(it["counts"]["train_normal"] or it["counts"]["val_normal"]
                   for it in cat_items):
            report["warnings"].append("没有任何 train 域正常图：预训练组将为空")
    elif fmt == "yolo_split":
        counts = {"train_normal": 0, "train_anomaly": 0,
                  "val_normal": 0, "val_anomaly": 0,
                  "test_normal": 0, "test_anomaly": 0}
        samples: List[str] = []
        for split_dir in ("train", "valid", "validation", "val", "test"):
            d = root / split_dir / "images"
            if not d.is_dir():
                continue
            sp = _SPLIT_DIRS[split_dir]
            for dp, _dn, fn in os.walk(d):
                for f in fn:
                    p = Path(dp) / f
                    if not _is_image(p):
                        continue
                    if report["n_images"] >= PROBE_MAX_IMAGES:
                        truncated = True
                        continue
                    report["n_images"] += 1
                    has_def = _yolo_label(p)
                    counts[f"{sp}_{'anomaly' if has_def else 'normal'}"] += 1
                    if len(samples) < 3:
                        samples.append(str(p.relative_to(root)))
        report["categories"] = [{"name": root.name, "dir": str(root),
                                 "counts": counts, "n_images": report["n_images"],
                                 "samples": samples}]
        if not counts["train_normal"]:
            report["warnings"].append(
                "train 域没有正常图（txt 全部非空或无 train）：预训练组将为空")
    else:
        # dir_rules / plain：单品类，顶层目录为分组
        groups: Dict[str, dict] = {}
        for dp, _dn, fn in os.walk(root):
            for f in fn:
                p = Path(dp) / f
                if not _is_image(p):
                    n_other += 1
                    continue
                if report["n_images"] >= PROBE_MAX_IMAGES:
                    truncated = True
                    continue
                rel = p.relative_to(root)
                group = rel.parts[0] if len(rel.parts) > 1 else ""
                c = _classify_dir_rules(rel)
                g = groups.setdefault(
                    group, {"name": group or "（根目录）", "dir": group,
                            "normal": 0, "anomaly": 0, "template": 0,
                            "unknown": 0, "samples": []})
                g[c["label"]] = g.get(c["label"], 0) + 1
                if len(g["samples"]) < 3:
                    g["samples"].append(str(rel))
                report["n_images"] += 1
        group_items = sorted(groups.values(),
                             key=lambda g: (g["name"] == "（根目录）", g["name"]))
        report["categories"] = [{"name": root.name, "dir": str(root),
                                 "counts": {"train_normal": 0,
                                            "test_anomaly": 0,
                                            "template": 0},
                                 "groups": group_items,
                                 "n_images": report["n_images"]}]
        n_unknown = sum(g["unknown"] for g in group_items)
        n_normal = sum(g["normal"] for g in group_items)
        if n_unknown:
            names = "、".join(g["name"] for g in group_items if g["unknown"])[:200]
            report["warnings"].append(
                f"有 {n_unknown} 张图片未能自动识别（分组：{names}…），"
                f"默认按缺陷导入，请在下方逐分组确认")
        if not n_normal and not any(g["template"] for g in group_items):
            report["warnings"].append(
                "未识别到任何正常/模板图：预训练组将为空（若数据集有 OK/良品/"
                "误报目录请检查命名）")
        if fmt == "plain":
            report["warnings"].insert(
                0, "未识别出已知数据集结构，全部图片待确认处理方式")
    report["n_other"] = n_other
    if truncated:
        report["warnings"].append(
            f"目录图片数超过 {PROBE_MAX_IMAGES}，识别已截断（导入同样只处理前 "
            f"{PROBE_MAX_IMAGES} 张）")
    return report


# ── import：按确认映射入库 ────────────────────────────────────
def import_adapted(root: str | Path, fmt: str,
                   category: Optional[str] = None,
                   ok_role: str = "train",
                   group_overrides: Optional[Dict[str, str]] = None,
                   categories: Optional[List[str]] = None,
                   dataset_name: Optional[str] = None,
                   name: Optional[str] = None,
                   note: Optional[str] = None,
                   datasource_id: Optional[int] = None,
                   progress_cb: Optional[Callable] = None) -> dict:
    """按确认后的映射导入。后台任务内执行，progress_cb(done, total)。

    category：单品类格式（yolo_split/dir_rules/plain）的品类名，默认根目录名。
    ok_role：dir_rules 识别为正常且无显式 split 的图片归属（train|template）。
    group_overrides：{分组名: anomaly|normal|skip}，作用于该分组「未识别」图片。
    categories：mvtec_like 要导入的品类名列表（None=全部有图的品类）。
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"目录不存在: {root}")
    group_overrides = group_overrides or {}
    ok_role = ok_role if ok_role in ("train", "template") else "train"
    dataset_name = dataset_name or root.name

    # 1) 扫描 + 分类，构建 (品类, [(path, split, label, defect_type)])
    per_cat: Dict[str, List[tuple]] = {}
    n_masks_total = 0

    def _collect(cat: str, path: Path, split: str, label: str,
                 defect: Optional[str]) -> None:
        per_cat.setdefault(cat, []).append((path, cat, split, label, defect))

    if fmt == "mvtec_like":
        _, cats = _detect_format(root)
        sel = set(categories) if categories is not None else None
        for cat_dir in cats:
            cat_name = cat_dir.name if cat_dir != root else (category or root.name)
            if sel is not None and cat_name not in sel:
                continue
            for dp, _dn, fn in os.walk(cat_dir):
                for f in fn:
                    p = Path(dp) / f
                    if not _is_image(p):
                        continue
                    c = _classify_mvtec(p.relative_to(cat_dir))
                    if c["label"] == "mask":
                        n_masks_total += 1
                        continue
                    if c["label"] == "template":
                        _collect(cat_name, p, "template", "normal", None)
                    elif c["label"] == "normal":
                        _collect(cat_name, p, c["split_hint"] or "train",
                                 "normal", None)
                    elif c["label"] == "anomaly":
                        _collect(cat_name, p, c["split_hint"] or "test",
                                 "anomaly", c["defect_type"])
                    else:
                        _collect(cat_name, p, c["split_hint"] or "unlabeled",
                                 "unknown", None)
    elif fmt == "yolo_split":
        cat = category or root.name
        for split_dir in ("train", "valid", "validation", "val", "test"):
            d = root / split_dir / "images"
            if not d.is_dir():
                continue
            sp = _SPLIT_DIRS[split_dir]
            for dp, _dn, fn in os.walk(d):
                for f in fn:
                    p = Path(dp) / f
                    if not _is_image(p):
                        continue
                    has_def = _yolo_label(p)
                    _collect(cat, p, sp,
                             "anomaly" if has_def else "normal", None)
    else:
        cat = category or root.name
        for dp, _dn, fn in os.walk(root):
            for f in fn:
                p = Path(dp) / f
                if not _is_image(p):
                    continue
                rel = p.relative_to(root)
                group = rel.parts[0] if len(rel.parts) > 1 else ""
                c = _classify_dir_rules(rel)
                if c["label"] == "template":
                    _collect(cat, p, "template", "normal", None)
                elif c["label"] == "normal":
                    sp = c["split_hint"] or ok_role
                    _collect(cat, p, sp, "normal", None)
                elif c["label"] == "anomaly":
                    _collect(cat, p, c["split_hint"] or "test", "anomaly",
                             c["defect_type"])
                else:
                    ov = group_overrides.get(group, "anomaly")
                    if ov == "skip":
                        continue
                    if ov == "normal":
                        _collect(cat, p, c["split_hint"] or "train",
                                 "normal", None)
                    else:
                        _collect(cat, p, c["split_hint"] or "test", "anomaly",
                                 c["defect_type"])

    total_records = sum(len(v) for v in per_cat.values())
    if not total_records:
        return {"imported": 0, "total": 0, "dataset_ids": {},
                "n_masks": n_masks_total, "format": fmt}

    # 2) 逐品类建批次 + 批量入库（复用快速指纹与幂等去重）
    dataset_ids: Dict[str, int] = {}
    imported = 0
    done_offset = 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for cat, records in per_cat.items():
        ds_id = create_dataset(
            cat, fmt, source_path=str(root),
            params={"format": fmt, "ok_role": ok_role,
                    "group_overrides": group_overrides, "root": str(root)},
            name=name or f"{cat}_{ts}", note=note or "",
            dataset_name=dataset_name, datasource_id=datasource_id)
        dataset_ids[cat] = ds_id

        def _cb(done: int, tot: int, _off=done_offset, _tot=total_records,
                _cat=cat) -> None:
            if progress_cb:
                progress_cb(_off + done, _tot, f"导入 {_cat} {done}/{tot}")

        r = _register_batch(records, "adapt", _cb, dataset_id=ds_id)
        imported += r.get("imported", 0)
        done_offset += len(records)
        refresh_dataset_count(ds_id)
        if n_masks_total:
            update_dataset_params(ds_id, {"n_masks": n_masks_total})
    return {"imported": imported, "total": total_records,
            "dataset_ids": dataset_ids, "n_masks": n_masks_total,
            "format": fmt, "categories": sorted(per_cat)}
