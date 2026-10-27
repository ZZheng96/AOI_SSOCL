"""Round2 发布运维报告（AC-011～016/028～035/060/061）。

纯函数、只读：盘点/审计/计划/门禁，不进推理热路径、不改生产指针/模型/库。
"""
import hashlib
import json
from copy import deepcopy

from .features import registry as freg
from .fusion.cde_report import (SLOT_CAPABILITIES, SLOT_ROLES, POST_FUSION_ROLES, slot_health)
from .feature_revision import digest

SEVEN = ("sem", "disc", "blob", "trad", "layout", "inp", "tpl")
WRITE_ROLES = ("engineer", "admin")
ADMISSION_GATE = ("gain", "redundancy", "regression", "g1", "g2", "version")


def _require_write(role):
    if role not in WRITE_ROLES:
        raise PermissionError("仅工程师可修改；操作员只读")


# ---------------------------------------------------------------- AC-011
def group_selection_plan(results, min_auroc=0.55, min_n=3):
    """按 (category, defect, view) 独立选组。results: [{category, defect, view,
    groups:{k:{auroc,n}}}]。整图差不否定局部；样本不足不选；只产出 overlay 建议。"""
    plan = {}
    for r in results:
        key = "/".join(str(r.get(k) or "*") for k in ("category", "defect", "view"))
        sel, rej = [], {}
        for k, g in (r.get("groups") or {}).items():
            a, n = g.get("auroc"), int(g.get("n") or 0)
            if n < min_n or a is None:
                rej[k] = "insufficient"
            elif a < 0.5:
                rej[k] = "reversed"
            elif a < min_auroc:
                rej[k] = "weak"
            else:
                sel.append(k)
        plan[key] = {"view": r.get("view"), "selected": sorted(sel), "rejected": rej,
                     "overlay": {"trad": {"groups": sorted(sel), "scoring": "grouped"}}
                     if sel else None,
                     "status": "proposal" if sel else "keep_current"}
    # 局部视图独立：global 弃用的组若在 tile 视图有效仍保留
    return {"plans": plan, "auto_apply": False, "force_all_groups": False,
            "semantics": "各(品类,缺陷,视图)独立；整图无增益不否定局部；仅建议，需准入后生效"}


# ---------------------------------------------------------------- AC-012
PROFILE_GROUP_FIELDS = ("single_auroc", "single_ci95", "direction", "fixed_loo_auroc",
                        "rebuild_loo_auroc", "permutation_auroc", "redundancy", "extract_ms")


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
                          .encode()).hexdigest()


def stamp_profile_lineage(profile, data_items, config=None, code_hash=None):
    """补齐档案血缘：code/data/config hash。data_items: [(sha256 或 path, label)]。"""
    if code_hash is None:
        from .persist import code_sha256
        code_hash = code_sha256()
    out = deepcopy(profile)
    meta = out.setdefault("meta", {})
    meta["code_sha256"] = code_hash
    meta["data_hash"] = _sha(sorted([str(a), int(b)] for a, b in data_items))
    meta["config_hash"] = _sha(config if config is not None else meta.get("config"))
    return out


def profile_completeness(profile, min_per_class=3):
    if not profile or profile.get("status") != "available":
        return {"status": "missing" if not profile else "insufficient",
                "reason": (profile or {}).get("reason", "无档案")}
    meta = profile.get("meta") or {}
    gaps = [k for k in ("registry_version", "registry_sha256", "code_sha256", "data_hash",
                        "config_hash", "category", "data_source") if not meta.get(k)]
    if "full_ci95" not in profile:
        gaps.append("full_ci95")
    if "cluster_loo" not in profile:
        gaps.append("cluster_loo")
    for k, g in (profile.get("groups") or {}).items():
        miss = [f for f in PROFILE_GROUP_FIELDS if f not in g]
        if g.get("enabled") is False:
            miss = [f for f in miss if not f.endswith("loo_auroc") and f != "permutation_auroc"]
        gaps += [f"{k}.{f}" for f in miss]
    small = min(meta.get("n_normal", 0), meta.get("n_defect", 0)) < min_per_class
    return {"status": "incomplete" if gaps else "insufficient" if small else "ok",
            "gaps": gaps, "small_sample": small,
            "n": {"normal": meta.get("n_normal"), "defect": meta.get("n_defect")}}


# ---------------------------------------------------------------- AC-013
def trad_protocol_audit(trad):
    """如实描述 trad 实际打分协议（不改分数）。"""
    rep = deepcopy(getattr(trad, "selection_report", None) or {})
    grouped = trad.scoring == "grouped" and bool(trad.group_weight)
    zero = sorted(k for k, w in (trad.group_weight or {}).items() if w <= 0)
    auto = rep.get("auto") or {}
    return {"scoring": trad.scoring,
            "distance": ("sum_k w_k*||Δ_k||/scale_k（组距离加权和）" if grouped
                         else "启用组全部维拼接后的单一 L2（legacy，与旧版逐位一致）"),
            "enabled": list(trad.groups),
            "near_const_dims_dropped": rep.get("near_const_dims", {}) if grouped else {},
            "near_const_note": None if grouped else "legacy 不剔除近零方差维（保数值一致）",
            "constant_groups": rep.get("constant_groups", []),
            "zero_weight_groups": zero if grouped else [],
            "reversed_groups": auto.get("reversed", []),
            "selection_data": rep.get("data"),
            "selection": auto.get("status", "manual" if trad.group_spec != "auto" else None),
            "test_used_for_selection": False}


# ---------------------------------------------------------------- AC-014
def box_attribution_audit(boxes, trad_explain=None):
    """逐框：区域证据 vs 整图回退复制；trad 参考来自 explain（若有）。"""
    rows, n_fb = [], 0
    for i, b in enumerate(boxes or []):
        ev = b.get("evidence") or {}
        fb = ev.get("level") == "image_fallback" or not b.get("dominant_slot")
        n_fb += fb
        rows.append({"i": i, "dominant_slot": b.get("dominant_slot"),
                     "level": "image_fallback" if fb else ev.get("level", "region"),
                     "copied_from_image": bool(fb)})
    ref = None
    if trad_explain:
        ref = {k: trad_explain.get(k) for k in ("group_breakdown", "top_dims", "neighbors")
               if k in trad_explain}
    return {"boxes": rows, "n_region": len(rows) - n_fb, "n_image_fallback": n_fb,
            "trad_reference": ref,
            "semantics": "image_fallback 框的主导槽位复制整图结论，不代表框内归因"}


# ---------------------------------------------------------------- AC-015 / AC-035
SNAPSHOT_REQUIRED = ("version", "cfg_hash", "artifact_hashes", "backbone_sha256",
                     "code_sha256", "components", "cache_inventory", "rng",
                     "python_version", "torch_version", "numpy_version",
                     "feature_revision_binding")


def snapshot_completeness(meta):
    missing = [k for k in SNAPSHOT_REQUIRED if meta.get(k) in (None, "", {}, [])]
    comps = meta.get("components") or {}
    absent = sorted(k for k, v in comps.items() if not v.get("present"))
    bind = meta.get("feature_revision_binding") or {}
    return {"status": "incomplete" if missing else "ok", "missing": missing,
            "absent_components": absent,
            "feature_revision": bind.get("status", "unknown"),
            "reproducible": not missing and bind.get("status") == "bound"}


def representation_change_plan(old_meta, new_meta):
    """表示变化（backbone/代码/特征注册/预处理）→ 重提/重建/重校准/G2/隔离。"""
    changed = [k for k in ("backbone_sha256", "code_sha256", "cfg_hash")
               if old_meta.get(k) and new_meta.get(k) and old_meta[k] != new_meta[k]]
    of = (old_meta.get("production_feature_metadata") or {})
    nf = (new_meta.get("production_feature_metadata") or {})
    if of and nf and of != nf:
        changed.append("feature_metadata")
    rep = "backbone_sha256" in changed or "feature_metadata" in changed
    steps = []
    if rep:
        steps = ["quarantine_old_banks", "re_extract_features", "rebuild_banks_and_heads",
                 "recalibrate", "g2_regression", "approve_then_bind"]
    elif changed:
        steps = ["recalibrate", "g2_regression", "approve_then_bind"]
    return {"changed": changed, "representation_changed": rep, "steps": steps,
            "mix_old_vectors": False if rep else None,
            "auto_apply": False}


# ---------------------------------------------------------------- AC-016 / AC-061
def report_status(profile, current_registry_sha=None, current_code_sha=None, min_per_class=3):
    """特征报告状态：missing / insufficient / stale / ok。"""
    if not profile:
        return "missing"
    if profile.get("status") != "available":
        return "insufficient"
    meta = profile.get("meta") or {}
    reg = current_registry_sha or freg.registry_hash()
    if meta.get("registry_sha256") != reg or (
            current_code_sha and meta.get("code_sha256") not in (None, current_code_sha)):
        return "stale"
    if min(meta.get("n_normal", 0), meta.get("n_defect", 0)) < min_per_class:
        return "insufficient"
    return "ok"


def revision_binding(store, model_snapshot):
    """AC-061：快照 meta.feature_revision_binding ← 生产指针（只读）；含血缘链。"""
    st = store.detection_stamp(model_snapshot)
    rid = st.get("feature_revision")
    return {"status": st["status"], "revision_id": rid,
            "lineage": store.lineage(rid) if rid else [],
            "reason": None if st["status"] == "bound" else
            "production_pointer_unbound" if rid is None else "model_snapshot_mismatch"}


def feature_ops_view(store, profiles=None, role="operator", **status_kw):
    """特征运维视图：revision 列表、生产指针、报告状态；operator 只读。"""
    rows = store.list()
    ptr = store.production_pointer()
    cur = (ptr or {}).get("current")
    out = []
    for rid, row in rows.items():
        out.append({"revision_id": rid, "state": row["state"],
                    "parent": row["manifest"].get("parent"),
                    "groups": [g["key"] for g in row["manifest"].get("groups", [])],
                    "production": rid == cur,
                    "report": report_status((profiles or {}).get(rid), **status_kw)})
    return {"revisions": out, "production": cur, "bound": cur is not None,
            "editable": role in WRITE_ROLES, "role": role}


def submit_toggle(store, base_rid, enable, disable, *, role, implementation_hashes=None):
    """启停组变更 → 仅登记 draft revision（parent=base），不切换生产。"""
    _require_write(role)
    base = store.list().get(base_rid)
    if not base:
        raise ValueError("基线 revision 不存在")
    m = deepcopy(base["manifest"])
    keys = [g["key"] for g in m["groups"]]
    keys = [k for k in keys if k not in set(disable or ())] + \
           [k for k in (enable or ()) if k not in keys]
    if not keys:
        raise ValueError("至少保留一个特征组")
    pos, groups = 0, []
    for k in keys:
        g = next((x for x in m["groups"] if x["key"] == k), None)
        if g is None:
            r = freg.group(k)
            g = {"key": k, "dim": int(r["dim"]), "params": dict(r.get("params") or {})}
        g = {**g, "slice": [pos, pos + int(g["dim"])]}
        pos += int(g["dim"])
        groups.append(g)
    m.update(groups=groups, dimension=pos, parent=base_rid)
    if implementation_hashes:
        m["implementation_hashes"] = implementation_hashes
    rid = store.register(m)
    return {"revision_id": rid, "state": "draft", "parent": base_rid,
            "production_changed": False}


# ---------------------------------------------------------------- AC-028
def _arr_hash(x):
    try:
        import numpy as np
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return hashlib.sha256(a.tobytes()).hexdigest()
    except Exception:
        return None


def memory_asset_inventory(pipe):
    """记忆资产清单：向量库/样例库是 memory 不是权重；头/路由/校准是参数。"""
    h = getattr(pipe, "handler", None)
    inv = {}
    sem = (getattr(pipe, "slots", {}) or {}).get("sem")
    if sem is not None:
        bank = getattr(sem, "bank", None)
        inv["sem_bank"] = {"kind": "memory", "memory_not_weights": True,
                           "n": int(len(bank)) if bank is not None else 0,
                           "sha256": _arr_hash(bank) if bank is not None else None,
                           "source": "train/good coreset"}
    nb = getattr(h, "normal_bank", None) if h is not None else None
    if nb is not None:
        inv["normal_bank"] = {**nb.lineage(), "kind": "memory", "memory_not_weights": True,
                              "sha256": _arr_hash(nb.core) if nb.core is not None else None}
    db = getattr(h, "defect_bank", None) if h is not None else None
    if db is not None:
        inv["defect_bank"] = {**db.lineage(), "kind": "memory", "memory_not_weights": True}
    hm = getattr(pipe, "head_mgr", None)
    if hm is not None:
        st = hm.status(db) if db is not None else {}
        inv["incubate_head"] = {"kind": "weights", "memory_not_weights": False,
                                "trained": bool(st.get("trained")),
                                "source": "纯 patch 样例孵育"}
    router = getattr(pipe, "router", None)
    inv["router"] = {"kind": "weights", "memory_not_weights": False,
                     "present": router is not None, "applied_to_fused": False}
    cals = getattr(pipe, "calibrators", {}) or {}
    inv["calibrators"] = {"kind": "statistics", "memory_not_weights": True,
                          "slots": sorted(cals),
                          "n_normal": {k: len(v) for k, v in
                                       (getattr(pipe, "_cal_normal_hist", {}) or {}).items()}}
    return inv


# ---------------------------------------------------------------- AC-033
def capability_card():
    """槽 + 融合后通道统一能力卡；字段缺省显式标 unknown。"""
    fields = ("role", "reference", "fit_type", "uses_defect", "train", "update",
              "localization", "domain_sensitivity", "not")
    card = {}
    for n, cap in SLOT_CAPABILITIES.items():
        row = {**cap, **SLOT_ROLES.get(n, {})}
        card[n] = {f: row.get(f, "unknown") for f in fields}
        card[n]["in_fusion"] = n != "open"
        card[n]["base_seven"] = n in SEVEN
    for n, r in POST_FUSION_ROLES.items():
        card[n] = {f: r.get(f, "unknown") for f in fields}
        card[n].update(in_fusion=False, base_seven=False, uses_defect=True)
    return card


# ---------------------------------------------------------------- AC-034
def admission_registration_gate(candidate, admission, snapshot_version, store=None,
                                manifest=None, role="engineer"):
    """证据齐全（gain/redundancy/regression/g1 时间/g2/version）才登记 draft；永不自动上线。"""
    _require_write(role)
    adm = admission or {}
    st = {k: (adm.get(k) or {}).get("status") for k in ADMISSION_GATE if k != "version"}
    st["version"] = ("passed" if adm.get("snapshot_version") is not None
                     and adm.get("snapshot_version") == snapshot_version else "failed")
    bad = [k for k, v in st.items() if v != "passed"]
    out = {"candidate": (candidate or {}).get("kind"), "checks": st, "missing": bad,
           "auto_online": False, "revision_id": None,
           "status": "blocked" if bad else "registered_draft"}
    if not bad and store is not None and manifest is not None:
        m = deepcopy(manifest)
        m["admission_required"] = True
        out["revision_id"] = store.register(m)
    elif not bad:
        out["status"] = "ready"
    return out


# ---------------------------------------------------------------- AC-060
def _cfg_hash(obj):
    cfg = getattr(obj, "cfg", None)
    if not isinstance(cfg, dict):
        return None
    try:
        return digest(cfg)
    except (TypeError, ValueError):
        return _sha(cfg)


def orchestration_table(pipe, prev=None, health=None):
    """槽编排总表：基础七槽 vs 候选分列；is_seven 仅七槽齐且 fitted。prev 为上次表→diff。"""
    health = health or slot_health(pipe)
    slots = getattr(pipe, "slots", {}) or {}
    weights = getattr(pipe, "weights", None) or {}
    rows = {}
    for n, h in health["slots"].items():
        obj = slots.get(n)
        rows[n] = {"tier": "base" if n in SEVEN else "candidate",
                   "role": SLOT_ROLES.get(n, {}).get("role") or h.get("reference"),
                   "enabled": n in slots or n == "open",
                   "in_fusion": h.get("in_fusion"),
                   "fitted": bool(h.get("fitted")),
                   "degraded": h.get("status") in ("reverse", "weak", "unverified"),
                   "health": h.get("status"),
                   "plugin": type(obj).__name__ if obj is not None else None,
                   "config_hash": _cfg_hash(obj),
                   "weight": float(weights.get(n, 0.0)) if weights else None,
                   "scope": "global",
                   "effective": bool(h.get("fitted")) and h.get("in_fusion") and n in weights}
    for n in SEVEN:
        rows.setdefault(n, {"tier": "base", "enabled": False, "fitted": False,
                            "health": "absent", "effective": False})
    base_ok = all(rows[n]["enabled"] and rows[n]["fitted"] for n in SEVEN)
    diff = None
    if prev is not None:
        pr = prev.get("rows", {})
        diff = {n: {k: [pr.get(n, {}).get(k), r.get(k)] for k in r
                    if pr.get(n, {}).get(k) != r.get(k)}
                for n, r in rows.items() if pr.get(n) != r}
        diff.update({n: "removed" for n in pr if n not in rows})
    return {"rows": rows, "base": [n for n in SEVEN],
            "candidates": sorted(n for n in rows if n not in SEVEN),
            "missing_base": [n for n in SEVEN if not rows[n]["enabled"]],
            "is_seven": base_ok, "post_fusion": health.get("post_fusion"), "diff": diff}
