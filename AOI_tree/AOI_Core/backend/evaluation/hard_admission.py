"""AC-017 难例池驱动的特征准入试算（只评价，不注册/不激活/不触发学习）。

难例池 = 漏检/误报/open 处置/人工确认缺陷反馈图（非 test）+ train 正常参照；
同图有效标签矛盾 → conflict（单列，不入正负池）；学习回滚/回归门拒绝 → drift 标签。
候选 revision 与当前生产 trad 在同一池上独立打分，输出：
  gain（AUROC 增益+配对 bootstrap CI）/ redundancy（新增组与已有组最大 |spearman|）/
  regression（分来源 AUROC 回退）/ G1（最近基准 p95 + 评分增量）/ G2（激活门锚定集）。
任何一项未 passed（含 unknown/skipped）→ acceptance≠passed；open 正样本占比过高 →
insufficient_evidence（高 open 分不得直接上线）；未注册的自动发现组 → planned。
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
from sqlalchemy import or_

from ..core.config import get_settings
from ..db.database import log_action, session_scope
from ..db.models import Detection, EvalRun, Feedback, Image as ImageRow

GAIN_MIN = 0.005          # 有新增组时最小增益
PRUNE_TOL = 0.005         # 纯裁剪/口径变更允许的最大退化
REDUNDANCY_MAX = 0.95     # 新增组与已有组最大 |spearman|
REGRESS_MARGIN = 0.02     # 分来源/锚定集 AUROC 回退容忍
OPEN_SHARE_MAX = 0.5      # open 来源正样本占比上限
MIN_SIDE = 4              # 正/负池最少样本（AUROC 分辨率与激活门同口径）
_POS_SOURCES = ("miss", "open_defect", "confirmed_defect")
_NEG_SOURCES = ("fp", "open_normal", "train_normal")


# ── 难例池 ────────────────────────────────────────────────────
def _fb_source(ftype: str, label: int) -> Optional[str]:
    if ftype in ("false_negative", "new_defect"):
        return "miss"
    if ftype == "false_positive":
        return "fp"
    if ftype == "open_alert":
        return {1: "open_defect", 0: "open_normal"}.get(label)
    return {1: "confirmed_defect"}.get(label)


def _is_drift(trace) -> bool:
    eu = (trace or {}).get("engine_update") or {}
    action, reason = str(eu.get("action") or ""), str(eu.get("reason") or "")
    return "rolled_back" in action or reason in ("regress_gate", "drift")


def build_hard_pool(category: str, limit: int = 60, n_reference: int = 20) -> Dict:
    """-> {items:[{path,label,sources,drift}], conflicts:[...], counts, sha256}。"""
    from algo.feature_revision import digest
    per: Dict[str, Dict] = {}
    with session_scope() as s:
        rows = (s.query(Detection.image_path, Feedback.feedback_type,
                        Feedback.operator_label, Feedback.learning_trace)
                .join(Feedback, Feedback.detection_id == Detection.id)
                .filter(Detection.category == category,
                        Feedback.invalidated.is_(False),
                        ~s.query(ImageRow.id)
                        .filter(ImageRow.split == "test",
                                or_(ImageRow.id == Detection.image_id,
                                    ImageRow.path == Detection.image_path))
                        .exists())
                .order_by(Feedback.id.desc()).limit(limit * 4).all())
        refs = (s.query(ImageRow.path).filter(ImageRow.category == category,
                                              ImageRow.split == "train",
                                              ImageRow.label == "normal")
                .order_by(ImageRow.id).limit(n_reference).all())
    causes: Dict[str, int] = {}
    for path, ftype, label, trace in rows:
        src = _fb_source(ftype, int(label) if label is not None else -1)
        if src is None:
            continue
        lab = 1 if src in _POS_SOURCES else 0
        e = per.setdefault(path, {"path": path, "labels": set(), "sources": set(),
                                  "drift": False})
        e["labels"].add(lab)
        e["sources"].add(src)
        cause = (trace or {}).get("open_cause")
        if cause:   # AC-034 人工原因归类单列计数
            causes[cause] = causes.get(cause, 0) + 1
        e["drift"] = e["drift"] or _is_drift(trace) or cause == "drift"
    items, conflicts = [], []
    for e in per.values():
        row = {"path": e["path"], "sources": sorted(e["sources"]), "drift": e["drift"]}
        if len(e["labels"]) > 1:
            conflicts.append(row)
        else:
            items.append({**row, "label": next(iter(e["labels"]))})
    items = sorted(items, key=lambda r: r["path"])[:limit]
    seen = {r["path"] for r in items} | {r["path"] for r in conflicts}
    for (p,) in refs:
        if p not in seen:
            seen.add(p)
            items.append({"path": p, "label": 0, "sources": ["train_normal"], "drift": False})
    counts = {src: sum(1 for r in items if src in r["sources"])
              for src in _POS_SOURCES + _NEG_SOURCES}
    counts.update(pos=sum(r["label"] for r in items),
                  neg=sum(1 - r["label"] for r in items),
                  conflict=len(conflicts), drift=sum(1 for r in items if r["drift"]),
                  open_causes=causes)
    sha = digest([[r["path"], r["label"], r["sources"]] for r in items])
    return {"items": items, "conflicts": conflicts, "counts": counts, "sha256": sha}


# ── 打分（与 TradSlot.score_image 同口径：跳最近邻取 [1:4] 均值）──────
def _group_sq(trad, Z: np.ndarray, keys: List[str]) -> Dict[str, np.ndarray]:
    from algo.features import registry as freg
    B = trad.bank_w.astype(np.float64)
    dim = B.shape[1]
    out = {}
    for k in keys:
        ids = trad.group_valid_dims.get(k) if trad.scoring == "grouped" else None
        ids = [i for i in (ids if ids is not None else freg.dim_index([k], dim)) if i < dim]
        if not ids:
            continue
        zb, bb = Z[:, ids], B[:, ids]
        out[k] = np.maximum((zb ** 2).sum(1)[:, None] + (bb ** 2).sum(1)[None, :]
                            - 2 * zb @ bb.T, 0.0)
    return out


def _combine(trad, sq: Dict[str, np.ndarray], keys: List[str], scoring: str, notes: List[str]):
    from algo.eval.contribution import _knn_skip1
    keys = [k for k in keys if k in sq]
    if not keys:
        return None
    if scoring == "grouped":
        w = dict(trad.group_weight or {})
        fitted = [v for v in w.values() if v > 0]
        mean_w = float(np.mean(fitted)) if fitted else 1.0
        acc = 0.0
        for k in keys:
            wk = w.get(k, 0.0)
            if wk <= 0:
                wk = mean_w
                notes.append(f"{k}: 未拟合组权重，按已启用组均值 {mean_w:.4f} 估计")
            sc = trad.group_scale.get(k)
            d = np.sqrt(sq[k])
            if not sc:
                sc = float(np.median(d)) or 1.0
                notes.append(f"{k}: 未拟合组尺度，按池距离中位数估计")
            acc = acc + wk * d / sc
        return _knn_skip1(acc)
    return _knn_skip1(np.sqrt(sum(sq[k] for k in keys)))


def _paired_ci(y, a, b, n_boot=200, seed=0):
    from algo.eval.contribution import _auc
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return None
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ab, aa = _auc(y[idx], b[idx]), _auc(y[idx], a[idx])
        if ab is not None and aa is not None:
            vals.append(ab - aa)
    if len(vals) < 10:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _source_regression(items, base, cand, margin):
    from algo.eval.contribution import _auc
    y = np.asarray([r["label"] for r in items])
    out, ok, any_eval = {}, True, False
    for src in _POS_SOURCES + _NEG_SOURCES:
        sel = np.asarray([src in r["sources"] for r in items])
        if src in _POS_SOURCES:
            mask = (sel & (y == 1)) | (y == 0)
            n_src = int((sel & (y == 1)).sum())
        else:
            mask = (sel & (y == 0)) | (y == 1)
            n_src = int((sel & (y == 0)).sum())
        if n_src < 2:
            out[src] = {"status": "insufficient", "n": n_src}
            continue
        ab, ac = _auc(y[mask], base[mask]), _auc(y[mask], cand[mask])
        if ab is None or ac is None:
            out[src] = {"status": "insufficient", "n": n_src}
            continue
        any_eval = True
        passed = ac >= ab - margin
        ok = ok and passed
        out[src] = {"status": "passed" if passed else "failed", "n": n_src,
                    "base_auroc": float(ab), "candidate_auroc": float(ac)}
    return {"status": ("passed" if ok else "failed") if any_eval else "insufficient",
            "margin": margin, "by_source": out}


def admission_metrics(trad, Z: np.ndarray, items: List[Dict], manifest: Dict,
                      Z_anchor: Optional[np.ndarray] = None,
                      anchor_labels: Optional[List[int]] = None,
                      bench: Optional[Dict] = None) -> Dict:
    """纯计算：给定池 z 向量与候选 manifest，产出准入判定块（不读写 DB/文件）。"""
    from scipy.stats import spearmanr
    from algo.eval.contribution import _auc
    from algo.features import registry as freg
    notes: List[str] = []
    base_keys = [k for k in freg.GROUP_KEYS if k in trad.groups]
    cand_keys = list(manifest.get("enabled_groups") or [])
    base_scoring = "grouped" if trad.scoring == "grouped" and trad.group_weight else "legacy"
    cand_scoring = manifest.get("scoring", "legacy")
    unknown = [k for k in cand_keys if k not in freg.GROUP_KEYS]
    added = [k for k in cand_keys if k not in base_keys]
    removed = [k for k in base_keys if k not in cand_keys]
    out: Dict = {"base_groups": base_keys, "candidate_groups": cand_keys,
                 "added": added, "removed": removed,
                 "base_scoring": base_scoring, "candidate_scoring": cand_scoring,
                 "notes": notes}
    if unknown or manifest.get("origin") == "auto_discovery":
        out.update(acceptance="planned", reason=(
            f"候选含未注册组 {unknown}" if unknown else "自动发现组尚未完成注册")
            + "：仅为规划项，不可准入")
        return out
    if cand_scoring == "grouped" and not trad.group_weight:
        out.update(acceptance="insufficient_evidence",
                   reason="当前快照未拟合组权重，grouped 候选需先重建表示后试算")
        return out
    y = np.asarray([int(r["label"]) for r in items])
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    out.update(n_pos=n_pos, n_neg=n_neg)
    if min(n_pos, n_neg) < MIN_SIDE:
        out.update(acceptance="insufficient_evidence",
                   reason=f"难例池正负各需≥{MIN_SIDE}（pos={n_pos} neg={n_neg}）")
        return out
    keys = sorted(set(base_keys) | set(cand_keys))
    sq = _group_sq(trad, Z, keys)
    t0 = time.perf_counter()
    base = _combine(trad, sq, base_keys, base_scoring, [])
    t_base = (time.perf_counter() - t0) * 1000 / len(Z)
    t0 = time.perf_counter()
    cand = _combine(trad, sq, cand_keys, cand_scoring, notes)
    t_cand = (time.perf_counter() - t0) * 1000 / len(Z)
    if base is None or cand is None:
        out.update(acceptance="insufficient_evidence", reason="候选/基线无可评分维度")
        return out
    a_base, a_cand = _auc(y, base), _auc(y, cand)
    gain = None if a_base is None or a_cand is None else float(a_cand - a_base)
    need = GAIN_MIN if added else -PRUNE_TOL
    out["auroc"] = a_cand
    out["gain"] = {"status": "passed" if gain is not None and gain >= need else "failed",
                   "base_auroc": a_base, "candidate_auroc": a_cand, "delta": gain,
                   "required": need, "ci95": _paired_ci(y, base, cand)}
    # 冗余：新增组单组分 vs 已有组单组分
    if added:
        from algo.eval.contribution import _knn_skip1
        single = {k: _knn_skip1(np.sqrt(sq[k])) for k in keys if k in sq}
        red = {}
        for a in added:
            cs = [abs(c) for b in base_keys if b in single and a in single
                  for c in [spearmanr(single[a], single[b]).statistic] if np.isfinite(c)]
            red[a] = float(max(cs)) if cs else None
        worst = max((v for v in red.values() if v is not None), default=None)
        out["redundancy"] = {"status": "passed" if worst is not None and worst <= REDUNDANCY_MAX
                             else "failed", "max_abs_spearman": red, "threshold": REDUNDANCY_MAX}
    else:
        out["redundancy"] = {"status": "passed", "note": "无新增组，不引入冗余"}
    out["regression"] = _source_regression(items, base, cand, REGRESS_MARGIN)
    # G1：最近一次基准 p95 + 评分增量
    if bench and bench.get("p95") is not None:
        est = float(bench["p95"]) + max(0.0, t_cand - t_base)
        budget = float(bench.get("budget_ms", 200))
        out["g1"] = {"status": "passed" if est < budget else "failed",
                     "baseline_p95_ms": float(bench["p95"]), "score_delta_ms": t_cand - t_base,
                     "estimated_p95_ms": est, "budget_ms": budget,
                     "benchmark_run_id": bench.get("eval_run_id"),
                     "basis": "提取全 200 维不随组启停变化，增量仅为评分耗时"}
    else:
        out["g1"] = {"status": "unknown", "reason": "无延迟基准记录，先跑 benchmark"}
    # G2：激活门锚定集
    if Z_anchor is not None and anchor_labels is not None:
        ya = np.asarray(anchor_labels)
        pa, na = int(ya.sum()), int(len(ya) - ya.sum())
        if min(pa, na) < MIN_SIDE:
            out["g2"] = {"status": "skipped", "reason": f"锚定集不足 pos={pa} neg={na}"}
        else:
            sqa = _group_sq(trad, Z_anchor, keys)
            ab = _auc(ya, _combine(trad, sqa, base_keys, base_scoring, []))
            ac = _auc(ya, _combine(trad, sqa, cand_keys, cand_scoring, []))
            ok = ab is not None and ac is not None and ac >= ab - REGRESS_MARGIN
            out["g2"] = {"status": "passed" if ok else "failed", "base_auroc": ab,
                         "candidate_auroc": ac, "margin": REGRESS_MARGIN, "n": int(len(ya))}
    else:
        out["g2"] = {"status": "skipped", "reason": "无锚定集"}
    checks = ("gain", "redundancy", "regression", "g1", "g2")
    open_pos = sum(1 for r in items if r["label"] == 1 and "open_defect" in r["sources"])
    out["open_share"] = open_pos / n_pos
    if out["open_share"] > OPEN_SHARE_MAX:
        out.update(acceptance="insufficient_evidence",
                   reason=f"正样本 open 来源占比 {out['open_share']:.0%}>{OPEN_SHARE_MAX:.0%}："
                          "高 open 分不得直接上线，需更多人工确认漏检/缺陷样本")
    elif all(out[c]["status"] == "passed" for c in checks):
        out["acceptance"] = "passed"
    else:
        out.update(acceptance="failed", reason="未通过: " + ",".join(
            f"{c}={out[c]['status']}" for c in checks if out[c]["status"] != "passed"))
    return out


# ── 编排 + 证据落盘 ──────────────────────────────────────────
def _latest_benchmark(category: str) -> Optional[Dict]:
    with session_scope() as s:
        row = (s.query(EvalRun).filter(EvalRun.category == category,
                                       EvalRun.run_type == "benchmark")
               .order_by(EvalRun.id.desc()).first())
        if row is None or not (row.latency or {}).get("p95"):
            return None
        return {"eval_run_id": row.id, "p95": row.latency["p95"],
                "budget_ms": (row.metrics or {}).get("budget_ms", 200)}


def _store_dir():
    return get_settings().storage("feature_revisions")


def evaluate_feature_admission(category: str, rid: str, limit: int = 60,
                               progress_cb: Optional[Callable] = None) -> Dict:
    """候选 revision 难例准入试算 → 证据 JSON（feature_revisions/admission_*.json）。

    只用 build_variant 独立 pipe 读快照，不改生产 pipe/DB 模型指针，不触发学习。"""
    from algo.common.io import load_image
    from algo.feature_revision import FeatureSetRevision
    from ..engine import get_engine
    from .service import _gate_anchor_items
    store = FeatureSetRevision(_store_dir() / "registry.json")
    row = store.list().get(rid)
    if row is None:
        raise ValueError(f"revision {rid} 不存在")
    manifest = row["manifest"]
    if manifest.get("category") not in (None, category):
        raise ValueError(f"revision 属于品类 {manifest.get('category')}，与 {category} 不符")
    eng = get_engine()
    version = int(manifest["model_snapshot"]["version"])
    if version not in eng.list_versions(category):
        raise ValueError(f"revision 绑定快照 v{version} 不存在")
    pool = build_hard_pool(category, limit=limit)
    anchors = _gate_anchor_items(category)
    total = len(pool["items"]) + len(anchors) + 1
    if progress_cb:
        progress_cb(0, total, f"难例池 {pool['counts']}，加载快照 v{version}...")
    pipe = eng.build_variant(category, version)
    trad = (getattr(pipe, "slots", None) or {}).get("trad")
    if trad is None or getattr(trad, "bank_w", None) is None:
        raise ValueError("快照无 trad 整图库，无法试算")

    def zvecs(paths, offset):
        zs, ok, missing = [], [], 0
        for i, p in enumerate(paths):
            try:
                img = pipe._preprocess(load_image(p))
            except (FileNotFoundError, OSError, ValueError):
                missing += 1
                continue
            zs.append(trad._z(trad._vec(img), trad.norm_w))
            ok.append(i)
            if progress_cb:
                progress_cb(offset + i + 1, total, f"提取特征 {offset + i + 1}/{total}")
        Z = np.stack(zs).astype(np.float64) if zs else np.zeros((0, trad.bank_w.shape[1]))
        return Z, ok, missing

    Z, ok, miss_pool = zvecs([r["path"] for r in pool["items"]], 0)
    items = [pool["items"][i] for i in ok]
    Za, oka, miss_anchor = zvecs([p for p, _ in anchors], len(pool["items"]))
    block = admission_metrics(trad, Z, items, manifest, Z_anchor=Za,
                              anchor_labels=[anchors[i][1] for i in oka],
                              bench=_latest_benchmark(category))
    block.update(hard_pool_sha256=pool["sha256"], sources=pool["counts"],
                 conflicts=len(pool["conflicts"]),
                 missing_images={"pool": miss_pool, "anchor": miss_anchor})
    acceptance = block.get("acceptance", "failed")
    with session_scope() as s:
        run = EvalRun(name=f"feature_admission_{category}_{datetime.now():%Y%m%d_%H%M%S}",
                      run_type="feature_admission", category=category,
                      dataset_desc=f"hard_pool n={len(items)} rev={rid[:12]}",
                      n_images=len(items),
                      metrics={"revision_id": rid, "acceptance": acceptance,
                               "admission": block})
        s.add(run)
        s.flush()
        run_id = run.id
    evidence = {
        "revision_id": rid, "dataset_split": manifest["dataset_split"],
        "model_snapshot": manifest["model_snapshot"], "seed": manifest["seed"],
        "eval_run_id": run_id, "status": "completed",
        "independent": True,
        "independence": "难例池取自反馈/open 处置/train 参照（排除 test），与候选 draft 的选择口径分离",
        "acceptance": acceptance, "auroc": block.get("auroc"),
        "admission": block, "created_at": datetime.now().isoformat(),
    }
    name = f"admission_{rid[:16]}_{run_id}.json"
    path = _store_dir() / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2, default=float)
    if progress_cb:
        progress_cb(total, total, f"完成：{acceptance}")
    log_action("feature_admission",
               f"category={category} revision={rid[:12]} acceptance={acceptance} run={run_id}")
    return {"eval_run_id": run_id, "evidence_path": name, "acceptance": acceptance,
            "admission": block}
