"""批次覆盖检查 + FPR 预算工作点（只读，不改阈值/快照/训练集）。

批次推断（infer_batch）：
  1. 父目录名不是通用目录（good/train/test/normal/...）时，用父目录名；
  2. 否则看文件名：取开头的字母段（如 "Image"、"t_fov"）；如果紧跟着 ≥6 位的
     日期型数字，再拼上它的前 4 位（Image_20260301_01 → "Image_2026"）；
     纯数字文件名（0598.png）记为 "numeric"。
域外比例 = 待检正常样本中分数 > train 上沿的比例。train 上沿优先用 train 分数最大值，
没有 train 分数时退回到 decider.tau_high（train/good 的 q99 分位数），并标明来源。
"""
from __future__ import annotations

import re
from pathlib import PurePath
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

_GENERIC_DIRS = {"good", "train", "test", "val", "valid", "normal", "anomaly",
                 "defect", "ng", "ok", "bad", "images", "image", "img", "imgs",
                 "data", "raw", "uploads", "upload", ""}
_SPLIT = re.compile(r"[_\-\s.]+")

SUGGEST_FMT = "将批次 {batch} 的正常样本纳入 train/good 后重建"


def infer_batch(path: str) -> str:
    p = PurePath(str(path).replace("\\", "/"))
    parent = p.parent.name
    if parent.lower() not in _GENERIC_DIRS:
        return parent
    tokens = [t for t in _SPLIT.split(p.stem) if t]
    if not tokens:
        return "unknown"
    head: List[str] = []
    for t in tokens:
        if t.isdigit():
            if head and len(t) >= 6:
                head.append(t[:4])
            break
        if not t.isalpha():
            m = re.match(r"[A-Za-z]+", t)
            if m:
                head.append(m.group(0))
            break
        head.append(t)
    return "_".join(head) if head else "numeric"


def parse_batch_list(text: Optional[str]) -> List[str]:
    return [b.strip() for b in str(text or "").split(",") if b.strip()]


def split_extra_good(paths: Sequence[str], approved: Iterable[str], seed: int = 0,
                     fit_ratio: float = 0.5,
                     batch_fn: Callable[[str], str] = infer_batch) -> Dict[str, List[str]]:
    """登记批次正常图按图像划分为互不相交的 fit / eval 子集（防泄漏）。

    只处理 batch_fn(path) ∈ approved 的路径；按批次分别洗牌（固定种子，路径先排序，
    与输入顺序无关），每批 fit 取 floor(n*fit_ratio) 张。未登记批次的路径不出现在结果中。
    """
    approved_set = set(approved)
    by_batch: Dict[str, List[str]] = {}
    for p in sorted(set(map(str, paths))):
        b = batch_fn(p)
        if b in approved_set:
            by_batch.setdefault(b, []).append(p)
    fit: List[str] = []
    ev: List[str] = []
    for b in sorted(by_batch):
        items = by_batch[b]
        order = np.random.default_rng(seed).permutation(len(items))
        k = int(np.floor(len(items) * fit_ratio))
        fit += [items[i] for i in order[:k]]
        ev += [items[i] for i in order[k:]]
    return {"fit": fit, "eval": ev}


def batch_coverage(train_paths: Sequence[str], eval_paths: Sequence[str],
                   eval_scores: Optional[Sequence[float]] = None,
                   eval_labels: Optional[Sequence[int]] = None,
                   train_scores: Optional[Sequence[float]] = None,
                   train_ref: Optional[float] = None,
                   approved: Iterable[str] = (),
                   batch_fn: Callable[[str], str] = infer_batch) -> Dict:
    """比较待检/评估集正常样本与 train/good 的批次来源和分数分布。

    eval_labels 给出时只统计 label==0（正常）样本；否则全部视为正常待检样本。
    Returns: {uncovered_batches, out_of_domain_ratio, train_ref, train_ref_source,
              per_batch{b: {n, covered, ood_ratio, approved}}, suggestions, ...}
    """
    approved_set = set(approved)
    train_batches = sorted({batch_fn(p) for p in train_paths})
    n = len(eval_paths)
    labels = (np.zeros(n, dtype=int) if eval_labels is None
              else np.asarray(eval_labels, dtype=int))
    scores = None if eval_scores is None else np.asarray(eval_scores, dtype=np.float64)

    if train_scores is not None and len(train_scores):
        ref, ref_src = float(np.max(train_scores)), "train_max"
    elif train_ref is not None and np.isfinite(train_ref):
        ref, ref_src = float(train_ref), "tau_high(q99)"
    else:
        ref, ref_src = None, None

    per: Dict[str, Dict] = {}
    for i, path in enumerate(eval_paths):
        if labels[i] != 0:
            continue
        b = batch_fn(path)
        d = per.setdefault(b, {"n": 0, "n_ood": 0})
        d["n"] += 1
        if ref is not None and scores is not None and scores[i] > ref:
            d["n_ood"] += 1
    known = bool(train_batches)
    for b, d in per.items():
        d["covered"] = (b in train_batches) if known else None
        d["approved"] = b in approved_set
        d["ood_ratio"] = (round(d["n_ood"] / d["n"], 4)
                          if ref is not None and scores is not None else None)
    uncovered = sorted(b for b, d in per.items() if d["covered"] is False)
    n_norm = sum(d["n"] for d in per.values())
    n_ood = sum(d["n_ood"] for d in per.values())
    ood = (round(n_ood / n_norm, 4)
           if n_norm and ref is not None and scores is not None else None)
    return {
        "train_batches": train_batches,
        "uncovered_batches": uncovered,
        "out_of_domain_ratio": ood,
        "n_eval_normal": n_norm,
        "train_ref": ref,
        "train_ref_source": ref_src,
        "per_batch": per,
        "approved_extra_good_batches": sorted(approved_set),
        "suggestions": [SUGGEST_FMT.format(batch=b) for b in uncovered
                        if b not in approved_set],
        "note": "只读检查；登记批次不会自动重建（下次准备模型时其 fit 子集并入 train/good）",
    }


def max_recall_at_fpr(scores: Sequence[float], labels: Sequence[int],
                      max_fpr: float) -> Optional[Dict]:
    """FPR ≤ max_fpr 时的最大 recall 及其阈值（判定口径 score >= thr，与 F1 扫描一致）。

    recall 相同时取 FPR 更低（阈值更高）的点。无正常或无缺陷样本时返回 None。
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return None
    cand = np.unique(s)
    cand = np.concatenate([cand, [cand[-1] + 1e-9]])   # 最后一点：全部判正常
    best = None
    for thr in cand[::-1]:                              # 阈值从高到低，recall 单调不减
        pred = s >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        if fp / n_neg > max_fpr + 1e-12:
            break
        rec = tp / n_pos
        if best is None or rec > best["recall"]:
            best = {"recall": round(rec, 4), "fpr": round(fp / n_neg, 4),
                    "threshold": float(thr), "tp": tp, "fp": fp,
                    "fn": n_pos - tp, "tn": n_neg - fp}
    if best is not None:
        best["max_fpr"] = float(max_fpr)
        best["note"] = "参考工作点，不修改生产阈值"
    return best
