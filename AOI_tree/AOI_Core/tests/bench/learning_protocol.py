"""自学习协议化实验（P0-3 证据工具，2026-10-02）。

目的：把"反馈驱动学习不退化"从口头结论变成可复跑协议。离线快照副本上
双臂对照（不动线上模型）：

  A. 正确反馈臂：操作员反馈真值（verdict 与预测对错一致）——
     断言锚定集 AUROC 不退化（final >= initial - tol），报告增益。
  B. 压力测试臂：仅当 --adv-ratio > 0 时按比例注入**错误标注**（label 翻转，
     模拟误标/恶意反馈）；比例为 0 时两臂均使用真实标签。压力臂断言回归门控挡住灾难性退化
     （final >= initial - adv_tol），并留痕门控否决次数。

协议要素：固定 holdout（锚定集只评估不反馈，选取种子与实验种子独立）、
多种子（--seeds）、正确/错误反馈注入、回归门禁断言、环境 manifest
（commit/python/torch/cuda/config_hash/数据集/预算/计时边界）。

诚实边界：本工具产出的是"当前机器+当前数据集品类"的证据；赛题答辩所需
指定硬件/多品类结论，需在对应环境复跑并归档报告 JSON。
报告写 storage/logs/learning_protocol_{时间戳}.json。

用法：
    python -m tests.bench.learning_protocol                      # 标准档
    python -m tests.bench.learning_protocol --quick              # 快速自检
    python -m tests.bench.learning_protocol --seeds 42,1,2,7,9 --n-rounds 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.testdata import pick_category, list_imgs  # noqa: E402

# 回归门禁预算（锚定集 AUROC 允许最大回撤）
CORRECT_TOL = 0.02        # 正确反馈臂：不得退化（留出数值噪声）
ADVERSARIAL_TOL = 0.05    # 错误反馈臂：门控应挡住灾难性退化
ZERO_ERROR_TOL = 0.0      # 流内已全对时，学习后不得新增错误
HOLDOUT_SEED = 20261002   # 锚定集选取种子（与实验种子独立，固定 holdout）


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _torch_manifest() -> dict:
    try:
        import torch
        return {"version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", None),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None)}
    except Exception:  # noqa: BLE001
        return {"version": "unavailable"}


def _file_hash(path: str) -> str:
    if not path or not os.path.isfile(path):
        return "unavailable"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_items(cat_dir: str):
    """按协议分域收集：训练异常只允许来自 train/valid/val，test 仅评估。"""
    def images(d):
        return sorted(list_imgs(d) + ([os.path.join(d, name) for name in os.listdir(d)
                                       if name.lower().endswith(".bmp")]
                                      if os.path.isdir(d) else []))

    if os.path.isdir(os.path.join(cat_dir, "train", "images")):
        # YOLO 检测格式（如 GYU-DET）：split/images + split/labels。
        # 有非空同名 .txt 标注（含框）→ 缺陷；无标注或空标注 → 正常。
        def yolo_split(split):
            img_dir = os.path.join(cat_dir, split, "images")
            lab_dir = os.path.join(cat_dir, split, "labels")
            normals, defects = [], []
            for img in images(img_dir):
                lab = os.path.join(lab_dir, os.path.splitext(os.path.basename(img))[0] + ".txt")
                has_box = os.path.isfile(lab) and bool(open(lab, encoding="utf-8").read().strip())
                (defects if has_box else normals).append(img)
            return normals, defects

        train_n, train_d = yolo_split("train")
        for split in ("valid", "val"):
            if os.path.isdir(os.path.join(cat_dir, split, "images")):
                train_d += yolo_split(split)[1]
        test_n, test_d = yolo_split("test")
        return train_n, sorted(set(train_d)), test_n, test_d

    normal = "good" if os.path.isdir(os.path.join(cat_dir, "train", "good")) else "ok"
    train_n = images(os.path.join(cat_dir, "train", normal))
    train_d = []
    for split in ("train", "valid", "val"):
        split_dir = os.path.join(cat_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            d = os.path.join(split_dir, name)
            if os.path.isdir(d) and name not in ("good", "ok"):
                train_d += images(d)
    test_n = images(os.path.join(cat_dir, "test", normal))
    test_d = []
    test_dir = os.path.join(cat_dir, "test")
    if os.path.isdir(test_dir):
        for name in sorted(os.listdir(test_dir)):
            d = os.path.join(test_dir, name)
            if os.path.isdir(d) and name not in ("good", "ok"):
                test_d += images(d)
    return train_n, sorted(set(train_d)), test_n, test_d


def _holdout_scores(pipe, eval_items, audit=False):
    """单趟复用 predict，返回 AUROC、判定指标及可选逐图审计证据。"""
    from sklearn.metrics import roc_auc_score
    import time
    labels, scores, records, lat, preds = [], [], [], [], []
    slot_raw = {}   # 单槽 raw 分（溯源：哪个槽位有区分力）
    tp = fp = tn = fn = 0
    for path, y in eval_items:
        t0 = time.perf_counter()
        rr = pipe.predict(path)
        lat.append(time.perf_counter() - t0)
        labels.append(y)
        scores.append(float(rr["fused"]))
        for k, v in rr.get("raw_scores", {}).items():
            slot_raw.setdefault(k, []).append(float(v))
        pred = int(rr["decision"] != "normal")  # gray 也视为异常
        preds.append(pred)
        if y == 1:
            if pred:
                tp += 1
            else:
                fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
        if audit:
            records.append({
                "path": path,
                "label": int(y),
                "fused": float(rr["fused"]),
                "slot_scores": {k: float(v) for k, v in rr.get("slot_scores", {}).items()},
                "raw_scores": {k: float(v) for k, v in rr.get("raw_scores", {}).items()},
                "decision": rr.get("decision"),
                "gate_evidence": {
                    "boost": float(rr.get("boost", 0.0)),
                    "router_w": {k: float(v) for k, v in rr.get("router_w", {}).items()},
                    "decision_trace": rr.get("decision_trace", []),
                    "align_warn": rr.get("align_warn"),
                    "open_alert": rr.get("open_alert"),
                },
            })
    metrics = {
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sec_per_img_mean": round(float(np.mean(lat)), 4) if lat else 0.0,
        "sec_per_img_max": round(float(np.max(lat)), 4) if lat else 0.0,
    }
    metrics.update(_bootstrap_ci(labels, scores, preds))
    if len(set(labels)) == 2:
        metrics["slot_auroc"] = {k: round(float(roc_auc_score(labels, v)), 4)
                                 for k, v in slot_raw.items() if len(v) == len(labels)}
    return float(roc_auc_score(labels, scores)), metrics, records


def _bootstrap_ci(labels, scores, preds, n_boot=1000, seed=0):
    """I6：AUROC / recall 的 95% bootstrap 置信区间（按类分层重采样）。"""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(labels)
    s = np.asarray(scores)
    p = np.asarray(preds)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(i1) < 2 or len(i0) < 2:
        return {}
    rng = np.random.default_rng(seed)
    au, rc = [], []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        au.append(roc_auc_score(y[b], s[b]))
        rc.append(float(p[b][y[b] == 1].mean()))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {"auroc_ci95": q(au), "recall_ci95": q(rc)}


def _metric_delta(current, baseline):
    return {key: round(current[key] - baseline[key], 4)
            for key in ("recall", "accuracy", "precision", "tp", "fp", "tn", "fn")}


def _decider_taus(pipe):
    d = getattr(pipe, "decider", None)
    return [getattr(d, "tau_gray", float("nan")), getattr(d, "tau_high", float("nan"))]


def _run_rounds(pipe, pool, eval_items, round_size, n_rounds, flip_paths,
                progress_tag="", audit_holdout=False, review_mode="full"):
    """单臂回放：逐轮反馈（flip_paths 内样本 label 翻转=错误反馈注入），
    每轮末在固定锚定集重算 AUROC 与判定指标。
    复核口径对齐业务（routes_review /review/queue）：
      full = 工单开启人工复核（review_enabled）：全部检测（含判正常）按真值复核即学；
      gray = 未开启人工复核：只有 decision=='gray' 的检测进复核队列给真值，其余无反馈。"""
    from algo.ssocl.learning_curve import _ft_summary
    initial, initial_metrics, initial_audit = _holdout_scores(pipe, eval_items, audit_holdout)
    aurocs = []
    previous_metrics = initial_metrics
    holdout_audit = {"initial": initial_audit} if audit_holdout else None
    n_fb = 0
    n_adv = 0
    round_stats = []
    n_rounds = min(n_rounds, len(pool) // max(round_size, 1))
    for r in range(n_rounds):
        chunk = pool[r * round_size:(r + 1) * round_size]
        errors_before = 0
        model_corrections = 0
        fb_trace = []
        h = getattr(pipe, "handler", None)
        rb_start = len(h.rollback_log) if h is not None else 0
        wh_start = len(h.weight_history) if h is not None else 0
        pt_start = len(h.pair_thresh_log) if h is not None else 0
        n_reviewed = n_unreviewed = n_missed_unseen = 0
        for path, y in chunk:
            rr = pipe.predict(path)
            pred = 1 if rr["decision"] != "normal" else 0
            if pred != y:
                errors_before += 1
            if review_mode == "gray" and rr["decision"] != "gray":
                n_unreviewed += 1
                if pred != y:
                    n_missed_unseen += 1                   # 错判但不在灰区：系统不可见
                continue
            n_reviewed += 1
            y_fb = (1 - y) if path in flip_paths else y   # 仅压力臂翻转标签
            if path in flip_paths:
                n_adv += 1
            correct = (pred == y_fb)
            if not correct:
                model_corrections += 1
            fb_entry = pipe.feedback(path, "correct" if correct else "wrong", label=y_fb)
            n_fb += 1
            if isinstance(fb_entry, dict):
                fb_trace.append({"path": os.path.basename(str(path)), "label": int(y_fb),
                                 "pred": pred, "fused": fb_entry.get("fused"),
                                 "action": fb_entry.get("action"),
                                 "decision": fb_entry.get("decision")})
        errors_after = 0
        for path, y in chunk:
            rr_after = pipe.predict(path)
            pred_after = 1 if rr_after["decision"] != "normal" else 0
            if pred_after != y:
                errors_after += 1
        au, metrics, round_audit = _holdout_scores(pipe, eval_items, audit_holdout)
        aurocs.append(round(au, 4))
        if audit_holdout:
            holdout_audit[f"round_{r + 1}"] = round_audit
        round_stats.append({
            "round": r + 1,
            "feedback_count": n_reviewed,
            "stream_count": len(chunk),
            "review_mode": review_mode,
            "n_unreviewed": n_unreviewed,
            "n_errors_unreviewed": n_missed_unseen,
            "prediction_errors_before_learning": errors_before,
            "prediction_errors_after_learning": errors_after,
            "errors_corrected": errors_before - errors_after,
            "model_correction_count": model_corrections,
            "feedback_prediction_correct_count": n_reviewed - model_corrections,
            "flipped_label_count": sum(path in flip_paths for path, _ in chunk),
            "holdout_auroc": round(au, 4),
            "holdout_metrics": metrics,
            "holdout_delta_previous": _metric_delta(metrics, previous_metrics),
            "holdout_delta_initial": _metric_delta(metrics, initial_metrics),
            "feedback_trace": fb_trace,
            "rollbacks": ([{k: v for k, v in e.items() if k != "t"}
                           for e in h.rollback_log[rb_start:]] if h is not None else []),
            "weight_events": ([{"n_fb": e["n_fb"], "action": e["action"],
                                "weights": {k: round(v, 4) for k, v in e["weights"].items()}}
                               for e in h.weight_history[wh_start:]] if h is not None else []),
            "handler_stats": dict(h.stats) if h is not None else {},
            "pair_thresh_events": ([{k: v for k, v in e.items() if k != "t"}
                                    for e in h.pair_thresh_log[pt_start:]]
                                   if h is not None else []),
            "tau": [round(float(x), 4) for x in _decider_taus(pipe)],
        })
        print(f"  [{progress_tag}] 轮 {r + 1} 动作={[t['action'] for t in fb_trace]} "
              f"回滚={[e.get('op') + ':' + str(e.get('reason')) for e in round_stats[-1]['rollbacks']]} "
              f"权重={[e['action'] for e in round_stats[-1]['weight_events']]} "
              f"tau={round_stats[-1]['tau']}", flush=True)
        previous_metrics = metrics
        print(f"  [{progress_tag}] 轮 {r + 1}/{n_rounds} 反馈={n_fb} "
              f"流内错检={errors_before} 模型纠错反馈={model_corrections} "
              f"翻转标签={sum(path in flip_paths for path, _ in chunk)} "
              f"锚定集AUROC={au:.4f} recall={metrics['recall']:.4f} "
              f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
              f"CI95(AUROC)={metrics.get('auroc_ci95')} 复核={n_reviewed}(未复核错判{n_missed_unseen}) "
              f"TP/FP/TN/FN={metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']} "
              f"单槽AUROC={metrics.get('slot_auroc')} "
              f"Δ上轮={round_stats[-1]['holdout_delta_previous']} "
              f"Δ初始={round_stats[-1]['holdout_delta_initial']}", flush=True)
    if aurocs and not audit_holdout:
        final, final_metrics, final_audit = aurocs[-1], previous_metrics, []
    elif not aurocs and not audit_holdout:
        final, final_metrics, final_audit = initial, initial_metrics, []
    else:
        final, final_metrics, final_audit = _holdout_scores(pipe, eval_items, audit_holdout)
    if audit_holdout:
        holdout_audit["final"] = final_audit
    ft = _ft_summary(pipe)
    gate = {"weight_apply": ft["weight_learn"].get("n_apply", 0),
            "weight_reject": ft["weight_learn"].get("n_reject", 0),
            "thresh_recal_reject": ft["thresh_recal"].get("n_reject", 0),
            "head_ft_reject": ft["head_ft"].get("n_reject", 0),
            "disc_ft_reject": ft["disc_ft"].get("n_reject", 0)}
    return (round(initial, 4), aurocs, round(final, 4), n_fb, n_adv,
            round_stats, gate, holdout_audit, initial_metrics, final_metrics)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2", help="实验种子（逗号分隔）")
    ap.add_argument("--round-size", type=int, default=8)
    ap.add_argument("--n-rounds", type=int, default=3)
    ap.add_argument("--eval-per-class", type=int, default=8,
                    help="锚定集每类张数（固定 holdout）")
    ap.add_argument("--adv-ratio", type=float, default=0.3,
                    help="错误反馈臂的标注翻转比例")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs"
                                            / "engine_fast.yaml"))
    ap.add_argument("--scenario", default="L1a", choices=["L0", "L1a", "L1b", "L2", "L3"],
                    help="场景层；L3 启用 tpl 模板槽（I2：无显式模板时从 init_normal 自动建库）")
    ap.add_argument("--quick", action="store_true",
                    help="快速自检（1 种子×2 轮×4 条，锚定集 2+2）")
    ap.add_argument("--strict-protocol", action="store_true",
                    help="严格检查赛题 100 正常+30 异常训练协议；不足时直接失败")
    ap.add_argument("--data-dir", default=None,
                    help="显式指定真实品类目录；允许低样本诊断且不回退合成数据")
    ap.add_argument("--ablate-weight", action="store_true",
                    help="消融在线槽位权重写回，仅用于机制定位")
    ap.add_argument("--ablate-intercept", action="store_true",
                    help="消融 DefectBank/head 拦截加分，仅用于机制定位")
    ap.add_argument("--ablate-reflow", action="store_true",
                    help="消融正常回流，仅用于机制定位")
    ap.add_argument("--ablate-cdf", action="store_true",
                    help="消融回流 CDF 重估，仅用于机制定位")
    ap.add_argument("--n-init", type=int, default=100,
                    help="初始训练正常图张数上限（缩小以制造冷启动误检，验证学习增益）")
    ap.add_argument("--no-ssocl-fix", action="store_true",
                    help="关闭 2026-10-05 SSOCL 根因修复（对照组）")
    ap.add_argument("--audit-holdout", action="store_true",
                    help="显式记录逐图 holdout 审计证据（初始/逐轮/最终）")
    ap.add_argument("--review-mode", choices=("full", "gray"), default="full",
                    help="full=开启人工复核（全部检测按真值复核）；gray=未开启（仅灰区进复核队列）")
    ap.add_argument("--pseudo-box", action="store_true",
                    help="开启 I4 伪框（默认关闭，见报告 12.11：无框缺陷用 sem 热图伪框驱动 disc/shead 微调）")
    ap.add_argument("--init-defect-source", choices=("train", "pool"), default="train",
                    help="train=只取 train/valid 缺陷；pool=不足 30 张时从反馈池（test 减 holdout）"
                         "补齐并移出反馈池，对齐赛题 100+30 与研究侧口径，holdout 不受影响")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.n_rounds, args.round_size, args.eval_per_class = \
            "42", 2, 4, 2

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.data_dir:
        cat_dir = str(Path(args.data_dir).resolve())
        cat_name, synthetic = Path(cat_dir).name, False
        if not Path(cat_dir).is_dir():
            raise SystemExit(f"指定的数据集目录不存在：{cat_dir}")
    else:
        cat_dir, cat_name, synthetic = pick_category()
    train_n, train_d, test_n, test_d = _collect_items(cat_dir)
    print(f"[protocol] 品类={cat_name} synthetic={synthetic} "
          f"train_normal={len(train_n)} train_defect={len(train_d)} "
          f"test_normal={len(test_n)} test_defect={len(test_d)}", flush=True)
    if args.strict_protocol:
        if len(train_n) < 100 or len(train_d) < 30 or synthetic or len(test_n) < 2:
            raise SystemExit(
                "严格协议不满足：需要至少100张训练正常图、30张train/valid异常图和独立评价集；"
                "test/anomaly 不能补齐训练异常。"
            )

    # 固定 holdout：与实验种子独立；异常取 test 缺陷、正常取 test/good。
    # 锚定集只评估不反馈，绝不进入 fit 域与反馈池。
    rng_h = np.random.default_rng(HOLDOUT_SEED)
    n_e = min(args.eval_per_class, len(test_d), len(test_n))
    if n_e < 2:
        raise SystemExit("样本不足：锚定集每类至少 2 张")
    eval_def = sorted(rng_h.choice(test_d, n_e, replace=False).tolist())
    eval_nor = sorted(rng_h.choice(test_n, n_e, replace=False).tolist())
    eval_items = [(p, 1) for p in eval_def] + [(p, 0) for p in eval_nor]
    eval_set = set(eval_def) | set(eval_nor)
    rest_test_n = [p for p in test_n if p not in eval_set]

    # 数据划分（满足隔离红线 guard_bundle）：
    #   init_normal 纯 train 域；init_defect 只取 train/valid/val 缺陷；
    #   test 项与 fit 集零交集；反馈池只取未反馈的 test 样本。
    n_init = min(args.n_init, len(train_n))
    if n_init < 2:
        raise SystemExit("样本不足：train/good 至少 2 张")
    init_normal = train_n[:n_init]
    rest_test_n = [p for p in test_n if p not in eval_set]
    rest_test_d = [p for p in test_d if p not in eval_set]
    if rest_test_n:
        test_item = (rest_test_n.pop(0), 0)
    elif rest_test_d:
        test_item = (rest_test_d.pop(0), 1)
    else:
        raise SystemExit("样本不足：无法构造与 fit 零交集的 test 项")
    init_defect = train_d[:min(30, len(train_d))]
    if args.init_defect_source == "pool" and len(init_defect) < 30 and rest_test_d:
        # 分类集（MVTec/BTAD/MPDD）train 无缺陷：从反馈池补齐（不含 holdout），
        # 补齐样本移出反馈池，保证 fit 集/反馈池/holdout 三者零交集。
        k = min(30 - len(init_defect), len(rest_test_d) // 2)
        take = set(np.random.default_rng(HOLDOUT_SEED + 1)
                   .choice(rest_test_d, k, replace=False).tolist())
        init_defect = init_defect + sorted(take)
        rest_test_d = [p for p in rest_test_d if p not in take]
    pool_pairs = ([(p, 0) for p in rest_test_n]
                  + [(p, 1) for p in rest_test_d])
    print(f"[protocol] 锚定集={len(eval_items)}（异常{n_e}/正常{n_e}） "
          f"训练集=正常{len(init_normal)}/异常{len(init_defect)} "
          f"反馈池={len(pool_pairs)}", flush=True)

    # 准备引擎（独立临时存储，不动线上存储）
    tmp_storage = tempfile.mkdtemp(prefix="aoi_lp_")
    from backend.engine import DetectionEngine
    engine = DetectionEngine(tmp_storage, args.config)
    bundle = {
        "init_normal": list(init_normal),
        "init_defect": list(init_defect),
        "val": [],
        "test": [test_item],
    }
    cat = f"{cat_name}_protocol"
    v = engine.prepare(cat, bundle, scenario=args.scenario, profile="fast",
                       trigger="bench", note="learning protocol")
    print(f"[protocol] prepare v{v} storage={tmp_storage}", flush=True)

    # 异常/正常交错（避免流前段全 normal）
    rng_i = np.random.default_rng(0)
    anoms = [p for p, y in pool_pairs if y == 1]
    norms = [p for p, y in pool_pairs if y == 0]
    rng_i.shuffle(anoms)
    rng_i.shuffle(norms)
    base_pool = []
    while anoms or norms:
        if anoms:
            base_pool.append((anoms.pop(), 1))
        if norms:
            base_pool.append((norms.pop(), 0))

    from algo.persist import current_version, load_snapshot
    cv = current_version(engine._cat_dir(cat))
    v_dir = engine._v_dir(cat, cv)
    device = engine.device
    _p0 = load_snapshot(v_dir, device)
    init_fusion = {"weights": {k: round(float(x), 4) for k, x in (_p0.weights or {}).items()},
                   "slot_sanity": dict(getattr(_p0, "slot_sanity", {}) or {}),
                   "slot_sanity_detail": dict(getattr(_p0, "slot_sanity_detail", {}) or {}),
                   "train_auroc": getattr(_p0, "train_auroc", None)}
    print(f"[protocol] init 融合: {init_fusion}", flush=True)
    del _p0

    arms = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # 按种子分层交错：先各自打乱再正常/异常交替，保证每轮都有双侧真值
        #（原实现交错后整体 shuffle，正常稀缺品类反馈流几乎全是缺陷，
        # 权重学习 reject_no_pair、回流/阈值重估从未触发）。
        need = args.round_size * args.n_rounds
        s_an = [p for p, y in base_pool if y == 1]
        s_no = [p for p, y in base_pool if y == 0]
        rng.shuffle(s_an)
        rng.shuffle(s_no)
        pool = []
        while (s_an or s_no) and len(pool) < need:
            if s_no:
                pool.append((s_no.pop(), 0))
            if s_an and len(pool) < need:
                pool.append((s_an.pop(), 1))
        # 错误反馈注入集：按 adv-ratio 抽样翻转 label（每个种子独立）
        n_adv = int(round(len(pool) * args.adv_ratio))
        flip_paths = set()
        if n_adv:
            idx = rng.choice(len(pool), n_adv, replace=False).tolist()
            flip_paths = {pool[i][0] for i in idx}

        # adv_ratio=0 时两臂完全相同，只跑 correct 臂
        for arm in (("correct", "adversarial") if n_adv else ("correct",)):
            pipe = load_snapshot(v_dir, device)
            if args.ablate_weight and pipe.handler is not None:
                pipe.handler._ablate_weight = True
            if pipe.handler is not None:
                pipe.handler.pseudo_box_on = bool(args.pseudo_box)
            if args.no_ssocl_fix and pipe.handler is not None:
                pipe.handler.labeled_reflow = False
                pipe.handler.pair_thresh_on = False
            pipe.set_ablation(
                intercept=args.ablate_intercept,
                reflow=args.ablate_reflow,
                cdf=args.ablate_cdf,
            )
            flips = flip_paths if arm == "adversarial" else set()
            (ini, aurocs, fin, n_fb, n_adv_done, round_stats, gate,
             holdout_audit, initial_metrics, final_metrics) = _run_rounds(
                pipe, pool, eval_items, args.round_size, args.n_rounds,
                flips, progress_tag=f"seed{seed}-{arm}", audit_holdout=args.audit_holdout,
                review_mode=args.review_mode)
            arms.setdefault(arm, []).append(
                {"seed": seed, "initial_auroc": ini, "final_auroc": fin,
                 "initial_holdout_metrics": initial_metrics,
                 "final_holdout_metrics": final_metrics,
                 "gain": round(fin - ini, 4), "round_aurocs": aurocs,
                 "n_feedback": n_fb, "n_adversarial": n_adv_done,
                 "round_stats": round_stats,
                 "gate": gate,
                 **({"holdout_audit": holdout_audit} if args.audit_holdout else {})})

    # ---- 回归门禁断言 ----
    failures = []
    for arm, tol in (("correct", CORRECT_TOL), ("adversarial", ADVERSARIAL_TOL)):
        for rec in arms.get(arm, []):
            if rec["final_auroc"] < rec["initial_auroc"] - tol:
                failures.append(f"{arm} seed={rec['seed']} AUROC "
                                f"{rec['initial_auroc']}→{rec['final_auroc']} "
                                f"回撤超 {tol}")
            for rs in rec.get("round_stats", []):
                if (rs["prediction_errors_before_learning"] == 0
                        and rs["prediction_errors_after_learning"] > ZERO_ERROR_TOL):
                    failures.append(
                        f"{arm} seed={rec['seed']} round={rs['round']} "
                        "初始流内全对但学习后新增错误")
    torch_m = _torch_manifest()
    report = {
        "manifest": {
            "commit": _git_commit(), "python": sys.version,
            "platform": platform.platform(), "torch": torch_m,
            "device": ("gpu" if torch_m.get("cuda_available") else "cpu"),
            "config": args.config, "config_hash": _file_hash(args.config),
            "dataset": cat_dir, "category": cat_name, "synthetic": synthetic,
            "seeds": seeds, "holdout_seed": HOLDOUT_SEED,
            "round_size": args.round_size, "n_rounds": args.n_rounds,
            "adv_ratio": args.adv_ratio,
            "review_mode": args.review_mode,
            "pseudo_box": bool(args.pseudo_box),
            "correct_tol": CORRECT_TOL, "adversarial_tol": ADVERSARIAL_TOL,
            "zero_error_tol": ZERO_ERROR_TOL,
            "ablations": {
                "weight": args.ablate_weight,
                "intercept": args.ablate_intercept,
                "reflow": args.ablate_reflow,
                "cdf": args.ablate_cdf,
            },
            "note": "离线快照副本回放，不动线上模型；指定硬件/多品类结论"
                    "需在对应环境复跑归档",
        },
        "init_fusion": init_fusion,
        "holdout": {"n_eval": len(eval_items), "n_per_class": n_e},
        "arms": arms, "failures": failures,
    }
    log_dir = REPO_ROOT / "storage" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S') + f"_{time.time_ns() % 1_000_000_000:09d}"
    safe_cat = ''.join(c if c.isalnum() or c in '-_' else '_' for c in cat_name)
    out = log_dir / f"learning_protocol_{safe_cat}_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    for arm in arms:
        gains = [r["gain"] for r in arms[arm]]
        print(f"[protocol] {arm}: gains={gains} "
              f"mean={float(np.mean(gains)):.4f}", flush=True)
    print(f"[protocol] 报告已写 {out}", flush=True)
    ok = not failures
    print(f"[protocol] 结论: {'PASS' if ok else 'FAIL'}"
          + ("" if ok else f"（{failures}）"), flush=True)
    import shutil
    shutil.rmtree(tmp_storage, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
