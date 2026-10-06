"""模型评价服务（赛题要求：模型评价 + 检测时间考核）。

两类评估：
  1. accuracy  : 在带标注数据集（DB中 label!=unknown 的图像）上计算
                 AUROC / AP / F1-max / 最优阈值 / 混淆矩阵，可分级统计
  2. benchmark : 延迟基准（预热 + 计时），输出 mean/p50/p95/max，
                 对照赛题指标：2060 GPU 2500x2500 <200ms，CPU <2s
结果写入 EvalRun 表。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np

from ..core.config import get_settings
from ..db.database import log_action, session_scope
from ..db.models import Detection, EvalRun, Feedback, Image as ImageRow
from ..pipeline.service import get_detection_service

logger = logging.getLogger(__name__)


def _latency_stats(values: List[float]) -> Dict:
    """延迟统计：mean/p50/p95/max/min。"""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def _best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> Dict:
    """扫描阈值求 F1-max 及对应混淆矩阵。"""
    best = {"f1": 0.0, "threshold": float(np.median(scores)),
            "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    # 以分数唯一值为候选阈值（含略低于最小值的一点保证全覆盖）
    candidates = np.unique(scores)
    candidates = np.concatenate([[candidates[0] - 1e-9], candidates])
    for thr in candidates:
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "threshold": float(thr),
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return best


def run_accuracy_eval(
    category: str,
    split: str = "test",
    name: Optional[str] = None,
    limit: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """对 DB 中指定 category/split 的有标注图像跑检测并计算指标。

    Returns: {"eval_run_id","n_images","metrics":{auroc,ap,f1,threshold,
              tn,fp,fn,tp},"per_image":[{image_id,score,label,is_anomaly}...]}
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    # 1. 取有标注图像（正常/异常分开取再均衡混合，避免 limit 截断成单一类别）
    with session_scope() as s:
        def _fetch(lbl: str, n: Optional[int]) -> list:
            q = (s.query(ImageRow)
                 .filter(ImageRow.category == category,
                         ImageRow.split == split,
                         ImageRow.label == lbl)
                 .order_by(ImageRow.id))
            if n:
                q = q.limit(n)
            return q.all()

        if limit:
            half = max(limit // 2, 1)
            rows = _fetch("anomaly", half) + _fetch("normal", limit - half)
        else:
            rows = _fetch("anomaly", None) + _fetch("normal", None)
        images = [(r.id, r.path, 1 if r.label == "anomaly" else 0) for r in rows]
    if not images:
        raise ValueError(f"category={category} split={split} 无有标注图像")

    # 2. 逐张检测（不写库；with_heatmap=False——评估不需要热力图，
    #    计时更接近纯推理口径，与延迟基准可比，且整体评估更快）
    service = get_detection_service()
    per_image: List[Dict] = []
    for i, (img_id, path, label) in enumerate(images):
        try:
            det = service.detect_image(path, category, image_id=img_id,
                                       persist=False, with_heatmap=False)
        except Exception as e:
            # 单张失败不中断整体评估
            logger.warning("跳过检测失败的图像 %s: %s", path, e)
            if progress_cb:
                progress_cb(i + 1, len(images))
            continue
        per_image.append({
            "image_id": img_id,
            "score": float(det["final_score"]),
            "label": label,
            "is_anomaly": bool(det["is_anomaly"]),
            "latency_ms": float(det["latency_ms"]),
        })
        if progress_cb:
            progress_cb(i + 1, len(images))

    if not per_image:
        raise ValueError(f"category={category} split={split} 全部图像检测失败")
    scores = np.array([p["score"] for p in per_image], dtype=np.float64)
    labels = np.array([p["label"] for p in per_image], dtype=np.int64)
    latencies = [p["latency_ms"] for p in per_image]

    # 3. 指标计算（单类别时 AUROC/AP 无定义，置 None）
    if len(np.unique(labels)) == 2:
        auroc: Optional[float] = float(roc_auc_score(labels, scores))
        ap: Optional[float] = float(average_precision_score(labels, scores))
    else:
        auroc = ap = None
        logger.warning("评估集仅含单一类别，AUROC/AP 无法计算")

    best = _best_f1_threshold(scores, labels)
    lat_stats = _latency_stats(latencies)
    budget = float(get_settings().get("pipeline", "latency_budget_ms", 200))

    # 混淆矩阵派生指标（2026-08-30）：除零时置 None（该指标无意义，
    # 如全正常样本下 precision/recall_defect/fnr 无定义），UI 显示 "-"
    tp, tn, fp, fn = best["tp"], best["tn"], best["fp"], best["fn"]
    n_total = tp + tn + fp + fn

    def _rate(a: int, b: int) -> Optional[float]:
        return round(a / b, 4) if b else None

    metrics = {
        "auroc": auroc, "ap": ap,
        "f1": best["f1"], "threshold": best["threshold"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        # 派生业务指标：accuracy=判对比例；precision=判缺陷中真缺陷占比；
        # recall_defect=缺陷检出率（召回）；fpr=误报率（正常误判缺陷）；
        # fnr=漏检率（缺陷漏判正常）；specificity=正常判正常比例
        "accuracy": _rate(tp + tn, n_total),
        "precision": _rate(tp, tp + fp),
        "recall_defect": _rate(tp, tp + fn),
        "fpr": _rate(fp, fp + tn),
        "fnr": _rate(fn, fn + tp),
        "specificity": _rate(tn, tn + fp),
        # 精度评估顺带产出延迟分布（一次跑完"精度+延迟"），供看板对照预算
        "budget_ms": budget,
        # 延迟达标判定口径：端到端（p95<budget；该次检测有 e2e 值时取 e2e）
        "latency_basis": "e2e",
        "latency_pass": bool(lat_stats["p95"] < budget),
        # 防膨胀，最多存200条
        "per_image": [{"image_id": p["image_id"], "score": p["score"],
                       "label": p["label"], "pred": int(p["is_anomaly"])}
                      for p in per_image[:200]],
    }
    latency = lat_stats

    # 4. 入库 EvalRun
    run_name = name or f"accuracy_{category}_{datetime.now():%Y%m%d_%H%M%S}"
    with session_scope() as s:
        run = EvalRun(name=run_name, run_type="accuracy", category=category,
                      dataset_desc=f"split={split}, n={len(per_image)}",
                      n_images=len(per_image), metrics=metrics, latency=latency)
        s.add(run)
        s.flush()
        run_id = run.id

    log_action("accuracy_eval",
               f"category={category} split={split} n={len(per_image)} "
               f"auroc={auroc} f1={best['f1']:.4f}")
    return {
        "eval_run_id": run_id,
        "n_images": len(per_image),
        "metrics": {k: v for k, v in metrics.items() if k != "per_image"},
        "per_image": per_image,
    }


def run_latency_benchmark(
    category: str,
    n_images: int = 30,
    warmup: int = 5,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """延迟基准。优先使用大图（>=2000px）以贴近 2500x2500 考核场景。

    Returns: {"eval_run_id","device","latency":{mean,p50,p95,max,min},
              "budget_ms":200,"pass":bool,"per_image_ms":[...]}
    """
    import torch

    # 1. 选图：优先 width>=2000 的大图，不足则取任意图补齐
    with session_scope() as s:
        big = (s.query(ImageRow)
               .filter(ImageRow.category == category, ImageRow.width >= 2000)
               .order_by(ImageRow.id).all())
        if len(big) < n_images + warmup:
            extra = (s.query(ImageRow)
                     .filter(ImageRow.category == category)
                     .order_by(ImageRow.id).all())
            seen = {r.id for r in big}
            big += [r for r in extra if r.id not in seen]
        paths = [r.path for r in big]
    if not paths:
        raise ValueError(f"category={category} 无可用图像")

    # 2. 预热 + 计时（图不够则循环复用）
    #    检测时间只计推理（with_heatmap=False 跳过热力图渲染/存盘，
    #    可视化展示不属模型推理时间）
    service = get_detection_service()
    # 诚实口径修正（2026-08-30，不足清单 #5）：逐张清特征缓存--
    # pipe._item_cache 命中会跳过读盘/DINO 前向，图数少于循环次数时
    # 实测值退化为"热缓存"口径且随图数波动（旧 t11 基准 52ms 即此）。
    # 产线每图只检一次、必为冷缓存，此处清缓存使口径确定。
    try:
        pipe = service.get_pipeline(category)
    except RuntimeError as e:
        raise ValueError(f"品类 '{category}' 未准备（{e}）")
    per_image_ms: List[float] = []
    per_e2e_ms: List[float] = []
    total = warmup + n_images
    for i in range(total):
        path = paths[i % len(paths)]
        pipe._item_cache.clear()
        det = service.detect_image(path, category, persist=False,
                                   with_heatmap=False)
        if i >= warmup:
            per_image_ms.append(float(det["latency_ms"]))
            per_e2e_ms.append(float(det.get("latency_e2e_ms")
                                    or det["latency_ms"]))
        if progress_cb:
            progress_cb(i + 1, total)

    # 3. 设备与预算判定（预算读 pipeline.latency_budget_ms——设置中心
    #    白名单项，与顶栏/看板同一来源；此前读 evaluation.latency_budget
    #    是 yaml 中不存在的键，永远落到默认 200，设置改了不生效）
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    device = f"cuda:{gpu_name}" if cuda else "cpu"
    budget = float(get_settings().get("pipeline", "latency_budget_ms", 200))
    lat_stats = _latency_stats(per_image_ms)
    e2e_stats = _latency_stats(per_e2e_ms)
    # 2026-08-30 质检核实修复：延迟达标统一按 p95<预算（与 accuracy 评估
    # latency_pass 同口径）；此前用 mean，mean 达标但 p95 超标时误判"通过"
    passed = lat_stats["p95"] < budget
    # 1s 红线（用户 2026-08-19 性能红线）按 e2e 口径判定并记录
    red_line_ms = 1000.0
    n_over_red = sum(1 for v in per_e2e_ms if v > red_line_ms)

    # 4. 入库 EvalRun
    run_name = f"benchmark_{category}_{datetime.now():%Y%m%d_%H%M%S}"
    with session_scope() as s:
        run = EvalRun(name=run_name, run_type="benchmark", category=category,
                      dataset_desc=f"warmup={warmup}, n={n_images}",
                      n_images=n_images,
                      metrics={"device": device, "gpu_name": gpu_name,
                               "budget_ms": budget, "pass": passed,
                               # 旧值 "e2e" 名不副实（实际采 engine latency_ms）
                               "latency_basis": "engine_cold_cache",
                               "latency_e2e": e2e_stats,
                               "red_line_ms": red_line_ms,
                               "e2e_over_red_line": n_over_red},
                      latency=lat_stats)
        s.add(run)
        s.flush()
        run_id = run.id

    log_action("latency_benchmark",
               f"category={category} device={device} mean={lat_stats['mean']:.1f}ms "
               f"e2e_mean={e2e_stats['mean']:.1f}ms budget={budget}ms "
               f"pass={passed} over_1s={n_over_red}")
    return {
        "eval_run_id": run_id,
        "device": device,
        "gpu_name": gpu_name,
        "latency": lat_stats,
        "latency_e2e": e2e_stats,
        "budget_ms": budget,
        "pass": passed,
        "red_line_ms": red_line_ms,
        "e2e_over_red_line": n_over_red,
        "per_image_ms": per_image_ms,
    }


def compare_eval_runs(run_ids: List[int]) -> Dict:
    """多次评估对比：{str(run_id): {name, run_type, metrics, latency, created_at}}。"""
    with session_scope() as s:
        rows = s.query(EvalRun).filter(EvalRun.id.in_(run_ids)).all()
        return {
            str(r.id): {
                "name": r.name,
                "run_type": r.run_type,
                "metrics": r.metrics,
                "latency": r.latency,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            } for r in rows
        }


def run_ab_compare(
    category: str,
    version_a: Optional[int] = None,
    version_b: Optional[int] = None,
    split: str = "test",
    limit: int = 100,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """A/B 影子对比（M2b）：version_a（默认当前激活） vs version_b 候选快照。

    同集平行打分（pipe.predict 直调，persist 不入库），算 AUROC/AP/F1 两组
    指标 + 分歧样本列表（两管决策不一致的图，上限 20 条）。
    用于新快照的灰度发布决策：先看对比指标，再决定是否激活。

    Returns: {"eval_run_id","category","version_a","version_b",
              "active_metrics":{auroc,ap,f1,...},"candidate_metrics":{...},
              "verdict":"better"|"worse"|"tie","disagreements":[...],
              "per_image":[...]}
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    from ..engine import get_engine

    if version_b is None:
        raise ValueError("version_b（候选版本号）必填")
    engine = get_engine()
    versions = engine.list_versions(category)
    if version_b not in versions:
        raise ValueError(f"版本 v{version_b} 不存在（现有: {versions}）")
    if version_a is not None and version_a not in versions:
        raise ValueError(f"版本 v{version_a} 不存在（现有: {versions}）")
    va = version_a if version_a is not None else engine.current_version(category)
    if va is None:
        raise ValueError(f"品类 {category} 无激活版本，请先 /models/prepare")
    if va == version_b:
        raise ValueError(f"A/B 双方版本相同（v{va}），无对比意义")

    # 1. 样本（均衡取样，上限 limit）
    with session_scope() as s:
        def _fetch(lbl: str, n: int) -> list:
            return (s.query(ImageRow)
                    .filter(ImageRow.category == category,
                            ImageRow.split == split,
                            ImageRow.label == lbl)
                    .order_by(ImageRow.id).limit(n).all())
        half = max(limit // 2, 1)
        rows = _fetch("anomaly", half) + _fetch("normal", limit - half)
        images = [(r.id, r.path, 1 if r.label == "anomaly" else 0) for r in rows]
    if len(images) < 2:
        raise ValueError(f"category={category} split={split} 标注样本不足")

    # 2. 构建双方流水线（A 缺省用激活 pipe；B 用 build_variant 不影响激活态）
    pipe_a = (engine.get_pipeline(category) if version_a is None
              else engine.build_variant(category, version_a))
    pipe_b = engine.build_variant(category, version_b)

    def _judge(pipeline, path: str):
        rec = pipeline.predict(path)
        return (float(rec.get("fused", 0.0)),
                rec.get("decision", "normal") == "anomaly")

    # 3. 同集平行打分
    per_image: List[Dict] = []
    for i, (img_id, path, label) in enumerate(images):
        try:
            sa, ja = _judge(pipe_a, path)
            sb, jb = _judge(pipe_b, path)
            per_image.append({"image_id": img_id, "path": path, "label": label,
                              "score_a": sa, "pred_a": int(ja),
                              "score_b": sb, "pred_b": int(jb)})
        except Exception as e:  # noqa: BLE001
            logger.warning("A/B 跳过失败图像 %s: %s", path, e)
        if progress_cb:
            progress_cb(i + 1, len(images))
    if not per_image:
        raise ValueError(f"category={category} split={split} 全部图像打分失败")

    # 4. 双方指标 + 分歧样本
    def _metrics(side: str) -> Dict:
        scores = np.array([d[f"score_{side}"] for d in per_image])
        labels = np.array([d["label"] for d in per_image])
        out = {"auroc": None, "ap": None}
        if labels.min() != labels.max():
            out["auroc"] = float(roc_auc_score(labels, scores))
            out["ap"] = float(average_precision_score(labels, scores))
        out.update(_best_f1_threshold(scores, labels))
        return out

    m_a, m_b = _metrics("a"), _metrics("b")
    disagreements = [{"image_id": d["image_id"], "path": d["path"],
                      "label": d["label"],
                      "score_a": d["score_a"], "score_b": d["score_b"]}
                     for d in per_image if d["pred_a"] != d["pred_b"]][:20]

    # 裁决：AUROC 为主，F1 次之（容差 0.01）
    verdict = "tie"
    if m_b["auroc"] is not None and m_a["auroc"] is not None:
        if m_b["auroc"] > m_a["auroc"] + 0.01:
            verdict = "better"
        elif m_b["auroc"] < m_a["auroc"] - 0.01:
            verdict = "worse"
        elif m_b["f1"] > m_a["f1"] + 0.01:
            verdict = "better"
        elif m_b["f1"] < m_a["f1"] - 0.01:
            verdict = "worse"

    # 5. 入库 EvalRun（run_type=ab_compare）
    run_name = f"ab_{category}_v{va}_vs_v{version_b}_{datetime.now():%Y%m%d_%H%M%S}"
    with session_scope() as s:
        run = EvalRun(name=run_name, run_type="ab_compare", category=category,
                      dataset_desc=f"v{va} vs v{version_b} "
                                   f"split={split} n={len(per_image)}",
                      n_images=len(per_image),
                      metrics={"version_a": va, "version_b": version_b,
                               "active": m_a, "candidate": m_b,
                               "verdict": verdict,
                               "n_disagree": len(disagreements),
                               "disagreements": disagreements,
                               "per_image": per_image[:200]})
        s.add(run)
        s.flush()
        run_id = run.id

    log_action("ab_compare",
               f"category={category} v{va} vs v{version_b} verdict={verdict} "
               f"auroc {m_a['auroc']} -> {m_b['auroc']}")
    return {
        "eval_run_id": run_id,
        "category": category,
        "version_a": va,
        "version_b": version_b,
        "active_metrics": m_a,
        "candidate_metrics": m_b,
        "verdict": verdict,
        "disagreements": disagreements,
        "per_image": per_image,
    }


# ── M5a：版本激活质量门控 ────────────────────────────────────
def _gate_anchor_items(category: str, n_per_side: int = 20) -> List:
    """激活门控锚定集：train 正常抽样 + (train 缺陷 ∪ 人工确认缺陷反馈图) 抽样。

    2026-08-31 评审修复（红线"test 只验不选"）：锚定集不再使用 test 缺陷图——
    门控是"选"（决定版本激活与否），用 test 缺陷分布做选择依据与自设红线冲突。
    缺陷侧来源改为 train/anomaly（协议内 init_defect）与人工确认反馈图；
    两侧任一侧不足 4 张时按原逻辑跳过门控（不降级为单正常侧漂移判定，
    避免与在线 RegressGate 的正常侧门控重复且口径含糊）。

    返回 [(path, label_int), ...]，按 path 去重。
    """
    from sqlalchemy import func as sa_func
    items: List = []
    seen = set()
    with session_scope() as s:
        normals = (s.query(ImageRow)
                   .filter(ImageRow.category == category,
                           ImageRow.split == "train",
                           ImageRow.label == "normal")
                   .order_by(sa_func.random()).limit(n_per_side).all())
        train_anoms = (s.query(ImageRow)
                       .filter(ImageRow.category == category,
                               ImageRow.split == "train",
                               ImageRow.label == "anomaly")
                       .order_by(sa_func.random()).limit(n_per_side).all())
        fb_rows = (s.query(Detection.image_path)
                   .join(Feedback, Feedback.detection_id == Detection.id)
                   .filter(Detection.category == category,
                           Feedback.operator_label == 1)
                   .order_by(sa_func.random()).limit(n_per_side).all())
    for r in normals:
        if r.path not in seen:
            seen.add(r.path)
            items.append((r.path, 0))
    for r in train_anoms:
        if r.path not in seen:
            seen.add(r.path)
            items.append((r.path, 1))
    for (p,) in fb_rows:
        if p not in seen:
            seen.add(p)
            items.append((p, 1))
    return items


def evaluate_activation_gate(category: str, candidate_version: int,
                             margin: float = 0.02) -> Dict:
    """版本激活质量门控：候选版本 vs 当前激活版本在锚定集上平行打分。

    锚定集 = train 正常抽样 20 + (test 缺陷 ∪ 人工确认缺陷反馈图) 抽样 20；
    有标注图不足 4 张、或正/负样本任一侧不足 4 张时跳过门控
    （skipped=True 直接放行）——AUROC 分辨率为 1/(n_pos×n_neg)，
    单侧 <4 时粒度已粗于 margin=0.02，门控结论纯属噪声。
    candidate_auroc < current_auroc - margin → passed=False（路由层返 409）。
    任一 AUROC 不可计算（单一类别）时不做阻断，passed=True。

    Returns gate_report: {skipped, passed, margin, n_images, current_version,
    candidate_version, current:{auroc,f1}, candidate:{auroc,f1}, divergent}
    """
    from sklearn.metrics import roc_auc_score

    from ..engine import get_engine

    engine = get_engine()
    current_version = engine.current_version(category)
    report: Dict = {"skipped": False, "passed": True, "margin": margin,
                    "n_images": 0,
                    "current_version": current_version,
                    "candidate_version": candidate_version,
                    "current": {"auroc": None, "f1": None},
                    "candidate": {"auroc": None, "f1": None},
                    "divergent": 0}
    if current_version is None:
        # 2026-08-30 首用质检 B1 修复：首次激活无当前版本可比，
        # 门控跳过直接放行——否则全新部署"准备→验收→激活"路径被 400 切断
        report["skipped"] = True
        report["skip_reason"] = "首次激活（无当前版本可比），门控跳过"
        log_action("activation_gate",
                   f"category={category} v{candidate_version} "
                   f"skipped (first activation)")
        return report

    items = _gate_anchor_items(category)
    report["n_images"] = len(items)
    if candidate_version == current_version:
        report["skipped"] = True
        report["skip_reason"] = "候选版本即当前激活版本，无需门控"
        return report
    n_pos = sum(1 for _, y in items if y == 1)
    n_neg = len(items) - n_pos
    if len(items) < 4 or min(n_pos, n_neg) < 4:
        report["skipped"] = True
        report["skip_reason"] = (f"锚定样本不足（total={len(items)} "
                                 f"pos={n_pos} neg={n_neg}，正负各需≥4），"
                                 f"跳过门控")
        log_action("activation_gate",
                   f"category={category} v{candidate_version} skipped "
                   f"n={len(items)} pos={n_pos} neg={n_neg}")
        return report

    pipe_cur = engine.get_pipeline(category)
    pipe_cand = engine.build_variant(category, candidate_version)

    labels = np.array([y for _, y in items], dtype=np.int64)
    scores: Dict[str, List[float]] = {"current": [], "candidate": []}
    preds: Dict[str, List[int]] = {"current": [], "candidate": []}
    for path, _y in items:
        for side, pipe in (("current", pipe_cur), ("candidate", pipe_cand)):
            try:
                rec = pipe.predict(path)
                scores[side].append(float(rec.get("fused", 0.0)))
                preds[side].append(int(rec.get("decision") == "anomaly"))
            except Exception as e:  # noqa: BLE001 单图失败置中性分
                logger.warning("门控打分失败 %s (%s): %s", path, side, e)
                scores[side].append(0.5)
                preds[side].append(0)

    for side in ("current", "candidate"):
        sc = np.array(scores[side], dtype=np.float64)
        if labels.min() != labels.max():
            report[side]["auroc"] = float(roc_auc_score(labels, sc))
        report[side]["f1"] = _best_f1_threshold(sc, labels)["f1"]
    report["divergent"] = int(sum(a != b for a, b in
                                  zip(preds["current"], preds["candidate"])))

    auc_cur, auc_cand = report["current"]["auroc"], report["candidate"]["auroc"]
    if auc_cur is not None and auc_cand is not None:
        report["passed"] = bool(auc_cand >= auc_cur - margin)

    log_action("activation_gate",
               f"category={category} v{current_version}→v{candidate_version} "
               f"auroc {auc_cur} -> {auc_cand} passed={report['passed']} "
               f"n={len(items)} divergent={report['divergent']}")
    return report


# ── 六维评价 · 维度2：噪声鲁棒性（BEND-BCI 迁移）───────────────
def run_robustness_eval(
    category: str,
    split: str = "test",
    limit: int = 60,
    noise_kinds: List[str] = ("brightness", "gauss", "blur"),
    levels: List[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """在检测流水线上施加图像扰动，统计各扰动档位 AUC → 分数-噪声曲线面积 A_rob。

    扰动图写临时目录再走统一 detect 路径（兼容任意引擎，不污染原图）。
    Returns: {"eval_run_id","category","curves":{kind:{auc_levels,a_rob}},
              "baseline_auroc"}
    """
    import uuid

    import cv2

    from sklearn.metrics import roc_auc_score

    from ..core.config import get_settings
    from ..pipeline.service import get_detection_service

    def _fetch(lbl: str, n: int) -> list:
        with session_scope() as s:
            return (s.query(ImageRow)
                    .filter(ImageRow.category == category,
                            ImageRow.split == split,
                            ImageRow.label == lbl)
                    .order_by(ImageRow.id).limit(n).all())

    rows = _fetch("anomaly", limit // 2) + _fetch("normal", limit - limit // 2)
    images = [(r.id, r.path, 1 if r.label == "anomaly" else 0) for r in rows]
    if len(images) < 8:
        raise ValueError(f"category={category} split={split} 标注样本不足")

    tmp_dir = get_settings().storage("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    service = get_detection_service()

    def _score(path: str):
        det = service.detect_image(path, category, persist=False,
                                   with_heatmap=False)
        return float(det["final_score"])

    def _perturb(img, kind: str, lv: float, seed: int) -> str:
        rng = np.random.default_rng(seed)
        out = img.astype(np.float32)
        if kind == "brightness":
            out = np.clip(out * (1.0 + lv), 0, 255)
        elif kind == "gauss":
            out = np.clip(out + rng.normal(0, 40 * lv, out.shape), 0, 255)
        elif kind == "blur":
            out = cv2.GaussianBlur(out.astype(np.uint8), (0, 0), 0.5 + 4.0 * lv)
        p = tmp_dir / f"rob_{uuid.uuid4().hex[:10]}_{seed}.png"
        cv2.imwrite(str(p), cv2.cvtColor(out.astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))
        return str(p)

    # 基线
    y = np.array([lbl for _, _, lbl in images])
    base = []
    for i, (img_id, path, lbl) in enumerate(images):
        base.append(_score(path))
        if progress_cb:
            progress_cb(i + 1, len(images))
    base_auc = float(roc_auc_score(y, np.array(base)))

    curves = {}
    for kind in noise_kinds:
        aucs = []
        for lv in levels:
            scores = []
            for i, (img_id, path, lbl) in enumerate(images):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    scores.append(0.5)
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tmp_path = _perturb(img, kind, lv, seed=int(i))
                try:
                    scores.append(_score(tmp_path))
                except Exception as e:  # noqa: BLE001
                    logger.warning("扰动图检测失败 %s: %s", tmp_path, e)
                    scores.append(0.5)
            aucs.append(float(roc_auc_score(y, np.array(scores))))
        arob = sum((aucs[i] + aucs[i + 1]) / 2 * (levels[i + 1] - levels[i])
                   for i in range(len(levels) - 1))
        curves[kind] = {"auc_levels": aucs, "a_rob": arob}

    run_name = f"robustness_{category}_{datetime.now():%Y%m%d_%H%M%S}"
    with session_scope() as s:
        run = EvalRun(name=run_name, run_type="robustness", category=category,
                      dataset_desc=f"split={split}, n={len(images)}, kinds={noise_kinds}",
                      n_images=len(images),
                      metrics={"baseline_auroc": base_auc, "curves": curves})
        s.add(run)
        s.flush()
        run_id = run.id

    log_action("robustness_eval",
               f"category={category} baseline={base_auc:.4f} "
               f"a_rob=" + ",".join(f"{k}={c['a_rob']:.4f}" for k, c in curves.items()))
    return {"eval_run_id": run_id, "category": category,
            "baseline_auroc": base_auc, "curves": curves}


# ── M2b：学习曲线回放 / 槽位贡献档案（Demo5 引擎评估能力）───────
def _fetch_eval_items(category: str, split: str, limit: int) -> List:
    """均衡取 split 内有标注图 [(path, label_int), ...]（上限 limit）。"""
    with session_scope() as s:
        def _fetch(lbl: str, n: int) -> list:
            return (s.query(ImageRow)
                    .filter(ImageRow.category == category,
                            ImageRow.split == split,
                            ImageRow.label == lbl)
                    .order_by(ImageRow.id).limit(n).all())
        half = max(limit // 2, 1)
        rows = _fetch("anomaly", half) + _fetch("normal", limit - half)
    return [(r.path, 1 if r.label == "anomaly" else 0) for r in rows]


def evaluate_learning_curve(
    category: str,
    n_stream: int = 20,
    n_eval: int = 60,
    feedback_ratio: float = 1.0,
    eval_every: int = 5,
    max_feedback: int = 20,
    mode: str = "rounds",
    round_size: int = 25,
    n_rounds: int = 4,
    eval_per_sample: int = 5,
    progress_cb: Optional[Callable] = None,
) -> Dict:
    """学习曲线离线回放（独立 pipe 副本，不动线上模型）。

    mode="rounds"（默认）：多轮持续学习（algo simulate_learning_rounds）——
    轮次轴锚定集 AUROC（举一反三）+ 已反馈重测指标（学过就会：重测准确率/
    错→对转化率/缺陷检出率），跨轮累积突破单轮流 ~53 条反馈上限与早期饱和
    （algo learning_curve.py 注释结论）。
    mode="stream"：旧版单轮流回放（algo simulate_learning），仅调试保留。

    eval_items：split=test 均衡抽样（锚定集，只评估不反馈）——取样规则与
    run_accuracy_eval（模型管理「历史最佳验证 AUC」来源）一致（test 按 id
     anomaly/normal 各取一半），初始 AUROC 与模型管理页同口径可比；
    2026-08-29 默认 n_eval 20→60：10+10 小样本 AUROC 分辨率仅 0.01 且
    极易饱和到 1.0（走查"AUC 恒 1"主因），30+30 分辨率 1/900。
    pool_items：品类其余有标注图（train/train_anomaly/未进锚定集的 test），
    异常/正常交错排列（避免流前段全 normal）。

    Returns: {"eval_run_id", ...回放结果}
    """
    from ..engine import get_engine

    eval_items = _fetch_eval_items(category, "test", n_eval)
    labels = {y for _, y in eval_items}
    if len(eval_items) < 2 or len(labels) < 2:
        raise ValueError(f"category={category} test 锚定集需同时含正常/异常标注图"
                         f"（当前 {len(eval_items)} 张）")
    eval_paths = {p for p, _ in eval_items}

    with session_scope() as s:
        rows = (s.query(ImageRow)
                .filter(ImageRow.category == category,
                        ImageRow.label.in_(("normal", "anomaly")))
                .order_by(ImageRow.id).all())
    rest = [(r.path, 1 if r.label == "anomaly" else 0)
            for r in rows if r.path not in eval_paths]

    def _interleaved(items: List, cap: int) -> List:
        anoms = [it for it in items if it[1] == 1]
        norms = [it for it in items if it[1] == 0]
        out: List = []
        while (anoms or norms) and len(out) < cap:
            if anoms and len(out) < cap:
                out.append(anoms.pop(0))
            if norms and len(out) < cap:
                out.append(norms.pop(0))
        return out

    engine = get_engine()

    if mode == "rounds":
        # 多轮持续学习：轮池 = 锚定集之外全部有标注图（上限留 1 轮余量），
        # 轮内无放回抽样；不足一轮也放行（algo 内部按可用条数反馈）。
        pool_items = _interleaved(rest, int(round_size) * (int(n_rounds) + 1))
        if not pool_items:
            raise ValueError(f"category={category} 锚定集之外无有标注图可回放")
        if progress_cb:
            progress_cb(0, 1, f"多轮学习回放中（{n_rounds} 轮 × {round_size} 条，"
                              f"轮池={len(pool_items)} 锚定集={len(eval_items)}，"
                              f"离线副本与产线并行）...")
        cb = (lambda d, t: progress_cb(d, t, "多轮学习回放中（前向进度）...")
              ) if progress_cb else None
        result = engine.simulate_learning_rounds(
            category, pool_items, eval_items,
            round_size=int(round_size), n_rounds=int(n_rounds),
            eval_per_sample=int(eval_per_sample), progress_cb=cb)
        if progress_cb:
            progress_cb(1, 1, "回放完成，入库...")
        result["mode"] = "rounds"
        learned = result.get("learned") or {}
        n_fb_total = int(learned.get("n_feedback_total") or 0)
        desc = (f"多轮 {n_rounds}×{round_size} 反馈={n_fb_total} "
                f"锚定集=test均衡{len(eval_items)}张"
                f"（异常{sum(1 for _, y in eval_items if y == 1)}"
                f"/正常{sum(1 for _, y in eval_items if y == 0)}）")
        log_action("learning_curve",
                   f"category={category} mode=rounds n_feedback={n_fb_total} "
                   f"auroc {result.get('initial_auroc')} -> "
                   f"{result.get('final_auroc')} gain={result.get('auroc_gain')}")
    else:
        # 旧版单轮流（调试保留）
        stream_items = _interleaved(rest, max(2, int(n_stream)))
        if not stream_items:
            raise ValueError(f"category={category} 锚定集之外无有标注图可回放")
        if progress_cb:
            progress_cb(0, 1, f"学习回放中（stream={len(stream_items)} "
                              f"锚定集={len(eval_items)}，离线副本与产线并行）...")
        cb = (lambda d, t: progress_cb(d, t, "学习回放中（前向进度）...")
              ) if progress_cb else None
        result = engine.simulate_learning(
            category, stream_items, eval_items,
            feedback_ratio=feedback_ratio, eval_every=eval_every,
            max_feedback=max_feedback, box_map=None, progress_cb=cb)
        if progress_cb:
            progress_cb(1, 1, "回放完成，入库...")
        result["mode"] = "stream"
        n_fb_total = int(result.get("n_feedback") or 0)
        pool_items = stream_items
        desc = (f"n_stream={len(stream_items)} "
                f"锚定集=test均衡{len(eval_items)}张"
                f"（异常{sum(1 for _, y in eval_items if y == 1)}"
                f"/正常{sum(1 for _, y in eval_items if y == 0)}） "
                f"feedback_ratio={feedback_ratio} eval_every={eval_every} "
                f"max_feedback={max_feedback}")
        log_action("learning_curve",
                   f"category={category} mode=stream n_feedback={n_fb_total} "
                   f"auroc {result.get('initial_auroc')} -> "
                   f"{result.get('final_auroc')} gain={result.get('auroc_gain')}")

    run_name = f"learning_curve_{category}_{datetime.now():%Y%m%d_%H%M%S}"
    with session_scope() as s:
        run = EvalRun(name=run_name, run_type="learning_curve", category=category,
                      dataset_desc=desc,
                      n_images=len(pool_items) + len(eval_items),
                      metrics=result)
        s.add(run)
        s.flush()
        run_id = run.id

    return {"eval_run_id": run_id, **result}


def evaluate_contribution(
    category: str,
    n_eval: int = 30,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """槽位贡献档案（engine.contribution_report）：单变量/冗余度/消融三层评价。

    Returns: {"eval_run_id","profile":{univariate,redundancy,ablation}}
    """
    from ..engine import get_engine

    eval_items = _fetch_eval_items(category, "test", n_eval)
    labels = {y for _, y in eval_items}
    if len(eval_items) < 2 or len(labels) < 2:
        raise ValueError(f"category={category} test 评估集需同时含正常/异常标注图"
                         f"（当前 {len(eval_items)} 张）")

    if progress_cb:
        progress_cb(0, 1, f"贡献档案计算中（eval={len(eval_items)}）...")
    profile = get_engine().contribution_report(category, eval_items)
    if progress_cb:
        progress_cb(1, 1, "完成")

    run_name = f"contribution_{category}_{datetime.now():%Y%m%d_%H%M%S}"
    with session_scope() as s:
        run = EvalRun(name=run_name, run_type="contribution", category=category,
                      dataset_desc=f"split=test n={len(eval_items)}",
                      n_images=len(eval_items), metrics=profile)
        s.add(run)
        s.flush()
        run_id = run.id

    log_action("contribution_eval",
               f"category={category} n={len(eval_items)}")
    return {"eval_run_id": run_id, "profile": profile}
