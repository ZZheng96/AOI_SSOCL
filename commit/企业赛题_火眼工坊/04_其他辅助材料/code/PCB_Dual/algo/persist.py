"""Demo5 Pipeline 快照持久化。

Demo5 fit/在线更新后状态全部驻留内存（无原生 save/load），
本模块把运行时状态序列化为 `storage/snapshots/<category>/v<n>/`，
为 AOI_sys 的模型版本管理（激活/回滚/A-B 对比）提供基础。

快照目录结构：
    v<version>/
        meta.json             # 版本信息：版本号、父版本、创建时间、触发源、配置哈希
        cfg.yaml              # pipeline 配置快照（yaml.safe_dump）
        slot_state.pkl        # 各槽位内部状态：
                              #   sem: {bank}(tensor), disc/shead: {head_state_dict}
                              #   blob: 无, trad: {bank_w, norm_w}(np.ndarray)
                              #   layout: {norm, centroids, patterns}
                              #   inp: 无, tpl: {templates, norm}
        calibrators.npz       # 每槽 CDFCalibrator.edges
        decider.json          # Decider 参数 {tau_high, tau_gray}
        weights.json          # 融合权重 {slot: float}
        sanity.json           # slot_sanity（init_defect AUROC，用于 sanity 加权报告）
        router.pt             # router.state_dict（可空）
        router_gate.json      # {enabled, base_auc, router_auc, ...}（可空）
        consensus.json        # consensus_gate + consensus_weights（可空）
        open.npz              # open_det PCA（components/mean/explained/topk_ratio/thresh）
                              #   + open_cal edges
        ssocl/                # 仅在线更新后有
            normal_bank.pkl   # NormalBank: core/ext/fuse_blocked/...
            defect_bank.pkl   # DefectBank: samples/topk/sim_thresh/boost/...
            handler.json      # regress_gate/weight_learner/trigger 状态
            head_mgr.pkl      # IncubatedHead 权重
            cal_hist.npz      # 每槽 _cal_normal_hist
    current.json              # {version: n}（激活版本指针，原子写）

设计原则：
- save 是"冷"快照：Pipeline 对象不再修改时调用，
  save 后 pipe 对象内部状态不变。
- load 时重新装配 backbone（按 cfg 走 factory.build，
  预训练权重 init_from 在 cfg 中给绝对路径），再把槽位状态注入。
  为保证加载后 predict 与保存前一致：
  - backbone 是 frozen（DINO 永冻），权重由 init_from + hub 决定，
    不在快照里重复存储。
  - 所有 fit/在线更新产生的可变状态都列在上面清单里；
    加载后调用 pipe._item_cache.clear()（运行时缓存不清无关紧要但
    与 save 前等价）。
"""
from __future__ import annotations

import os
import json
import pickle
import shutil
import tempfile
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from .eval.offline import Pipeline


def _slot_state(slot) -> Dict[str, Any]:
    """提取单个槽位的 fit 后可变状态。"""
    name = slot.name
    if name == "sem":
        bank = getattr(slot, "bank", None)
        bank_t = bank if isinstance(bank, torch.Tensor) else (
            torch.tensor(bank, dtype=torch.float32) if bank is not None else None
        )
        return {"bank": bank_t}
    if name == "disc":
        head = getattr(slot, "head", None)
        state: Dict[str, Any] = {"head_state_dict": head.state_dict() if head is not None else None}
        # DiscHead 额外 fit 时态（输入维度由 cfg + stat 开关决定）：
        for attr in ("in_dim", "score_mode", "block_score_topk", "guided_score_topk",
                     "score_overlap", "score_max_crops", "use_context_crops", "use_adjacent_agg",
                     "crop_size", "adaptive_crop_size",
                     "chan_stats", "patch_stats", "focus_topk", "focus_score_only",
                     "loss_history"):
            if hasattr(slot, attr):
                state[attr] = getattr(slot, attr)
        return state
    if name == "shead":
        head = getattr(slot, "head", None)
        state = {"head_state_dict": head.state_dict() if head is not None else None}
        # SheadSlot 额外 fit 时态（含 fit 内可能覆盖的 crop_size）
        for attr in ("in_dim", "score_mode", "block_score_topk", "guided_score_topk",
                     "score_overlap", "score_max_crops", "use_context_crops", "use_adjacent_agg",
                     "crop_size", "adaptive_crop_size",
                     "chan_stats", "patch_stats", "focus_topk", "focus_score_only",
                     "rand_sliding_train", "n_pos_slides_per_defect", "slide_jitter",
                     "pos_frac_min", "block_pos_thresh", "n_score_grid_max",
                     "n_fit_normal_crops", "crop_pad",
                     "loss_history"):
            if hasattr(slot, attr):
                state[attr] = getattr(slot, attr)
        return state
    if name == "trad":
        return {"bank_w": getattr(slot, "bank_w", None),
                "norm_w": getattr(slot, "norm_w", None),
                "config": getattr(slot, "cfg", None)}
    if name == "layout":
        return {"norm": getattr(slot, "norm", None),
                "centroids": getattr(slot, "centroids", None),
                "patterns": getattr(slot, "patterns", None),
                "n_components": getattr(slot, "n_components", None),
                "pca": getattr(slot, "pca", None)}
    if name == "blob":
        return {"sigmas": getattr(slot, "sigmas", None)}
    if name == "inp":
        return {}
    if name == "tpl":
        return {"templates": getattr(slot, "templates", None),
                "norm": getattr(slot, "norm", None)}
    if name == "color":
        return {"hist_mean": getattr(slot, "hist_mean", None),
                "hist_std": getattr(slot, "hist_std", None)}
    return {}


def _restore_slot_state(slot, state: Dict[str, Any]) -> None:
    """把快照状态注入槽位。"""
    name = slot.name
    if not state:
        return
    if name == "sem":
        if "bank" in state and state["bank"] is not None:
            b = state["bank"]
            if isinstance(b, torch.Tensor):
                cur = getattr(slot, "bank", None)
                dev = cur.device if isinstance(cur, torch.Tensor) else None
                slot.bank = b.to(dev) if dev is not None else b
            else:
                slot.bank = torch.tensor(b, dtype=torch.float32)
    elif name in ("disc", "shead"):
        head = getattr(slot, "head", None)
        if head is not None and state.get("head_state_dict") is not None:
            missing, extra = head.load_state_dict(state["head_state_dict"], strict=False)
            if missing:
                print(f"[persist] {name} head missing keys: {missing[:3]}", flush=True)
        # 恢复 fit 时可能被覆盖/计算出的运行时态（crop_size、统计口径、loss 曲线等）
        for attr in ("in_dim", "score_mode", "block_score_topk", "guided_score_topk",
                     "score_overlap", "score_max_crops", "use_context_crops", "use_adjacent_agg",
                     "crop_size", "adaptive_crop_size",
                     "chan_stats", "patch_stats", "focus_topk", "focus_score_only",
                     "rand_sliding_train", "n_pos_slides_per_defect", "slide_jitter",
                     "pos_frac_min", "block_pos_thresh", "n_score_grid_max",
                     "n_fit_normal_crops", "crop_pad",
                     "loss_history"):
            if attr in state and state[attr] is not None:
                setattr(slot, attr, state[attr])
        # 关键：fit 完成后 head 会被显式切到 eval（见 shead.py:560/723/860/1026 与 disc.py:150）。
        # 快照 load 不重跑 fit，槽位 __init__ 默认 head 为 train 模式，dropout/bn 会随机化打分，
        # 导致 save/load 同图 predict 不一致——这里统一切换到 eval。
        if head is not None:
            head.eval()
    elif name == "trad":
        for k in ("bank_w", "norm_w", "cfg"):
            if k in state and state[k] is not None:
                setattr(slot, k if k != "cfg" else "cfg", state[k])
    elif name == "layout":
        for k in ("norm", "centroids", "patterns", "n_components", "pca"):
            if k in state and state[k] is not None:
                setattr(slot, k, state[k])
    elif name == "blob":
        if "sigmas" in state and state["sigmas"] is not None:
            slot.sigmas = list(state["sigmas"])
    elif name == "tpl":
        for k in ("templates", "norm"):
            if k in state and state[k] is not None:
                setattr(slot, k, state[k])
    elif name == "color":
        for k in ("hist_mean", "hist_std"):
            if k in state and state[k] is not None:
                setattr(slot, k, state[k])


def _atomic_write(target_path: str, write_fn) -> None:
    """原子写：tmp 目录 + rename，避免崩溃造成快照损坏。"""
    d = os.path.dirname(target_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    os.close(fd)
    try:
        write_fn(tmp)
        os.replace(tmp, target_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_snapshot(pipe: Pipeline, snapshot_dir: str,
                  *, version: int, parent_version: Optional[int] = None,
                  trigger: str = "fit", note: str = "",
                  ssocl_cfg: Optional[Dict[str, Any]] = None) -> str:
    """保存 Pipeline 为独立快照目录。返回 snapshot_dir。

    ssocl_cfg：引擎侧在线学习配置（banks/gate/head_ft 等），随快照落盘
    ssocl/ssocl_cfg.json，保证快照自包含（加载不再依赖硬编码模板）。"""
    os.makedirs(snapshot_dir, exist_ok=True)

    # 1. cfg.yaml（配置副本，离线加载 backbone 用）
    with open(os.path.join(snapshot_dir, "cfg.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(pipe.cfg, f, allow_unicode=True, sort_keys=False)

    # 2. slot_state.pkl
    slots_state = {name: _slot_state(s) for name, s in pipe.slots.items()}
    slots_state["_order"] = list(pipe.slots.keys())
    with open(os.path.join(snapshot_dir, "slot_state.pkl"), "wb") as f:
        pickle.dump(slots_state, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. calibrators.npz
    cal = {}
    for name, c in pipe.calibrators.items():
        cal[f"{name}__edges"] = np.asarray(c.edges, dtype=np.float64)
    np.savez_compressed(os.path.join(snapshot_dir, "calibrators.npz"), **cal)

    # 4. decider.json / weights.json / sanity.json / consensus.json
    dec = {"tau_high": float(pipe.decider.tau_high) if pipe.decider else 0.99,
           "tau_gray": float(pipe.decider.tau_gray) if pipe.decider else 0.95}
    with open(os.path.join(snapshot_dir, "decider.json"), "w", encoding="utf-8") as f:
        json.dump(dec, f, ensure_ascii=False, indent=2)
    with open(os.path.join(snapshot_dir, "weights.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in (pipe.weights or {}).items()}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(snapshot_dir, "sanity.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in (pipe.slot_sanity or {}).items()}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(snapshot_dir, "consensus.json"), "w", encoding="utf-8") as f:
        json.dump({"gate": pipe.consensus_gate}, f, ensure_ascii=False, indent=2)

    # 5. router
    if getattr(pipe, "router", None) is not None:
        torch.save(pipe.router.state_dict(), os.path.join(snapshot_dir, "router.pt"))
    rg_path = os.path.join(snapshot_dir, "router_gate.json")
    with open(rg_path, "w", encoding="utf-8") as f:
        json.dump(pipe.router_gate, f, ensure_ascii=False, indent=2)

    # 6. open_det
    open_path = os.path.join(snapshot_dir, "open.npz")
    open_dict: Dict[str, Any] = {"enabled": pipe.open_det is not None}
    if pipe.open_det is not None:
        pca = getattr(pipe.open_det, "pca", None)
        for attr in ("components_", "mean_", "explained_variance_ratio_"):
            if pca is not None and hasattr(pca, attr):
                open_dict[attr] = np.asarray(getattr(pca, attr), dtype=np.float64)
        open_dict["topk_ratio"] = float(getattr(pipe.open_det, "topk_ratio", 0.05))
        open_dict["thresh"] = float(getattr(pipe.open_det, "thresh", 0.9))
    if pipe.open_cal is not None:
        open_dict["open_cal_edges"] = np.asarray(pipe.open_cal.edges, dtype=np.float64)
    np.savez_compressed(open_path, **{k: v for k, v in open_dict.items() if isinstance(v, np.ndarray)})
    open_meta = {k: v for k, v in open_dict.items() if not isinstance(v, np.ndarray)}
    with open(os.path.join(snapshot_dir, "open_meta.json"), "w", encoding="utf-8") as f:
        json.dump(open_meta, f, ensure_ascii=False, indent=2)

    # 7.5 对位预警参考（M11d）：有则随快照存取，保证 load 后预警行为一致
    if getattr(pipe, "_align_ref", None) is not None:
        np.save(os.path.join(snapshot_dir, "align_ref.npy"), pipe._align_ref)

    # 7. SSCL（在线状态）—— handler 非空才有
    ssocl_dir = os.path.join(snapshot_dir, "ssocl")
    if pipe.handler is not None:
        os.makedirs(ssocl_dir, exist_ok=True)
        if pipe.handler.normal_bank is not None:
            with open(os.path.join(ssocl_dir, "normal_bank.pkl"), "wb") as f:
                pickle.dump(pipe.handler.normal_bank, f, protocol=pickle.HIGHEST_PROTOCOL)
        if pipe.handler.defect_bank is not None:
            with open(os.path.join(ssocl_dir, "defect_bank.pkl"), "wb") as f:
                pickle.dump(pipe.handler.defect_bank, f, protocol=pickle.HIGHEST_PROTOCOL)
        # handler 轻量状态（门控/学习器参数）
        hstate = {
            "gate_snapshot": getattr(pipe.handler, "gate_snapshot", None),
            "gate_tol": getattr(pipe.handler, "gate_tol", None),
            "weight_learner": {
                "weights": getattr(getattr(pipe.handler, "weight_learner", None),
                                   "weights", None),
            },
            "pending_defect_feedback": len(getattr(pipe.handler, "_pending_defect", [])),
            "pending_normal_feedback": len(getattr(pipe.handler, "_pending_normal", [])),
            "cooling": getattr(pipe.handler, "_cooling", {}),
        }
        try:
            with open(os.path.join(ssocl_dir, "handler.json"), "w", encoding="utf-8") as f:
                json.dump(hstate, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        # gate_snapshot: RegressGate 基准 + 锚定路径（load 后无需重新 _predict_fused_only）
        try:
            gate = getattr(pipe.handler, "gate", None)
            if gate is not None:
                snap = {
                    "tol": float(getattr(gate, "tol", 0.05)),
                    "anchor_size": int(getattr(gate, "anchor_size", 20)),
                    "anchor_paths": list(getattr(gate, "anchor_paths", []) or []),
                    "base_mean": float(getattr(gate, "base_mean")) if getattr(gate, "base_mean", None) is not None else None,
                    "base_std": float(getattr(gate, "base_std")) if getattr(gate, "base_std", None) is not None else None,
                    "history": list(getattr(gate, "history", []) or []),
                }
                with open(os.path.join(ssocl_dir, "regress_gate.json"), "w", encoding="utf-8") as f:
                    json.dump(snap, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        # weight_learner 完整状态（只保存真实数据字段；"weights" 是方法名，绝不能落盘/覆盖）
        try:
            wl = getattr(pipe.handler, "weight_learner", None)
            if wl is not None:
                wl_state = {}
                for attr in ("slots", "min_neg", "n", "pos", "neg", "n_updates",
                             "height_suppress", "height_log", "_height_flagged",
                             "initial_weights", "prior_fade"):
                    if hasattr(wl, attr):
                        v = getattr(wl, attr)
                        if callable(v):
                            continue
                        try:
                            if isinstance(v, np.ndarray):
                                v = v.tolist()
                            elif hasattr(v, "item"):
                                v = v.item()
                            elif isinstance(v, set):
                                v = list(v)
                        except Exception:
                            pass
                        wl_state[attr] = v
                with open(os.path.join(ssocl_dir, "weight_learner.json"), "w", encoding="utf-8") as f:
                    json.dump(wl_state, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        # 权重演化历史（学习效果页曲线；滚动 300，FeedbackHandler._record_weight 落点）
        try:
            wh = list(getattr(pipe.handler, "weight_history", []) or [])
            if wh:
                with open(os.path.join(ssocl_dir, "weight_history.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(wh[-300:], f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        if pipe.head_mgr is not None and getattr(pipe.head_mgr, "heads", None):
            with open(os.path.join(ssocl_dir, "head_mgr.pkl"), "wb") as f:
                pickle.dump(pipe.head_mgr, f, protocol=pickle.HIGHEST_PROTOCOL)
        # 每槽 _cal_normal_hist
        hist_dict = {n: np.asarray(v, dtype=np.float64)
                     for n, v in (pipe._cal_normal_hist or {}).items() if v}
        if hist_dict:
            np.savez_compressed(os.path.join(ssocl_dir, "cal_hist.npz"), **hist_dict)

    # 8. meta.json（版本信息，原子写）
    meta = {"version": int(version),
            "parent_version": int(parent_version) if parent_version is not None else None,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trigger": trigger,
            "note": note,
            "fit_seconds": float(getattr(pipe, "fit_seconds", 0.0)),
            "ssocl_enabled": pipe.handler is not None,
            "weights_mode": pipe.cfg.get("fusion", {}).get("weight_mode", "equal"),
            "slot_names": list(pipe.slots.keys())}
    meta_path = os.path.join(snapshot_dir, "meta.json")
    _atomic_write(meta_path, lambda tmp: (
        open(tmp, "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=2))))
    return snapshot_dir


def list_versions(category_dir: str) -> List[int]:
    """已存版本号（sorted int）。"""
    if not os.path.isdir(category_dir):
        return []
    vs = []
    for name in os.listdir(category_dir):
        if name.startswith("v") and os.path.isdir(os.path.join(category_dir, name)):
            try:
                vs.append(int(name[1:]))
            except ValueError:
                pass
    return sorted(vs)


def current_version(category_dir: str) -> Optional[int]:
    """读取激活版本指针 current.json。"""
    cur = os.path.join(category_dir, "current.json")
    if not os.path.exists(cur):
        return None
    try:
        with open(cur, "r", encoding="utf-8") as f:
            d = json.load(f)
        return int(d["version"])
    except Exception:
        return None


def set_current_version(category_dir: str, version: int) -> None:
    """原子写激活版本指针。"""
    os.makedirs(category_dir, exist_ok=True)
    target = os.path.join(category_dir, "current.json")
    _atomic_write(target, lambda tmp: (
        open(tmp, "w", encoding="utf-8").write(
            json.dumps({"version": int(version), "set_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                       ensure_ascii=False, indent=2))))


def load_snapshot(snapshot_dir: str, device: str,
                  *, _rebuild_backbone=True) -> Pipeline:
    """按快照目录加载 Pipeline。返回的 pipe.predict 应与 save 前一致。

    backbone/disc/shead 的预训练 init_from 在 cfg.yaml 中保留原绝对路径，
    集成时由引擎层在传入 cfg 时改写为 AOI_sys 内部路径即可。
    """
    with open(os.path.join(snapshot_dir, "cfg.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from .factory import build
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)

    # slot state
    with open(os.path.join(snapshot_dir, "slot_state.pkl"), "rb") as f:
        slots_state = pickle.load(f)
    order = slots_state.pop("_order", list(pipe.slots.keys()))
    for name in order:
        if name not in pipe.slots:
            continue
        _restore_slot_state(pipe.slots[name], slots_state.get(name, {}))

    # calibrators
    from .fusion.calibrate import CDFCalibrator
    cal_npz = np.load(os.path.join(snapshot_dir, "calibrators.npz"))
    pipe.calibrators = {}
    for arr_key, edges in cal_npz.items():
        if not arr_key.endswith("__edges"):
            continue
        n = arr_key[: -len("__edges")]
        cal = CDFCalibrator(n_bins=len(edges) - 1 if edges.ndim else 256)
        cal.edges = np.asarray(edges, dtype=np.float64)
        pipe.calibrators[n] = cal
    cal_npz.close()

    # decider
    with open(os.path.join(snapshot_dir, "decider.json"), "r", encoding="utf-8") as f:
        dec = json.load(f)
    from .decide import Decider
    pipe.decider = Decider(tau_high=float(dec["tau_high"]), tau_gray=float(dec["tau_gray"]))
    pipe.train_auroc = dec.get("train_auroc")   # U108：旧快照无此键 → None（权重学习按无判据放行）

    with open(os.path.join(snapshot_dir, "weights.json"), "r", encoding="utf-8") as f:
        pipe.weights = {k: float(v) for k, v in json.load(f).items()}
    # U96：全局类型判别器（跨品类共享模型，非品类状态，不入快照文件）——
    # 加载时从 cfg.attribution.global_clf 或随包 assets 恢复，保证 load 后
    # 归因行为与 fit 后一致（否则回退规则映射，save/load 语义不一致）。
    try:
        from .eval.offline import _global_clf_path
        from .decision import defect_classifier as _dc  # noqa: F401 注册 src.* pkl 别名
        _clf_path = _global_clf_path(cfg)
        if os.path.exists(_clf_path):
            with open(_clf_path, "rb") as _f:
                pipe.defect_clf = pickle.load(_f)
    except Exception:
        pipe.defect_clf = None
    with open(os.path.join(snapshot_dir, "sanity.json"), "r", encoding="utf-8") as f:
        pipe.slot_sanity = {k: float(v) for k, v in json.load(f).items()}
    with open(os.path.join(snapshot_dir, "consensus.json"), "r", encoding="utf-8") as f:
        cc = json.load(f)
        pipe.consensus_gate = cc.get("gate")

    # 派生状态（来自 fit 入口，不单独持久化）
    fm0 = pipe.cfg.get("fusion", {})
    if fm0.get("skip_zero_weight") and fm0.get("only_slots"):
        pipe._score_slots = [n for n in pipe.slots if n in fm0["only_slots"]]
    else:
        pipe._score_slots = list(pipe.slots.keys())
    pipe._last_eval = None
    pipe._ablate_intercept = False
    # M11d 对位预警参考（旧快照无此文件 → None，预警自动停用）
    _ar_path = os.path.join(snapshot_dir, "align_ref.npy")
    if os.path.exists(_ar_path):
        try:
            pipe._align_ref = np.load(_ar_path)
        except Exception:  # noqa: BLE001 参考损坏仅停用预警
            pipe._align_ref = None

    # router
    r_path = os.path.join(snapshot_dir, "router.pt")
    rg_path = os.path.join(snapshot_dir, "router_gate.json")
    pipe.router = None
    pipe.router_gate = None
    if os.path.exists(r_path) and getattr(pipe, "router", None) is None:
        from .fusion.router import LinearRouter
        n_slots = len([n for n in pipe.slots if n != "open"])
        if n_slots:
            pipe.router = LinearRouter(n_slots).to(device)
            missing, extra = pipe.router.load_state_dict(torch.load(r_path, map_location=device, weights_only=False),
                                                     strict=False)
            pipe.router.eval()
    if os.path.exists(rg_path):
        with open(rg_path, "r", encoding="utf-8") as f:
            pipe.router_gate = json.load(f)

    # open_det
    open_meta_path = os.path.join(snapshot_dir, "open_meta.json")
    open_npz_path = os.path.join(snapshot_dir, "open.npz")
    if os.path.exists(open_meta_path):
        with open(open_meta_path, "r", encoding="utf-8") as f:
            om = json.load(f)
        if om.get("enabled"):
            from .slots.open import OpenDetector
            pipe.open_det = OpenDetector(
                n_components=float(om.get("thresh", 0.95)),
                topk_ratio=float(om.get("topk_ratio", 0.05)))
            pipe.open_det.thresh = float(om.get("thresh", 0.9))
            if os.path.exists(open_npz_path):
                open_npz = np.load(open_npz_path)
                if "components_" in open_npz and pipe.open_det.pca is None:
                    try:
                        from sklearn.decomposition import PCA
                        pipe.open_det.pca = PCA()
                        pipe.open_det.pca.components_ = open_npz["components_"]
                        pipe.open_det.pca.mean_ = open_npz["mean_"]
                        pipe.open_det.pca.explained_variance_ratio_ = open_npz["explained_variance_ratio_"]
                    except Exception:
                        pass
                if "open_cal_edges" in open_npz:
                    from .fusion.calibrate import CDFCalibrator
                    pipe.open_cal = CDFCalibrator(n_bins=len(open_npz["open_cal_edges"]) - 1)
                    pipe.open_cal.edges = np.asarray(open_npz["open_cal_edges"], dtype=np.float64)
                open_npz.close()

    # SSCL
    ssocl_dir = os.path.join(snapshot_dir, "ssocl")
    if os.path.isdir(ssocl_dir):
        # 先 enable_ssocl（空壳配置足以让 handler/selector/head_mgr 非空）
        from .ssocl.banks import NormalBank, DefectBank
        from .ssocl.feedback import FeedbackHandler, RegressGate
        from .ssocl.active import ActiveSelector
        from .ssocl.incubate import HeadManager
        nb_path = os.path.join(ssocl_dir, "normal_bank.pkl")
        db_path = os.path.join(ssocl_dir, "defect_bank.pkl")
        hm_path = os.path.join(ssocl_dir, "head_mgr.pkl")
        hist_path = os.path.join(ssocl_dir, "cal_hist.npz")
        gate_path = os.path.join(ssocl_dir, "regress_gate.json")
        wl_path = os.path.join(ssocl_dir, "weight_learner.json")
        tmp_cfg = {"banks": {"ext_max": 64, "cluster_k": 3, "sim_thresh": 0.75,
                             "intercept_topk": 8, "intercept_boost": 0.25,
                             "intercept_sim": 0.15},
                   "gate": {"tol": 0.05, "anchor_size": 20},
                   "active": {"top_n": 10}, "incubate": {"min_samples": 8},
                   "router_finetune": {"trigger": 6, "lr": 1e-4, "epochs": 5},
                   "weight_min_samples": 8,
                   "height_suppress": False, "height_recal": True}
        # 2026-08-31 评审修复（快照自包含）：优先读快照内 ssocl_cfg.json
        # （save_snapshot 随快照落盘），缺失时才回退硬编码模板并告警——
        # 模板与引擎配置重复维护，漂移即行为改变。
        _cfg_path = os.path.join(ssocl_dir, "ssocl_cfg.json")
        if os.path.exists(_cfg_path):
            try:
                with open(_cfg_path, "r", encoding="utf-8") as f:
                    tmp_cfg = json.load(f)
            except Exception as e:  # noqa: BLE001 配置损坏回退模板
                print(f"[persist][警告] ssocl_cfg.json 读取失败（{e}），"
                      "回退硬编码模板", flush=True)
        else:
            print(f"[persist][警告] 快照缺 ssocl_cfg.json（旧版快照），"
                  f"按硬编码模板恢复：{snapshot_dir}", flush=True)
        _bk = tmp_cfg.get("banks", {})
        # 直接装配（不调用 enable_ssocl，避免重复锚定集采样）
        pipe.handler = FeedbackHandler(pipe, tmp_cfg)
        if os.path.exists(nb_path):
            with open(nb_path, "rb") as f:
                pipe.handler.normal_bank = pickle.load(f)
            # 2026-08-30：缺陷率滑动窗是当前决策流的实时属性，不随快照传播
            # （防历史污染窗口锁死新会话回流，MPDD 评测 fuse 继承锁死教训）
            pipe.handler.normal_bank.reset_stream_stats()
            # v7-4：快照内 bank 携带构建期参数（旧快照 cluster_k/sim_thresh
            # 过严），内容保留、可调参数按当前配置刷新
            _nb = pipe.handler.normal_bank
            _nb.cluster_k = _bk.get("cluster_k", _nb.cluster_k)
            _nb.sim_thresh = _bk.get("sim_thresh", _nb.sim_thresh)
            _nb.ext_max = _bk.get("ext_max", _nb.ext_max)
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                pipe.handler.defect_bank = pickle.load(f)
            # v7-4：旧快照 defect_bank 的 sim_thresh=0.85 近乎不命中，
            # 内容（样例）保留、拦截参数按当前配置刷新
            _db = pipe.handler.defect_bank
            _db.topk = _bk.get("intercept_topk", _db.topk)
            _db.boost = _bk.get("intercept_boost", _db.boost)
            _db.sim_thresh = _bk.get("intercept_sim", _db.sim_thresh)
        pipe.selector = ActiveSelector(tmp_cfg.get("active", {}))
        if os.path.exists(hm_path):
            with open(hm_path, "rb") as f:
                pipe.head_mgr = pickle.load(f)
        else:
            pipe.head_mgr = HeadManager(tmp_cfg.get("incubate", {}))
        if os.path.exists(hist_path):
            hist_npz = np.load(hist_path)
            pipe._cal_normal_hist = {k: list(hist_npz[k]) for k in hist_npz.files}
            hist_npz.close()
        # RegressGate 恢复：gate_snapshot 里保留 anchor_paths + base_mean/std，
        # 避免 load 后再重跑锚定集 _predict_fused_only（慢且引入数值差异）。
        if os.path.exists(gate_path):
            try:
                with open(gate_path, "r", encoding="utf-8") as f:
                    gs = json.load(f)
                gate = RegressGate(
                    tol=float(gs.get("tol", 0.05)),
                    anchor_size=int(gs.get("anchor_size", 20)),
                )
                gate.anchor_paths = list(gs.get("anchor_paths", []) or [])
                gate.base_mean = float(gs["base_mean"]) if gs.get("base_mean") is not None else None
                gate.base_std = float(gs["base_std"]) if gs.get("base_std") is not None else None
                gate.history = list(gs.get("history", []) or [])
                pipe.handler.gate = gate
            except Exception as e:
                print(f"[persist] RegressGate 恢复失败，跳过: {e}", flush=True)
        # SlotWeightLearner 恢复（跳过方法名；"_height_flagged" 恢复为 set）
        if os.path.exists(wl_path):
            try:
                with open(wl_path, "r", encoding="utf-8") as f:
                    wl_state = json.load(f)
                wl = getattr(pipe.handler, "weight_learner", None)
                if wl is not None:
                    for attr, val in wl_state.items():
                        if attr in ("weights",):   # 方法名，禁止 setattr
                            continue
                        cur = getattr(wl, attr, None)
                        if callable(cur):          # 不覆盖任何方法
                            continue
                        try:
                            if attr == "_height_flagged" and isinstance(val, list):
                                val = set(val)
                            setattr(wl, attr, val)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[persist] weight_learner 恢复失败，跳过: {e}", flush=True)
        # 权重演化历史恢复（旧快照无此文件 → 保持 __init__ 的空列表）
        wh_path = os.path.join(ssocl_dir, "weight_history.json")
        if os.path.exists(wh_path):
            try:
                with open(wh_path, "r", encoding="utf-8") as f:
                    pipe.handler.weight_history = list(json.load(f) or [])
            except Exception as e:
                print(f"[persist] weight_history 恢复失败，跳过: {e}", flush=True)
        # 2026-08-30：重建锚定集内存特征（_anchor_items）——gate.check 经
        # _predict_fused_only 的锚定直算依赖它；装载时一次性 ~1s（20 张提
        # 特征），换在线期间每次门控 ~1.1s → ~0.1s（reflow 1.6s 超红线修复）。
        # 必须在兜底 gate2.setup（内部调 _predict_fused_only ×20）之前。
        try:
            aps = list(getattr(pipe.handler.gate, "anchor_paths", []) or [])
            if aps and not getattr(pipe, "_anchor_items", None):
                pipe._anchor_items = pipe._tile_and_extract(aps)
        except Exception as e:
            print(f"[persist] 锚定集特征重建失败（门控回落缓存路径）: {e}",
                  flush=True)
        # gate.base_mean==None（快照无 regress_gate.json 或 anchor_paths 空）的兜底：
        # 用 pipeline._train_normal_paths 抽样 anchor_size 个正常路径重算 base_mean/std。
        try:
            gate2 = getattr(pipe.handler, "gate", None)
            if gate2 is not None and (gate2.base_mean is None or gate2.base_std is None):
                tpaths = list(getattr(pipe, "_train_normal_paths", []) or [])
                if tpaths:
                    rng2 = np.random.default_rng(0)
                    asz = int(getattr(gate2, "anchor_size", 20) or 20)
                    sel = tpaths if len(tpaths) <= asz else list(rng2.choice(tpaths, asz, replace=False))
                    gate2.setup(pipe, sel)
        except Exception as e:
            print(f"[persist] RegressGate 兜底 setup 失败: {e}", flush=True)

    # 清运行时缓存
    pipe._item_cache.clear()
    return pipe
