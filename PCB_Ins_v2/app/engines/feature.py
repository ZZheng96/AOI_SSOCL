"""特征引擎（AOI_sys demo5 引擎迁入）：按品类管理 pipeline 实例。

接口（面向双检闭环）：
- prepare(category, bundle, scenario, profile) -> version   品类准备（fit+快照）
- predict_image_path / predict_ndarray -> DetectionResult   推理
- submit_feedback(category, path, verdict, label, box)       反馈即学
- consolidate / activate / rollback / list_snapshots        模型版本管理
- get_engine() 进程单例

来源：AOI_sys backend/engine/__init__.py，去掉 DB 依赖（learning_insight /
active_suggestions 由 API 层按需实现），其余逻辑保持一致。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


# ---- 结果契约 ----
@dataclass
class DetectionResult:
    """特征引擎推理结果。"""
    score: float = 0.0                      # fused（含拦截加分）
    decision: str = "normal"                # normal/gray/anomaly
    is_anomaly: bool = False
    threshold: float = 0.0                  # tau_high
    gray_threshold: float = 0.0             # tau_gray
    slot_scores: Dict[str, float] = field(default_factory=dict)
    raw_scores: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    router_w: Dict[str, float] = field(default_factory=dict)
    defect_boxes: List[Dict[str, Any]] = field(default_factory=list)
    mask: Optional[Any] = None
    heatmap_paths: Dict[str, str] = field(default_factory=dict)
    types: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    n_tiles: int = 0
    triggered_slot: str = ""
    open_alert: bool = False
    align_offset: Optional[List[float]] = None
    align_warn: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


# ---- 场景分层 → 槽位配置改写（与 AOI_sys 一致） ----
SCENARIO_SLOTS = {
    "L0": {"sem": True, "inp": True, "trad": True, "blob": True,
           "layout": True, "disc": False, "shead": False, "open": True, "tpl": False},
    "L1a": {"sem": True, "disc": True, "shead": True, "blob": True,
            "trad": True, "layout": True, "inp": True, "open": True, "tpl": False},
    "L1b": {"sem": True, "disc": True, "shead": True, "blob": True,
            "trad": True, "layout": True, "inp": True, "open": True, "tpl": False},
    "L2": {"sem": True, "disc": True, "shead": True, "blob": True,
           "trad": True, "layout": True, "inp": True, "open": True, "tpl": False},
    "L3": {"sem": True, "disc": True, "shead": True, "blob": True,
           "trad": True, "layout": True, "inp": True, "open": True, "tpl": True},
}

PROFILE_SLOTS = {
    "fast": ("sem", "disc", "shead"),
    "accuracy": None,
    "cpu": ("blob", "trad", "layout"),
}

SSOCL_CFG_TEMPLATE = {
    "banks": {"core_max": 2048, "ext_max": 8192, "cluster_k": 8,
              "sim_thresh": 0.85, "defect_ratio_thresh": 0.60},
    "gate": {"tol": 0.05},
    "active": {"top_ratio": 0.25, "max_queue": 200},
    "incubate": {"min_samples": 8},
    "router_finetune": {"trigger": 10, "epochs": 30, "lr": 1e-3, "margin": 0.05},
    "weight_learner": {"enabled": True, "alpha": 0.05, "balance": 0.5},
    "height_suppress": False,
    "height_recal": True,
    "head_ft": {"min_pos": 9999, "min_neg": 9999},
    "disc_ft": {"min_pos": 9999, "min_neg": 9999},
    "thresh_recal": {"min_normals": 9999},
}


def apply_scenario_and_profile(cfg: Dict[str, Any], scenario: str, profile: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(cfg))
    scenario = scenario or "L1a"
    profile = profile or "fast"
    scenario_map = SCENARIO_SLOTS[scenario]
    for slot_name, scn_enabled in scenario_map.items():
        if slot_name not in cfg["slots"]:
            continue
        cfg["slots"][slot_name]["enabled"] = bool(scn_enabled)
    if PROFILE_SLOTS[profile] is not None:
        keep = set(PROFILE_SLOTS[profile])
        for slot_name in list(cfg["slots"].keys()):
            if slot_name == "tpl" and profile == "fast":
                continue
            if slot_name == "open":
                continue
            cfg["slots"][slot_name]["enabled"] = cfg["slots"][slot_name].get("enabled", False) and (slot_name in keep)
    if profile == "cpu":
        cfg["backbone"]["enabled"] = False
    return cfg


class FeatureEngine:
    """按品类管理 pipeline（进程单例，线程锁）。"""

    def __init__(self, storage_dir: str, base_cfg_path: str, ssocl_cfg: Optional[Dict] = None):
        self.storage_dir = os.path.abspath(storage_dir)
        self.snap_root = os.path.join(self.storage_dir, "snapshots")
        os.makedirs(self.snap_root, exist_ok=True)
        self.base_cfg_path = os.path.abspath(base_cfg_path)
        self.ssocl_cfg = ssocl_cfg or SSOCL_CFG_TEMPLATE
        self.base_cfg: Dict[str, Any] = {}
        self._load_base_cfg()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipes: Dict[str, Tuple[Any, int]] = {}
        self._write_lock = threading.RLock()

    def _load_base_cfg(self) -> None:
        with open(self.base_cfg_path, "r", encoding="utf-8") as f:
            self.base_cfg = yaml.safe_load(f)
        _cfg_dir = os.path.dirname(self.base_cfg_path)
        for _slot in ("disc", "shead"):
            _if = (self.base_cfg.get("slots", {}).get(_slot) or {}).get("init_from")
            if _if and not os.path.isabs(_if):
                _abs = os.path.join(_cfg_dir, _if)
                self.base_cfg["slots"][_slot]["init_from"] = os.path.normpath(_abs)

    # ----- 目录辅助 -----
    def _cat_dir(self, category: str) -> str:
        return os.path.join(self.snap_root, category)

    def _v_dir(self, category: str, version: int) -> str:
        return os.path.join(self._cat_dir(category), f"v{int(version)}")

    def list_snapshots(self, category: str) -> List[Dict[str, Any]]:
        from algo.persist import list_versions, current_version
        items = []
        for v in list_versions(self._cat_dir(category)):
            meta_path = os.path.join(self._v_dir(category, v), "meta.json")
            d = {"version": v, "active": v == current_version(self._cat_dir(category))}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        d.update(json.load(f))
                except Exception:
                    pass
            items.append(d)
        return items

    def current_version(self, category: str) -> Optional[int]:
        from algo.persist import current_version
        return current_version(self._cat_dir(category))

    # ----- 准备新品类 -----
    def prepare(self, category: str, bundle: Dict[str, Any], *,
                scenario: str = "L1a", profile: str = "fast",
                trigger: str = "prepare", note: str = "",
                force: bool = False) -> int:
        """品类准备：fit + 存快照 + 激活。bundle 见 algo.factory.build / Pipeline.fit。"""
        from algo.factory import build
        from algo.eval.offline import Pipeline
        from algo.persist import save_snapshot, list_versions, set_current_version

        with self._write_lock:
            cat_dir = self._cat_dir(category)
            os.makedirs(cat_dir, exist_ok=True)
            if not force:
                cv = self.current_version(category)
                if cv is not None and os.path.isdir(self._v_dir(category, cv)):
                    self._ensure_loaded(category)
                    return cv
            cfg = apply_scenario_and_profile(self.base_cfg, scenario, profile)
            backbone, slots = build(cfg, self.device)
            pipe = Pipeline(cfg, backbone, slots)
            pipe.fit(bundle)
            rng = np.random.default_rng(int(cfg.get("seed", 42)))
            pipe.enable_ssocl(self.ssocl_cfg,
                              train_normal_paths=bundle.get("init_normal", []),
                              rng=rng)
            vs = list_versions(cat_dir)
            new_v = max(vs) + 1 if vs else 1
            snap_dir = self._v_dir(category, new_v)
            if os.path.exists(snap_dir):
                shutil.rmtree(snap_dir)
            save_snapshot(pipe, snap_dir, version=new_v,
                          parent_version=None, trigger=trigger,
                          note=f"scenario={scenario}, profile={profile}. {note}")
            set_current_version(cat_dir, new_v)
            self._pipes[category] = (pipe, new_v)
            return new_v

    # ----- 装载 -----
    def _ensure_loaded(self, category: str) -> None:
        from algo.persist import load_snapshot, current_version
        with self._write_lock:
            if category in self._pipes:
                return
            cv = current_version(self._cat_dir(category))
            if cv is None:
                raise RuntimeError(f"品类 {category} 尚未准备模型。")
            snap = self._v_dir(category, cv)
            pipe = load_snapshot(snap, self.device)
            self._pipes[category] = (pipe, cv)

    def get_pipeline(self, category: str):
        self._ensure_loaded(category)
        return self._pipes[category][0]

    # ----- 激活 / 回滚 -----
    def activate(self, category: str, version: int) -> None:
        from algo.persist import load_snapshot, set_current_version, list_versions
        with self._write_lock:
            if version not in list_versions(self._cat_dir(category)):
                raise ValueError(f"版本 v{version} 不存在")
            pipe = load_snapshot(self._v_dir(category, version), self.device)
            self._pipes[category] = (pipe, version)
            set_current_version(self._cat_dir(category), version)

    def rollback(self, category: str, target_version: Optional[int] = None) -> int:
        vs = self.current_version(category)
        if not vs:
            raise RuntimeError(f"品类 {category} 无快照可回滚")
        cv = self.current_version(category) or vs[-1]
        idx = vs.index(cv) if cv in vs else len(vs) - 1
        target = target_version if target_version is not None else vs[max(0, idx - 1)]
        self.activate(category, target)
        return target

    # ----- 推理 -----
    def predict_image_path(self, category: str, path: str) -> DetectionResult:
        self._ensure_loaded(category)
        pipe, _ = self._pipes[category]
        t0 = time.perf_counter()
        rec = pipe.predict(path)
        dt_ms = (time.perf_counter() - t0) * 1000
        return self._to_result(pipe, rec, dt_ms)

    def predict_ndarray(self, category: str, rgb: np.ndarray, path_hint: Optional[str] = None) -> DetectionResult:
        self._ensure_loaded(category)
        pipe, _ = self._pipes[category]
        t0 = time.perf_counter()
        rec = pipe.predict_frame(rgb, path=path_hint)
        dt_ms = (time.perf_counter() - t0) * 1000
        return self._to_result(pipe, rec, dt_ms)

    def _to_result(self, pipe, rec: Dict[str, Any], dt_ms: float) -> DetectionResult:
        slot_scores = dict(rec.get("slot_scores", {}))
        weights = dict(getattr(pipe, "weights", {}) or {})
        decision = rec.get("decision", "normal")
        boxes = [{"bbox": list(b.get("bbox", [0, 0, 0, 0])),
                  "area": int(b.get("area", 0)),
                  "dominant_slot": self._dominant_slot(slot_scores, weights)}
                 for b in (rec.get("boxes", []) or [])]
        return DetectionResult(
            score=float(rec.get("fused", 0.0)),
            decision=decision,
            is_anomaly=(decision == "anomaly"),
            threshold=float(pipe.decider.tau_high) if pipe.decider else 0.99,
            gray_threshold=float(pipe.decider.tau_gray) if pipe.decider else 0.95,
            slot_scores=slot_scores,
            raw_scores=dict(rec.get("raw_scores", {})),
            weights=weights,
            router_w=dict(rec.get("router_w", {}) or {}),
            defect_boxes=boxes,
            mask=rec.get("mask"),
            types=list(rec.get("types", []) or []),
            latency_ms=float(dt_ms),
            n_tiles=int(rec.get("n_tiles", 0)),
            triggered_slot=self._dominant_slot(slot_scores, weights),
            open_alert=bool(rec.get("open_alert", False)),
        )

    @staticmethod
    def _dominant_slot(slot_scores: Dict[str, float], weights: Dict[str, float]) -> str:
        best = ("", -1e9)
        for n, s in slot_scores.items():
            w = weights.get(n, 0.0)
            val = float(s) * max(0.0, float(w))
            if val > best[1]:
                best = (n, val)
        return best[0]

    # ----- 反馈即学 -----
    def submit_feedback(self, category: str, image_path: str,
                        verdict: str, label: Optional[int] = None,
                        box: Optional[List[float]] = None) -> Dict[str, Any]:
        self._ensure_loaded(category)
        pipe, _ = self._pipes[category]
        t0 = time.perf_counter()
        entry = pipe.feedback(image_path, verdict=verdict, label=label, box=box)
        dt_ms = (time.perf_counter() - t0) * 1000
        info = {"path": image_path, "verdict": verdict, "label": label,
                "feedback_ms": round(dt_ms, 2)}
        if isinstance(entry, dict):
            for k in ("action", "drift", "boost", "defect_ratio"):
                if k in entry:
                    info[k] = entry[k]
        return info

    # ----- 巩固（落盘新版） -----
    def consolidate(self, category: str, *, note: str = "") -> int:
        from algo.persist import save_snapshot, set_current_version, list_versions
        with self._write_lock:
            self._ensure_loaded(category)
            pipe, active = self._pipes[category]
            cat_dir = self._cat_dir(category)
            vs = list_versions(cat_dir)
            new_v = max(vs) + 1 if vs else 1
            snap = self._v_dir(category, new_v)
            save_snapshot(pipe, snap, version=new_v,
                          parent_version=active, trigger="consolidate", note=note)
            set_current_version(cat_dir, new_v)
            self._pipes[category] = (pipe, new_v)
            return new_v


# ---- 进程级单例 ----
_engine: Optional[FeatureEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> FeatureEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from app.config import get_settings
                settings = get_settings()
                _engine = FeatureEngine(str(settings.snapshots_dir.parent),
                                        str(settings.feature_base_cfg))
    return _engine
