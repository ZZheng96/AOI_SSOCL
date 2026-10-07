"""FeatureClient：通过 HTTP 调用树干 AOI_Core 的特征学习能力。

树枝唯一的特征引擎实现（get_engine 单例），
树枝进程不 import 树干代码、不加载 torch。格式差异在本层抹平：

- 框：Core ``[x,y,w,h]`` ↔ 树枝 ``{"bbox":[x0,y0,x1,y1], ...}``
- 版本：Core ``"vN"`` ↔ 树枝 ``int``
- 反馈：树枝 (path, verdict, label) → Core ``detection_id + feedback_type``；
  detection_id 由最近一次检测按 path 缓存，也随结果放在 ``extra``
"""
from __future__ import annotations

import hashlib
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.engines.feature import DetectionResult


class CoreError(RuntimeError):
    """AOI_Core 调用失败（带 HTTP 状态码）。"""

    def __init__(self, msg: str, status: int = 0):
        super().__init__(msg)
        self.status = status


def _ver_num(v: Any) -> Optional[int]:
    if v is None:
        return None
    s = str(v).lstrip("vV")
    return int(s) if s.isdigit() else None


class FeatureClient:
    _DET_CACHE_MAX = 2000

    def __init__(self, base_url: str, timeout: float = 30.0):
        import requests

        self.base = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._http = requests.Session()
        self._ready = False
        self._ready_lock = threading.Lock()
        self._det_ids: "OrderedDict[str, int]" = OrderedDict()
        self._det_lock = threading.Lock()

    # ----- HTTP 基础 -----
    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._ready_lock:
            if self._ready:
                return
            from app.core.aoi_core_launcher import core_healthy, ensure_core_running
            if not core_healthy(self.base):
                ok, msg = ensure_core_running()
                if not ok:
                    raise CoreError(f"AOI_Core 不可用：{msg}")
            self._ready = True

    def _req(self, method: str, path: str, *, timeout: Optional[float] = None, **kw) -> Any:
        import requests

        self._ensure_ready()
        try:
            r = self._http.request(method, f"{self.base}/api{path}",
                                   timeout=timeout or self.timeout, **kw)
        except requests.RequestException as e:
            self._ready = False  # 下次调用重新探活/拉起
            raise CoreError(f"AOI_Core 连接失败：{e}") from e
        if not r.ok:
            try:
                detail = (r.json() or {}).get("detail", r.text)
            except ValueError:
                detail = r.text
            raise CoreError(f"AOI_Core {method} {path} → {r.status_code}：{detail}", r.status_code)
        return r.json() if r.content else {}

    def _wait_task(self, task_id: int, timeout_s: float = 1800.0) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            t = self._req("GET", f"/tasks/{task_id}")
            st = t.get("status")
            if st == "done":
                return t.get("result") or {}
            if st in ("failed", "cancelled"):
                raise CoreError(f"AOI_Core 任务 {task_id} {st}：{t.get('message') or t.get('error')}")
            time.sleep(0.5)
        raise CoreError(f"AOI_Core 任务 {task_id} 超时")

    # ----- 推理 -----
    def _remember(self, path: Optional[str], det_id: Optional[int]) -> None:
        if not path or det_id is None:
            return
        with self._det_lock:
            self._det_ids[str(path)] = int(det_id)
            self._det_ids.move_to_end(str(path))
            while len(self._det_ids) > self._DET_CACHE_MAX:
                self._det_ids.popitem(last=False)

    def _to_result(self, d: Dict[str, Any]) -> DetectionResult:
        trig = d.get("triggered_slot") or ""
        boxes = []
        for b in d.get("defect_boxes") or []:
            if len(b) < 4:
                continue
            x, y, w, h = (int(v) for v in b[:4])
            boxes.append({"bbox": [x, y, x + w, y + h], "area": w * h, "dominant_slot": trig})
        n_tiles = d.get("n_tiles")
        extra = {"detection_id": d.get("detection_id"),
                 "feedback_supported": d.get("feedback_supported"),
                 "backend": "aoi_core"}
        if not isinstance(n_tiles, int):
            extra["n_tiles"] = n_tiles
            n_tiles = 0
        heat = {k: d[f"{k}_path"] for k in ("heatmap", "overlay") if d.get(f"{k}_path")}
        decision = d.get("decision") or ("anomaly" if d.get("is_anomaly") else "normal")
        return DetectionResult(
            score=float(d.get("final_score") or 0.0),
            decision=decision,
            is_anomaly=bool(d.get("is_anomaly", decision == "anomaly")),
            threshold=float(d.get("threshold") or 0.0),
            gray_threshold=float(d.get("gray_threshold") or 0.0),
            slot_scores=dict(d.get("slots") or {}),
            raw_scores=dict(d.get("raw_scores") or {}),
            weights=dict(d.get("weights") or {}),
            router_w=dict(d.get("router_w") or {}),
            defect_boxes=boxes,
            heatmap_paths=heat,
            types=list(d.get("defect_types") or []),
            latency_ms=float(d.get("latency_ms") or 0.0),
            n_tiles=n_tiles,
            triggered_slot=trig,
            open_alert=bool(d.get("open_alert", False)),
            align_offset=d.get("align_offset"),
            align_warn=bool(d.get("align_warn", False)),
            extra=extra,
        )

    def predict_image_path(self, category: str, path: str) -> DetectionResult:
        p = str(Path(path).resolve())
        d = self._req("POST", "/detect/image",
                      json={"path": p, "category": category, "with_heatmap": True})
        self._remember(str(path), d.get("detection_id"))
        self._remember(p, d.get("detection_id"))
        return self._to_result(d)

    def predict_ndarray(self, category: str, rgb: np.ndarray,
                        path_hint: Optional[str] = None) -> DetectionResult:
        import cv2

        img = rgb
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise CoreError("图像编码失败")
        name = Path(path_hint).name if path_hint else "frame.png"
        d = self._req("POST", "/detect/upload",
                      files={"file": (name, buf.tobytes(), "image/png")},
                      data={"category": category})
        self._remember(path_hint, d.get("detection_id"))
        return self._to_result(d)

    # ----- 反馈 -----
    def detection_id_for(self, image_path: str) -> Optional[int]:
        with self._det_lock:
            return (self._det_ids.get(str(image_path))
                    or self._det_ids.get(str(Path(image_path).resolve())))

    def submit_feedback(self, category: str, image_path: str,
                        verdict: str, label: Optional[int] = None,
                        box: Optional[List[float]] = None,
                        detection_id: Optional[int] = None,
                        defect_type: Optional[str] = None,
                        comment: Optional[str] = None,
                        operator: str = "PCB_Dual") -> Dict[str, Any]:
        """verdict: correct / wrong；label: 操作员认定 0=正常 1=异常。"""
        det_id = detection_id or self.detection_id_for(image_path)
        if det_id is None:
            # 本进程未检测过该图：先检测落库拿 detection_id
            det_id = self.predict_image_path(category, image_path).extra.get("detection_id")
        if det_id is None:
            raise CoreError("无 detection_id，Core 未落库该次检测，无法反馈")
        if verdict == "correct":
            fb_type = "confirmed"
        elif label == 1:
            fb_type = "false_negative"
        elif label == 0:
            fb_type = "false_positive"
        else:
            fb_type, label = "uncertain", -1
        if label is None:
            label = -1
        # Core 以 source_event_id 幂等：同一次检测重复点判定只学一次
        body: Dict[str, Any] = {"detection_id": int(det_id), "feedback_type": fb_type,
                                "operator_label": int(label), "operator": operator,
                                "source_event_id": f"pcb_dual:det{int(det_id)}"}
        if defect_type:
            body["defect_type"] = defect_type
        if comment:
            body["comment"] = comment
        if box is not None and len(box) >= 4:
            x0, y0, x1, y1 = (int(v) for v in box[:4])
            body["region"] = [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]
        out = self._req("POST", "/feedback", json=body)
        info = {"path": image_path, "verdict": verdict, "label": label,
                "detection_id": int(det_id), "feedback_type": fb_type,
                "feedback_id": out.get("id"), "note": out.get("note"),
                "learning_status": out.get("learning_status"),
                "duplicate": out.get("note") == "已存在相同来源事件"}
        eu = out.get("engine_update")
        if isinstance(eu, dict):
            for k in ("action", "drift", "boost", "defect_ratio", "feedback_ms"):
                if k in eu:
                    info[k] = eu[k]
        return info

    # ----- 模型版本 -----
    def list_categories(self) -> List[Dict[str, Any]]:
        return list(self._req("GET", "/categories") or [])

    def _models(self, category: str) -> List[Dict[str, Any]]:
        items = (self._req("GET", "/models") or {}).get("items", [])
        return [m for m in items if m.get("category") == category]

    def list_snapshots(self, category: str) -> List[Dict[str, Any]]:
        out = []
        for m in self._models(category):
            v = _ver_num(m.get("version"))
            if v is None:
                continue
            d = dict(m.get("metrics") or {})
            d.update(version=v, active=bool(m.get("is_active")), model_id=m.get("id"),
                     origin=m.get("origin"), parent_version=_ver_num(m.get("parent_id")))
            out.append(d)
        return sorted(out, key=lambda x: x["version"])

    def current_version(self, category: str) -> Optional[int]:
        for s in self.list_snapshots(category):
            if s["active"]:
                return s["version"]
        return None

    def activate(self, category: str, version: int) -> None:
        mid = next((s["model_id"] for s in self.list_snapshots(category)
                    if s["version"] == int(version)), None)
        if mid is None:
            raise ValueError(f"版本 v{version} 不存在")
        try:
            self._req("POST", f"/models/{mid}/activate", params={"gate": "true", "force": "false"},
                      timeout=max(self.timeout, 120))
        except CoreError as e:
            if e.status == 409:
                raise RuntimeError(f"激活门控未通过（v{version}）：{e}") from e
            raise

    def rollback(self, category: str, target_version: Optional[int] = None) -> int:
        if target_version is not None:
            self.activate(category, target_version)
            return int(target_version)
        out = self._req("POST", "/models/rollback", json={"category": category},
                        timeout=max(self.timeout, 120))
        v = _ver_num(out.get("active_version"))
        if v is None:
            raise RuntimeError(f"回滚失败：{out}")
        return v

    # ----- 品类准备 -----
    def _stage(self, category: str, paths: List[str], sub: str) -> Optional[Path]:
        """内容寻址暂存：同内容→同路径（Core 按 path 幂等，不重复入库），
        路径含 train 域标记以满足 guard_bundle 红线。"""
        if not paths:
            return None
        from app.config import get_settings
        root = get_settings().storage("core_stage") / category / sub
        root.mkdir(parents=True, exist_ok=True)
        for p in paths:
            src = Path(p)
            h = hashlib.sha1(src.read_bytes()).hexdigest()[:16]
            dst = root / f"{h}{src.suffix.lower() or '.png'}"
            if not dst.exists():
                shutil.copy2(src, dst)
        return root

    def _import(self, folder: Path, category: str, split: str, label: str) -> None:
        out = self._req("POST", "/images/import", json={
            "folder": str(folder), "category": category, "split": split, "label": label,
            "source": "PCB_Dual", "copy_to_storage": False,
            "dataset_name": f"PCB_Dual_{split}"})
        self._wait_task(out["task_id"])

    def prepare(self, category: str, bundle: Dict[str, Any], *,
                scenario: str = "L1a", profile: str = "fast",
                trigger: str = "prepare", note: str = "",
                force: bool = False) -> int:
        """登记样本到 Core（train/good、train_anomaly/defect）后调 /models/prepare。

        注意 Core 语义：prepare 取该品类 DB 中全部 train 正常图，历史已登记的样本也会参与。
        """
        if not force:
            cv = self.current_version(category)
            if cv is not None:
                return cv
        normals = [str(p) for p in bundle.get("init_normal") or []]
        defects = [str(p) for p in bundle.get("init_defect") or []]
        if len(normals) < 3:
            raise ValueError(f"正常图不足（{len(normals)}<3），AOI_Core 生产准备至少需要 3 张")
        nd = self._stage(category, normals, "train/good")
        self._import(nd, category, "train", "normal")
        dd = self._stage(category, defects, "train_anomaly/defect")
        if dd is not None:
            self._import(dd, category, "train_anomaly", "anomaly")
        out = self._req("POST", "/models/prepare", json={
            "category": category, "force": True, "scenario": scenario,
            "profile": profile, "note": f"[PCB_Dual:{trigger}] {note}".strip()})
        res = self._wait_task(out["task_id"])
        v = res.get("version_num") or _ver_num(res.get("version"))
        if v is None:
            raise RuntimeError(f"AOI_Core prepare 未返回版本：{res}")
        return int(v)

    # ----- 学习 -----
    def consolidate(self, category: str, *, note: str = "") -> int:
        out = self._req("POST", "/self_learning/update",
                        json={"category": category, "note": note})
        res = self._wait_task(out["task_id"])
        v = _ver_num((res or {}).get("version")) or _ver_num((res or {}).get("new_version"))
        return v if v is not None else (self.current_version(category) or 0)

    def learning_insight(self, category: str) -> Dict[str, Any]:
        return self._req("GET", "/self_learning/insight", params={"category": category})

