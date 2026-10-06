"""产线服务：running 工单自动消费数据流检测（双检落库）。

- 本地数据流：running 工单按队列（回队批次优先）逐张「未检测」图送检。
- 暂停（paused）跳过消费；恢复不重启不重配（状态在工单 pipeline_status）。
- 检测方式：工单绑定 PCB_Ins 模板（template_ref）→ 双检；否则单特征引擎。
- 实时流（rtsp / 视频模拟）：可选，逐帧送检。

AOI_sys PipelineService 精简版（去掉 DB 依赖耦合，改用 v2 模型）。
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class PipelineService:
    _instance: "PipelineService | None" = None
    _lock = threading.Lock()

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = False
        self._thread: threading.Thread | None = None
        self._rtsp: dict[int, threading.Thread] = {}
        self._rtsp_stop: dict[int, bool] = {}

    @classmethod
    def get(cls) -> "PipelineService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = PipelineService()
            return cls._instance

    # ── 生命周期 ──────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="pipeline-service")
        self._thread.start()
        log.info("[pipeline] 产线服务已启动")

    def stop(self) -> None:
        self._stop = True
        for dsid in list(self._rtsp.keys()):
            self._rtsp_stop[dsid] = True

    def _loop(self) -> None:
        while not self._stop:
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001
                log.warning("[pipeline] tick 异常：%s", e)
            time.sleep(self.interval)

    def _tick(self) -> None:
        self._sync_rtsp()
        self._consume_one()

    # ── 本地数据流消费 ────────────────────────────────
    def _consume_one(self) -> None:
        from datetime import datetime as dt

        from app.db.database import session_scope
        from app.db.models import (Dataset, Detection, Image as ImageRow,
                                   WorkOrder, WorkOrderSource)
        from app.inspect.detect_service import get_detection_service
        from app.template.store import TemplateStore

        with session_scope() as s:
            wos = (s.query(WorkOrder)
                   .filter(WorkOrder.pipeline_status == "running").all())
            job = None
            for wo in wos:
                qcfg = dict(wo.queue_json or {})
                requeued_ids = {str(k): list(v) for k, v in
                                (qcfg.get("requeued_ids") or {}).items()}
                src_ids = [r[0] for r in s.query(WorkOrderSource.datasource_id)
                           .filter(WorkOrderSource.workorder_id == wo.id).all()]
                if not src_ids:
                    continue

                # 1) 回队批次优先（允许插队）
                for bk in sorted(requeued_ids,
                                 key=lambda k: (qcfg.get("requeued_at") or {}).get(k, ""),
                                 reverse=True):
                    ids = requeued_ids[bk]
                    picked = None
                    all_redone = True
                    for img in (s.query(ImageRow)
                                .filter(ImageRow.id.in_(ids))
                                .order_by(ImageRow.id.asc()).all()):
                        latest = (s.query(Detection.created_at)
                                  .filter(Detection.image_id == img.id)
                                  .order_by(Detection.id.desc()).first())
                        if latest is None:
                            picked = img
                            all_redone = False
                            break
                    if picked is None:
                        if all_redone:  # 回队批次已全部重检 → 自动出队
                            requeued_ids.pop(bk, None)
                            qcfg["requeued_ids"] = requeued_ids
                            qcfg["requeued"] = sorted(requeued_ids)
                            wo.queue_json = qcfg
                        continue
                    job = (wo, picked)
                    break
                if job:
                    break

                # 2) 普通检测组：按批次找首张「未检测」图
                ds_q = (s.query(Dataset)
                        .filter(Dataset.datasource_id.in_(src_ids))
                        .order_by(Dataset.id.desc()).all())
                for ds in ds_q:
                    det_exists = (s.query(Detection.id)
                                  .filter(Detection.image_id == ImageRow.id)
                                  .exists())
                    img = (s.query(ImageRow)
                           .filter(ImageRow.dataset_id == ds.id, ~det_exists)
                           .order_by(ImageRow.id.asc()).first())
                    if img is None or img.id is None:
                        continue
                    job = (wo, img)
                    break
                if job:
                    break
            if job is None:
                return
            _wo, img = job
            image_path = img.path
            image_id = img.id
            template_ref = _wo.template_ref
            workorder_id = _wo.id
            category = str(img.category or "default")

        # 检测（锁外执行）
        try:
            svc = get_detection_service()
            if template_ref:
                tpl = TemplateStore().load(template_ref)
                if tpl is None:
                    log.warning("[pipeline] 工单 %s 模板 %s 不存在", workorder_id, template_ref)
                    return
                out = svc.detect_dual(tpl, image_path, image_id=image_id,
                                      workorder_id=workorder_id,
                                      target_type="pipeline", persist=True)
                log.info("[pipeline] 双检工单=%s img=%s overall=%s (%.0fms)",
                         workorder_id, Path(image_path).name,
                         out.get("overall"), out.get("elapsed_ms"))
            else:
                # 无模板 → 单特征引擎（按品类模型）
                from app.engines.feature import get_engine
                from app.inspect.fusion import feature_to_dict
                r = get_engine().predict_image_path(category, image_path)
                _persist_feature(category, image_path, feature_to_dict(r),
                                 workorder_id=workorder_id, image_id=image_id)
        except Exception as e:  # noqa: BLE001
            log.warning("[pipeline] 送检失败 %s：%s", image_path, e)

    # ── 实时流（rtsp / 视频模拟）──────────────────────
    def _sync_rtsp(self) -> None:
        from app.db.database import session_scope
        from app.db.models import DataSource, WorkOrder, WorkOrderSource

        with session_scope() as s:
            rows = (s.query(DataSource, WorkOrder.id)
                    .join(WorkOrderSource,
                          WorkOrderSource.datasource_id == DataSource.id)
                    .join(WorkOrder, WorkOrder.id == WorkOrderSource.workorder_id)
                    .filter(DataSource.source_type == "rtsp",
                            WorkOrder.pipeline_status == "running").all())
            active = {src.id for src, _wo_id in rows}
        for dsid in list(self._rtsp.keys()):
            if dsid not in active:
                self._rtsp_stop[dsid] = True
        for src, wo_id in rows:
            if src.id not in self._rtsp or not self._rtsp[src.id].is_alive():
                self._rtsp_stop[src.id] = False
                t = threading.Thread(target=self._rtsp_loop, args=(src, wo_id),
                                     daemon=True, name=f"rtsp-{src.id}")
                self._rtsp[src.id] = t
                t.start()

    def _rtsp_loop(self, src, workorder_id=None) -> None:
        import os
        import tempfile

        import cv2

        cfg = src.stream_config or {}
        url = str(cfg.get("url") or "")
        path = str(cfg.get("path") or "")
        category = str(cfg.get("category") or "default")
        try:
            cap = cv2.VideoCapture(path if path else url)
        except Exception as e:  # noqa: BLE001
            log.warning("[pipeline] rtsp 打开失败 source=%s：%s", src.id, e)
            return
        if not cap.isOpened():
            log.warning("[pipeline] rtsp 无法打开 source=%s", src.id)
            return
        try:
            while not self._stop and not self._rtsp_stop.get(src.id):
                ok, frame = cap.read()
                if not ok:
                    if path:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                fd, tmp = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                try:
                    cv2.imwrite(tmp, frame)
                    from app.db.database import session_scope as _ss
                    from app.db.models import WorkOrder
                    with _ss() as s2:
                        wo = s2.get(WorkOrder, workorder_id)
                        template_ref = wo.template_ref if wo else None
                    svc = get_detection_service()
                    if template_ref:
                        from app.template.store import TemplateStore
                        tpl = TemplateStore().load(template_ref)
                        if tpl is not None:
                            svc.detect_dual(tpl, tmp, workorder_id=workorder_id,
                                            target_type="stream", persist=True)
                            continue
                    from app.engines.feature import get_engine
                    from app.inspect.fusion import feature_to_dict
                    r = get_engine().predict_image_path(category, tmp)
                    _persist_feature(category, tmp, feature_to_dict(r),
                                     workorder_id=workorder_id)
                except Exception as e:  # noqa: BLE001
                    log.warning("[pipeline] rtsp 送检异常 source=%s：%s", src.id, e)
                finally:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                time.sleep(self.interval)
        finally:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass


def _persist_feature(category: str, image_path: str, feat: dict,
                     *, workorder_id=None, image_id=None) -> None:
    from app.db.database import session_scope
    from app.db.models import Detection, StatsDaily
    from datetime import date
    with session_scope() as s:
        row = Detection(
            target_type="pipeline", workorder_id=workorder_id,
            image_id=image_id,
            image_path=image_path, category=category,
            engine_mode="feature",
            feature_decision=feat.get("decision"),
            final_score=float(feat.get("score") or 0.0),
            is_anomaly=bool(feat.get("is_anomaly")),
            latency_ms=float(feat.get("latency_ms") or 0),
            n_tiles={"slot_scores": feat.get("slot_scores"),
                     "weights": feat.get("weights"),
                     "triggered_slot": feat.get("triggered_slot"),
                     "decision": feat.get("decision")},
            dual_json=feat,
        )
        s.add(row)
        s.flush()
        stats = s.query(StatsDaily).filter(StatsDaily.date == date.today()).first()
        if stats is None:
            stats = StatsDaily(date=date.today())
            s.add(stats)
            s.flush()
        stats.n_inspected += 1
        stats.n_anomaly += int(row.is_anomaly)
