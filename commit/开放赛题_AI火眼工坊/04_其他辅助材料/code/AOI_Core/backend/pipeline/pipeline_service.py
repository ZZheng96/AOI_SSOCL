"""产线服务（前端反馈 v5）：工单产线自动消费数据流检测。

- 本地数据流：running 工单按队列（错检回队批次优先）逐张「未检测」图片
  送检（detect_image persist 落库），interval 节流；暂停（paused）跳过消费。
- 实时数据流：rtsp 源（stream_config.url RTSP 地址 / path 本地视频文件循环
  模拟）用 cv2.VideoCapture 拉流逐帧送检，品类取 stream_config.category。
- 暂停/恢复不重启不重配：状态保存在工单 pipeline_status，服务轮询读取。
- 复判背压（2026-08-30 前端反馈 v6-3）：开启人工复判（review_enabled）的
  工单，待复核积压 ≥ pipeline.review_backlog_high（默认 30）时自动暂停送检，
  积压消化到 ≤ pipeline.review_backlog_low（默认 10）时自动恢复。
  防止产线跑太快质检员跟不上、复核队列无限积压。
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class PipelineService:
    """产线后台服务（单例，daemon 线程）。"""

    _instance: "PipelineService | None" = None
    _lock = threading.Lock()

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = False
        self._thread: threading.Thread | None = None
        self._rtsp: dict[int, threading.Thread] = {}   # datasource_id -> thread
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

    # ── 主循环 ────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop:
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001
                log.warning("[pipeline] tick 异常：%s", e)
            time.sleep(self.interval)

    def _tick(self) -> None:
        self._apply_review_backpressure()
        self._sync_rtsp()
        self._consume_one()

    # ── 复判背压（前端反馈 v6-3）──────────────────────
    def _apply_review_backpressure(self) -> None:
        """review_enabled 工单的待复核积压高水位暂停 / 低水位恢复。

        只影响 review_enabled=True 的工单（这些工单的全部检测都进复核队列，
        质检员跟不上时队列会无限膨胀）；普通工单不受影响。
        """
        from ..core.config import get_settings
        from ..db.database import log_action, session_scope
        from ..db.models import Detection, Feedback, WorkOrder

        cfg = get_settings()
        high = int(cfg.get("pipeline", "review_backlog_high", 30))
        low = int(cfg.get("pipeline", "review_backlog_low", 10))
        if high <= low:
            return  # 配置非法时不启用，避免抖动

        with session_scope() as s:
            # 已有有效反馈的 detection_id（复核队列出队口径）
            fb_ids = {r[0] for r in s.query(Feedback.detection_id)
                      .filter(Feedback.invalidated == False).all()}  # noqa: E712
            wos = (s.query(WorkOrder)
                   .filter(WorkOrder.review_enabled.is_(True)).all())
            for wo in wos:
                # 该工单名下待复核检测数（产线路径 + 无反馈）
                rows = (s.query(Detection)
                        .filter(Detection.workorder_id == wo.id,
                                Detection.target_type.in_(
                                    ("pipeline", "stream", "plc")))
                        .all())
                n_pending = sum(1 for d in rows if d.id not in fb_ids)
                if (wo.pipeline_status == "running"
                        and n_pending >= high):
                    # 背压自动暂停用独立状态 auto_paused，与人工暂停 paused
                    # 区分：自动恢复只能解除自己设置的暂停，不能覆盖人工操作
                    # （前端反馈 v7-1「暂停了还被自动恢复」的根因）。
                    wo.pipeline_status = "auto_paused"
                    log_action(
                        "pipeline_backpressure_pause",
                        f"wo={wo.id} pending={n_pending} >= high={high}",
                        extra={"workorder_id": wo.id, "pending": n_pending})
                    log.info("[pipeline] 复判积压 %d ≥ %d，工单 %d 自动暂停",
                             n_pending, high, wo.id)
                elif (wo.pipeline_status == "auto_paused"
                        and n_pending <= low):
                    wo.pipeline_status = "running"
                    log_action(
                        "pipeline_backpressure_resume",
                        f"wo={wo.id} pending={n_pending} <= low={low}",
                        extra={"workorder_id": wo.id, "pending": n_pending})
                    log.info("[pipeline] 复判积压 %d ≤ %d，工单 %d 自动恢复",
                             n_pending, low, wo.id)

    # ── 本地数据流消费 ────────────────────────────────
    def _consume_one(self) -> None:
        from datetime import datetime as dt

        from ..db.database import session_scope
        from ..db.models import (Dataset, Detection, Image as ImageRow,
                                 WorkOrder, WorkOrderSource)
        from .service import get_detection_service

        with session_scope() as s:
            wos = (s.query(WorkOrder)
                   .filter(WorkOrder.pipeline_status == "running").all())
            job = None
            from ..engine import get_engine
            engine = get_engine()
            for wo in wos:
                qcfg = dict(wo.queue_json or {})
                requeued_ids = {str(k): list(v) for k, v in
                                (qcfg.get("requeued_ids") or {}).items()}
                requeued_at = {str(k): v for k, v in
                               (qcfg.get("requeued_at") or {}).items()}
                src_ids = [r[0] for r in s.query(WorkOrderSource.datasource_id)
                           .filter(WorkOrderSource.workorder_id == wo.id).all()]
                if not src_ids:
                    continue

                # 1) 回队检测批次优先：逐批找首张「待重检」图（允许插队）
                for bk in sorted(requeued_ids,
                                 key=lambda k: requeued_at.get(k, ""),
                                 reverse=True):
                    ids = requeued_ids[bk]
                    try:
                        t_req = dt.fromisoformat(requeued_at[bk])
                    except (KeyError, TypeError, ValueError):
                        t_req = None
                    picked = None
                    all_redone = True
                    for img in (s.query(ImageRow)
                                .filter(ImageRow.id.in_(ids))
                                .order_by(ImageRow.id.asc()).all()):
                        latest = (s.query(Detection.created_at)
                                  .filter(Detection.image_id == img.id)
                                  .order_by(Detection.id.desc()).first())
                        if latest is None or (t_req is not None
                                              and latest[0] < t_req):
                            picked = img
                            all_redone = False
                            break
                    if picked is None:
                        if all_redone:   # 回队批次已全部重检 → 自动出队
                            requeued_ids.pop(bk, None)
                            requeued_at.pop(bk, None)
                            qcfg["requeued_ids"] = requeued_ids
                            qcfg["requeued_at"] = requeued_at
                            qcfg["requeued"] = sorted(requeued_ids)
                            wo.queue_json = qcfg
                        continue
                    try:
                        engine.get_pipeline(picked.category)
                    except Exception:  # noqa: BLE001 未准备品类跳过
                        continue
                    job = (wo, picked)
                    break
                if job:
                    break

                # 2) 普通检测组：按导入批次找首张「未检测」图
                # （回队批次已有检测记录，天然不会被此处 ~det_exists 选中，
                #   与回队重检互不冲突）
                ds_q = (s.query(Dataset)
                        .filter(Dataset.datasource_id.in_(src_ids))
                        .order_by(Dataset.id.desc()).all())
                for ds in ds_q:
                    try:
                        engine.get_pipeline(ds.category)
                    except Exception:  # noqa: BLE001
                        # 品类未准备：整批次跳过（否则同一张图反复送检失败
                        # 无限重试，产线被卡死；用户准备该品类后自动恢复）
                        continue
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
            category = str(img.category or "default")
            path = img.path
            image_id = img.id
        try:
            svc = get_detection_service()
            # 产线检测关闭热力图落盘：单图耗时红线 ≤1s，逐帧热力图
            # 渲染/存盘是最大开销（0.5-2.1s），产线只看判定与分数
            # target_type="pipeline" + workorder_id：监控页实时流与复核
            # 队列按产线路径/工单归属精确订阅
            svc.detect_image(path, category, image_id=image_id, persist=True,
                             with_heatmap=False, target_type="pipeline",
                             workorder_id=_wo.id)
        except Exception as e:  # noqa: BLE001
            log.warning("[pipeline] 送检失败 %s (%s)：%s", path, category, e)

    # ── 实时流（rtsp / 视频模拟）──────────────────────
    def _sync_rtsp(self) -> None:
        from ..db.database import session_scope
        from ..db.models import DataSource, WorkOrder, WorkOrderSource

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
        from .service import get_detection_service

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
        svc = get_detection_service()
        try:
            while not self._stop and not self._rtsp_stop.get(src.id):
                ok, frame = cap.read()
                if not ok:
                    if path:  # 本地视频循环模拟实时流
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                fd, tmp = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                try:
                    cv2.imwrite(tmp, frame)
                    svc.detect_image(tmp, category, persist=True,
                                     with_heatmap=False, target_type="stream",
                                     workorder_id=workorder_id)
                except Exception as e:  # noqa: BLE001
                    log.warning("[pipeline] rtsp 送检异常 source=%s：%s", src.id, e)
                finally:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                time.sleep(interval)
        finally:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
