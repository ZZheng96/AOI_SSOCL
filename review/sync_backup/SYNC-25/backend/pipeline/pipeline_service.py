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
import uuid
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


class PipelineService:
    """产线后台服务（单例，daemon 线程）。"""

    _instance: "PipelineService | None" = None
    _lock = threading.Lock()

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = False
        self._thread: threading.Thread | None = None
        # 一个 RTSP 源可被多个工单使用，线程必须按 (source, workorder) 隔离。
        self._rtsp: dict[tuple[int, int], threading.Thread] = {}
        self._rtsp_stop: dict[tuple[int, int], threading.Event] = {}

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
        for stop_event in list(self._rtsp_stop.values()):
            stop_event.set()

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
        from sqlalchemy import exists

        cfg = get_settings()
        high = int(cfg.get("pipeline", "review_backlog_high", 30))
        low = int(cfg.get("pipeline", "review_backlog_low", 10))
        if high <= low:
            return  # 配置非法时不启用，避免抖动

        with session_scope() as s:
            wos = (s.query(WorkOrder)
                   .filter(WorkOrder.review_enabled.is_(True)).all())
            for wo in wos:
                feedback_exists = exists().where(
                    Feedback.detection_id == Detection.id,
                    Feedback.invalidated.is_(False))
                n_pending = (s.query(Detection.id)
                              .filter(Detection.workorder_id == wo.id,
                                      Detection.target_type.in_(
                                          ("pipeline", "stream", "plc")),
                                      ~feedback_exists)
                              .count())
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
        from ..db.database import session_scope
        from ..db.models import (Dataset, Detection, Image as ImageRow,
                                 WorkOrder, WorkOrderSource)
        from .service import get_detection_service

        # 在数据库会话外执行检测，避免推理耗时期间长期占用事务
        with session_scope() as s:
            wos = (s.query(WorkOrder)
                   .filter(WorkOrder.pipeline_status == "running").all())
            job = None
            requeue_marker = None
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
                    picked = None
                    all_redone = True
                    for img in (s.query(ImageRow)
                                .filter(ImageRow.id.in_(ids))
                                .order_by(ImageRow.id.asc()).all()):
                        # 只认当前回队代次的持久化结果；回队前启动的 worker
                        # 即便在回队后提交，也不能满足本轮重检。
                        marker = requeued_at.get(bk)
                        done_q = (s.query(Detection.id)
                                  .filter(Detection.image_id == img.id,
                                          Detection.workorder_id == wo.id,
                                          Detection.target_type == "pipeline"))
                        # marker 缺失的旧回队记录仍需按 NULL 代次判重，
                        # 否则成功后会被同一回队批次无限重复消费。
                        done = done_q.filter(
                            Detection.claim_requeued_at == marker).first()
                        if done is not None:
                            continue
                        all_redone = False
                        if img.processing_status in ("claiming", "failed"):
                            continue
                        if img.processing_status in ("pending", "succeeded"):
                            picked = img
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
                    requeue_marker = requeued_at.get(bk)
                    break
                if job:
                    break

                # 2) 普通检测组：按导入批次找首张「尚未产生检测记录」图
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
                    det_for_wo = (s.query(Detection.id)
                                  .filter(Detection.image_id == ImageRow.id,
                                          Detection.workorder_id == wo.id,
                                          Detection.target_type == "pipeline")
                                  .exists())
                    # Image.processing_status 是跨工单全局字段；旧检测不能证明
                    # 当前 claiming 属于本工单，也不能安全恢复可能已死锁的 claim。
                    det_exists = det_for_wo
                    img = (s.query(ImageRow)
                           .filter(ImageRow.dataset_id == ds.id,
                                   ImageRow.processing_status.in_(
                                       ("pending", "succeeded")),
                                   ~det_exists)
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
            now = datetime.now()
            claim_token = uuid.uuid4().hex
            # 不重领超时 claiming：旧 worker 可能仍在推理，需人工核验后恢复。
            claimed = (s.query(ImageRow)
                       .filter(ImageRow.id == img.id)
                       .filter(ImageRow.processing_status.in_(
                           ("pending", "succeeded")))
                       .update({
                           ImageRow.processing_status: "claiming",
                           ImageRow.claim_token: claim_token,
                           ImageRow.claim_workorder_id: _wo.id,
                           ImageRow.claim_requeued_at: requeue_marker,
                           ImageRow.claimed_at: now,
                           ImageRow.attempt_count: ImageRow.attempt_count + 1,
                           ImageRow.last_error: None,
                       }, synchronize_session=False))
            if claimed != 1:
                return
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
                             workorder_id=_wo.id, claim_token=claim_token)
        except Exception as e:  # noqa: BLE001
            error = str(e)[:512]
            with session_scope() as s:
                s.query(ImageRow).filter(
                    ImageRow.id == image_id,
                    ImageRow.claim_token == claim_token,
                    ImageRow.claim_workorder_id == _wo.id).update({
                        ImageRow.processing_status: "failed",
                        ImageRow.last_error: error,
                    }, synchronize_session=False)
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
            active = {(src.id, wo_id) for src, wo_id in rows}

        # 先停止不再运行的 (source, workorder)，但不要立即删除记录；旧线程
        # 退出时只允许按 identity 清理，避免竞态下删掉新线程。
        for key, thread in list(self._rtsp.items()):
            if key not in active:
                stop_event = self._rtsp_stop.get(key)
                if stop_event is not None:
                    stop_event.set()
            if not thread.is_alive():
                if self._rtsp.get(key) is thread:
                    self._rtsp.pop(key, None)
                    self._rtsp_stop.pop(key, None)

        for src, wo_id in rows:
            key = (src.id, wo_id)
            thread = self._rtsp.get(key)
            if thread is not None and thread.is_alive():
                continue
            stop_event = threading.Event()
            t = threading.Thread(target=self._rtsp_loop,
                                 args=(src, wo_id, stop_event), daemon=True,
                                 name=f"rtsp-{src.id}-{wo_id}")
            self._rtsp_stop[key] = stop_event
            self._rtsp[key] = t
            t.start()

    def _rtsp_loop(self, src, workorder_id=None, stop_event=None) -> None:
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
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
            return
        if stop_event is None:
            stop_event = threading.Event()
        thread = threading.current_thread()
        try:
            svc = get_detection_service()
            while not self._stop and not stop_event.is_set():
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
                time.sleep(self.interval)
        finally:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
            key = (src.id, workorder_id)
            if self._rtsp.get(key) is thread:
                self._rtsp.pop(key, None)
                self._rtsp_stop.pop(key, None)
