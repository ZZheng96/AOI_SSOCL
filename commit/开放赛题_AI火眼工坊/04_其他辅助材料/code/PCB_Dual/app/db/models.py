"""数据库 ORM 模型：覆盖原始图像/视频、伪异常、检测、反馈、模型、评估、统计、操作日志、后台任务。"""
from __future__ import annotations

from datetime import datetime, date

from sqlalchemy import (JSON, Boolean, Date, DateTime, Float, ForeignKey,
                        Integer, String, Text)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


# ── 数据集批次（M8a 数据资产治理）─────────────────────────────
class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    dataset_name: Mapped[str] = mapped_column(String(64), default="")  # 数据集组名（如 mvtec / MPDD）
    category: Mapped[str] = mapped_column(String(64), default="default", index=True)
    source_type: Mapped[str] = mapped_column(String(32), default="folder")  # folder/mvtec/upload/pseudo/augment/video/feedback/legacy
    source_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)   # 导入/生成参数
    n_images: Mapped[int] = mapped_column(Integer, default=0)
    note: Mapped[str] = mapped_column(String(256), default="")
    # W-source：批次归属的数据源（数据按数据源管理）
    datasource_id: Mapped[int | None] = mapped_column(
        ForeignKey("data_sources.id"), nullable=True, index=True)


# ── 数据源（W-source：独立登记的数据实体，可被多个工单复用）──
class DataSource(Base, TimestampMixin):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    modality: Mapped[str] = mapped_column(String(16), default="image")   # image/video
    # 采样配置（视频源：关键帧间隔/采样帧率等；图片源可为空）
    sampling: Mapped[dict] = mapped_column(JSON, default=dict)
    # 标注档位声明（L0/L1a/L1b，前端反馈 v4：条件下沉到数据源层；
    # 导入数据后按实际支撑体检自动校正）
    label_tier: Mapped[str] = mapped_column(String(8), default="L0")
    # 数据流类型（前端反馈 v5）：local 本地数据流 / rtsp 实时拉流
    source_type: Mapped[str] = mapped_column(String(16), default="local")
    # 数据流配置：rtsp -> {url}；video_sim -> {path}（本地视频循环模拟实时流）
    stream_config: Mapped[dict] = mapped_column(JSON, default=dict)
    # 前端反馈 v6：数据源级声明——数据已分品类（导入按目录自动识别）/ 提供模板图
    per_category: Mapped[bool] = mapped_column(Boolean, default=False)
    has_template: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str] = mapped_column(String(256), default="")
    # 预训练组配置（前端反馈 v9：数据源内分「预训练组」+「检测组」；
    # 预训练组按品类取前 N 正常 + M 异常，其余进检测组，检测组按 batch_size 分批）
    pretrain_normal: Mapped[int] = mapped_column(Integer, default=100)
    pretrain_anomaly: Mapped[int] = mapped_column(Integer, default=30)
    batch_size: Mapped[int] = mapped_column(Integer, default=30)   # 检测组每批图片数
    # 分组方案（另存的可复用分配：预训练组图片清单等；JSON）
    plan_json: Mapped[dict] = mapped_column(JSON, default=dict)


# ── 数据源分组方案（前端反馈 v9：另存整理好的数据集，下次导入可复用）──
class DataSourcePlan(Base, TimestampMixin):
    __tablename__ = "data_source_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    source_root: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # 方案内容：{pretrain_normal, pretrain_anomaly, batch_size,
    #           pretrain_ids:[image_id...] 或 rel_paths:[相对路径...],
    #           version}
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    note: Mapped[str] = mapped_column(String(256), default="")


# ── 工单（W-workorder：产线任务实例，含多个品类，条件在工单级声明）──
class WorkOrder(Base, TimestampMixin):
    __tablename__ = "work_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    # 数据条件三档（标注信息量递进）：L0 仅正常图 / L1a 图像级标注 / L1b 缺陷位置标注
    label_tier: Mapped[str] = mapped_column(String(8), default="L1a")
    # 两个可叠加开关：品类独立（数据已分品类）/ 模板比对（提供模板图）
    per_category: Mapped[bool] = mapped_column(Boolean, default=True)
    has_template: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="active")    # active/closed
    note: Mapped[str] = mapped_column(String(256), default="")
    # 人工复判开关（前端反馈 v4）：开启后该工单全部检测进待复核队列，
    # 复核后才计入统计与持续学习（复核提交自动生成 feedback 回流学习）。
    review_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # 产线状态（前端反馈 v5）：running 运行 / paused 暂停（学习提升/编排数据流）
    pipeline_status: Mapped[str] = mapped_column(String(16), default="running")
    # 学习完恢复方式（工单级配置）：True 自动恢复产线 / False 手动恢复
    auto_resume: Mapped[bool] = mapped_column(Boolean, default=False)
    # 数据流队列配置（前端反馈 v5）：{"requeued":[dataset_id...], "priority":[ds_id...]}
    queue_json: Mapped[dict] = mapped_column(JSON, default=dict)
    template_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_order_templates.id"), nullable=True)  # 从哪个任务模板创建
    # v2：关联 PCB_Ins 检测模板 ID（InspectionTemplate，字符串），工单检测可选绑定
    template_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # ── v2 板级产线补字段（借鉴 Java AOI work_order）────────────────
    order_code: Mapped[str | None] = mapped_column(String(64), nullable=True)   # 产线工单号
    customer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    track_index: Mapped[int | None] = mapped_column(Integer, nullable=True)     # 轨道号 1/2
    order_state: Mapped[str] = mapped_column(String(16), default="open")        # open/running/closed
    fixture_width: Mapped[float | None] = mapped_column(Float, nullable=True)   # 治具尺寸/偏移
    fixture_height: Mapped[float | None] = mapped_column(Float, nullable=True)
    fixture_offset_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    fixture_offset_y: Mapped[float | None] = mapped_column(Float, nullable=True)


class WorkOrderSource(Base, TimestampMixin):
    """工单-数据源关联（多对多）：工单可挂多个数据源。"""
    __tablename__ = "work_order_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workorder_id: Mapped[int] = mapped_column(
        ForeignKey("work_orders.id", ondelete="CASCADE"), index=True)
    datasource_id: Mapped[int] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"), index=True)


# ── 任务模板（W-template：工单配置存档，新建工单可导入）──
class WorkOrderTemplate(Base, TimestampMixin):
    __tablename__ = "work_order_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)   # {label_tier, per_category, has_template}
    note: Mapped[str] = mapped_column(String(256), default="")


# ── 原始数据 ─────────────────────────────────────────────────
class Image(Base, TimestampMixin):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(512), unique=True)
    category: Mapped[str] = mapped_column(String(64), default="default", index=True)
    split: Mapped[str] = mapped_column(String(32), default="unlabeled", index=True)  # train/test/feedback/pseudo/unlabeled
    label: Mapped[str] = mapped_column(String(16), default="unknown", index=True)   # normal/anomaly/unknown
    defect_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="manual")  # mvtec/dagm/upload/pseudo/video
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sha1: Mapped[str | None] = mapped_column(String(40), nullable=True)
    dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id"), nullable=True, index=True)  # M8a 所属批次


class Video(Base, TimestampMixin):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(512), unique=True)
    category: Mapped[str] = mapped_column(String(64), default="default")
    label: Mapped[str] = mapped_column(String(16), default="unknown")
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id"), nullable=True)  # M8a 该视频抽帧所属批次


class VideoFrame(Base):
    __tablename__ = "video_frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), index=True)
    frame_idx: Mapped[int] = mapped_column(Integer)
    timestamp_ms: Mapped[float] = mapped_column(Float, default=0.0)
    image_id: Mapped[int | None] = mapped_column(ForeignKey("images.id"), nullable=True)


# ── 伪异常数据（赛题问题二）──────────────────────────────────
class PseudoAnomaly(Base, TimestampMixin):
    __tablename__ = "pseudo_anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    base_image_id: Mapped[int] = mapped_column(ForeignKey("images.id"))
    method: Mapped[str] = mapped_column(String(32))          # cutpaste/scar/noise/color_shift
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    image_path: Mapped[str] = mapped_column(String(512))
    mask_path: Mapped[str | None] = mapped_column(String(512), nullable=True)


# ── 检测记录（赛题问题一）────────────────────────────────────
class Detection(Base, TimestampMixin):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_type: Mapped[str] = mapped_column(String(16), default="image")  # image/video_frame/upload/pipeline/stream/plc
    workorder_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_orders.id"), nullable=True, index=True)  # 产线消费的工单归属（2026-08-29 精确口径）
    image_id: Mapped[int | None] = mapped_column(ForeignKey("images.id"), nullable=True)
    video_id: Mapped[int | None] = mapped_column(ForeignKey("videos.id"), nullable=True)
    frame_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_path: Mapped[str] = mapped_column(String(512))     # 冗余路径，便于展示
    category: Mapped[str] = mapped_column(String(64), default="default")
    model_id: Mapped[int | None] = mapped_column(ForeignKey("models.id"), nullable=True)
    final_score: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean)
    level1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    level2_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    level3_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    triggered_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    n_tiles: Mapped[dict] = mapped_column(JSON, default=dict)  # {"l1":36,"l2":5,"l3":2}
    heatmap_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # ── v2 双检扩展（PCB_Ins 融合）────────────────────────────
    engine_mode: Mapped[str] = mapped_column(String(16), default="dual")  # traditional/feature/dual
    template_id: Mapped[str | None] = mapped_column(String(64), nullable=True)   # PCB_Ins 模板 ID
    template_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    traditional_overall: Mapped[str | None] = mapped_column(String(16), nullable=True)  # OK/NG/ERROR
    feature_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)     # normal/gray/anomaly
    dual_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # DualSummary 全量摘要
    # ── v2 板级/坐标扩展（P0/P2 补全，借鉴 Java AOI inspect/defect）────
    inspect_id: Mapped[int | None] = mapped_column(
        ForeignKey("inspects.id"), nullable=True, index=True)  # 所属板级检测
    position_on: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 位号
    fov_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    board_x: Mapped[float | None] = mapped_column(Float, nullable=True)  # 物理坐标（mm）
    board_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    align_offset_x: Mapped[float | None] = mapped_column(Float, nullable=True)  # 对位偏移（px）
    align_offset_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    align_warn: Mapped[bool] = mapped_column(Boolean, default=False)  # 对位偏移超阈值
    synced: Mapped[bool] = mapped_column(Boolean, default=False, index=True)  # 已上报 MES/上位机


# ── 板级检测（P0：借鉴 Java AOI inspect，一板一条，detections 为框/块级明细）──
class Inspect(Base, TimestampMixin):
    __tablename__ = "inspects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workorder_id: Mapped[int | None] = mapped_column(
        ForeignKey("work_orders.id"), nullable=True, index=True)
    template_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    template_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    number: Mapped[str | None] = mapped_column(String(64), nullable=True)       # 二维码/板号
    board_barcode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    track_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_count: Mapped[int] = mapped_column(Integer, default=0)   # 检测器件/图数
    ng_count: Mapped[int] = mapped_column(Integer, default=0)
    pass_: Mapped[bool] = mapped_column(Boolean, default=True)     # 整板判定
    inspect_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)  # 检测结果目录
    inspect_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finish_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    repair_state: Mapped[str] = mapped_column(String(16), default="UNREPAIRED")  # UNREPAIRED/REPAIRED
    repair_user: Mapped[str | None] = mapped_column(String(64), nullable=True)
    material_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pack: Mapped[str | None] = mapped_column(String(64), nullable=True)
    synced: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    dual_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # 汇总摘要


# ── 缺陷明细（P0：借鉴 Java AOI defect，逐缺陷记录，支撑复判/SPC/追溯）──
class DefectRecord(Base, TimestampMixin):
    __tablename__ = "defect_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    detection_id: Mapped[int] = mapped_column(
        ForeignKey("detections.id"), index=True)
    inspect_id: Mapped[int | None] = mapped_column(
        ForeignKey("inspects.id"), nullable=True, index=True)
    defect_type: Mapped[str] = mapped_column(String(64))         # 多锡(包锡)/AOI·sem/起皱/缺件...
    engine: Mapped[str] = mapped_column(String(16), default="traditional")  # traditional/feature
    algorithm_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    algorithm_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    x: Mapped[float | None] = mapped_column(Float, nullable=True)   # 图像坐标（px）
    y: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[float | None] = mapped_column(Float, nullable=True)
    height: Mapped[float | None] = mapped_column(Float, nullable=True)
    board_x: Mapped[float | None] = mapped_column(Float, nullable=True)   # 物理坐标（mm）
    board_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_on: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 位号
    fov_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    material_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pack: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)  # 缺陷小图
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rejudged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)  # 已复判
    final_error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 复判后最终类型
    synced: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


# ── 不良过滤规则（P2：借鉴 Java AOI ng_filter，全局库+模板绑定）──
class NgFilter(Base, TimestampMixin):
    __tablename__ = "ng_filters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    defect_type: Mapped[str] = mapped_column(String(64), index=True)   # 命中哪类缺陷
    name: Mapped[str] = mapped_column(String(128), default="")
    filter_content: Mapped[str] = mapped_column(String(64), default="area")  # 过滤内容：area/width/height/score...
    filter_condition: Mapped[str] = mapped_column(String(16), default="LT")  # GT/LT/EQ/BETWEEN/CONTAINS
    value_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    value_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str] = mapped_column(String(16), default="px")
    rule_json: Mapped[dict] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    template_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 绑定模板；空=全局


# ── 告警记录（P2：借鉴 Java AOI alarm_record）──────────────
class AlarmRecord(Base, TimestampMixin):
    __tablename__ = "alarm_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alarm_type: Mapped[str] = mapped_column(String(32), index=True)  # detection_ng/task_failed/system
    level: Mapped[str] = mapped_column(String(8), default="WARN")    # INFO/WARN/ERROR
    source: Mapped[str] = mapped_column(String(64), default="")      # workorder/task/detect...
    content: Mapped[str] = mapped_column(Text, default="")
    ref_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 关联检测/任务/板 ID
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    process_status: Mapped[str] = mapped_column(String(16), default="pending")  # pending/processed/ignored
    remark: Mapped[str | None] = mapped_column(String(256), nullable=True)


# ── 操作员反馈（赛题问题三）──────────────────────────────────
class Feedback(Base, TimestampMixin):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), index=True)
    feedback_type: Mapped[str] = mapped_column(String(32))   # false_positive/false_negative/confirmed/new_defect
    operator_label: Mapped[int] = mapped_column(Integer)     # 0=正常 1=异常
    defect_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    region: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # 漏检框选区域 [x,y,w,h]
    operator: Mapped[str] = mapped_column(String(64), default="operator")
    consumed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)  # 是否已用于增量更新
    calibration_excluded: Mapped[bool] = mapped_column(Boolean, default=False)  # 误检反馈是否被移出校准集
    invalidated: Mapped[bool] = mapped_column(Boolean, default=False, index=True)  # 已作废（标错撤回）：不计入统计/曲线/复核已判
    # ── v2 复判扩展（借鉴 Java AOI rejudgment）───────────────────
    final_state: Mapped[str | None] = mapped_column(String(16), nullable=True)  # NG/OK（复判终判）
    rejudge_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    device_key: Mapped[str | None] = mapped_column(String(64), nullable=True)   # 复判终端标识
    device_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    device_url: Mapped[str | None] = mapped_column(String(256), nullable=True)


# ── 模型注册表 ───────────────────────────────────────────────
class Model(Base, TimestampMixin):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    level: Mapped[str] = mapped_column(String(16), default="cascade")  # 1/2/3/cascade
    category: Mapped[str] = mapped_column(String(64), default="default")
    version: Mapped[str] = mapped_column(String(32), default="v1")
    path: Mapped[str] = mapped_column(String(512))
    format: Mapped[str] = mapped_column(String(16), default="pt")      # pkl/pt/onnx
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    origin: Mapped[str] = mapped_column(String(32), default="demo5_fit")   # demo5_fit/self_learned_demo5/uploaded
    parent_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


# ── 评估运行 ─────────────────────────────────────────────────
class EvalRun(Base, TimestampMixin):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    run_type: Mapped[str] = mapped_column(String(32), default="accuracy")  # accuracy/benchmark
    category: Mapped[str] = mapped_column(String(64), default="default")
    dataset_desc: Mapped[str] = mapped_column(String(256), default="")
    n_images: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)  # auroc/ap/f1/threshold...
    latency: Mapped[dict] = mapped_column(JSON, default=dict)  # mean/p50/p95/max ms


# ── 统计聚合 ─────────────────────────────────────────────────
class StatsDaily(Base):
    __tablename__ = "stats_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, unique=True, index=True)
    n_inspected: Mapped[int] = mapped_column(Integer, default=0)
    n_anomaly: Mapped[int] = mapped_column(Integer, default=0)
    n_feedback: Mapped[int] = mapped_column(Integer, default=0)
    n_false_positive: Mapped[int] = mapped_column(Integer, default=0)
    n_false_negative: Mapped[int] = mapped_column(Integer, default=0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)


# ── 操作日志 ─────────────────────────────────────────────────
class OperationLog(Base, TimestampMixin):
    __tablename__ = "operation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user: Mapped[str] = mapped_column(String(64), default="operator")
    action: Mapped[str] = mapped_column(String(64), index=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # M8a 结构化字段


# ── 后台任务 ─────────────────────────────────────────────────
class Task(Base, TimestampMixin):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_type: Mapped[str] = mapped_column(String(32))       # detect_video/pseudo/eval/benchmark/self_update/import
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending/running/done/failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(String(512), default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
