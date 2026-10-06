"""API 层 Pydantic 请求模型与通用序列化辅助。"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ── 通用辅助 ─────────────────────────────────────────────────
def to_dict(obj: Any, extra: Optional[Dict] = None) -> Dict:
    """ORM 对象 → dict（遍历 __table__.columns，datetime/date 转 ISO 字符串）。"""
    d: Dict[str, Any] = {}
    for col in obj.__table__.columns:
        v = getattr(obj, col.name)
        if isinstance(v, (datetime, date)):
            v = v.isoformat()
        d[col.name] = v
    if extra:
        d.update(extra)
    return d


def device_info() -> Dict[str, Any]:
    """探测运行设备：{"device","gpu_name","torch_version"}（不触发模型加载）。"""
    try:
        import torch
        if torch.cuda.is_available():
            return {"device": "cuda",
                    "gpu_name": torch.cuda.get_device_name(0),
                    "torch_version": torch.__version__}
        return {"device": "cpu", "gpu_name": None,
                "torch_version": torch.__version__}
    except Exception:  # noqa: BLE001
        return {"device": "unknown", "gpu_name": None, "torch_version": None}


# ── 图片 / 视频导入 ─────────────────────────────────────────
class DataSourceCreateRequest(BaseModel):
    """W-source：新建数据源（声明模态 + 标注档位 + 采样配置 + 数据流）。"""
    name: str
    modality: str = "image"               # image/video
    label_tier: str = "L0"                # 标注档位声明（导入后体检校正）
    sampling: Optional[Dict] = None       # 视频：{frame_interval, sample_fps}
    source_type: str = "local"            # local 本地数据流 / rtsp 实时拉流
    stream_config: Optional[Dict] = None  # rtsp:{url}；video_sim:{path}
    per_category: bool = False            # 声明数据已分品类（导入按目录自动识别）
    has_template: bool = False            # 声明提供模板图（A_tpl.png 命名识别）
    note: Optional[str] = None
    # 前端反馈 v9：预训练组/检测组配置与复用方案
    pretrain_normal: Optional[int] = None     # 预训练组正常图数（默认 100）
    pretrain_anomaly: Optional[int] = None    # 预训练组异常图数（默认 30）
    batch_size: Optional[int] = None          # 检测组每批图片数（默认 30）
    plan_id: Optional[int] = None             # 复用已保存分组方案（预训练组固定）

    class Config:
        extra = "ignore"


class DataSourceUpdateRequest(BaseModel):
    """W-source：编辑数据源（模态锁定，名称/档位/备注/采样/数据流可改）。"""
    name: Optional[str] = None
    label_tier: Optional[str] = None
    sampling: Optional[Dict] = None
    source_type: Optional[str] = None
    stream_config: Optional[Dict] = None
    per_category: Optional[bool] = None
    has_template: Optional[bool] = None
    note: Optional[str] = None
    pretrain_normal: Optional[int] = None
    pretrain_anomaly: Optional[int] = None
    batch_size: Optional[int] = None
    plan_id: Optional[int] = None

    class Config:
        extra = "ignore"


class PlanSaveRequest(BaseModel):
    """另存数据源分组方案。"""
    name: str
    note: Optional[str] = None


class PlanImportRequest(BaseModel):
    """从 JSON 导入分组方案。"""
    name: Optional[str] = None
    config: Dict                     # {pretrain_normal, pretrain_anomaly, batch_size, pretrain_ids,...}
    note: Optional[str] = None


class WorkOrderCreateRequest(BaseModel):
    """W-workorder：新建工单（数据源必选；条件由数据源属性自动聚合，
    保留 label_tier/per_category/has_template 仅作兼容，后端不再使用）。"""
    name: str
    label_tier: str = "L1a"               # 兼容字段（已废弃，聚合取代）
    per_category: bool = True             # 兼容字段（已废弃）
    has_template: bool = False            # 兼容字段（已废弃）
    review_enabled: bool = False          # 人工复判：全部检测进待复核队列
    auto_resume: bool = False             # 学习完自动恢复产线（工单级配置）
    note: Optional[str] = None
    datasource_ids: Optional[List[int]] = None   # 新建时直接挂数据源（必选）
    from_template: Optional[int] = None   # 从任务模板导入配置


class WorkOrderUpdateRequest(BaseModel):
    """W-workorder：修改工单（全可选，只改传入字段；datasource_ids=None 不动挂接）。"""
    name: Optional[str] = None
    label_tier: Optional[str] = None
    per_category: Optional[bool] = None
    has_template: Optional[bool] = None
    review_enabled: Optional[bool] = None
    auto_resume: Optional[bool] = None
    status: Optional[str] = None
    note: Optional[str] = None
    datasource_ids: Optional[List[int]] = None


class WorkOrderSourceRequest(BaseModel):
    """W-workorder：工单挂接数据源（整体替换列表）。"""
    datasource_ids: List[int]


class QueueDatasetRequest(BaseModel):
    """W-workorder：数据流队列检测批次操作（回队/取消回队）。

    前端反馈：队列统一到「检测组 30 图/批」粒度，用 batch_key
    （"{datasource_id}:{category}:{batch_index}"）标识批次；
    dataset_ids 仅作旧契约兼容（导入批次粒度，已废弃）。
    """
    batch_keys: Optional[List[str]] = None
    dataset_ids: Optional[List[int]] = None


class WorkOrderTemplateRequest(BaseModel):
    """W-template：把工单配置存为任务模板（条件随源自动算，不存条件；
    存数据源挂接列表 + 复判开关 + 备注）。"""
    name: str
    datasource_ids: Optional[List[int]] = None
    review_enabled: bool = False
    note: Optional[str] = None


class ImageImportRequest(BaseModel):
    folder: str
    category: str = "default"
    split: str = "unlabeled"
    label: str = "unknown"
    source: str = "manual"
    copy_to_storage: bool = True
    name: Optional[str] = None            # M8a 批次名（默认 {source_type}_{时间戳}）
    note: Optional[str] = None            # M8a 批次备注
    dataset_name: Optional[str] = None    # U-opsflow 数据集组名（默认取文件夹名）
    datasource_id: Optional[int] = None   # W-source 归属数据源


class MvtecImportRequest(BaseModel):
    root: str
    categories: Optional[List[str]] = None
    name: Optional[str] = None            # M8a 批次名
    note: Optional[str] = None            # M8a 批次备注
    dataset_name: Optional[str] = None    # U-opsflow 数据集组名（默认取根目录名）
    datasource_id: Optional[int] = None   # W-source 归属数据源


class VideoImportRequest(BaseModel):
    path: str
    category: str = "default"
    label: str = "unknown"
    name: Optional[str] = None            # M8a 批次名
    note: Optional[str] = None            # M8a 批次备注
    datasource_id: Optional[int] = None   # W-source 归属数据源


# ── 数据集适配（识别 + 确认导入）─────────────────────────────
class DatasetProbeRequest(BaseModel):
    """只读扫描识别数据集内容（不写库），返回确认报告。"""
    root: str


class DatasetAdaptImportRequest(BaseModel):
    """按确认后的映射导入（后台任务）。

    category：单品类格式（yolo_split/dir_rules/plain）的品类名；
    ok_role：dir_rules 识别为正常且无显式 split 的图片归属 train|template；
    group_overrides：{分组名: anomaly|normal|skip}，作用于该分组未识别图片；
    categories：mvtec_like 要导入的品类名列表（None=全部有图品类）。
    """
    root: str
    format: str = "dir_rules"
    category: Optional[str] = None
    ok_role: str = "train"
    group_overrides: Optional[Dict[str, str]] = None
    categories: Optional[List[str]] = None
    dataset_name: Optional[str] = None
    name: Optional[str] = None
    note: Optional[str] = None
    datasource_id: Optional[int] = None


class ImageAnnotateRequest(BaseModel):
    """M8a 标注流转：改 label/split/defect_type（只改传入字段）。"""
    label: Optional[str] = None           # normal/anomaly/unknown
    split: Optional[str] = None           # train/test/val/feedback/unlabeled...
    defect_type: Optional[str] = None


class OrphanCleanRequest(BaseModel):
    """M8a 孤儿文件清理：仅允许删除 storage 目录内的文件。"""
    paths: List[str]


# ── 数据增强配置（v10.29：伪异常 / 预处理 / 训练增强）─────────
class PseudoPreviewRequest(BaseModel):
    image_id: int
    method: str = "defect_transplant"   # defect_transplant/cutpaste/color_blot
    params: Optional[Dict] = None


class AugmentConfigUpdateRequest(BaseModel):
    """数据增强配置写回（augment 节：pseudo/preprocess/enhance，§15.29）。"""
    pseudo: Optional[Dict] = None       # 伪异常合成（methods/transplant_scale/feather）
    preprocess: Optional[Dict] = None   # 图像预处理（enabled/gray/clahe/median）
    enhance: Optional[Dict] = None      # 训练增强（flip/rot90/brightness）


# ── 检测 / 监控 ─────────────────────────────────────────────
class DetectImageRequest(BaseModel):
    image_id: Optional[int] = None          # image_id 与 path 二选一
    path: Optional[str] = None
    category: str
    with_heatmap: bool = True               # False=高速模式（跳过热力图生成/存盘）


class DetectVideoRequest(BaseModel):
    video_id: int
    category: str
    interval: Optional[int] = None


class ABCompareRequest(BaseModel):
    """A/B 影子对比：version_a 为空时取当前激活版本，version_b 为候选快照版本号。"""
    category: str
    version_a: Optional[int] = None     # 空=当前激活版本
    version_b: int                      # 候选版本号
    split: str = "test"
    limit: int = 100


class CalibrationExcludeRequest(BaseModel):
    feedback_id: int
    excluded: bool = True


class ModelPrepareRequest(BaseModel):
    category: str
    force: bool = True                  # 已有激活版本时是否强制重备
    scenario: str = "L1a"               # L0/L1a/L1b/L2/L3（demo5 场景分层）
    profile: str = "fast"               # fast/accuracy/cpu（demo5 工作点）
    note: str = ""
    datasource_id: Optional[int] = None  # 前端反馈 v9：从数据源预训练组取样本


# ── 反馈 / 自学习 ───────────────────────────────────────────
class FeedbackCreateRequest(BaseModel):
    detection_id: int
    feedback_type: str                      # false_positive/false_negative/confirmed/new_defect/uncertain(M15a)
    operator_label: int                     # 0=正常 1=异常 -1=无法确认（uncertain）
    defect_type: Optional[str] = None
    comment: Optional[str] = None
    region: Optional[List[int]] = None      # 漏检框选区域 [x,y,w,h]
    operator: str = "operator"


class SelfUpdateRequest(BaseModel):
    category: str
    note: Optional[str] = None          # 巩固备注（写入快照 meta）
    # 以下字段为旧 EWC 语义遗留，demo5 巩固落盘不再使用
    epochs: Optional[int] = None
    auto_activate: Optional[bool] = None


# ── 模型 / 评估 ─────────────────────────────────────────────
class RollbackRequest(BaseModel):
    category: str


class EvalAccuracyRequest(BaseModel):
    category: str
    split: str = "test"
    name: Optional[str] = None
    limit: Optional[int] = None


class EvalBenchmarkRequest(BaseModel):
    category: str
    n_images: int = 30
    warmup: int = 5


class EvalCompareRequest(BaseModel):
    run_ids: List[int]


class EvalRobustnessRequest(BaseModel):
    """六维评价·维度2：噪声鲁棒性（光照/高斯/模糊扰动下 AUC 衰减 + A_rob）"""
    category: str
    split: str = "test"
    limit: int = 60
    noise_kinds: List[str] = ["brightness", "gauss", "blur"]
    levels: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8]


# ── 学习曲线 / 贡献档案（M2b）────────────────────────────────
class LearningCurveRequest(BaseModel):
    """学习曲线离线回放（独立 pipe 副本，与产线并行，不影响线上激活模型）。

    mode="rounds"（默认）：多轮持续学习——轮次轴锚定集 AUROC + 已反馈重测
    指标（重测准确率/错→对转化率/缺陷检出率）， algo simulate_learning_rounds。
    mode="stream"：旧版单轮流回放（流一次性、反馈锁死，仅调试用）。
    """
    category: str
    mode: str = "rounds"                # rounds/stream
    n_eval: int = 60                    # 锚定评估集上限（test 均衡抽样；
                                        # 2026-08-29 20→60：小样本 AUROC 易饱和恒 1）
    round_size: int = 25                # rounds：每轮新反馈条数
    n_rounds: int = 4                   # rounds：轮数
    eval_per_sample: int = 5            # rounds：每类锚定样本数（0=用 n_eval 全集）
    # 以下仅 mode="stream" 使用（旧契约保留）
    n_stream: int = 20                  # 回放流长度上限
    feedback_ratio: float = 1.0         # 反馈率
    eval_every: int = 5                 # 每几条反馈重估一次锚定集
    max_feedback: int = 20              # 反馈数上限


class ContributionRequest(BaseModel):
    """槽位贡献档案（单变量/冗余度/消融三层评价）。"""
    category: str
