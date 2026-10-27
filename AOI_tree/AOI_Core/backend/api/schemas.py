"""API 层 Pydantic 请求模型与通用序列化辅助。"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ..core.config import get_settings


class FeatureSpaceRequest(BaseModel):
    category: str = Field(min_length=1, max_length=64)
    datasource_id: Optional[int] = Field(default=None, gt=0)
    dataset_id: Optional[int] = Field(default=None, gt=0)
    sample_limit: int = Field(default=8, ge=3, le=12)
    pseudo: Dict[str, Any] = Field(default_factory=dict)
    preprocess: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('category')
    @classmethod
    def concrete_category(cls, value):
        value = value.strip()
        if not value or value.lower() in {'all', '*', '全部', '所有品类'}:
            raise ValueError('必须指定具体品类')
        return value

    @field_validator('pseudo')
    @classmethod
    def valid_pseudo(cls, value):
        methods = value.get('methods', [])
        if not isinstance(methods, list) or any(m not in ('defect_transplant', 'cutpaste', 'color_blot') for m in methods):
            raise ValueError('不支持的伪异常方式')
        if 'feather' in value and not isinstance(value['feather'], bool):
            raise ValueError('feather必须为布尔值')
        return value

    @field_validator('preprocess')
    @classmethod
    def valid_preprocess(cls, value):
        if not 0 <= float(value.get('clahe') or 0) <= 8 or value.get('median', 0) not in (0, 3, 5):
            raise ValueError('预处理仅支持clahe 0~8、median 0/3/5')
        for key in ('enabled', 'gray'):
            if key in value and not isinstance(value[key], bool):
                raise ValueError(f'{key}必须为布尔值')
        return value


# ── 通用辅助 ─────────────────────────────────────────────────
def detection_interpretation(decision: Optional[str], is_anomaly: bool = False) -> Dict[str, Any]:
    status = {"normal": "PASS", "anomaly": "ANOMALY",
              "gray": "REVIEW", "error": "ERROR"}.get(decision)
    if status is None:
        status = "ANOMALY" if is_anomaly else "REVIEW"
    return {"final_status": status, "is_defect": status == "ANOMALY",
            "requires_review": status == "REVIEW",
            "is_system_error": status == "ERROR"}


def task_interpretation(status: Optional[str], result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """任务未产生检测结果时不伪造 PASS；失败/超时属于系统错误。"""
    result = result or {}
    if result.get("final_status") in {"PASS", "ANOMALY", "REVIEW", "ERROR"}:
        return {k: result.get(k) for k in
                ("final_status", "is_defect", "requires_review", "is_system_error")}
    if status in {"failed", "error", "timeout"}:
        return {"final_status": "ERROR", "is_defect": False,
                "requires_review": False, "is_system_error": True}
    return {"final_status": None, "is_defect": False,
            "requires_review": False, "is_system_error": False}


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
    if obj.__table__.name == "detections":
        nt = d.get("n_tiles")
        if not isinstance(nt, dict):
            nt = {}
        rv = nt.get("region_review")
        if isinstance(rv, dict):  # AC-063：与 /regions 同一派生视图（存储不变）
            from algo.decision.localization import annotate_review
            rv = annotate_review(rv)
        d.update({
            "open_alert": bool(nt.get("open_alert")),
            "e_open_score": nt.get("e_open_score"),
            "e_open_threshold": nt.get("e_open_threshold"),
            "open_alert_item": nt.get("open_alert_item"),
            "model_version": nt.get("model_version"),
            "threshold": nt.get("threshold"),
            "gray_threshold": nt.get("gray_threshold"),
            "defect_boxes": nt.get("boxes") or [],
            "defect_types": nt.get("types") or [],
            "decision": nt.get("decision"),
            "slots": nt.get("slots") or {},
            "weights": nt.get("weights") or {},
            "raw_scores": nt.get("raw") or {},
            "cal_scores": nt.get("cal") or nt.get("slots") or {},
            "actual_router": nt.get("actual_router") or nt.get("router_w") or {},
            "candidate_router": nt.get("candidate_router"),
            "feature_metadata": nt.get("feature_metadata"),
            "triggered_slot": nt.get("triggered_slot"),
            "box_evidence": nt.get("box_evidence") or [],
            "localization": nt.get("localization") or {"status": "legacy_unknown"},
            "region_review": rv,
            "decision_trace": nt.get("decision_trace") or [],
            "detection_id": d.get("id"),
            "persisted": True,
            "feedback_supported": True,
        })
        d.update(detection_interpretation(nt.get("decision"),
                                          bool(d.get("is_anomaly"))))
    return d


# ── 推理算力探测（顶栏「算力」徽标）────────────────────────────
# 赛题指定硬件：RTX 2060（30 SM / 1920 CUDA 核心 / boost 1.68GHz）。
# 本机算力占该基准的百分比 = 同口径估算峰值之比，用于一眼判断"这台机器够不够"。
_BASELINE = {"name": "RTX 2060", "sm": 30, "cores_per_sm": 64, "clock_ghz": 1.68}
_PROFILE_CACHE: Dict[str, Any] | None = None


def _gpu_max_clock_ghz() -> float | None:
    """读 GPU 最大 SM 主频（nvidia-smi，MHz→GHz）；读不到返回 None。

    只在首次探测时调用（结果被 compute_profile 缓存），不会随 /api/health
    轮询反复拉起子进程。
    """
    try:
        import shutil
        import subprocess
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        r = subprocess.run(
            [exe, "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        mhz = float((r.stdout or "").strip().splitlines()[0].strip())
        return mhz / 1000.0 if mhz > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _short_gpu_name(raw: str) -> str:
    """去掉厂商前缀，得到专业简洁的型号名（NVIDIA GeForce GTX 1660 SUPER → GTX 1660 SUPER）。"""
    s = str(raw or "").strip()
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "GeForce "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s or "GPU"


def compute_profile() -> Dict[str, Any]:
    """探测推理算力（首次调用后缓存；不触发模型加载）。

    输出**专业可核对的口径**，不使用「算力 高/中/低」这类含义模糊的分级：
    - gpu_short：简洁型号名（如 `GTX 1660 SUPER`）；
    - tops_int8：估算 INT8 吞吐（SM × 每 SM FP32 核心 × 2 × 主频 × 2 打包）；
    - baseline_ratio_pct：相对赛题基准 `RTX 2060` 的同口径估算算力占比；
    - tflops_fp32 / vram_gb / compute_capability / sm_count / clock_ghz：硬件事实或估算值。
    所有吞吐/占比均为**估算值**，非厂商标称值（tooltip 会如实标注）。
    """
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE

    out: Dict[str, Any] = {
        "device": "unknown", "gpu_short": None, "tops_int8": None,
        "baseline_name": _BASELINE["name"], "baseline_ratio_pct": None,
        "tflops_fp32": None, "vram_gb": None,
        "compute_capability": None, "sm_count": None, "clock_ghz": None,
    }
    try:
        import torch
        if not torch.cuda.is_available():
            out["device"] = "cpu"
            _PROFILE_CACHE = out
            return out
        props = torch.cuda.get_device_properties(0)
        major = int(getattr(props, "major", 0) or 0)
        minor = int(getattr(props, "minor", 0) or 0)
        sm = int(getattr(props, "multi_processor_count", 0) or 0)
        vram = getattr(props, "total_memory", None)
        # 每 SM 的 FP32 核心数：Volta/Turing 及更早为 64，Ampere/Ada 为 128
        cores_per_sm = 64 if major < 8 else 128
        clock = _gpu_max_clock_ghz()
        if not clock:
            clock = float(getattr(props, "clock_rate", 0) or 0) / 1e6  # kHz → GHz
        clock_from_smi = bool(clock)
        if not clock or clock <= 0:
            clock = 1.5  # 读不到主频时的保守默认值（仅用于估算）
        base_flops = sm * cores_per_sm * 2 * clock / 1000.0 if sm else None
        base_ref = (_BASELINE["sm"] * _BASELINE["cores_per_sm"] * 2
                    * _BASELINE["clock_ghz"] / 1000.0)
        out.update({
            "device": "cuda",
            "gpu_short": _short_gpu_name(torch.cuda.get_device_name(0)),
            "tops_int8": round(base_flops * 2, 1) if base_flops else None,
            "baseline_ratio_pct": (round(base_flops / base_ref * 100)
                                   if base_flops else None),
            # 耗时系数：基准硬件耗时 × 该系数 ≈ 本机耗时（>1 表示本机比基准慢）
            "latency_factor": (round(base_ref / base_flops, 2)
                               if base_flops else None),
            "tflops_fp32": round(base_flops, 1) if base_flops else None,
            "vram_gb": round(vram / (1024 ** 3), 1) if vram else None,
            "compute_capability": f"{major}.{minor}" if major else None,
            "sm_count": sm or None,
            "clock_ghz": round(clock, 2),
            "clock_source": "nvidia-smi" if clock_from_smi else "默认估算",
        })
        _PROFILE_CACHE = out
        return out
    except Exception:  # noqa: BLE001
        _PROFILE_CACHE = out
        return out


def device_info() -> Dict[str, Any]:
    """探测运行设备：{"device","gpu_name","torch_version", + 算力字段}
    （不触发模型加载）。"""
    try:
        import torch
        if torch.cuda.is_available():
            info = {"device": "cuda",
                    "gpu_name": torch.cuda.get_device_name(0),
                    "torch_version": torch.__version__}
        else:
            info = {"device": "cpu", "gpu_name": None,
                    "torch_version": torch.__version__}
    except Exception:  # noqa: BLE001
        info = {"device": "unknown", "gpu_name": None, "torch_version": None}
    profile = compute_profile()
    info.update({k: v for k, v in profile.items() if k != "device"})
    if info["device"] == "unknown" and profile["device"] != "unknown":
        info["device"] = profile["device"]
    return info


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


class PreprocessPreviewRequest(BaseModel):
    image_id: int
    preprocess: Dict[str, Any]


class AugmentConfigUpdateRequest(BaseModel):
    """数据增强配置写回（augment 节：pseudo/preprocess/enhance，§15.29）。"""
    pseudo: Optional[Dict] = None       # 伪异常合成（methods/transplant_scale/feather）
    preprocess: Optional[Dict] = None   # 图像预处理（enabled/gray/clahe/median）
    enhance: Optional[Dict] = None      # 训练增强（flip/rot90/brightness）


class AugmentAblationRequest(AugmentConfigUpdateRequest):
    category: str = Field(min_length=1, max_length=64)
    factor: str
    split: str = 'test'
    train_limit: int = Field(default=8, ge=2, le=100)
    defect_limit: int = Field(default=4, ge=0, le=30)
    eval_limit: int = Field(default=12, ge=2, le=100)

    @field_validator('category')
    @classmethod
    def concrete_category(cls, value):
        if not value.strip() or value.strip() == '*':
            raise ValueError('请选择具体品类')
        return value.strip()

    @field_validator('split')
    @classmethod
    def heldout_split(cls, value):
        if value not in ('test', 'holdout'):
            raise ValueError('只允许 test/holdout 留出集')
        return value

    @field_validator('factor')
    @classmethod
    def supported_factor(cls, value):
        if value not in ('pseudo.enabled', 'preprocess.enabled', 'preprocess.gray',
                         'preprocess.clahe', 'preprocess.median', 'enhance.flip',
                         'enhance.rot90', 'enhance.brightness'):
            raise ValueError('未知消融因素')
        return value


# ── 检测 / 监控 ─────────────────────────────────────────────
class DetectImageRequest(BaseModel):
    image_id: Optional[int] = None          # image_id 与 path 二选一
    path: Optional[str] = None
    category: str
    with_heatmap: bool = True               # False=高速模式（跳过热力图生成/存盘）
    pair_template: Optional[str] = None     # 仅使用显式指定的配对模板路径


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
    scenario: str = "L1a"               # L0/L1a/L1b/L2/L3（场景分层）
    profile: str = "fast"               # fast/accuracy/cpu（工作点）
    note: str = ""
    datasource_id: Optional[int] = None  # 前端反馈 v9：从数据源预训练组取样本
    allow_experimental: bool = False    # 显式实验拟合：不进入生产激活
    # AC-015 表示重建：按当前代码/注册表重提特征、重建库、重校准，生成未激活新版本，
    # 须经激活门控（G2/锚定 AUROC）后才上线；旧版本保留可回滚
    rebuild_representation: bool = False


# ── 反馈 / 自学习 ───────────────────────────────────────────
class FeedbackCreateRequest(BaseModel):
    detection_id: int
    feedback_type: str                      # false_positive/false_negative/confirmed/new_defect/uncertain(M15a)
    operator_label: int                     # 0=正常 1=异常 -1=无法确认（uncertain）
    defect_type: Optional[str] = None
    comment: Optional[str] = None
    region: Optional[List[int]] = None      # 漏检框选区域 [x,y,w,h]
    operator: str = "operator"
    source_event_id: Optional[str] = None


class SelfUpdateRequest(BaseModel):
    category: str
    note: Optional[str] = None          # 巩固备注（写入快照 meta）
    # 以下字段为旧 EWC 语义遗留，引擎巩固落盘不再使用
    epochs: Optional[int] = None
    auto_activate: Optional[bool] = None  # False=仅生成候选版本，不切换激活指针


# ── 模型 / 评估 ─────────────────────────────────────────────
class RollbackRequest(BaseModel):
    category: str
    allow_stale: bool = False   # AC-015/073：保留兼容；HTTP 生产入口传 true 返回 409 STALE_OVERRIDE_FORBIDDEN


class EvalAccuracyRequest(BaseModel):
    category: str
    version: Optional[int] = None  # None=调用时解析当前激活版本并固定本次评估
    split: str = "test"
    name: Optional[str] = None
    limit: Optional[int] = None


class EvalBenchmarkRequest(BaseModel):
    category: str
    n_images: int = Field(default_factory=lambda: int(get_settings().get(
        "evaluation", "benchmark_n_images", 30)), gt=0)
    warmup: int = Field(default_factory=lambda: int(get_settings().get(
        "evaluation", "benchmark_warmup", 5)), ge=0)


class EvalCompareRequest(BaseModel):
    run_ids: List[int]


class EvalRobustnessRequest(BaseModel):
    """六维评价·维度2：噪声鲁棒性（光照/高斯/模糊扰动下 AUC 衰减 + A_rob）"""
    category: str
    version: Optional[int] = None  # None=调用时解析当前激活版本并固定本次评估
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
