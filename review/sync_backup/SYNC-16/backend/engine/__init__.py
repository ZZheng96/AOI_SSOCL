"""检测引擎适配层：把 algo.Pipeline 包装成系统级服务。

职责：
1. 引擎启动时按 storage/snapshots/{category}/current.json 激活快照；
   未激活/目录空时返回"未准备"友好态，不崩溃。
2. 暴露 prepare(category, bundle, scenario, profile) -> version 新品准备，
   内部用 algo.factory.build + Pipeline.fit + persist.save_snapshot + set_current。
3. predict_image/predict_frame 统一返回 DetectionResult（结果契约）。
4. feedback(detection_id, path, verdict, label, box) 即学（秒级生效），
   返回更新条目（是否触发了拦截/重校准/权重学习/门控回滚）。
5. consolidate(category) -> version："批量巩固"——累积反馈后的整管整理，
   落盘快照为新版本，供 A-B 对比 & 激活。
6. list_snapshots / activate / rollback / build_variant（A-B）模型管理。
7. evaluate / contribution_report / learning_curve 评估。
"""
from __future__ import annotations

import logging
import os
import contextlib
import shutil
import threading
import time
import json
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ---- 结果契约 ----
@dataclass
class DetectionResult:
    """与原 CascadeResult 同构的字段，附加 引擎专属字段。"""
    score: float = 0.0                      # fused（含拦截加分）
    decision: str = "normal"                # normal/gray/anomaly
    is_anomaly: bool = False                # 二分类判定（gray 也可当异常，UI 可选）
    threshold: float = 0.0                  # tau_high
    gray_threshold: float = 0.0             # tau_gray

    slot_scores: Dict[str, float] = field(default_factory=dict)   # CDF 校准分
    raw_scores: Dict[str, float] = field(default_factory=dict)    # 原始分
    weights: Dict[str, float] = field(default_factory=dict)       # 当前融合权重
    router_w: Dict[str, float] = field(default_factory=dict)      # 路由权重（可空）

    defect_boxes: List[Dict[str, Any]] = field(default_factory=list)  # [{bbox:[x0,y0,x1,y1], area, dominant_slot}]
    mask: Optional[Any] = None               # HxW uint8 热力图低分辨率掩码（内部用，外部渲染）
    fused_heatmap: Optional[Any] = None      # AC-063 工作网格连续融合热图（渲染用，不落库）
    heatmap_paths: Dict[str, str] = field(default_factory=dict)  # 渲染后落盘的路径
    types: List[Dict[str, Any]] = field(default_factory=list)     # 规则化缺陷类型归因

    latency_ms: float = 0.0                 # 本次推理全耗时
    n_tiles: int = 0                        # tile 数
    triggered_slot: str = ""                # 主导槽位（命中的最高权重高分槽）
    open_alert: bool = False                # 不完备哨兵
    align_offset: Optional[List[float]] = None   # M11d 对位偏移 (dx,dy)（参考分辨率像素）
    align_warn: bool = False                # M11d 偏移超阈值预警
    model_version: Optional[int] = None     # 本次推理实际使用的激活版本
    extra: Dict[str, Any] = field(default_factory=dict)           # 预留扩展


# ---- 场景分层 → 槽位配置改写 ----
SCENARIO_SLOTS = {
    "L0": {          # 仅正常图
        "sem": True, "inp": True, "trad": True, "blob": True,
        "layout": True, "disc": False, "shead": False,
        "open": True, "tpl": False,
    },
    "L1a": {         # 赛题标准：100+30 图像级标注
        "sem": True, "disc": True, "shead": True, "blob": True,
        "trad": True, "layout": True, "inp": True,
        "open": True, "tpl": False,
    },
    "L1b": {         # +定位标注（框级反馈，反馈路径消费，槽位启用同 L1a）
        "sem": True, "disc": True, "shead": True, "blob": True,
        "trad": True, "layout": True, "inp": True,
        "open": True, "tpl": False,
    },
    "L2": {          # +品类（系统天然按 category 隔离，槽位启用同 L1a）
        "sem": True, "disc": True, "shead": True, "blob": True,
        "trad": True, "layout": True, "inp": True,
        "open": True, "tpl": False,
    },
    "L3": {          # +模板（激活 tpl 槽位）
        "sem": True, "disc": True, "shead": True, "blob": True,
        "trad": True, "layout": True, "inp": True,
        "open": True, "tpl": True,
    },
}

PROFILE_SLOTS = {
    "fast": ("sem", "disc", "shead"),               # 速度工作点（默认）
    "accuracy": None,                                # 精度模式：全槽位启用
    "cpu": ("blob", "trad", "layout"),              # CPU 降配
}

# 与算法工程 scripts/m5_learning_curve.py 的 SSOCL_CFG 对齐（前端反馈 v7-4）。
# 此前模板与代码默认值偏离：head_ft/disc_ft 被 9999 阈值永久锁死、
# banks 缺 intercept_sim（代码默认 0.85，相对差分布仅 0.1~0.3 → 缺陷样例
# 拦截加分近乎失效）、cluster_k/sim_thresh 过严 → 在线学习两大分数级杠杆
# 与拦截通道形同虚设，是「AOI_sys 达不到 算法基线学习水平」的主因。
SSOCL_CFG_TEMPLATE = {
    # 候选路径整轮累计预算；旧 submit_feedback 不读取此节。
    "candidate_budget": {"max_events": 16, "max_loo": 8, "max_search": 16,
                         "max_steps": 10000, "max_patch_bytes": 64 * 1024 * 1024,
                         "max_input_pixels": 25_000_000},
    "banks": {"ext_max": 64, "cluster_k": 3, "sim_thresh": 0.75,
              "intercept_topk": 8, "intercept_boost": 0.25,
              "intercept_sim": 0.15},
    "gate": {"tol": 0.05, "anchor_size": 20},
    "active": {"top_n": 10},
    "incubate": {"min_samples": 8},
    "router_finetune": {"trigger": 6, "lr": 1e-4, "epochs": 5},
    "height_suppress": False,
    "height_recal": True,
    "fb_aug_k": 0,
    "weight_min_samples": 8,
    "head_ft": {"enabled": True, "min_pos": 20, "min_neg": 40,
                "lr": 1e-4, "epochs": 3, "anchor_n": 20, "anchor_lambda": 1.0,
                "opt": "adam", "tol_down": 0.35},
    "disc_ft": {"enabled": True, "min_pos": 60, "min_neg": 120,
                "lr": 1e-4, "epochs": 3, "batch": 128, "tol_down": 0.35},
    "thresh_recal": {"enabled": True, "min_n": 20, "max_fp_rate": 0.5,
                     "cooldown": 10},
}


def _weights_semantics() -> Dict[str, str]:
    from algo.fusion.cde_report import WEIGHTS_SEMANTICS
    return dict(WEIGHTS_SEMANTICS)


def _lineage_summary(bank) -> Optional[Dict[str, Any]]:
    """AC-028：bank 血缘摘要（按 op/source 计数、legacy 数、最近 3 条）。"""
    if bank is None or not hasattr(bank, "lineage"):
        return None
    lin = bank.lineage()
    items = lin.get("ext") if lin.get("kind") == "normal_bank" else lin.get("samples")
    items = items or []
    by_op: Dict[str, int] = {}
    by_src: Dict[str, int] = {}
    for m in items:
        by_op[str(m.get("op"))] = by_op.get(str(m.get("op")), 0) + 1
        by_src[str(m.get("source"))] = by_src.get(str(m.get("source")), 0) + 1
    out = {"kind": lin.get("kind"), "memory_not_weights": True,
           "version": lin.get("version"), "n": len(items),
           "n_legacy": sum(1 for m in items if m.get("legacy")),
           "by_op": by_op, "by_source": by_src,
           "recent": [{k: m.get(k) for k in ("op", "source", "version", "ts",
                                              "hash", "image_id")}
                      for m in items[-3:]]}
    if lin.get("kind") == "normal_bank":
        out["core"] = lin.get("core")
        out["pending_clusters"] = len(lin.get("pending_clusters") or {})
    else:
        out["n_pure_images"] = lin.get("n_pure_images")
    return out


def apply_scenario_and_profile(cfg: Dict[str, Any], scenario: str, profile: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(cfg))     # 深拷贝
    scenario = scenario or "L1a"
    profile = profile or "fast"
    if scenario not in SCENARIO_SLOTS:
        raise ValueError(f"未知场景层: {scenario}")
    if profile not in PROFILE_SLOTS:
        raise ValueError(f"未知工作点: {profile}")
    scenario_map = SCENARIO_SLOTS[scenario]
    for slot_name, scn_enabled in scenario_map.items():
        if slot_name not in cfg["slots"]:
            continue
        cfg["slots"][slot_name]["enabled"] = bool(scn_enabled)
    # profile 白名单：场景是"能不能用"，profile 是"开不开"——profile=fast 强制收敛到子集
    # 冲突时以 profile 为准（速度是赛题硬红线）。L0 场景用 accuracy profile 即留手工槽。
    if PROFILE_SLOTS[profile] is not None:
        keep = set(PROFILE_SLOTS[profile])
        for slot_name in list(cfg["slots"].keys()):
            if slot_name == "tpl" and profile == "fast":
                # tpl 对场景 L3 也保持（不是速度的主瓶颈，配准是前置操作）
                continue
            if slot_name == "open":
                # open 哨兵几乎零开销，常开
                continue
            cfg["slots"][slot_name]["enabled"] = cfg["slots"][slot_name].get("enabled", False) and (slot_name in keep)
    # CPU profile → backbone 禁用
    if profile == "cpu":
        cfg["backbone"]["enabled"] = False
        # 2026-10-11 修复：禁用主干后 DINO 特征为空，E_open（PCA on DINO patch
        # tokens）fit 必崩（ValueError: expected 4, got 0）——随主干一并关闭。
        if "open" in cfg["slots"]:
            cfg["slots"]["open"]["enabled"] = False
    return cfg


# ---- 引擎主类 ----
class DetectionEngine:
    """按品类管理 pipeline 实例（进程单例，线程锁）。"""
    # 学习/推理并行开关（2026-10-12）：None=按 self_learning.learning_parallel
    # 热读取（默认 False）。False 时后台学习在合作式检查点让行前台推理。
    # 类级默认值：兼容测试中绕过 __init__ 构造的引擎。
    learning_parallel: Optional[bool] = None
    learning_idle_ms = 300       # 最近一次推理结束后多久视为空闲
    _fg_cond = threading.Condition()
    _fg_active = 0
    _fg_last = 0.0

    def __init__(self, storage_dir: str, base_cfg_path: str, ssocl_cfg: Optional[Dict] = None,
                 max_cached_categories: int = 8):
        self.storage_dir = os.path.abspath(storage_dir)
        self.snap_root = os.path.join(self.storage_dir, "snapshots")
        os.makedirs(self.snap_root, exist_ok=True)
        # A10：快照根注册进 pickle 反序列化白名单（algo.common.safe_pickle）
        from algo.common.safe_pickle import register_pickle_root
        register_pickle_root(self.snap_root)
        self.base_cfg_path = os.path.abspath(base_cfg_path)
        self.ssocl_cfg = ssocl_cfg or SSOCL_CFG_TEMPLATE
        self.base_cfg: Dict[str, Any] = {}
        self._load_base_cfg()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2026-10-03 A9 修复：_pipes 改 OrderedDict LRU，超上限驱逐最久未用品类。
        # 原实现只增不减——多品类长期运行内存/显存单调上涨直至 OOM。
        # 驱逐安全性：未巩固的即学均先落 WAL（learning_journal.jsonl），
        # 驱逐后下次访问走 _ensure_loaded 重载 + _replay_journal 重放，
        # 与崩溃恢复同路径，学习不丢；在途 predict 持 pipe 本地引用不受影响。
        self._pipes: "OrderedDict[str, Tuple[Any, int]]" = OrderedDict()  # category -> (Pipeline, active_version)
        self._max_cached_categories = max(1, int(max_cached_categories))
        self._write_lock = threading.RLock()           # 快照目录写/版本指针切换/_pipes 替换串行（仅管理操作，不护推理）
        # 2026-08-30 并发修复：品类级细粒度锁。
        # 原 routes_detect._detect_lock 是全局锁且只护 REST 同步接口，
        # 产线 daemon / RTSP 线程直接调 engine 绕过它 → predict 与 feedback
        # 即学（改 NormalBank/融合权重）并发竞态；全局锁还把不同品类串行化。
        # 现改在引擎层按品类加锁：同品类 predict/feedback/consolidate 串行，
        # 不同品类互不阻塞。锁序约定：品类锁 → _write_lock（不可反向）。
        self._pipe_locks: Dict[str, threading.RLock] = {}    # category -> 推理/反馈锁
        self._prepare_locks: Dict[str, threading.Lock] = {}  # category -> prepare 防重锁
        self._locks_guard = threading.Lock()                 # 护上面两个字典的惰性创建
        self._fg_cond = threading.Condition()
        self._last_journal_replay_audit: List[Dict[str, Any]] = []

    def _load_base_cfg(self) -> None:
        """从 base_cfg_path 读取基础配置并修正 init_from 相对路径。"""
        with open(self.base_cfg_path, "r", encoding="utf-8") as f:
            self.base_cfg = yaml.safe_load(f)
        # 修正 init_from 的相对路径（如 ./assets/disc_pretrain.pt）为基于
        # base_cfg 所在目录的绝对路径（M14 独立版：权重随 AOI_sys 自带，不
        # 再依赖 外部文件夹；绝对路径保持原样——客户环境可直接配置）。
        _cfg_dir = os.path.dirname(self.base_cfg_path)
        for _slot in ("disc", "shead"):
            _if = (self.base_cfg.get("slots", {}).get(_slot) or {}).get("init_from")
            if _if and not os.path.isabs(_if):
                _abs = os.path.join(_cfg_dir, _if)
                self.base_cfg["slots"][_slot]["init_from"] = os.path.normpath(_abs)

    def reload_config(self) -> None:
        """伪异常/槽位配置修改后重载基础配置（get_engine 单例内存缓存）。

        注意：仅影响**下一次 prepare/consolidate**（新 Pipeline 构建时读取），
        已加载的 Pipeline 实例不热更新。
        """
        with self._write_lock:
            self._load_base_cfg()

    # ----- 品类级锁（2026-08-30 并发修复） -----
    def _pipe_lock(self, category: str) -> threading.RLock:
        """取品类推理锁（惰性创建）。所有接触该品类 pipe 实例的入口
        （predict/feedback/consolidate 等）必须持此锁，保证同品类串行、
        不同品类并行。"""
        lock = self._pipe_locks.get(category)
        if lock is None:
            with self._locks_guard:
                lock = self._pipe_locks.setdefault(category, threading.RLock())
        return lock

    # ----- 学习让行前台推理（2026-10-12） -----
    @contextlib.contextmanager
    def _foreground(self):
        """标记一次前台推理；串行模式下后台学习在检查点等待其结束。"""
        with self._fg_cond:
            self._fg_active += 1
        try:
            yield
        finally:
            with self._fg_cond:
                self._fg_active -= 1
                self._fg_last = time.monotonic()
                self._fg_cond.notify_all()

    def _learning_parallel(self) -> bool:
        if self.learning_parallel is not None:
            return bool(self.learning_parallel)
        try:
            from ..core.config import get_settings
            return bool(get_settings().get("self_learning", "learning_parallel", False))
        except Exception:
            return False

    def _learning_yield(self, cancel_event=None) -> float:
        """串行模式：前台推理进行中或刚结束 learning_idle_ms 内则阻塞等待。
        返回等待秒数（候选墙钟预算扣除该段，避免持续推理负载误判超时）。
        单步（一次前向/保存）不可中断，故让行粒度为合作式检查点。"""
        if self._learning_parallel():
            return 0.0
        # 持品类锁/写锁时不得等待前台（前台推理可能正等这把锁 → 死锁）。
        locks = [getattr(self, "_write_lock", None),
                 *list(getattr(self, "_pipe_locks", {}).values())]
        if any(getattr(lock, "_is_owned", lambda: False)() for lock in locks):
            return 0.0
        t0 = time.monotonic()
        idle = self.learning_idle_ms / 1000.0
        with self._fg_cond:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    break
                remain = (0.0 if self._fg_active else
                          self._fg_last + idle - time.monotonic())
                if not self._fg_active and remain <= 0:
                    break
                self._fg_cond.wait(timeout=min(0.5, remain) if remain > 0 else 0.5)
        return time.monotonic() - t0

    def _prepare_lock(self, category: str) -> threading.Lock:
        """取品类 prepare 防重锁（惰性创建）：同品类并发 prepare 串行，
        后到者在幂等检查处命中已激活版本直接返回，不重复 fit。"""
        lock = self._prepare_locks.get(category)
        if lock is None:
            with self._locks_guard:
                lock = self._prepare_locks.setdefault(category, threading.Lock())
        return lock

    # ----- 目录辅助 -----
    def _cat_dir(self, category: str) -> str:
        return os.path.join(self.snap_root, category)

    def _v_dir(self, category: str, version: int) -> str:
        return os.path.join(self._cat_dir(category), f"v{int(version)}")

    def list_versions(self, category: str) -> List[int]:
        from algo.persist import list_versions
        return list_versions(self._cat_dir(category))

    def current_version(self, category: str) -> Optional[int]:
        from algo.persist import current_version
        return current_version(self._cat_dir(category))

    def list_snapshots(self, category: str) -> List[Dict[str, Any]]:
        items = []
        for v in self.list_versions(category):
            meta_path = os.path.join(self._v_dir(category, v), "meta.json")
            d = {"version": v, "active": v == self.current_version(category)}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        d.update(json.load(f))
                except Exception:
                    pass
            items.append(d)
        return items

    # ----- 准备新品类 -----
    def prepare(self, category: str, bundle: Dict[str, Any], *,
                scenario: str = "L1a", profile: str = "fast",
                trigger: str = "prepare", note: str = "",
                force: bool = False, activate: bool = True,
                parent_version: Optional[int] = None,
                progress_cb=None) -> int:
        """准备新品类：fit + 存快照 + 激活。返回新版本号。

        bundle = {"init_normal": List[path], "init_defect": List[path],
                  "val": [(path, label)], "test": []}
        progress_cb(frac, message)：可选进度回调（前端反馈 2026-09-13 #5），
        frac 为 prepare 全程占比 [0,1]：fit 占 0.05~0.85，在线学习器初始化
        0.85~0.95，落盘激活 0.95~1.0。
        """
        from algo.factory import build
        from algo.eval.offline import Pipeline
        from algo.persist import save_snapshot, list_versions, set_current_version

        cat_dir = self._cat_dir(category)
        os.makedirs(cat_dir, exist_ok=True)
        # 同品类并发 prepare 防重：持本品类 prepare 锁走全程，后到者在下面
        # 的幂等检查处命中已激活版本直接返回，不会重复 fit（其它品类不受影响）。
        with self._prepare_lock(category):
            # 幂等快路径：已有激活版本且不 force 时直接返回（只读 current 需轻量锁）
            with self._write_lock:
                if not force:
                    cv = self.current_version(category)
                    if cv is not None and os.path.isdir(self._v_dir(category, cv)):
                        self._ensure_loaded(category)
                        return cv

            # 昂贵部分（build/fit/enable_ssocl）在 _write_lock 外进行：
            # 不阻塞其它已加载品类的实时检测（只持本品类 prepare 锁）。
            cfg = apply_scenario_and_profile(self.base_cfg, scenario, profile)
            backbone, slots = build(cfg, self.device)
            pipe = Pipeline(cfg, backbone, slots)

            def _prog(frac, msg):
                if progress_cb is not None:
                    progress_cb(min(1.0, max(0.0, frac)), msg)
            _prog(0.05, "模型结构初始化完成，开始训练")
            pipe.fit(bundle, progress_cb=(
                lambda f, m: _prog(0.05 + 0.80 * f, m)))
            _prog(0.87, "在线学习器初始化")
            rng = np.random.default_rng(int(cfg.get("seed", 42)))
            pipe.enable_ssocl(self.ssocl_cfg,
                              train_normal_paths=bundle.get("init_normal", []),
                              rng=rng)

            # 落盘 + 激活：写 current.json + 换入 _pipes 需要 _write_lock，但很快
            _prog(0.95, "模型落盘并激活")
            with self._write_lock:
                vs = list_versions(cat_dir)
                new_v = max(vs) + 1 if vs else 1
                snap_dir = self._v_dir(category, new_v)
                # remove if exists（force case）
                if os.path.exists(snap_dir):
                    shutil.rmtree(snap_dir)
                save_snapshot(pipe, snap_dir, version=new_v,
                              parent_version=parent_version, trigger=trigger,
                              note=f"scenario={scenario}, profile={profile}. {note}",
                              ssocl_cfg=self.ssocl_cfg)
                with open(os.path.join(snap_dir, "meta.json"), encoding="utf-8") as f:
                    pipe.production_feature_metadata = json.load(f)["production_feature_metadata"]
                if activate:
                    set_current_version(cat_dir, new_v)
                    self._clear_journal(category)   # 新品准备：旧版本线学习日志作废
                    self._cache_put(category, pipe, new_v)
                return new_v

    # ----- 装载 -----
    def _cache_put(self, category: str, pipe: Any, version: int) -> None:
        """写入/刷新缓存并置顶为最近使用；超上限驱逐最久未用品类（LRU）。

        调用方须已持 _write_lock。驱逐仅摘缓存条目：在途 predict/feedback
        持 pipe 本地引用照常完成；未巩固学习由 WAL 兜底，重载时重放。"""
        self._pipes[category] = (pipe, version)
        self._pipes.move_to_end(category)
        while len(self._pipes) > self._max_cached_categories:
            evicted, (ev_pipe, ev_v) = self._pipes.popitem(last=False)
            logger.info("[engine] LRU 驱逐品类缓存 %s (v%s)，缓存上限 %d",
                        evicted, ev_v, self._max_cached_categories)
            del ev_pipe  # 释放引用；GPU 显存随 GC 回收

    def _pipe_entry(self, category: str) -> Tuple[Any, int]:
        """取 (pipe, active_version)；与 LRU 驱逐竞态时重载（有限次重试）。

        所有"先 _ensure_loaded 再索引 _pipes"的入口统一走这里：fast-path
        命中与后续索引之间存在被并发驱逐摘除的窗口（仅缓存超上限时发生），
        直接索引会 KeyError。"""
        for _ in range(3):
            self._ensure_loaded(category)
            entry = self._pipes.get(category)
            if entry is not None:
                return entry
        # 仅在 max_cached_categories 被误配为 0 之类的病态并发下才可能到达
        raise RuntimeError(f"品类 {category} 缓存条目反复被驱逐，请调大 "
                           "system.engine_max_cached_categories")

    def _ensure_loaded(self, category: str) -> None:
        # 快路径：已加载品类直接返回，不抢 _write_lock，避免 prepare/consolidate
        # 长时间持锁时阻塞其它已加载品类的实时检测（dict 读取在 GIL 下安全）。
        if category in self._pipes:
            self._pipes.move_to_end(category)  # LRU 触碰（O(1)，GIL 下原子）
            return
        from algo.persist import load_snapshot, current_version
        with self._write_lock:
            if category in self._pipes:  # 双重检查：等待锁期间可能已被加载
                self._pipes.move_to_end(category)
                return
            cv = current_version(self._cat_dir(category))
            if cv is None:
                raise RuntimeError(f"品类 {category} 尚未准备模型。")
            snap = self._v_dir(category, cv)
            pipe = load_snapshot(snap, self.device)
            # 崩溃恢复（2026-08-31 评审修复）：重放上次快照之后、进程崩溃
            # 之前的未巩固在线学习（此前 consumed=True 的反馈随进程死亡
            # 永久丢失，无重放机制）。LRU 驱逐后的重载同走此路径。
            self._replay_journal(category, pipe, cv)
            self._cache_put(category, pipe, cv)

    # ----- 学习日志（WAL：未巩固的在线学习崩溃不丢） -----
    def _journal_path(self, category: str) -> str:
        return os.path.join(self._cat_dir(category), "learning_journal.jsonl")

    def _append_journal(self, category: str, rec: Dict[str, Any]) -> None:
        """即学成功后追加一条学习日志（base=学习时激活版本）。"""
        os.makedirs(self._cat_dir(category), exist_ok=True)
        with open(self._journal_path(category), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _feedback_receipts(self):
        # 独立于可清空的品类 WAL，巩固/重启后仍保留已完成事件身份。
        conn = sqlite3.connect(os.path.join(self.storage_dir, "feedback_receipts.sqlite3"),
                               timeout=30)
        conn.execute("CREATE TABLE IF NOT EXISTS receipts "
                     "(event_key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS candidate_events "
                     "(event_key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                     "parent INTEGER NOT NULL, state TEXT NOT NULL, result TEXT)")
        return conn

    @staticmethod
    def _feedback_identity(category, image_path, verdict, label, box, feedback_id, event_id=None):
        key = (f"feedback:{feedback_id}" if feedback_id is not None
               else f"event:{event_id}" if event_id is not None else None)
        payload = json.dumps([category, image_path, verdict, label, box],
                             sort_keys=True, ensure_ascii=False)
        return key, payload

    def _replay_journal(self, category: str, pipe, version: int) -> int:
        """装载快照后重放未巩固的反馈学习（仅 base == 当前版本的条目）。

        consolidate 成功后日志清空（学习已固化进新版本）；手动回滚/激活
        到其它版本时该版本线上的条目自然被 base 过滤，不会误重放。
        """
        path = self._journal_path(category)
        audit = []
        self._last_journal_replay_audit = audit
        if not os.path.exists(path):
            return 0
        n_ok = n_skip = 0
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        seen = {}
        for line_no, ln in enumerate(lines, 1):
            try:
                rec = json.loads(ln)
            except Exception as exc:  # noqa: BLE001 坏行不可静默丢失
                n_skip += 1
                audit.append({"line": line_no, "status": "invalid_json",
                              "error": str(exc)})
                continue
            if not isinstance(rec, dict):
                n_skip += 1
                audit.append({"line": line_no, "status": "invalid_record",
                              "error": "WAL记录必须是JSON对象"})
                continue
            if rec.get("base") != version:
                continue
            key, payload = self._feedback_identity(
                category, rec.get("path"), rec.get("verdict"), rec.get("label"),
                rec.get("box"), rec.get("feedback_id"), rec.get("event_id"))
            if key in seen:
                if seen[key] != payload:
                    audit.append({"line": line_no, "status": "conflict"})
                    raise ValueError("WAL反馈事件ID对应不同payload")
                continue
            p = rec.get("path")
            if not p or not os.path.exists(p):
                n_skip += 1
                audit.append({"line": line_no, "status": "missing_image", "path": p})
                continue
            try:
                pipe.feedback(p, verdict=rec.get("verdict"),
                              label=rec.get("label"), box=rec.get("box"))
                n_ok += 1
                audit.append({"line": line_no, "status": "replayed"})
                if key is not None:
                    seen[key] = payload
                    if hasattr(self, "storage_dir"):
                        conn = self._feedback_receipts()
                        try:
                            conn.execute("INSERT OR IGNORE INTO receipts VALUES (?, ?)", (key, payload))
                            conn.commit()
                        finally:
                            conn.close()
            except Exception as exc:  # noqa: BLE001 单条失败不可静默丢失
                n_skip += 1
                audit.append({"line": line_no, "status": "replay_failed",
                              "error": str(exc)})
        if n_ok or n_skip:
            print(f"[engine] {category}: 重放未巩固学习 {n_ok} 条"
                  + (f"（跳过 {n_skip} 条：图像缺失/重放失败，详见内存审计）" if n_skip else ""),
                  flush=True)
        return n_ok

    def _journal_mark(self, category: str) -> Optional[Tuple[int, int]]:
        """WAL 水位（size, mtime_ns）；仅追加写，变化即说明有新学习。"""
        try:
            st = os.stat(self._journal_path(category))
        except FileNotFoundError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def _clear_journal(self, category: str) -> None:
        try:
            if os.path.exists(self._journal_path(category)):
                os.remove(self._journal_path(category))
        except Exception:  # noqa: BLE001
            pass

    def get_pipeline(self, category: str):
        return self._pipe_entry(category)[0]

    def learning_insight(self, category: str) -> Dict[str, Any]:
        """M14b：在线学习内部状态透出（引擎设计 §2 H/I/J 的可视化数据源）。

        双库规模 / 缺陷样例库 / 孵育头进度 / 主动选样队列 / 权重门控统计。
        未准备品类返回 enabled=False（不抛错，UI 显示空态）。"""
        try:
            self._ensure_loaded(category)
        except Exception:
            return {"category": category, "enabled": False}
        # 品类锁：以下读取的 bank/head/selector 状态会被 feedback 即学并发修改，
        # 持锁读出一致快照（均为快读，不会长时间占用）
        with self._pipe_lock(category):
            pipe, version = self._pipe_entry(category)
            h = getattr(pipe, "handler", None)
            out: Dict[str, Any] = {
                "category": category, "enabled": h is not None,
                "version": f"v{version}",
                "train_auroc": getattr(pipe, "train_auroc", None),
            }
            if h is None:
                return out
            nb = h.normal_bank
            out["normal_bank"] = {
                "core_patches": int(nb.core.shape[0])
                if getattr(nb, "core", None) is not None else 0,
                "ext_samples": len(getattr(nb, "ext", []) or []),
                "fuse_blocked": bool(getattr(nb, "fuse_blocked", False)),
                # 2026-08-30：熔断可观测——透出在线缺陷率估值与阈值
                "defect_ratio": round(float(getattr(nb, "defect_ratio", 0.0)), 4),
                "defect_ratio_thresh": float(getattr(nb, "defect_ratio_thresh", 0.0)),
            }
            db = h.defect_bank
            n_def = len(getattr(db, "samples", []) or [])
            out["defect_bank"] = {"samples": n_def,
                                  "intercept_min_hit": getattr(db, "min_hit", 1)}
            # AC-028：库血缘摘要（记忆资产≠训练权重；逐条清单走 bank.lineage()）
            out["lineage"] = {k: _lineage_summary(b) for k, b in
                              (("normal_bank", nb), ("defect_bank", db))}
            hm = getattr(pipe, "head_mgr", None)
            min_samples = int(hm.cfg.get("min_samples", 8)) if hm else 8
            # AC-032：进度按纯 patch 独立图计数（整图 add / legacy 不计）
            hs = hm.status(db) if (hm is not None and hasattr(hm, "status")) else {}
            n_img = hs.get("n_pure_images")
            n_img = n_def if n_img is None else n_img
            out["incubate"] = {
                "trained": bool(hm is not None and hm.head is not None),
                "min_samples": min_samples,
                "n_pos": int(hm.head.n_pos) if (hm and hm.head) else 0,
                "n_pure_images": int(n_img),
                "progress": round(min(1.0, n_img / max(min_samples, 1)), 2),
                "separation": hs.get("separation"),
                "last_decision": hs.get("last_decision"),
                "log_tail": list(hm.log[-3:]) if hm else [],
            }
            sel = getattr(pipe, "selector", None)
            out["active_queue"] = len(getattr(sel, "queue", []) or []) if sel else 0
            st = getattr(h, "stats", {}) or {}
            out["weight_gate"] = {"apply": int(st.get("weight_apply", 0)),
                                  "reject": int(st.get("weight_reject", 0))}
            out["stats"] = {k: int(v) for k, v in st.items()
                            if isinstance(v, (int, float))}
            return out

    def feature_set(self, category: str) -> Dict[str, Any]:
        """AC-016：当前生产 trad 特征集只读快照（启用组/权重/口径/签名/表示状态）。

        未准备品类 enabled=False；无 trad 槽位 status=not_applicable。纯读取，
        不触发提取、不改变生产配置。"""
        from algo.features import registry as freg
        try:
            self._ensure_loaded(category)
        except Exception:
            return {"category": category, "enabled": False}
        with self._pipe_lock(category):
            pipe, version = self._pipe_entry(category)
            trad = (getattr(pipe, "slots", None) or {}).get("trad")
            out: Dict[str, Any] = {"category": category, "enabled": True,
                                   "version": int(version)}
            if trad is None:
                out["status"] = "not_applicable"
                return out
            enabled = set(trad.groups)
            weight = dict(getattr(trad, "group_weight", {}) or {})
            scale = dict(getattr(trad, "group_scale", {}) or {})
            out["status"] = "available"
            out["groups"] = [{
                "key": g["key"], "name": g["name"], "category": g["category"],
                "dim": int(g["dim"]), "enabled": g["key"] in enabled,
                "weight": (round(float(weight[g["key"]]), 4)
                           if g["key"] in weight else None),
                "scale": (round(float(scale[g["key"]]), 6)
                          if g["key"] in scale else None),
            } for g in freg.TRAD_GROUPS]
            out["enabled_groups"] = list(trad.groups)
            out["enabled_dim"] = int(sum(g["dim"] for g in freg.TRAD_GROUPS
                                         if g["key"] in enabled))
            out["total_dim"] = int(freg.TRAD_DIM)
            out["scoring"] = trad.scoring
            out["group_spec"] = trad.group_spec
            out["signature"] = trad.signature()
            out["feature_representation"] = dict(
                getattr(pipe, "feature_representation", None) or {})
            out["rebuild_required"] = out["feature_representation"].get("status") == "stale"
            out["selection_report"] = getattr(trad, "selection_report", None)
            pfm = getattr(pipe, "production_feature_metadata", None) or {}
            out["snapshot_signature"] = pfm.get("feature_signature")
            return json.loads(json.dumps(out, ensure_ascii=False, default=str))

    def active_suggestions(self, category: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """M15b：主动选样清单透出（引擎设计 §6 赛题点名"主动学习"）。

        pipe.active_select 语义：灰区中最不确定×最有代表性 top-N，
        每张只问一次（select 消耗队列，防重复打扰）。
        返回项关联最新 Detection id（复核/反馈提交需要 detection_id）。
        品类未准备或无选样器 → 空列表（不抛错）。"""
        try:
            pipe = self.get_pipeline(category)
        except Exception:
            return []
        if getattr(pipe, "selector", None) is None:
            return []
        # active_select 会消耗选样队列（写操作），持品类锁与 predict/feedback 串行
        with self._pipe_lock(category):
            picked = pipe.active_select(top_n=top_n) or []
        paths = [p["path"] for p in picked if p.get("path")]
        det_map: Dict[str, int] = {}
        if paths:
            from ..db.database import session_scope
            from ..db.models import Detection
            with session_scope() as s:
                rows = (s.query(Detection)
                        .filter(Detection.image_path.in_(paths))
                        .order_by(Detection.id.desc()).all())
                for r in rows:
                    det_map.setdefault(r.image_path, r.id)
        return [{**p, "image_path": p.get("path"),
                 "detection_id": det_map.get(p.get("path"))} for p in picked]

    @staticmethod
    def _require_current_representation(pipe, op: str) -> None:
        """AC-015：特征表示过期（代码/注册表/预处理与快照签名不一致）时，
        禁止在线学习/巩固写入——否则新表示向量混入旧库，或巩固把旧库盖上新签名“洗白”。"""
        rep = getattr(pipe, "feature_representation", None) or {}
        if rep.get("status") == "stale":
            raise ValueError(f"特征表示已过期（{','.join(rep.get('drift') or [])}），"
                             f"禁止{op}；须先表示重建（重提特征→重建库→重校准→过 G2）后再激活")

    # ----- 激活 / 回滚 / 变体 -----
    def activate(self, category: str, version: int, *, allow_stale: bool = False) -> None:
        from algo.persist import load_snapshot, set_current_version, list_versions
        candidate_meta = os.path.join(self._v_dir(category, version), "meta.json")
        if os.path.exists(candidate_meta):
            with open(candidate_meta, encoding="utf-8") as f:
                if json.load(f).get("trigger") == "isolated_candidate":
                    from ..self_learning.candidate_release import CandidateRelease
                    return CandidateRelease(self).switch(category, version)
        with self._pipe_lock(category), self._write_lock:
            if version not in list_versions(self._cat_dir(category)):
                raise ValueError(f"版本 v{version} 不存在")
            meta = {}
            meta_path = os.path.join(self._v_dir(category, version), "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            if meta.get("trigger") == "isolated_candidate":
                if meta.get("parent_version") != self.current_version(category):
                    raise ValueError("候选父版本已过期")
                if meta.get("candidate_state") != "completed":
                    raise ValueError("候选尚未完成")
                from ..self_learning.candidate_release import CandidateRelease
                return CandidateRelease(self).switch(category, version)
            pipe = load_snapshot(self._v_dir(category, version), self.device)
            rep = getattr(pipe, "feature_representation", None) or {}
            if rep.get("status") == "stale" and not allow_stale:
                raise ValueError(f"v{version} 特征表示已过期（{','.join(rep.get('drift') or [])}），"
                                 "拒绝激活；须表示重建后激活新版本（或 allow_stale 强制）")
            set_current_version(self._cat_dir(category), version)
            self._cache_put(category, pipe, version)

    def rollback(self, category: str, target_version: Optional[int] = None,
                 *, allow_stale: bool = False) -> int:
        """回滚：target_version 给定时直接指；否则取次新版本。"""
        current = self.current_version(category)
        if target_version is not None and target_version == current:
            from ..self_learning.candidate_release import CandidateRelease
            release = CandidateRelease(self)
            with release.connection() as c:
                event = c.execute("SELECT action,state,payload FROM release_events WHERE category=? AND action IN ('activate','rollback') AND state!='aborted' ORDER BY id DESC LIMIT 1", (category,)).fetchone()
            if event and event[:2] == ("rollback", "committed") and json.loads(event[2])["target_version"] == current:
                release.switch(category, current, rollback=True)
                return current
        if current is not None:
            meta_path = os.path.join(self._v_dir(category, current), "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("trigger") == "isolated_candidate":
                    from ..self_learning.candidate_release import CandidateRelease
                    target = target_version if target_version is not None else meta["parent_version"]
                    CandidateRelease(self).switch(category, target, rollback=True)
                    return target
        vs = self.list_versions(category)
        if not vs:
            raise RuntimeError(f"品类 {category} 无快照可回滚")
        if target_version is None:
            cv = self.current_version(category) or vs[-1]
            idx = vs.index(cv) if cv in vs else len(vs) - 1
            target = vs[max(0, idx - 1)]
        else:
            target = target_version
        self.activate(category, target, allow_stale=allow_stale)
        return target

    def build_variant(self, category: str, version: int):
        """A/B 用：加载指定版本 pipe（不影响激活态），外部用完丢。"""
        from algo.persist import load_snapshot
        return load_snapshot(self._v_dir(category, version), self.device)

    # ----- 推理 -----
    def predict_image_path(self, category: str, path: str) -> DetectionResult:
        import cv2
        from algo.common.io import load_image
        self._ensure_loaded(category)
        # 品类锁：与同品类 feedback 即学/consolidate 串行（不同品类并行不互斥）
        with self._foreground(), self._pipe_lock(category):
            pipe, active_version = self._pipe_entry(category)
            t0 = time.perf_counter()
            rec = pipe.predict(path)
            dt_ms = (time.perf_counter() - t0) * 1000
            return self._to_result(pipe, rec, dt_ms, active_version)

    def predict_ndarray(self, category: str, rgb: np.ndarray, path_hint: Optional[str] = None) -> DetectionResult:
        self._ensure_loaded(category)
        # 品类锁：与同品类 feedback 即学/consolidate 串行（不同品类并行不互斥）
        with self._foreground(), self._pipe_lock(category):
            pipe, active_version = self._pipe_entry(category)
            t0 = time.perf_counter()
            rec = pipe.predict_frame(rgb, path=path_hint)
            dt_ms = (time.perf_counter() - t0) * 1000
            return self._to_result(pipe, rec, dt_ms, active_version)

    def explain_image(self, category: str, path: str) -> Dict[str, Any]:
        """AC-014：按需图像解释（trad 测量值/正常范围/近邻/组距离），不写追溯、不进热路径。"""
        self._ensure_loaded(category)
        with self._foreground(), self._pipe_lock(category):
            pipe, active_version = self._pipe_entry(category)
            trad = pipe.explain_trad(path=path)
        return {"model_version": active_version, "trad": trad}

    def _to_result(self, pipe, rec: Dict[str, Any], dt_ms: float,
                   model_version: Optional[int] = None) -> DetectionResult:
        slot_scores = dict(rec.get("slot_scores", {}))
        raw_scores = dict(rec.get("raw_scores", {}))
        # AC-027：weights = 本次判定实际生效的融合权重（trace fusion 段），非 pipe 当前值
        fusion_tr = next((t for t in (rec.get("decision_trace") or [])
                          if t.get("stage") == "fusion"), None) or {}
        weights = dict(fusion_tr.get("effective_weights")
                       or getattr(pipe, "weights", {}) or {})
        router_w = dict(rec.get("router_w", {}) or {})

        decision = rec.get("decision", "normal")
        is_anomaly = (decision == "anomaly")  # gray 不直接当异常，UI 可另显橙
        triggered_slot = self._dominant_slot(slot_scores, weights)
        boxes = []
        for b in rec.get("boxes", []) or []:
            # AC-014：框级主导槽位来自区域证据；无区域证据时明示回退整图口径
            ev = b.get("evidence")
            if ev and b.get("dominant_slot"):
                dom = b["dominant_slot"]
            else:
                dom = triggered_slot
                ev = {**(ev or {}), "level": "image_fallback",
                      "semantics": "无框内区域证据，主导槽位取整图口径"}
            box = {"bbox": list(b.get("bbox", [0, 0, 0, 0])),
                   "area": int(b.get("area", 0)), "dominant_slot": dom, "evidence": ev}
            # AC-063：区域记录字段（来源/分数/后处理版本/候选状态）原样透出
            for k in ("source", "region_id", "score_peak", "score_mean", "area_work",
                      "method", "postproc", "status", "confirmed"):
                if b.get(k) is not None:
                    box[k] = b[k]
            boxes.append(box)

        r = DetectionResult(
            score=float(rec.get("fused", 0.0)),
            decision=decision,
            is_anomaly=is_anomaly,
            threshold=float(pipe.decider.tau_high) if pipe.decider else 0.99,
            gray_threshold=float(pipe.decider.tau_gray) if pipe.decider else 0.95,
            slot_scores=slot_scores,
            raw_scores=raw_scores,
            weights=weights,
            router_w=router_w,
            defect_boxes=boxes,
            mask=rec.get("mask"),
            fused_heatmap=rec.get("fused_heatmap"),
            types=list(rec.get("types", []) or []),
            latency_ms=float(dt_ms),
            n_tiles=int(rec.get("n_tiles", 0)),
            triggered_slot=triggered_slot,
            open_alert=bool(rec.get("open_alert", False)),
            align_offset=(list(rec["align_offset"]) if rec.get("align_offset") else None),
            align_warn=bool(rec.get("align_warn", False)),
            model_version=model_version,
            extra={"e_open_score": rec.get("open_score"),
                   "e_open_threshold": getattr(pipe, "open_thresh", None) if "open_score" in rec else None,
                   "decision_trace": list(rec.get("decision_trace", []) or []),
                   "candidate_router": {**(getattr(pipe, "router_gate", None) or {}),
                                        "applied": False,
                                        "candidate_weights": fusion_tr.get(
                                            "router_candidate_weights")}
                   if getattr(pipe, "router_gate", None) else None,
                   "weights_semantics": _weights_semantics(),
                   "localization": rec.get("localization") or {"status": "legacy_unknown"},
                   "feature_metadata": getattr(pipe, "production_feature_metadata",
                                                {"status": "legacy_unknown"})},
        )
        return r

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
    # 即学超时预算（评审问题#7：触发判别头微调+门控的最坏路径曾达 ~1.1s，
    # 同步等待会拖住 API 与同品类检测）。超出预算学习转后台完成——
    # 后台线程持品类锁，predict 排队而不读撕裂态；学习成功才记 WAL。
    feedback_timeout_ms = 3000

    def submit_feedback(self, category: str, image_path: str,
                        verdict: str, label: Optional[int] = None,
                        box: Optional[List[float]] = None,
                        timeout_ms: Optional[int] = None,
                        on_complete=None,
                        feedback_id: Optional[int] = None,
                        event_id: Optional[str] = None,
                        on_result=None) -> Dict[str, Any]:
        """on_result(result, error) 回传最终结果；on_complete 保留成功无参契约。"""
        self._ensure_loaded(category)
        timeout_s = (timeout_ms or self.feedback_timeout_ms) / 1000.0
        done: Dict[str, Any] = {}

        def result_info():
            info = {"path": image_path, "verdict": verdict, "label": label,
                    "feedback_ms": round(done.get("dt_ms", 0.0), 2)}
            if done.get("idempotent"):
                info["idempotent"] = True
            entry = done.get("entry")
            if isinstance(entry, dict):
                for k in ("action", "drift", "boost", "defect_ratio", "attribution", "evidence"):
                    if k in entry:
                        info[k] = entry[k]
            if done.get("fuse_blocked") is not None:
                info["fuse_blocked"] = done["fuse_blocked"]
            return info

        def _learn():
            # 品类锁：feedback 即学会改 NormalBank/融合权重/缺陷样例库，
            # 必须与同品类 predict 串行，否则 predict 读到撕裂的中间态
            try:
                with self._pipe_lock(category):
                    pipe, active_v = self._pipe_entry(category)
                    key, payload = self._feedback_identity(
                        category, image_path, verdict, label, box, feedback_id, event_id)
                    conn = self._feedback_receipts() if key else None
                    try:
                        if conn is not None:
                            conn.execute("BEGIN IMMEDIATE")
                            previous = conn.execute(
                                "SELECT payload FROM receipts WHERE event_key=?", (key,)).fetchone()
                            if previous and previous[0] != payload:
                                raise ValueError("反馈事件ID对应不同payload")
                        else:
                            previous = None
                        t0 = time.perf_counter()
                        if previous:
                            entry = {}
                            done["idempotent"] = True
                        else:
                            self._require_current_representation(pipe, "在线学习")
                            entry = pipe.feedback(image_path, verdict=verdict, label=label, box=box)
                            self._append_journal(category, {
                                "t": time.time(), "base": active_v, "path": image_path,
                                "verdict": verdict, "label": label, "box": box,
                                "feedback_id": feedback_id, "event_id": event_id,
                                "category": category})
                            if conn is not None:
                                conn.execute("INSERT INTO receipts VALUES (?, ?)", (key, payload))
                        if conn is not None:
                            conn.commit()
                        done["dt_ms"] = (time.perf_counter() - t0) * 1000
                    finally:
                        if conn is not None:
                            conn.close()
                    h = getattr(pipe, "handler", None)
                    done["fuse_blocked"] = (bool(h.normal_bank.fuse_blocked)
                                            if h is not None else None)
                    done["entry"] = entry
                if on_complete is not None:
                    try:
                        on_complete()
                    except Exception:  # noqa: BLE001 学习已成功，账务失败须保留待处理
                        logger.exception("反馈学习完成，但消费状态回写失败")
            except Exception as e:  # noqa: BLE001 线程内异常回传主线程抛出
                done["error"] = e
            finally:
                if on_result is not None:
                    try:
                        on_result(None if "error" in done else result_info(), done.get("error"))
                    except Exception:
                        logger.exception("反馈最终结果回写失败")

        th = threading.Thread(target=_learn, daemon=True,
                              name=f"feedback-learn-{category}")
        th.start()
        th.join(timeout_s)
        if th.is_alive():
            return {"path": image_path, "verdict": verdict, "label": label,
                    "async": True,
                    "note": f"即学超 {int(timeout_s * 1000)}ms 转后台完成"
                            "（同品类检测经品类锁排队，不会读撕裂态）"}
        if "error" in done:
            raise done["error"]
        return result_info()

    @staticmethod
    def _candidate_files(path, *, sync=False, cached=False):
        # cached=True 用 (abspath,size,mtime_ns) 缓存哈希：只用于不可变的已发布父快照，
        # 作为并发变更守卫而非防篡改边界；暂存区与恢复完整性校验仍读全文。
        import hashlib
        from algo.persist import _sha256_file_cached
        files = {}
        for root, dirs, names in os.walk(path):
            dirs.sort()
            if any(os.path.islink(os.path.join(root, name)) for name in dirs):
                raise ValueError("候选快照不允许符号链接")
            for name in sorted(names):
                full = os.path.join(root, name)
                if os.path.islink(full):
                    raise ValueError("候选快照不允许符号链接")
                if cached and not sync:
                    files[os.path.relpath(full, path)] = _sha256_file_cached(full)
                    continue
                with open(full, "r+b" if sync else "rb") as f:
                    files[os.path.relpath(full, path)] = hashlib.sha256(f.read()).hexdigest()
                    if sync:
                        os.fsync(f.fileno())
        if not files:
            raise ValueError("候选快照缺失")
        return files

    def _candidate_checkpoint(self, point):
        pass  # TEMP故障注入接缝，不参与生产状态决策

    from algo.ssocl.candidate_budget import CandidateBudget, CandidateDeferred

    @staticmethod
    def _candidate_load(path, device, budget):
        budget.add("reloads", point="snapshot_reload")
        size = sum(os.path.getsize(os.path.join(root, name))
                   for root, _, names in os.walk(path) for name in names)
        budget.add("snapshot_bytes", size, "snapshot_work")
        from algo.persist import load_snapshot
        pipe = load_snapshot(path, device, candidate=True)
        try:
            budget.check("after_reload")
        except BaseException:
            pipe._cpu_exec.shutdown()
            raise
        return pipe

    def learn_candidate(self, category: str, image_path: str, *, verdict: str,
                        label: Optional[int], box=None, event_id: str,
                        parent_version: int, recover: bool = False,
                        cancel_event=None) -> Dict[str, Any]:
        # recover=True 是显式恢复/拒绝请求；启动和普通重试不猜测任务死亡。
        # OS持有锁随进程退出释放；不能根据training或PID猜测持有者已死。
        from contextlib import ExitStack
        os.makedirs(self._cat_dir(category), exist_ok=True)
        with ExitStack() as stack:
            lock = stack.enter_context(open(os.path.join(self._cat_dir(category),
                                                        ".candidate.lock"), "a+b"))
            if os.path.getsize(lock.name) == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    stack.callback(lambda: (lock.seek(0), msvcrt.locking(
                        lock.fileno(), msvcrt.LK_UNLCK, 1)))
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    stack.callback(lambda: fcntl.flock(lock, fcntl.LOCK_UN))
            except OSError as e:
                raise ValueError("品类候选任务仍由其他执行者持锁") from e
            from algo.ssocl.candidate_budget import candidate_scope
            budget = self.CandidateBudget(self, category=category, event_id=event_id,
                                          cancel_event=cancel_event)
            with candidate_scope(budget):
                return self._learn_candidate_locked(category, image_path, verdict=verdict,
                    label=label, box=box, event_id=event_id, parent_version=parent_version,
                    recover=recover, cancel_event=cancel_event, budget=budget)

    def _sweep_candidate_stages(self, category: str, conn) -> List[str]:
        """清理崩溃残留的 .candidate-* 暂存；调用方须已持本品类候选 OS 锁。

        只有 prepared 意图的暂存可被显式 recover 补记，必须保留；training 意图
        恢复时只能拒绝、failed/rejected/deferred 不再使用，均可安全删除。
        """
        import glob
        norm = lambda p: os.path.normcase(os.path.abspath(p))
        keep = set()
        for (result,) in conn.execute(
                "SELECT result FROM candidate_events WHERE state='prepared'"):
            try:
                stage = json.loads(result).get("stage")
            except (TypeError, ValueError, AttributeError):
                continue
            if stage:
                keep.add(norm(stage))
        removed = []
        for path in glob.glob(os.path.join(self._cat_dir(category), ".candidate-*")):
            if os.path.isdir(path) and norm(path) not in keep:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path)
        return removed

    def _learn_candidate_locked(self, category: str, image_path: str, *, verdict: str,
                        label: Optional[int], box=None, event_id: str,
                        parent_version: int, recover: bool = False,
                        cancel_event=None, budget=None) -> Dict[str, Any]:
        """显式单事件候选：持久意图先于学习，不触碰生产实例/生产 WAL。"""
        import tempfile
        from algo.persist import load_snapshot, save_snapshot

        if not event_id or verdict not in ("correct", "wrong", "none") or label not in (0, 1, None):
            raise ValueError("候选事件ID/verdict/label无效")
        key = "candidate:" + event_id
        _, payload = self._feedback_identity(
            category, image_path, verdict, label, box, None, event_id)
        conn = self._feedback_receipts()
        conn.execute("PRAGMA busy_timeout=0")
        stage = None
        claimed = False
        preserve_stage = False
        pipe = None
        try:
            self._sweep_candidate_stages(category, conn)
            # 锁外预热父快照哈希缓存（已发布快照不可变；锁内仍按 size/mtime 复核），
            # 避免首个候选在推理锁内做数百毫秒冷哈希。
            if not recover and os.path.isdir(self._v_dir(category, parent_version)):
                self._candidate_files(self._v_dir(category, parent_version), cached=True)
            # 仅持锁冻结父版本和WAL边界；装载、重放、训练与保存均在推理锁外。
            with self._pipe_lock(category), self._write_lock:
                conn.execute("BEGIN IMMEDIATE")
                previous = conn.execute(
                    "SELECT payload,parent,state,result FROM candidate_events WHERE event_key=?",
                    (key,)).fetchone()
                if previous:
                    if previous[0] != payload or previous[1] != parent_version:
                        raise ValueError("候选事件ID对应不同payload/父版本")
                    if previous[2] in ("prepared", "completed"):
                        if previous[2] == "prepared" and not recover:
                            raise ValueError("候选待显式recover，禁止普通重试补记")
                        intent = json.loads(previous[3])
                        if "result" not in intent:
                            raise ValueError("旧候选缺恢复完整性证据，明确拒绝")
                        result = intent["result"]
                        target = self._v_dir(category, result["version_num"])
                        source = target if os.path.isdir(target) else intent["stage"]
                        try:
                            wal_now = b""
                            if os.path.exists(self._journal_path(category)):
                                with open(self._journal_path(category), "rb") as f:
                                    wal_now = f.read(65537)
                            import hashlib
                            if self.current_version(category) != parent_version:
                                raise ValueError("候选父版本已变化")
                            if hashlib.sha256(wal_now).hexdigest() != intent["wal_sha256"]:
                                raise ValueError("候选父WAL已变化")
                            if self._candidate_files(self._v_dir(category, parent_version)) != intent["parent_files"]:
                                raise ValueError("候选父快照已变化")
                            if self._candidate_files(source) != intent["files"]:
                                raise ValueError("候选快照完整性校验失败")
                            restored = self._candidate_load(source, self.device, budget)
                            restored._cpu_exec.shutdown()
                            if source != target:
                                os.rename(source, target)
                            conn.execute("UPDATE candidate_events SET state='completed' WHERE event_key=?", (key,))
                            conn.commit()
                        except self.CandidateDeferred:
                            conn.rollback()
                            raise  # prepared 工件保留，后续仍只能显式 recover
                        except Exception:
                            conn.execute("UPDATE candidate_events SET state='rejected' WHERE event_key=?", (key,))
                            conn.commit()
                            raise
                        result["idempotent"] = True
                        return result
                    if previous[2] == "training" and recover:
                        # 已取得本品类OS锁，才允许显式拒绝不完整意图；不猜测PID。
                        conn.execute("UPDATE candidate_events SET state='rejected' WHERE event_key=?", (key,))
                        conn.commit()
                    raise ValueError("候选事件未完成或失败，需显式recover拒绝后新事件重新构建")
                if recover:
                    raise ValueError("候选恢复意图不存在，拒绝创建新训练")
                if self.current_version(category) != parent_version:
                    raise ValueError("候选父版本已变化")
                pending = conn.execute("SELECT payload FROM candidate_events WHERE state IN ('training','prepared')").fetchall()
                if any(json.loads(row[0])[0] == category for row in pending):
                    raise ValueError("品类存在未恢复候选，请先用原事件ID恢复或明确拒绝")
                records = []
                wal = self._journal_path(category)
                wal_bytes = b""
                if os.path.exists(wal):
                    if os.path.getsize(wal) > 65536:
                        raise ValueError("父WAL超过候选底座64KiB预算，需后续微批协议")
                    with open(wal, "rb") as f:
                        wal_bytes = f.read()
                    for line in wal_bytes.decode("utf-8").splitlines():
                        rec = json.loads(line)
                        if rec.get("base") == parent_version:
                            records.append(rec)
                parent_files = self._candidate_files(self._v_dir(category, parent_version),
                                                     cached=True)
                stage = tempfile.mkdtemp(prefix=".candidate-", dir=self._cat_dir(category))
                conn.execute("INSERT INTO candidate_events VALUES (?,?,?,'training',?)",
                             (key, payload, parent_version, json.dumps(
                                 {"stage": stage, "candidate_budget": budget.snapshot()})))
                conn.commit()  # 持久预写意图；失败/重启不误报消费完成
                claimed = True
            budget.add("events", len(records), "parent_events")
            budget.add("events", 1, "new_event")
            budget.add("patch_bytes", sum(os.path.getsize(p) for p in
                       [r.get("path") for r in records] + [image_path]
                       if p and os.path.isfile(p)), "asset_bytes")
            pipe = self._candidate_load(self._v_dir(category, parent_version), self.device, budget)
            seen = {}
            for rec in records:
                identity, rec_payload = self._feedback_identity(
                    category, rec.get("path"), rec.get("verdict"), rec.get("label"),
                    rec.get("box"), rec.get("feedback_id"), rec.get("event_id"))
                if identity in seen:
                    if seen[identity] != rec_payload:
                        raise ValueError("父WAL事件payload冲突")
                    continue
                if not rec.get("path") or not os.path.isfile(rec["path"]):
                    raise ValueError("父WAL资产缺失，拒绝不完整候选")
                budget.check("parent_feedback")
                pipe.feedback(rec["path"], verdict=rec.get("verdict"),
                              label=rec.get("label"), box=rec.get("box"))
                if identity is not None:
                    seen[identity] = rec_payload
            budget.check("new_feedback")
            entry = pipe.feedback(image_path, verdict=verdict, label=label, box=box)
            budget.check("after_feedback")
            self._candidate_checkpoint("before_save")
            save_snapshot(pipe, stage, version=0, parent_version=parent_version,
                          trigger="isolated_candidate", note="独立验收未实现，禁止激活",
                          ssocl_cfg=self.ssocl_cfg)
            # 保存产物必须真实可重新装载，否则不发完成凭据。
            self._candidate_checkpoint("after_save")
            restored = self._candidate_load(stage, self.device, budget)
            restored._cpu_exec.shutdown()
            # 暂存区私有：全文哈希+fsync 在锁外完成，锁内只重写/重哈希 meta.json。
            stage_files = self._candidate_files(stage, sync=True)
            import hashlib
            with self._pipe_lock(category), self._write_lock:
                if self.current_version(category) != parent_version:
                    raise ValueError("训练期间父版本已变化，拒绝候选")
                current_wal = b""
                if os.path.exists(wal):
                    with open(wal, "rb") as f:
                        current_wal = f.read(65537)
                if current_wal != wal_bytes:
                    raise ValueError("训练期间父WAL已变化，拒绝过期候选")
                versions = self.list_versions(category)
                version = max(versions) + 1 if versions else 1
                meta_path = os.path.join(stage, "meta.json")
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                meta.update(version=version, candidate_state="completed",
                            candidate_event=key, acceptance="unavailable",
                            parent_wal_records=len(records), candidate_budget=budget.snapshot())
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, allow_nan=False)
                    f.flush()
                    os.fsync(f.fileno())
                if self._candidate_files(self._v_dir(category, parent_version),
                                         cached=True) != parent_files:
                    raise ValueError("候选父快照已变化")
                with open(meta_path, "rb") as f:
                    stage_files[os.path.relpath(meta_path, stage)] = \
                        hashlib.sha256(f.read()).hexdigest()
                result = {"category": category, "version_num": version,
                          "parent_version": parent_version, "event_id": event_id,
                          "learning_status": "candidate", "activated": False,
                          "candidate_state": "completed", "acceptance": "unavailable",
                          "entry": entry, "candidate_budget": budget.snapshot()}
                intent = {"result": result, "stage": stage,
                          "files": stage_files,
                          "parent_files": parent_files,
                          "wal_sha256": hashlib.sha256(wal_bytes).hexdigest()}
                self._candidate_checkpoint("before_prepared")
                budget.check("before_prepared")
                conn.execute("UPDATE candidate_events SET state='prepared',result=? WHERE event_key=?",
                             (json.dumps(intent, ensure_ascii=False, allow_nan=False), key))
                conn.commit()
                claimed = False  # prepared后保留工件，重试验证补记，不回退failed
                preserve_stage = True
                self._candidate_checkpoint("after_prepared")
                self._candidate_checkpoint("before_rename")
                os.rename(stage, self._v_dir(category, version))
                stage = None
                self._candidate_checkpoint("after_rename")
                self._candidate_checkpoint("before_completed")
                conn.execute("UPDATE candidate_events SET state='completed' WHERE event_key=?", (key,))
                conn.commit()
                self._candidate_checkpoint("after_completed")
            return result
        except self.CandidateDeferred as exc:
            conn.rollback()
            if claimed:
                evidence = {"candidate_state": exc.state, "reason": str(exc),
                            "reason_code": "CANDIDATE_BUILD_DEFERRED",
                            "violations": [{"code": "CANDIDATE_BUILD_DEFERRED",
                                            "message": str(exc)}],
                            "retryable": True,
                            "candidate_budget": budget.snapshot(), "retry": "new_event_id"}
                conn.execute("UPDATE candidate_events SET state=?,result=? WHERE event_key=?",
                             (exc.state, json.dumps(evidence, ensure_ascii=False), key))
                conn.commit()
            raise
        except Exception as exc:
            conn.rollback()
            if claimed:
                conn.execute("UPDATE candidate_events SET state='failed',result=? WHERE event_key=?",
                             (json.dumps({"reason": str(exc),
                                          "reason_code": "CANDIDATE_BUILD_FAILED",
                                          "violations": [{"code": "CANDIDATE_BUILD_FAILED",
                                                          "message": str(exc)}],
                                          "retryable": False,
                                          "candidate_budget": budget.snapshot()},
                                         ensure_ascii=False), key))
                conn.commit()
            raise
        finally:
            conn.close()
            if pipe is not None:
                pipe._cpu_exec.shutdown()
            if stage is not None and not preserve_stage:
                shutil.rmtree(stage)

    # ----- 巩固（落盘新版） -----
    def consolidate(self, category: str, *, note: str = "", activate: bool = True) -> int:
        """把当前内存态固化为新版本；候选巩固可暂不切换激活指针。"""
        from algo.persist import save_snapshot, set_current_version, list_versions, load_snapshot
        # 2026-08-30 并发修复：
        # 1) _ensure_loaded 移出 _write_lock（已加载品类快路径不抢锁）；
        # 2) save_snapshot 读取 pipe 内存态（NormalBank/融合权重可能正被
        #    feedback 即学修改），必须持品类锁与同品类 predict/feedback 串行，
        #    否则快照是撕裂的中间态；
        # 3) 耗时 IO 不持全局 _write_lock 阻塞推理——其它品类实时检测不受影响
        #    （_write_lock 仅剩版本号分配/指针切换等管理操作的互斥）。
        self._ensure_loaded(category)
        if activate:
            # 激活路径：写快照、切指针、清 WAL 必须与同品类 feedback 原子串行。
            with self._pipe_lock(category):
                pipe, active = self._pipe_entry(category)
                self._require_current_representation(pipe, "巩固")
                cat_dir = self._cat_dir(category)
                with self._write_lock:
                    vs = list_versions(cat_dir)
                    new_v = max(vs) + 1 if vs else 1
                    snap = self._v_dir(category, new_v)
                    save_snapshot(pipe, snap, version=new_v,
                                  parent_version=active, trigger="consolidate",
                                  note=note, ssocl_cfg=self.ssocl_cfg)
                    with open(os.path.join(snap, "meta.json"), encoding="utf-8") as f:
                        pipe.production_feature_metadata = json.load(f)["production_feature_metadata"]
                    set_current_version(cat_dir, new_v)
                    self._clear_journal(category)
                    self._cache_put(category, pipe, new_v)
                    return new_v
        # 候选路径（2026-10-12）：品类锁内只把 pipe 状态写入私有暂存目录；
        # 全文哈希 + meta 写入（~340ms）移到锁外；_write_lock 只护版本号分配 + rename。
        # 暂存目录以 "." 开头，list_versions 不可见，失败即删除。
        import tempfile
        from algo.persist import write_snapshot_state, finalize_snapshot, set_snapshot_version
        cat_dir = self._cat_dir(category)
        # 崩溃残留的巩固暂存（不可见、不可恢复，仅占盘）；正常存活仅数秒，
        # 按 10 分钟陈旧阈值清理，避免误删同目录另一引擎正在写的暂存。
        import glob
        for old in glob.glob(os.path.join(cat_dir, ".consolidate-*")):
            try:
                if time.time() - os.path.getmtime(old) > 600:
                    shutil.rmtree(old, ignore_errors=True)
            except OSError:
                pass
        stage = tempfile.mkdtemp(prefix=".consolidate-", dir=cat_dir)
        try:
            with self._pipe_lock(category):
                pipe, active = self._pipe_entry(category)
                self._require_current_representation(pipe, "巩固")
                meta = write_snapshot_state(pipe, stage, parent_version=active,
                                            trigger="consolidate", note=note,
                                            ssocl_cfg=self.ssocl_cfg)
                pipe.production_feature_metadata = json.loads(
                    json.dumps(meta["production_feature_metadata"], ensure_ascii=False))
                wal_mark = self._journal_mark(category)
            finalize_snapshot(stage, meta, version=0)
            with self._write_lock:
                vs = list_versions(cat_dir)
                new_v = max(vs) + 1 if vs else 1
                set_snapshot_version(stage, new_v)  # meta.json 不参与 artifact 哈希
                os.rename(stage, self._v_dir(category, new_v))
                stage = None
        finally:
            if stage is not None and os.path.isdir(stage):
                shutil.rmtree(stage, ignore_errors=True)
        # 候选快照不得进入生产推理缓存；保留 WAL，避免候选未激活前的在线反馈
        # 在重启/驱逐后丢失。与冷装载一致：先恢复同版本 WAL 再发布缓存，否则
        # 巩固后暂时丢失反馈、LRU/重启又恢复，同版本预测随缓存变化。
        # 2026-10-12：load_snapshot+重放（秒级）移出品类锁，避免阻塞同品类推理；
        # 锁内只复查“缓存条目与 WAL 均未变化”后原子发布，变化则保留现缓存
        # （现缓存已含全部在线学习，下次冷装载自然按 WAL 重建）。
        self._learning_yield()  # 串行模式：锁外重装载/重放前让行前台推理
        active_pipe = load_snapshot(self._v_dir(category, active), self.device)
        self._learning_yield()
        self._replay_journal(category, active_pipe, active)
        with self._pipe_lock(category):
            with self._write_lock:
                entry = self._pipes.get(category)
                if (entry is not None and entry[0] is pipe and entry[1] == active
                        and self._journal_mark(category) == wal_mark):
                    self._cache_put(category, active_pipe, active)
                else:
                    logger.info("[engine] %s 巩固期间缓存/WAL 已变化，保留现缓存", category)
        return new_v

    # ----- 评估 / 贡献档案 / 学习曲线 -----
    # 注（2026-08-30 并发修复）：evaluate/contribution_report 是手动触发的
    # 批量只读评估（内部逐图 predict，只读不改 pipe 状态），刻意**不**整程持
    # 品类锁——否则同品类产线 predict 会被阻塞整个评估时长（可达分钟级），
    # 突破单图 1s 红线。残留风险：与 feedback 即学并发时单图可能读到更新中
    # 的权重（与 predict×predict 并发同级，可接受）；需严格一致的评估请用
    # simulate_learning 的独立快照副本路径。
    def evaluate(self, category: str, split_items: List[Tuple[str, int]]) -> Dict[str, Any]:
        pipe, _ = self._pipe_entry(category)
        return pipe.evaluate(split_items)

    def contribution_report(self, category: str, split_items: List[Tuple[str, int]],
                            out_path: Optional[str] = None) -> Dict[str, Any]:
        pipe, _ = self._pipe_entry(category)
        return pipe.contribution_report(split_items, out_path)

    def simulate_learning(self, category: str, stream_items, eval_items,
                          *, feedback_ratio: float = 0.2, eval_every: int = 5,
                          max_feedback: Optional[int] = None,
                          box_map: Optional[Dict[str, Any]] = None,
                          progress_cb=None) -> Dict[str, Any]:
        """离线回放：用给定 stream + eval 产出学习曲线（不影响线上 pipe）。

        2026-08-29 走查修复：原实现整个回放（加载快照 + 模拟全程数分钟）
        持有 _write_lock，而产线 predict 每次都要经 _ensure_loaded 抢同一把
        锁 → 回放期间产线实质停摆且 UI 无提示。快照版本目录不可变、回放用
        独立 pipe 副本，锁只需护住 current_version 元信息读取；load 与模拟
        移出锁外，回放与产线并行（产线单图延迟可能因 GPU 抢占略升，属预期）。
        """
        from algo.persist import load_snapshot, current_version
        with self._write_lock:
            cv = current_version(self._cat_dir(category))
        if cv is None:
            raise RuntimeError(f"品类 {category} 未准备")
        # 用独立 pipe 回放，避免串改线上态（锁外：快照目录只读、副本独立）
        tmp = load_snapshot(self._v_dir(category, cv), self.device)
        from algo.ssocl.learning_curve import simulate_learning
        result = simulate_learning(
            tmp, stream_items, eval_items,
            feedback_ratio=feedback_ratio, eval_every=eval_every,
            max_feedback=max_feedback, box_map=box_map,
            progress_cb=progress_cb,
        )
        result["replay_evidence"] = self._replay_evidence(
            category, cv, {"mode": "stream", "feedback_ratio": feedback_ratio,
                           "eval_every": eval_every, "max_feedback": max_feedback,
                           "box_map": bool(box_map)}, stream_items, eval_items)
        return result

    def simulate_learning_rounds(self, category: str, pool_items, eval_items,
                                 *, round_size: int = 30, n_rounds: int = 10,
                                 box_map: Optional[Dict[str, Any]] = None,
                                 relearn_every: int = 1,
                                 eval_per_sample: int = 5,
                                 progress_cb=None) -> Dict[str, Any]:
        """多轮持续学习回放（独立 pipe 副本，锁外加载，与产线并行）。"""
        from algo.persist import load_snapshot, current_version
        with self._write_lock:
            cv = current_version(self._cat_dir(category))
        if cv is None:
            raise RuntimeError(f"品类 {category} 未准备")
        tmp = load_snapshot(self._v_dir(category, cv), self.device)
        from algo.ssocl.learning_curve import simulate_learning_rounds
        result = simulate_learning_rounds(
            tmp, pool_items, eval_items,
            round_size=round_size, n_rounds=n_rounds,
            box_map=box_map, relearn_every=relearn_every,
            eval_per_sample=eval_per_sample, progress_cb=progress_cb,
        )
        result["replay_evidence"] = self._replay_evidence(
            category, cv, {"mode": "rounds", "round_size": round_size,
                           "n_rounds": n_rounds, "relearn_every": relearn_every,
                           "eval_per_sample": eval_per_sample, "box_map": bool(box_map)},
            pool_items, eval_items)
        return result

    def _replay_evidence(self, category, parent_version, config, *datasets):
        """生成独立回放证据；仅内存返回，不写生产 DB/快照。"""
        import hashlib
        import uuid
        from algo.persist import current_version
        with self._write_lock:
            actual = current_version(self._cat_dir(category))
        if actual != parent_version:
            raise ValueError("回放父版本已变化")
        items = []
        for dataset in datasets:
            items.extend(dataset)
        summary = [{"path": item[0], "label": item[1]} if isinstance(item, tuple) else item
                   for item in items]
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
        payload = {"category": category, "parent_version": parent_version,
                   "wal_watermark": self._journal_mark(category),
                   "input_summary": {"count": len(summary), "sha256": hashlib.sha256(
                       json.dumps(summary, sort_keys=True, default=str).encode()).hexdigest()},
                   "config_hash": config_hash}
        payload["replay_run_id"] = "replay-" + uuid.uuid4().hex
        return payload

    def weight_history(self, category: str) -> Dict[str, Any]:
        """融合权重演化历史（学习效果页「权重随反馈演化」曲线数据源）。

        快照在 algo 侧 FeedbackHandler.weight_apply/reject 时落点并随快照持久化；
        未准备品类返回 enabled=False 空点列（UI 空态）。"""
        try:
            self._ensure_loaded(category)
        except Exception:
            return {"category": category, "enabled": False, "points": []}
        # 品类锁：weight_history/weights 会被 feedback 即学并发修改，持锁读一致快照
        with self._pipe_lock(category):
            pipe, version = self._pipe_entry(category)
            h = getattr(pipe, "handler", None)
            pts = list(getattr(h, "weight_history", []) or []) if h else []
            return {"category": category, "enabled": h is not None,
                    "version": f"v{version}", "points": pts,
                    "current": {k: float(v) for k, v in
                                (getattr(pipe, "weights", {}) or {}).items()},
                    "n_labeled_fb": int(getattr(h, "_n_labeled_fb", 0)) if h else 0}

    def slot_health(self, category: str) -> Dict[str, Any]:
        """AC-025/026 槽位健康页：能力清单+运行时状态+cde_report 摘要+更新准入审计。
        只读、零前向；未准备品类返回 enabled=False（UI 空态）。"""
        from algo.fusion.cde_report import slot_health
        try:
            self._ensure_loaded(category)
        except Exception:
            return {"category": category, "enabled": False}
        with self._pipe_lock(category):
            pipe, version = self._pipe_entry(category)
            out = slot_health(pipe)
            rep = getattr(pipe, "cde_report", None) or {}
            out["router_admission"] = rep.get("router_admission")
            out["domain_shift"] = rep.get("domain_shift")
            out["fusion_baselines"] = rep.get("fusion_baselines")
            h = getattr(pipe, "handler", None)
            out["update_audit"] = (h.update_audit_report()
                                   if h is not None and hasattr(h, "update_audit_report")
                                   else None)
            out.update({"category": category, "enabled": True, "version": f"v{version}"})
            from algo.api.trace import _jsonable
            return _jsonable(out)


# ---- 进程级单例（线程安全，仿 demo1_adapter.get_registry 模式） ----
_engine: Optional[DetectionEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> DetectionEngine:
    """DetectionEngine 进程单例；storage/base_cfg 取自 Settings。"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from ..core.config import get_settings
                settings = get_settings()
                base_cfg = settings.engine_base_cfg
                if base_cfg is None:
                    raise RuntimeError(
                        "引擎基础配置不存在：请在 configs/default.yaml 的 "
                        "system.engine_base_cfg 配置有效的 engine_fast.yaml 路径")
                _engine = DetectionEngine(str(settings.engine_storage),
                                      str(base_cfg),
                                      max_cached_categories=settings.get(
                                          "system", "engine_max_cached_categories", 8))
    return _engine
