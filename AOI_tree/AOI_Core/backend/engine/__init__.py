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
import shutil
import threading
import time
import json
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
    heatmap_paths: Dict[str, str] = field(default_factory=dict)  # 渲染后落盘的路径
    types: List[Dict[str, Any]] = field(default_factory=list)     # 规则化缺陷类型归因

    latency_ms: float = 0.0                 # 本次推理全耗时
    n_tiles: int = 0                        # tile 数
    triggered_slot: str = ""                # 主导槽位（命中的最高权重高分槽）
    open_alert: bool = False                # 不完备哨兵
    align_offset: Optional[List[float]] = None   # M11d 对位偏移 (dx,dy)（参考分辨率像素）
    align_warn: bool = False                # M11d 偏移超阈值预警
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
    return cfg


# ---- 引擎主类 ----
class DetectionEngine:
    """按品类管理 pipeline 实例（进程单例，线程锁）。"""

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
                              parent_version=None, trigger=trigger,
                              note=f"scenario={scenario}, profile={profile}. {note}",
                              ssocl_cfg=self.ssocl_cfg)
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
        try:
            os.makedirs(self._cat_dir(category), exist_ok=True)
            with open(self._journal_path(category), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001 日志失败不影响学习本身，但必须留痕
            print(f"[engine][警告] 学习日志写入失败（{category}）：{e}", flush=True)

    def _replay_journal(self, category: str, pipe, version: int) -> int:
        """装载快照后重放未巩固的反馈学习（仅 base == 当前版本的条目）。

        consolidate 成功后日志清空（学习已固化进新版本）；手动回滚/激活
        到其它版本时该版本线上的条目自然被 base 过滤，不会误重放。
        """
        path = self._journal_path(category)
        if not os.path.exists(path):
            return 0
        n_ok = n_skip = 0
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for ln in lines:
            try:
                rec = json.loads(ln)
            except Exception:  # noqa: BLE001 坏行跳过
                continue
            if rec.get("base") != version:
                continue
            p = rec.get("path")
            if not p or not os.path.exists(p):
                n_skip += 1
                continue
            try:
                pipe.feedback(p, verdict=rec.get("verdict"),
                              label=rec.get("label"), box=rec.get("box"))
                n_ok += 1
            except Exception:  # noqa: BLE001 单条重放失败不阻塞装载
                n_skip += 1
        if n_ok or n_skip:
            print(f"[engine] {category}: 重放未巩固学习 {n_ok} 条"
                  + (f"（跳过 {n_skip} 条：图像缺失/重放失败）" if n_skip else ""),
                  flush=True)
        return n_ok

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
            hm = getattr(pipe, "head_mgr", None)
            min_samples = int(hm.cfg.get("min_samples", 8)) if hm else 8
            out["incubate"] = {
                "trained": bool(hm is not None and hm.head is not None),
                "min_samples": min_samples,
                "n_pos": int(hm.head.n_pos) if (hm and hm.head) else 0,
                "progress": round(min(1.0, n_def / max(min_samples, 1)), 2),
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

    # ----- 激活 / 回滚 / 变体 -----
    def activate(self, category: str, version: int) -> None:
        from algo.persist import load_snapshot, set_current_version, list_versions
        with self._write_lock:
            if version not in list_versions(self._cat_dir(category)):
                raise ValueError(f"版本 v{version} 不存在")
            pipe = load_snapshot(self._v_dir(category, version), self.device)
            self._cache_put(category, pipe, version)
            set_current_version(self._cat_dir(category), version)

    def rollback(self, category: str, target_version: Optional[int] = None) -> int:
        """回滚：target_version 给定时直接指；否则取次新版本。"""
        vs = self.list_versions(category)
        if not vs:
            raise RuntimeError(f"品类 {category} 无快照可回滚")
        if target_version is None:
            cv = self.current_version(category) or vs[-1]
            idx = vs.index(cv) if cv in vs else len(vs) - 1
            target = vs[max(0, idx - 1)]
        else:
            target = target_version
        self.activate(category, target)
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
        with self._pipe_lock(category):
            pipe, _ = self._pipe_entry(category)
            t0 = time.perf_counter()
            rec = pipe.predict(path)
            dt_ms = (time.perf_counter() - t0) * 1000
            return self._to_result(pipe, rec, dt_ms)

    def predict_ndarray(self, category: str, rgb: np.ndarray, path_hint: Optional[str] = None) -> DetectionResult:
        self._ensure_loaded(category)
        # 品类锁：与同品类 feedback 即学/consolidate 串行（不同品类并行不互斥）
        with self._pipe_lock(category):
            pipe, _ = self._pipe_entry(category)
            t0 = time.perf_counter()
            rec = pipe.predict_frame(rgb, path=path_hint)
            dt_ms = (time.perf_counter() - t0) * 1000
            return self._to_result(pipe, rec, dt_ms)

    def _to_result(self, pipe, rec: Dict[str, Any], dt_ms: float) -> DetectionResult:
        slot_scores = dict(rec.get("slot_scores", {}))
        raw_scores = dict(rec.get("raw_scores", {}))
        weights = dict(getattr(pipe, "weights", {}) or {})
        router_w = dict(rec.get("router_w", {}) or {})

        decision = rec.get("decision", "normal")
        is_anomaly = (decision == "anomaly")  # gray 不直接当异常，UI 可另显橙
        boxes = []
        for b in rec.get("boxes", []) or []:
            boxes.append({"bbox": list(b.get("bbox", [0, 0, 0, 0])),
                          "area": int(b.get("area", 0)),
                          "dominant_slot": self._dominant_slot(slot_scores, weights)})

        triggered_slot = self._dominant_slot(slot_scores, weights)

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
            types=list(rec.get("types", []) or []),
            latency_ms=float(dt_ms),
            n_tiles=int(rec.get("n_tiles", 0)),
            triggered_slot=triggered_slot,
            open_alert=bool(rec.get("open_alert", False)),
            align_offset=(list(rec["align_offset"]) if rec.get("align_offset") else None),
            align_warn=bool(rec.get("align_warn", False)),
            extra={"decision_trace": list(rec.get("decision_trace", []) or [])},
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
                        feedback_id: Optional[int] = None) -> Dict[str, Any]:
        """verdict ∈ {correct, wrong, none}; label ∈ {0=正常, 1=缺陷, None}。"""
        self._ensure_loaded(category)
        timeout_s = (timeout_ms or self.feedback_timeout_ms) / 1000.0
        done: Dict[str, Any] = {}

        def _learn():
            # 品类锁：feedback 即学会改 NormalBank/融合权重/缺陷样例库，
            # 必须与同品类 predict 串行，否则 predict 读到撕裂的中间态
            try:
                with self._pipe_lock(category):
                    pipe, active_v = self._pipe_entry(category)
                    t0 = time.perf_counter()
                    entry = pipe.feedback(image_path, verdict=verdict, label=label, box=box)
                    done["dt_ms"] = (time.perf_counter() - t0) * 1000
                    # WAL：学习成功才记日志（崩溃/重启可重放，见 _replay_journal）
                    self._append_journal(category, {
                        "t": time.time(), "base": active_v, "path": image_path,
                        "verdict": verdict, "label": label, "box": box,
                        "feedback_id": feedback_id, "category": category})
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
        entry = done.get("entry")
        info = {"path": image_path, "verdict": verdict, "label": label,
                "feedback_ms": round(done.get("dt_ms", 0.0), 2)}
        if isinstance(entry, dict):
            # 2026-08-30：透出 handler 真实动作键（action/drift/boost）——
            # 原透传的 triggered_* 键在 handler entry 中从不存在（死键），
            # 导致 UI 恒显"已学习"，熔断拦截/待成簇等状态静默（MPDD 评测
            # 静默失败根因之一）。
            for k in ("action", "drift", "boost", "defect_ratio"):
                if k in entry:
                    info[k] = entry[k]
        if done.get("fuse_blocked") is not None:
            info["fuse_blocked"] = done["fuse_blocked"]
        return info

    # ----- 巩固（落盘新版） -----
    def consolidate(self, category: str, *, note: str = "") -> int:
        """把当前内存态（已在线更新）固化为新版本。用于 A-B 对比与回滚。"""
        from algo.persist import save_snapshot, set_current_version, list_versions
        # 2026-08-30 并发修复：
        # 1) _ensure_loaded 移出 _write_lock（已加载品类快路径不抢锁）；
        # 2) save_snapshot 读取 pipe 内存态（NormalBank/融合权重可能正被
        #    feedback 即学修改），必须持品类锁与同品类 predict/feedback 串行，
        #    否则快照是撕裂的中间态；
        # 3) 耗时 IO 不持全局 _write_lock 阻塞推理——其它品类实时检测不受影响
        #    （_write_lock 仅剩版本号分配/指针切换等管理操作的互斥）。
        self._ensure_loaded(category)
        with self._pipe_lock(category):
            pipe, active = self._pipe_entry(category)
            cat_dir = self._cat_dir(category)
            with self._write_lock:
                vs = list_versions(cat_dir)
                new_v = max(vs) + 1 if vs else 1
                snap = self._v_dir(category, new_v)
                save_snapshot(pipe, snap, version=new_v,
                              parent_version=active, trigger="consolidate",
                              note=note, ssocl_cfg=self.ssocl_cfg)
                set_current_version(cat_dir, new_v)
                # 学习已固化进新版本 → 清空 WAL（重放只对新版本之后的条目有意义）
                self._clear_journal(category)
                self._cache_put(category, pipe, new_v)
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
        return simulate_learning(
            tmp, stream_items, eval_items,
            feedback_ratio=feedback_ratio, eval_every=eval_every,
            max_feedback=max_feedback, box_map=box_map,
            progress_cb=progress_cb,
        )

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
        return simulate_learning_rounds(
            tmp, pool_items, eval_items,
            round_size=round_size, n_rounds=n_rounds,
            box_map=box_map, relearn_every=relearn_every,
            eval_per_sample=eval_per_sample, progress_cb=progress_cb,
        )

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
