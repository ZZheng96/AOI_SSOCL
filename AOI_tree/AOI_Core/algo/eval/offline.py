"""M0 离线评测管线：fit(train/good+30缺陷) -> CDF(train/good) -> eval(val/test)

全程经 guard_bundle 红线断言；每步落盘追溯。
"""
import os
import time
import json
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import cv2

from ..common.isolation import guard_bundle
from ..common.tiling import compute_tiles, extract_tiles, block_aggregate
from ..common.io import load_image
from ..common.preprocess import preprocess_image
from ..fusion.calibrate import CDFCalibrator, TplCalibrator
from ..fusion.fixed import fuse
from ..decide import Decider, TraceLogger
from .metrics import image_metrics


def _global_clf_path(cfg):
    """U96 全局类型判别器路径：cfg.attribution.global_clf 覆盖（不存在则回落），
    默认按 DINO 同款优先级查找：<包根>/assets → <包根>/../assets（AOI_tree 三组件
    共享一份）→ algo/assets（旧位置兼容）。pkl 为算法工程 scripts/train_defect_classifier.py
    生成的跨品类 508 张 init_defect 随机森林。"""
    p = (cfg.get("attribution") or {}).get("global_clf")
    if p and os.path.exists(p):
        return p
    from ..common.safe_pickle import asset_roots
    cands = [os.path.join(r, "defect_clf.pkl") for r in asset_roots()]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[-1]


def _load_or_train_defect_clf(cfg, defect_paths, def_items, seed=42):
    """U96（2026-08-24）：有监督类型判别器——优先加载全局预训练模型（跨品类
    聚合 init_defect 训练，解决 per-category 30 张样本太少导致的色彩归因 0 /
    外观归因 0.30），不存在时回退 per-category 训练。predict 复用 item["feats"]
    零额外前向。"""
    import pickle as _pickle  # noqa: F401 训练分支 pickle.dump 仍用
    from ..decision import defect_classifier as _dc  # noqa: F401 注册 src.* pkl 别名
    clf_path = _global_clf_path(cfg)
    if os.path.exists(clf_path):
        try:
            from ..common.safe_pickle import safe_pickle_load
            clf = safe_pickle_load(clf_path)  # A10：白名单+留痕（自定义
            # global_clf 路径须在白名单根内，或经 AOI_PICKLE_EXTRA_ROOTS 追加）
            print(f"  [fit] 加载全局类型判别器（{clf_path}）", flush=True)
            return clf
        except Exception as e:
            print(f"  [fit] 加载类型判别器失败（回退规则映射）: {e}", flush=True)
            return None
    if len(def_items) >= 5:
        try:
            from ..decision.defect_classifier import DefectClassifier
            clf = DefectClassifier(seed=seed)
            clf.fit(defect_paths,
                    [it["img"] for it in def_items],
                    [it["feats"] for it in def_items])
            print(f"  [fit] 类型判别器{'完成' if clf.fitted else '单类跳过'} "
                  f"({len(def_items)} 缺陷)", flush=True)
            return clf
        except Exception as e:
            print(f"  [fit] 类型判别器失败（回退规则映射）: {e}", flush=True)
    return None


class Pipeline:
    def __init__(self, cfg, backbone, slots):
        self.cfg = cfg
        self.backbone = backbone
        self.slots = {s.name: s for s in slots if s is not None}
        self.calibrators = {}
        self.decider = None
        self.defect_clf = None       # U96 有监督类型判别器（fit 或 load 时注入）
        self._last_eval = None
        t = cfg["tiling"]
        self.tiling = t
        # ---- M3 SSOCL（§7）：三类反馈 + 双库制 + 主动学习 ----
        self.handler = None          # FeedbackHandler（enable_ssocl 后非空）
        self.selector = None         # ActiveSelector
        self.head_mgr = None         # AHL 孵育头管理（§7.6）
        self._item_cache = {}        # path -> item（特征缓存，零重算工程前提 §7.3）
        self._item_cache_max = 128   # 缓存上限（回放场景由 learning_curve 调大）
        self._cal_normal_hist = {}   # 槽位 CDF 拟合用正常分数历史（回流重估）
        self._router_snapshot = None
        self.trace_api = None        # M4 追溯 API（§9.2，attach_trace_api 后启用）
        # ---- M11d 对位偏移预警（评审 §11.1）：fit 锚定首张训练正常图为参考，
        # predict 相位相关求 (dx,dy)，超阈值透出 align_warn（治具/传送带漂移早警）----
        self._align_ref = None       # 256² 灰度 float32 参考图（persist: align_ref.npy）
        # U108（2026-08-25，M13 同步）：初始融合权重质量（train 域 fused AUROC，
        # 协议内 fit 数据）——在线权重学习自适应启停判据（persist: decider.json）
        self.train_auroc = None
        # ---- U100 TTA（2026-08-24 引擎，M12b 同步）：对位偏移鲁棒性——判 anomaly/gray
        # 时做平移搜索取最小 fused，若找到"对齐"低分（< tau_gray）改判 normal
        #（修复 U95 平移 10px 误报 93%）。
        # 红线预估：触发时 = 4 次额外完整前向，fast profile 基线 ~200-320ms →
        # 最坏 ~1.6s 超 1s 红线 → 默认关；仅 accuracy 模式/复检工位（节拍允许）开启。----
        tta = cfg.get("decision", {}).get("tta", {})
        self.tta_enabled = bool(tta.get("enabled", False))
        self.tta_shift = int(tta.get("shift", 10))
        self.tta_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        # ---- E_open 开放通道（§3.3 完备性）：不完备报告哨兵，不进融合 ----
        self.open_det = None
        self.open_cal = None
        self.open_thresh = 0.9       # open 校准分超阈值 → 未解释信号
        # ---- 速度工程（U56，2026-08-16）：CPU 槽位（trad/blob/layout）与 GPU
        # 槽位线程并行——trad 特征提取 ~120ms 是最大单槽开销，纯 CPU 计算，
        # 与 DINO 前向/GPU 槽位重叠可把单图推理从 ~240ms 压到 ~160ms。
        # 线程安全：trad._vec 用全局 np.random（seed(0)+restore），由 _trad_lock
        # 串行化；blob/layout 无全局 RNG，直接并行。数值与串行版完全一致。----
        self._cpu_exec = ThreadPoolExecutor(max_workers=2)
        self._trad_lock = threading.Lock()
        # ---- 图像预处理（augment.preprocess，§15.29）：训练/检测同口径 ----
        self._pre_cfg = (cfg.get("augment", {}) or {}).get("preprocess", {}) or {}
        # ---- 学习机制消融开关（U24 诊断，§7.4 学习曲线攻坚）----
        self._ablate_intercept = False   # 关缺陷样例拦截加分
        self._ablate_reflow = False      # 关正常回流（含 sem 扩展与 CDF 重估）
        self._ablate_sem = False         # 回流时关 sem 扩展
        self._ablate_cdf = False         # 回流时关 CDF 重估

    def set_ablation(self, intercept=False, reflow=False, sem=False, cdf=False):
        """U24 机制消融：逐项关闭在线更新机制，定位 AUROC 退化源。"""
        self._ablate_intercept = intercept
        self._ablate_reflow = reflow
        self._ablate_sem = sem
        self._ablate_cdf = cdf

    def _preprocess(self, img):
        """图像预处理（augment.preprocess，§15.29）：训练/检测同口径——
        _tile_and_extract（fit 侧）与 predict/predict_frame（推理侧）读图后统一调用，
        确保两侧像素分布一致（灰度/CLAHE/中值，O(像素) CPU 操作，单图几 ms~几十 ms）。
        未启用时零开销直通。"""
        return preprocess_image(img, self._pre_cfg)

    def _tile_and_extract(self, paths, mode=None, masks=False, progress_cb=None):
        """-> list[dict]: 每图 tiles 图像与特征（backbone=None 时 feats=None，纯手工槽位模式）；
        masks=True 时同步提取每 tile 的缺陷位置掩码（load_defect_mask，None=该图无位置标注）；
        progress_cb(done, total) 逐图回报（前端反馈 2026-09-13 #5 进度细化）"""
        mode = mode or self.tiling["mode"]
        # U75 速度红线：backbone 批大小可配（multi-tile 30 块用 32 -> 1 批前向）
        bb = self.cfg.get("backbone", {})
        bb_batch = int(bb.get("extract_batch", 16))
        from ..slots.disc import load_defect_mask
        out = []
        t0 = time.time()
        for i, p in enumerate(paths):
            img = self._preprocess(load_image(p))
            tiles = compute_tiles(img.shape[:2], mode,
                                  self.tiling["tile_size"], self.tiling["stride"])
            tile_imgs = extract_tiles(img, tiles)
            tile_masks = None
            if masks:
                # 缺陷位置掩码按同一 tiling 网格切块（§15.30）；
                # 无掩码图 tile_masks=[None]*n（该图不参与按位置移植）
                m = load_defect_mask(p, img.shape[:2])
                if m is not None:
                    m3 = np.repeat(m[..., None], 3, axis=2)
                    tile_masks = extract_tiles(m3, tiles)[..., 0]
                else:
                    tile_masks = [None] * len(tiles)
            feats = None
            if self.backbone is not None:
                feats = self.backbone.extract_tiles(tile_imgs, batch=bb_batch).cpu()  # 显存：特征驻留 CPU
            out.append({"path": p, "img": img, "tiles": tiles,
                        "tile_imgs": tile_imgs, "tile_masks": tile_masks,
                        "feats": feats})
            if progress_cb is not None:
                progress_cb(i + 1, len(paths))
            if (i + 1) % 20 == 0:
                print(f"    [extract] {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)",
                      flush=True)
        return out

    def _prep_align_ref(self, img):
        """M11d：对位参考预处理——灰度 + 缩到 ref_size 方阵 + float32（相位相关输入）。"""
        size = int((self.cfg.get("alignment", {}) or {}).get("ref_size", 256))
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        return g.astype(np.float32)

    def _align_offset(self, img):
        """相位相关求相对参考图的平移 (dx,dy)（参考分辨率下像素）。
        无参考/异常返回 None；256² 下 <5ms，不破 1s 红线。预警通道故障不影响主检测。"""
        if self._align_ref is None:
            return None
        try:
            cur = self._prep_align_ref(img)
            (dx, dy), _resp = cv2.phaseCorrelate(self._align_ref, cur)
            return (round(float(dx), 2), round(float(dy), 2))
        except Exception:  # noqa: BLE001
            return None

    def _slot_image_score(self, slot, item):
        """统一评分：槽位返回 patch 分数图(T, G, G)，图像分数由 multiscale 聚合（零额外前向）。
        trad/blob 无 patch 图，用 tile 分数 top-k 兜底。"""
        from ..common.tiling import scoremap_multiscale_aggregate
        # U27：整图打分优先（trad：整图 3NN 均值 >> tile 级聚合，diag_trad_score 实证）
        r = slot.score_image(item["img"])
        if r is not None:
            return r
        if slot.needs_dino:
            ts, hms = slot.score_tiles(item["feats"], item["tile_imgs"])
        else:
            ts, hms = slot.score_tiles(None, item["tile_imgs"])
        ms = self.cfg.get("tiling", {})
        if hms and hms[0] is not None and hasattr(hms[0], "shape"):
            # 有 patch 分数图：多尺度聚合（对整图/单 tile 的分数图取 max 再聚合）
            gmap = np.stack([np.asarray(h) for h in hms])
            if gmap.ndim == 3 and len(hms) > 1:
                merged = gmap.max(axis=0)          # 跨 tile 取 max（异常局域性）
            else:
                merged = gmap[0] if gmap.ndim == 3 else gmap
            s = scoremap_multiscale_aggregate(
                merged, ms.get("multiscale_scales", [1, 2, 4]),
                ms.get("multiscale_weights", [0.6, 0.3, 0.1]))
            return s, ts, hms
        return block_aggregate(ts, self.cfg["slots"].get("sem", {}).get("block_topk", 3)), ts, hms

    def fit(self, bundle, progress_cb=None):
        """progress_cb(frac, message)：可选进度回调（前端反馈 2026-09-13 #5 进度细化）。
        frac 为 fit 内部阶段占比 [0,1]，调用方自行映射到任务总进度。"""
        guard_bundle(bundle, self.cfg["protocol"])
        t0 = time.time()

        def _prog(frac, msg):
            if progress_cb is not None:
                progress_cb(min(1.0, max(0.0, frac)), msg)
        # ---- U75 速度红线（2026-08-19）：skip_zero_weight=true 时只 fit/打分/
        # 校准/融合 only_slots 白名单槽位。动机：only_slots=[shead] 主导融合下，
        # 对权重 0 的槽位（trad/sem/disc 等逐块 CPU 计算）照常打分是纯浪费
        # （实测占单图耗时 ~40%，multi-tile 30 块下超 1s 红线）。诊断需要
        # 全槽 per-slot 报告时关此开关重跑。----
        fm0 = self.cfg.get("fusion", {})
        if fm0.get("skip_zero_weight") and fm0.get("only_slots"):
            self._score_slots = [n for n in self.slots if n in fm0["only_slots"]]
            print(f"[speed] skip_zero_weight: 仅保留槽位 {self._score_slots} "
                  f"(跳过其余槽位的 fit/打分/校准)", flush=True)
        else:
            self._score_slots = list(self.slots.keys())
        # U24：fit 全程可复现——固定 torch 全局 RNG + cuDNN 确定性
        #（sem coreset / disc 训练 / DINO 卷积在 GPU 上的非确定性曾导致同 seed 下
        #  fit 结果波动，initial AUROC 漂移 0.35~0.45，消融对比不可信）
        seed = int(self.cfg.get("seed", 42))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)     # U25：demo4 traditional.py 用全局 np.random.choice，必须固定
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False     # U25：TF32 非确定，关闭
        torch.backends.cuda.matmul.allow_tf32 = False
        train_items = self._tile_and_extract(
            bundle["init_normal"],
            progress_cb=(lambda d, t: _prog(0.05 + 0.30 * d / max(1, t),
                                            f"提取正常图特征 {d}/{t}")))
        print(f"  [fit] init_normal 提取 {len(train_items)} 张完成 "
              f"({time.time() - t0:.0f}s)", flush=True)
        # ---- M11d 对位预警参考：锚定第一张训练正常图 ----
        al_cfg = self.cfg.get("alignment", {}) or {}
        if al_cfg.get("enabled", True) and train_items:
            self._align_ref = self._prep_align_ref(train_items[0]["img"])
        ctx = {"train_tile_feats": [it["feats"] for it in train_items],
               "train_tile_imgs": [it["tile_imgs"] for it in train_items],
               "train_imgs": [it["img"] for it in train_items]}   # U27：整图库（trad 主路径）
        def_items = []
        if bundle["init_defect"]:
            def_items = self._tile_and_extract(
                bundle["init_defect"], masks=True,
                progress_cb=(lambda d, t: _prog(0.35 + 0.10 * d / max(1, t),
                                                f"提取缺陷图特征 {d}/{t}")))
            ctx["defect_tile_imgs"] = [it["tile_imgs"] for it in def_items]
            # §15.30：tile 级缺陷位置掩码（按位置抠缺陷移植；None=该图无位置标注）
            ctx["defect_tile_masks"] = [it["tile_masks"] for it in def_items]
            ctx["defect_paths"] = bundle["init_defect"]  # U72: crop 模式 YOLO 框解析用
            # U75 块模式（2026-08-19）：块位置（整图坐标，框重叠比例算块级标签用）
            # + 整图尺寸（YOLO 归一化框 -> 整图像素坐标，multi-tile 下 tile 尺寸≠整图尺寸）
            ctx["defect_tiles"] = [it["tiles"] for it in def_items]
            ctx["defect_img_shapes"] = [it["img"].shape[:2] for it in def_items]
            print(f"  [fit] init_defect 提取 {len(def_items)} 张完成 "
                  f"({time.time() - t0:.0f}s)", flush=True)
            self.defect_clf = _load_or_train_defect_clf(
                self.cfg, bundle["init_defect"], def_items, seed=seed)
        if bundle.get("templates"):       # L3 场景：用户显式模板图（红线：非隐式配对）
            ctx["templates"] = bundle["templates"]
        _fit_slots = [n for n in self.slots if n in self._score_slots]
        for si, name in enumerate(_fit_slots):
            _prog(0.45 + 0.15 * si / max(1, len(_fit_slots)), f"槽位 {name} 拟合")
            self.slots[name].fit(ctx)
            print(f"  [fit] 槽位 {name} 完成 ({time.time() - t0:.0f}s)", flush=True)
        # 红线 3：CDF 与阈值只用 train/good
        _prog(0.60, "正常库打分（校准分布）")
        normal_slot_scores = {n: [] for n in self._score_slots}
        for ii, it in enumerate(train_items):
            for n, slot in self.slots.items():
                if n not in self._score_slots:
                    continue
                s, _, _ = self._slot_image_score(slot, it)
                normal_slot_scores[n].append(s)
            if ii % 5 == 4 or ii == len(train_items) - 1:
                _prog(0.60 + 0.18 * (ii + 1) / max(1, len(train_items)),
                      f"正常库打分 {ii + 1}/{len(train_items)}")
        _prog(0.78, "分数校准")
        for n in self._score_slots:
            # 2026-08-30（P0）：tpl 槽位正常分布退化（train 图在模板库自匹配≈0），
            # CDF 校准会把正常分也映射到 1.0 → 误报；改用线性归一化（正常≈0）。
            if n == "tpl":
                self.calibrators[n] = TplCalibrator().fit(normal_slot_scores[n])
            else:
                self.calibrators[n] = CDFCalibrator(self.cfg["calibration"]["cdf_bins"]) \
                    .fit(normal_slot_scores[n])
            self._cal_normal_hist[n] = list(normal_slot_scores[n])   # M3 回流重估历史
        # ---- E_open 开放通道（§3.3）：不完备报告哨兵（不进融合，红线 3 只用 train/good）----
        oc = self.cfg["slots"].get("open", {})
        if oc.get("enabled"):
            from ..slots.open import OpenDetector
            self.open_det = OpenDetector(
                n_components=oc.get("n_components", 0.95),
                topk_ratio=oc.get("topk_ratio", 0.05))
            self.open_det.fit([it["feats"] for it in train_items])
            open_scores = [self.open_det.score(it["feats"])[0][0] for it in train_items]
            self.open_cal = CDFCalibrator(self.cfg["calibration"]["cdf_bins"]) \
                .fit(open_scores)
            self.open_thresh = oc.get("thresh", 0.9)
        # 槽位体检：init_defect 上各槽位校准 AUROC（协议内训练数据，诚实来源）
        _prog(0.84, "缺陷库打分（槽位体检）")
        self.slot_sanity = {}
        if def_items:
            from sklearn.metrics import roc_auc_score
            defect_slot_scores = {n: [] for n in self._score_slots}
            for it in def_items:
                for n, slot in self.slots.items():
                    if n not in self._score_slots:
                        continue
                    s, _, _ = self._slot_image_score(slot, it)
                    defect_slot_scores[n].append(s)
            for n in self._score_slots:
                cal_n = self.calibrators[n].transform(normal_slot_scores[n])
                cal_d = self.calibrators[n].transform(defect_slot_scores[n])
                self.slot_sanity[n] = round(float(roc_auc_score(
                    [0] * len(cal_n) + [1] * len(cal_d),
                    np.concatenate([cal_n, cal_d]))), 4)
        # 融合权重（§5.1）：默认等权保底；weight_mode=sanity 时用 init_defect 上
        # 的槽位 AUROC 加权（w=max(0,auc-0.5) 归一化，反向槽位置零剔除）。
        # U34 依据：BTAD/MPDD/mvtec 上 sem 是"全品类一致强槽"，sanity 与 test 一致；
        # U26 警示：域差品类（data_local gold_finger）sanity 不可外推，故默认关、按品类验证。
        fm = self.cfg.get("fusion", {})
        _prog(0.92, "融合权重计算")
        if fm.get("weight_mode") == "sanity" and self.slot_sanity:
            w = np.array([max(0.0, self.slot_sanity[n] - 0.5) for n in self._score_slots],
                         dtype=np.float64)
            if w.sum() > 0:
                w = w / w.sum()
                cand = {n: float(wi) for n, wi in zip(self._score_slots, w)}
                # M9b 排查记录（U26/U34 结论复核）：solder_smt AUROC 0.39 的
                # 根因不是 sanity 权重误导——正常侧融合均值恰为 0.50（CDF 校准
                # 按设计工作），且"用锚定集自检"是循环论证（必然 1.0）。真因是
                # 35 张正常图的少样本瓶颈（M9 已量化：20 张时 AUROC 低估 ~0.1）
                # 叠加域差。sold_smt 作为"少样本+域差"真实边界如实记录，不加
                # 无依据的回退机制凑过。
                print("[fusion] sanity 加权: " + " ".join(
                    f"{n}={cand[n]:.3f}" for n in self._score_slots), flush=True)
                # 2026-08-30（P0）：L3 模板场景 tpl 权重提升——tpl 差分是唯一
                # 域不变槽位（gold_finger 全库模板单槽 AUROC=1.0，其余槽位对
                # test 域正常图打高分导致阈值漂移误报）。tpl sanity AUROC 显著
                # 强时权重 ×2 再归一化，让"与金样板一致"信号主导决策。
                if "tpl" in cand and cand.get("tpl", 0) > 0 \
                        and self.slot_sanity.get("tpl", 0) >= 0.9:
                    cand["tpl"] *= 2.0
                    ws = sum(cand.values())
                    if ws > 0:
                        cand = {n: v / ws for n, v in cand.items()}
                    print("[fusion] L3 tpl 权重提升: " + " ".join(
                        f"{n}={cand[n]:.3f}" for n in cand), flush=True)
                self.weights = cand
            else:
                self.weights = {n: 1.0 / len(self._score_slots) for n in self._score_slots}
                print("[fusion] sanity 全 ≤0.5，回退等权", flush=True)
        else:
            self.weights = {n: 1.0 / len(self._score_slots) for n in self._score_slots}
        # ---- U73 白名单融合（shead 主导实验，2026-08-19）：只保留 only_slots 内
        # 槽位，权重在白名单内重新归一化（承接 sanity/等权基础权重），其余置零。
        # 动机：U72v3 shead 单槽 test AUROC=0.8370 但等权 fused 仅 0.4867--
        # 弱槽（disc 0.35/blob 0.35/layout 0.43/trad 0.40）稀释强槽信号，
        # 融合校准是当前最大瓶颈。槽位照常 fit/打分（per-slot 报告完整），
        # 仅融合权重变化。----
        only = fm.get("only_slots")
        if only:
            keep = [n for n in self.slots if n in only]
            if keep:
                s = sum(self.weights.get(n, 0.0) for n in keep)
                self.weights = {n: (self.weights.get(n, 0.0) / s if n in keep else 0.0)
                                for n in self.slots} if s > 0 else \
                               {n: (1.0 / len(keep) if n in keep else 0.0)
                                for n in self.slots}
                print(f"[fusion] only_slots={keep} 权重: " + " ".join(
                    f"{n}={self.weights[n]:.3f}" for n in self.slots
                    if self.weights[n] > 0), flush=True)
        # ---- U108（2026-08-25，M13 同步）：初始融合权重质量（train 域 fused
        # AUROC，协议内 fit 数据）——供在线权重学习自适应启停：易品类初始
        # fused 已优（≥0.99）→ 冻结权重学习防扰动（U105-U107 负增益根因 =
        # 在线权重劣于 fit 权重）；难品类初始不足 → 保留学习。零额外前向
        #（复用 normal/defect 校准分）。----
        self.train_auroc = None
        if def_items and self.weights:
            from sklearn.metrics import roc_auc_score
            names = list(self._score_slots)
            cal_n = {n: self.calibrators[n].transform(normal_slot_scores[n])
                     for n in names}
            cal_d = {n: self.calibrators[n].transform(defect_slot_scores[n])
                     for n in names}
            f_n = np.array([fuse({n: float(cal_n[n][i]) for n in names}, self.weights)
                            for i in range(len(normal_slot_scores[names[0]]))])
            f_d = np.array([fuse({n: float(cal_d[n][i]) for n in names}, self.weights)
                            for i in range(len(defect_slot_scores[names[0]]))])
            self.train_auroc = round(float(roc_auc_score(
                [0] * len(f_n) + [1] * len(f_d),
                np.concatenate([f_n, f_d]))), 4)
            print(f"[fusion] train 域 fused AUROC={self.train_auroc}"
                  f"（初始权重质量，协议内）", flush=True)
        # ---- M2 软路由（§5/§5.1）：L1a 用 init_normal+init_defect 训练 + 门控 ----
        self.router = None
        self.router_gate = None     # {"enabled": bool, "base_auc", "router_auc", ...}
        rcfg = self.cfg.get("router", {})
        if rcfg.get("enabled") and def_items:
            from ..fusion.router import (LinearRouter, train_router_on_matrix,
                                         router_gate)
            from sklearn.metrics import roc_auc_score
            names = sorted(self._score_slots)
            def _feats(slot_scores):
                return np.stack([[float(slot_scores[n][i]) for n in names]
                                 for i in range(len(next(iter(slot_scores.values()))))])
            # 修复：路由训练/推理必须同分布--均用校准分 [0,1]
            # （原代码喂原始分，尺度不一，线性权重被大尺度槽位主导）
            cal_normal = {n: self.calibrators[n].transform(normal_slot_scores[n])
                          for n in self._score_slots}
            cal_defect = {n: self.calibrators[n].transform(defect_slot_scores[n])
                          for n in self._score_slots}
            neg_feats = _feats(cal_normal)   # train/good 校准分矩阵
            pos_feats = _feats(cal_defect)   # init_defect 校准分矩阵
            router = LinearRouter(len(names))
            router, info = train_router_on_matrix(
                router, pos_feats, neg_feats,
                epochs=rcfg.get("epochs", 50), lr=rcfg.get("lr", 1e-3),
                margin=rcfg.get("margin", 0.1),
                lam_entropy=rcfg.get("lam_entropy", 0.1), seed=self.cfg.get("seed", 42))
            # 门控：路由在 init_defect 上 AUC vs 保底等权 AUC（U15：仅保守防线，
            # 是否真增准必须看 test 侧消融）。两侧都用校准分，尺度一致。
            with torch.no_grad():
                r_fused = router.fused(torch.tensor(pos_feats, dtype=torch.float32))[0].numpy()
                r_neg = router.fused(torch.tensor(neg_feats, dtype=torch.float32))[0].numpy()
            b_fused = np.array([fuse({n: float(cal_defect[n][i]) for n in names},
                                     self.weights) for i in range(len(pos_feats))])
            b_fused_all = np.concatenate([
                np.array([fuse({n: float(cal_normal[n][i]) for n in names},
                               self.weights) for i in range(len(neg_feats))]), b_fused])
            r_fused_all = np.concatenate([r_neg, r_fused])
            base_auc = float(roc_auc_score(
                [0] * len(neg_feats) + [1] * len(pos_feats), b_fused_all))
            router_auc = float(roc_auc_score(
                [0] * len(neg_feats) + [1] * len(pos_feats), r_fused_all))
            enabled = router_gate(base_auc, router_auc, rcfg.get("gate_margin", 0.005))
            self.router = router
            self.router_gate = {"enabled": enabled, "base_auc": round(base_auc, 4),
                                "router_auc": round(router_auc, 4),
                                "gate_margin": rcfg.get("gate_margin", 0.005),
                                "train_info": info}
            print(f"[route] base={base_auc:.4f} router={router_auc:.4f} "
                  f"enabled={enabled} (margin={rcfg.get('gate_margin', 0.005)})")
        # ---- M2 共识自适应（§5.3，U14）：无标签槽位权重 ----
        # 权重只用 train/good 校准分计算（无标签路径，不受 30 张 init_defect 限制）。
        # U15 实证（2026-08-15）：门控评估集 init_defect 不可靠——路由 train 门控通过
        # （0.8985）但 test 崩（0.2021 vs 保底 0.3385）。故共识**直接启用**（demo3 同款），
        # init_defect 对比数字仅作报告，不决定启用。
        self.consensus_gate = None
        cc = self.cfg.get("consensus", {})
        # U40（2026-08-16）：weight_mode=sanity 时跳过共识——共识权重混合校准分与
        # 等权（blend 0.5），会把 sanity 加权稀释/覆盖；sanity 与 consensus 二选一。
        if cc.get("enabled") and fm.get("weight_mode") != "sanity":
            from ..fusion.consensus import consensus_weights
            cal_normal = {n: self.calibrators[n].transform(normal_slot_scores[n])
                          for n in self._score_slots}
            cw = consensus_weights(cal_normal, blend=cc.get("blend", 0.5),
                                   hard_thresh=cc.get("hard_thresh", 0.0))
            self.weights = {k: float(v) for k, v in cw.items()}
            gate = {"enabled": True, "blend": cc.get("blend", 0.5),
                    "weights": {k: round(float(v), 4) for k, v in cw.items()}}
            if def_items:  # 仅报告：共识 vs 等权在 init_defect 上对比（U15 不可靠，不拦截）
                from sklearn.metrics import roc_auc_score
                cal_defect = {n: self.calibrators[n].transform(defect_slot_scores[n])
                              for n in self._score_slots}
                names = sorted(self._score_slots)
                n_, p_ = len(next(iter(cal_normal.values()))), len(next(iter(cal_defect.values())))
                eq = {n: 1.0 / len(names) for n in names}
                c_fused = [fuse({n: float(cal_normal[n][i]) for n in names}, cw)
                           for i in range(n_)] + \
                          [fuse({n: float(cal_defect[n][i]) for n in names}, cw)
                           for i in range(p_)]
                e_fused = [fuse({n: float(cal_normal[n][i]) for n in names}, eq)
                           for i in range(n_)] + \
                          [fuse({n: float(cal_defect[n][i]) for n in names}, eq)
                           for i in range(p_)]
                lab = [0] * n_ + [1] * p_
                base_auc = float(roc_auc_score(lab, e_fused))
                cons_auc = float(roc_auc_score(lab, c_fused))
                gate.update({"base_auc": round(base_auc, 4),
                             "consensus_auc": round(cons_auc, 4)})
                print(f"[consensus] (报告) base={base_auc:.4f} consensus={cons_auc:.4f} "
                      f"启用=是 blend={cc.get('blend', 0.5)}")
            self.consensus_gate = gate
        # ---- U113 机制C（2026-08-26，M14 同步）：域差品类域不变权重倾斜 ----
        # 理论：U97 实证 blob（DoG 局部亮度结构）是域不变特征——跨域下唯一稳定
        # 正向槽位（solder blob 0.74 / gold blob 0.82 实测，sem/shead/trad/layout
        # 跨域反向 0.06-0.38），等权/共识融合被反向槽位主导。
        # 机制：train_auroc < 阈值（协议内信号：初始融合明显欠优 = 域差/反向场景）
        # → 融合向域不变特征（blob/disc/inp）倾斜，权重来自 cfg fusion.domain_weights
        # （域不变特征先验，非 test 域反推——红线 test 只验不选）。
        # 位置：consensus 之后（共识覆盖权重，需在其后兜底）、decider 之前。
        dc = fm.get("domain_weights")
        if dc and self.train_auroc is not None \
                and self.train_auroc < fm.get("domain_train_auroc_thresh", 0.85):
            names = list(self._score_slots)
            s = sum(dc.get(n, 0.0) for n in names)
            if s > 0:
                self.weights = {n: (dc.get(n, 0.0) / s) for n in names}
                print(f"[fusion] 机制C 域差倾斜（train_auroc={self.train_auroc}"
                      f"<{fm.get('domain_train_auroc_thresh', 0.85)}）: "
                      + " ".join(f"{n}={self.weights[n]:.3f}" for n in names), flush=True)
        # 阈值只用 train/good：与 predict 实际生效权重同口径（共识启用时即共识权重）
        _prog(0.97, "判定阈值标定")
        fused_normal = []
        for i in range(len(normal_slot_scores[next(iter(self._score_slots))])):
            fused_normal.append(fuse(
                {n: float(self.calibrators[n].transform([normal_slot_scores[n][i]])[0])
                 for n in self._score_slots}, self.weights))
        self.decider = Decider.from_normal_scores(
            fused_normal, self.cfg["decision"]["tau_quantile"], self.cfg["decision"]["tau_gray"])
        self.fit_seconds = time.time() - t0
        # 内存工程（M0 实测：tiles36 全量 train 特征驻留 CPU 可达 5.6GB）：
        # fit 完成后原始图/特征不再被任何槽位引用，立即释放
        self._train_normal_paths = list(bundle["init_normal"])   # M3 enable_ssocl 用
        self.train_items = None
        self.def_items = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def predict(self, path):
        """推理：槽位打分 → 校准 → 融合（+M3 缺陷样例拦截加分）→ 分层输出。
        U56：读图后立即提交 CPU 槽位（trad/blob/layout），DINO 前向期间后台并行。
        2026-08-30 提速：特征缓存命中时跳过 load_image/compute_tiles/
        extract_tiles（缓存内已有 img/tiles/tile_imgs/feats），与 _get_item
        缓存语义一致——离线回放每轮重复推理同图时省掉重复读图/切块，
        打分/校准/融合流程不变，行为一致。"""
        item = self._item_cache.get(path)
        if item is not None:
            pre = self._prefetch_cpu(item)
            item["_clf_imgfeat"] = self._prefetch_clf_imgfeat(item["img"])
            rec = self._record(item, cpu_futures=pre)
            rec = self._finalize(item, rec)
            rec = self._tta_refine(item["img"], rec)
            if self.trace_api is not None:
                self.trace_api.record(rec)
            return rec
        img = self._preprocess(load_image(path))
        tiles = compute_tiles(img.shape[:2], self.tiling["mode"],
                              self.tiling["tile_size"], self.tiling["stride"])
        tile_imgs = extract_tiles(img, tiles)
        pre = self._prefetch_cpu({"img": img, "path": path,
                                  "tiles": tiles, "tile_imgs": tile_imgs})
        item = self._get_item(path, img=img, tiles=tiles, tile_imgs=tile_imgs)
        item["_clf_imgfeat"] = self._prefetch_clf_imgfeat(img)
        rec = self._record(item, cpu_futures=pre)
        rec = self._finalize(item, rec)
        rec = self._tta_refine(img, rec)
        if self.trace_api is not None:
            self.trace_api.record(rec)
        return rec

    def _predict_fused_only(self, path):
        """返回拦截前基础融合分；供需要隔离在线拦截影响的诊断使用。"""
        item = self._item_cache.get(path) or self._get_item(path)
        pre = self._prefetch_cpu(item)
        return float(self._record(item, cpu_futures=pre)["fused"])

    def _predict_final_fused(self, path):
        """返回与正式 predict 相同的最终 fused 分（含 DefectBank/head/TTA）。"""
        return float(self.predict(path)["fused"])

    def predict_frame(self, rgb, path=None):
        """视频帧推理（§6）：numpy 帧（RGB HxWx3 uint8）走同一打分管线。
        U56：同 predict——CPU 槽位与 DINO 前向并行。"""
        rgb = self._preprocess(rgb)
        tiles = compute_tiles(rgb.shape[:2], self.tiling["mode"],
                              self.tiling["tile_size"], self.tiling["stride"])
        tile_imgs = extract_tiles(rgb, tiles)
        pre = self._prefetch_cpu({"img": rgb, "path": path,
                                  "tiles": tiles, "tile_imgs": tile_imgs})
        item = self._frame_item(rgb, path, tiles, tile_imgs)
        item["_clf_imgfeat"] = self._prefetch_clf_imgfeat(rgb)
        rec = self._record(item, cpu_futures=pre)
        rec = self._finalize(item, rec)
        rec = self._tta_refine(rgb, rec)
        if self.trace_api is not None:
            self.trace_api.record(rec)
        return rec

    def _prefetch_clf_imgfeat(self, rgb):
        """U113：defect_clf 图像侧特征（颜色/几何/纹理，实测 ~60ms CPU）提交线程池
        与 DINO 前向并行，仅 anomaly/gray 归因时消费（_attribute_types）。
        _img_features 纯函数无状态，数值与串行完全一致。未 fitted 时返回 None。"""
        if self.defect_clf is None or not getattr(self.defect_clf, "fitted", False):
            return None
        from ..decision.defect_classifier import _img_features
        return self._cpu_exec.submit(_img_features, rgb)

    def _tta_refine(self, rgb, rec):
        """U100：对位偏移 TTA——判 anomaly/gray 时做平移搜索取最小 fused，若找到
        "对齐"低分（< tau_gray）改判 normal（修复 U95 平移 10px 误报 93%）。
        TTA 默认关（cfg decision.tta.enabled），开启时仅对非 normal 样本触发。"""
        if not self.tta_enabled or rec.get("decision") == "normal":
            return rec
        best = self._tta_min_fused(rgb)
        if best < self.decider.tau_gray:
            previous = rec.get("decision")
            fused_before = float(rec.get("fused", best))
            rec["decision"] = "normal"
            rec["fused"] = float(best)
            rec["tta_refined"] = True
            rec["boxes"], rec["mask"], rec["types"] = [], None, []
            rec.setdefault("decision_trace", []).append({
                "stage": "tta",
                "fused_before": fused_before,
                "fused_after": float(best),
                "thresholds": {"tau_gray": float(self.decider.tau_gray),
                               "tau_high": float(self.decider.tau_high)},
                "rule_reason": "tta_alignment_override",
                "status": rec["decision"],
                "previous_status": previous,
            })
        return rec

    def _tta_min_fused(self, rgb):
        """对平移 + 光照/亮度候选打分，返回最小 fused 分数（不含 (0,0)/gamma=1.0 原始）。"""
        best = float("inf")
        for di, dj in self.tta_offsets:
            if di == 0 and dj == 0:
                continue
            dx, dy = di * self.tta_shift, dj * self.tta_shift
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]),
                                     borderMode=cv2.BORDER_REPLICATE)
            best = min(best, self._tta_score_shifted(shifted))
        if self.tta_lighting:                       # U101：光照候选
            for g in self.tta_gammas:
                lit = np.clip((rgb.astype(np.float32) / 255.0) ** g * 255.0,
                              0, 255).astype(np.uint8)
                best = min(best, self._tta_score_shifted(lit))
        if self.tta_brightness:                     # U104：亮度候选
            for d in self.tta_brights:
                lit = np.clip(rgb.astype(np.float32) + d, 0, 255).astype(np.uint8)
                best = min(best, self._tta_score_shifted(lit))
        return best

    def _tta_score_shifted(self, shifted):
        """对单个扰动图打分，返回 fused（复用 _frame_item + _record，无 finalize）。"""
        tiles = compute_tiles(shifted.shape[:2], self.tiling["mode"],
                              self.tiling["tile_size"], self.tiling["stride"])
        tile_imgs = extract_tiles(shifted, tiles)
        pre = self._prefetch_cpu({"img": shifted, "path": None,
                                  "tiles": tiles, "tile_imgs": tile_imgs})
        item = self._frame_item(shifted, None, tiles, tile_imgs)
        rec = self._record(item, cpu_futures=pre)
        return rec["fused"]

    def _finalize(self, item, rec):
        """融合后处理（predict/predict_frame 共用）：拦截加分 + 决策 + 分层输出 + 留痕。"""
        # ---- M3 缺陷样例拦截通道（§7.3）：确认缺陷入库后即时生效 ----
        # U24/U25（2026-08-16）：绝对相似加分污染 AUROC 排序（正常样本被误加分，
        # full 组 gain 为负）；v2 改相对相似——加分 ∝（缺陷库相似−正常库相似），
        # 正常样本被正常库抵消不加分，缺陷侧学习真实反映到 fused/AUROC。
        # U70（2026-08-18）：fused cap 饱和修复——U69 定位 min(fused+boost,1.0)
        # 使 gyudet 正常+拦截样本大量饱和 1.0，0.99/0.95 分位阈值退化为 1.0
        # 决策失效区（阈值重估被缺陷侧门控全否决）。cfg decision.no_cap_fused
        # （默认 false，防影响既有成绩）：true 时不 cap（允许 fused>1.0，决策
        # 用比较运算即可、AUROC 只排序，均不依赖 [0,1]），false 保持原样。
        fused_before_rules = float(rec["fused"])
        boost = 0.0
        if self.handler is not None and not self._ablate_intercept:
            boost = self.handler.defect_bank.intercept_score(
                item["feats"], normal_bank=self.handler.normal_bank)
            if self.head_mgr is not None:
                boost += self.head_mgr.boost(item["feats"])
            rec["boost"] = round(boost, 5)
            fused_boosted = rec["fused"] + boost
            if not self.cfg.get("decision", {}).get("no_cap_fused", False):
                fused_boosted = min(fused_boosted, 1.0)
            rec["fused"] = float(fused_boosted)
        # ---- 分层输出 L2/L3（§6.2）：定位 + 类型归因（仅 anomaly/gray 时做，省时）----
        decision = self.decider.decide(rec["fused"])
        rule_reasons = ["threshold"]
        # 2026-08-30（P0）：L3 模板闸门——tpl 差分是域不变槽位（单槽 AUROC 1.0），
        # 校准分 < 1.0（正常≈0，缺陷≈2）即"与金样板一致"的正常强证据，压过其他
        # 槽位的域差虚高分（test 域正常图 fused 0.44-0.71 > train 域阈值 0.3-0.4）。
        # ③（同日 P0 完善）：小缺陷保护——topk 均值会被 ≤10% 面积的缺陷稀释，
        # 用 tpl 热力图最大 patch 差分（raw）判断"是否存在显著异常 patch"：
        # 仅当全图每个 patch 都与某金样板一致（raw_max < 校准 ref）才强判 normal，
        # 避免把小缺陷（个别 patch 高）误压为 normal。
        # 仅当 tpl 参与融合且无缺陷拦截加分时生效（避免压制缺陷库命中的真实缺陷）。
        if "tpl" in rec["slot_scores"] and decision != "normal" \
                and self.weights.get("tpl", 0) > 0 \
                and rec["slot_scores"]["tpl"] < 1.0 \
                and rec.get("boost", 0.0) < 0.1:
            tpl_hm = (self._last_hms or {}).get("tpl")
            tpl_ref = float(getattr(self.calibrators.get("tpl"), "ref", 1.0))
            hm_ok = True
            if tpl_hm:
                try:
                    hm_max = max(
                        float(np.max(np.abs(np.asarray(h, dtype=np.float64))))
                        for h in tpl_hm if h is not None)
                    hm_ok = hm_max < tpl_ref
                except ValueError:
                    hm_ok = False
            if hm_ok:
                decision = "normal"
                rule_reasons.append("tpl_consistency_override")
        if boost:
            rule_reasons.append("defect_bank_intercept")
        trace = rec.setdefault("decision_trace", [])
        trace[-1].update({
            "fused_before": fused_before_rules,
            "fused_after": float(rec["fused"]),
            "boost": float(boost),
            "thresholds": {"tau_gray": float(self.decider.tau_gray),
                           "tau_high": float(self.decider.tau_high)},
            "rule_reason": rule_reasons,
            "status": decision,
        })
        boxes, mask, types = [], None, []
        if decision != "normal" and getattr(self, "hier", True):
            boxes, mask, types = self._hier(item, rec["slot_scores"])
        rec.update(decision=decision, boxes=boxes, mask=mask, types=types)
        # E_open 不完备报告（§3.3）：open 信号高但命名槽位判 normal → 疑似体系外信号
        if self.open_det is not None and "open_score" in rec:
            rec["open_alert"] = bool(rec["open_score"] >= self.open_thresh
                                     and decision == "normal")
        # M11d 对位偏移预警：有参考时每帧估算平移，超阈值透出 align_warn
        if self._align_ref is not None:
            off = self._align_offset(item["img"])
            if off is not None:
                warn_px = float((self.cfg.get("alignment", {}) or {})
                                .get("warn_px", 4.0))
                rec["align_offset"] = [off[0], off[1]]
                rec["align_warn"] = bool(max(abs(off[0]), abs(off[1])) > warn_px)
        trace[-1]["status"] = decision
        if rec.get("open_alert"):
            trace[-1].setdefault("rule_reason", []).append("open_alert")
        if rec.get("align_warn"):
            trace[-1].setdefault("rule_reason", []).append("alignment_warning")
        if self.handler is not None:          # 回流熔断用在线缺陷率（§7.2）
            # 口径修复（2026-08-30）：只把 decision=="anomaly" 计为缺陷，
            # 与 engine._to_result 的 is_anomaly 口径一致——原 `!= "normal"`
            # 把 gray（待复核）也当缺陷，正常面右移（灰区积压）时缺陷率虚高，
            # 回流熔断被误触发锁死（MPDD 评测 fuse_blocked 恒 true）。
            self.handler.normal_bank._observe_decision(decision == "anomaly")
        return rec

    def attach_trace_api(self, log_dir):
        """M4 追溯 API（§9.2）：挂载后每次 predict 自动记录，供上位机查询。"""
        from ..api.trace import TraceAPI
        self.trace_api = TraceAPI(self, log_dir)
        return self.trace_api

    def _get_item(self, path, img=None, feats=None, tiles=None, tile_imgs=None):
        """特征缓存（§7.3 零重算工程前提）：命中直接返回，未命中提取并入缓存。
        U56：支持外部传入 img/tiles/tile_imgs（predict 里读图后立即提交 CPU 槽位，
        DINO 前向期间 CPU 槽位在后台并行，避免顺序执行）。"""
        if path in self._item_cache:
            return self._item_cache[path]
        if img is None:
            img = self._preprocess(load_image(path))
        if tiles is None or tile_imgs is None:
            tiles = compute_tiles(img.shape[:2], self.tiling["mode"],
                                  self.tiling["tile_size"], self.tiling["stride"])
            tile_imgs = extract_tiles(img, tiles)
        if feats is None:
            bb = self.cfg.get("backbone", {})
            feats = self.backbone.extract_tiles(
                tile_imgs, batch=int(bb.get("extract_batch", 16))).cpu() \
                if self.backbone is not None else None
        item = {"path": path, "img": img, "tiles": tiles,
                "tile_imgs": tile_imgs, "feats": feats}
        if len(self._item_cache) > self._item_cache_max:  # 缓存上限：FIFO 逐出
            self._item_cache.pop(next(iter(self._item_cache)))
        self._item_cache[path] = item
        return item

    def _prefetch_cpu(self, item):
        """U56：提前提交 CPU 槽位（trad/blob/layout，只需 img/tile_imgs，不依赖 DINO）
        到线程池，与主线程的 DINO 前向并行。返回 {slot: future}，_record 复用。
        数值与串行版完全一致（trad 由 _trad_lock 保护全局 RNG）。
        U75：skip 模式下非白名单槽位不提交。"""
        pre = {}
        for n, slot in self.slots.items():
            if n not in self._score_slots:
                continue
            if not slot.needs_dino:
                if n == "trad":
                    pre[n] = self._cpu_exec.submit(self._score_trad, slot, item)
                else:
                    pre[n] = self._cpu_exec.submit(self._slot_image_score, slot, item)
        return pre

    def _frame_item(self, rgb, path=None, tiles=None, tile_imgs=None):
        """从 numpy 帧构造 item（不走 load_image/缓存）。"""
        if tiles is None or tile_imgs is None:
            tiles = compute_tiles(rgb.shape[:2], self.tiling["mode"],
                                  self.tiling["tile_size"], self.tiling["stride"])
            tile_imgs = extract_tiles(rgb, tiles)
        feats = None
        if self.backbone is not None:
            bb = self.cfg.get("backbone", {})
            feats = self.backbone.extract_tiles(
                tile_imgs, batch=int(bb.get("extract_batch", 16))).cpu()
        return {"path": path, "img": rgb, "tiles": tiles,
                "tile_imgs": tile_imgs, "feats": feats}

    def _score_trad(self, slot, item):
        """U56：trad 提取用全局 np.random（seed(0)+restore），锁串行化防线程间污染。"""
        with self._trad_lock:
            return self._slot_image_score(slot, item)

    def _record(self, item, cpu_futures=None):
        """打分 + 校准 + 基础融合（拦截前）。M3 反馈/门控/归因共用，避免重复前向。
        U56：CPU 槽位（trad/blob/layout，needs_dino=False）提交线程池与 GPU 槽位并行；
        cpu_futures 由 predict 的 _prefetch_cpu 提供（DINO 前向期间已后台算完），
        数值与串行完全一致（trad 由 _trad_lock 保护全局 RNG）。"""
        slot_scores, raw = {}, {}
        self._last_hms = {}                        # 热力图缓存（_hier 复用，避免重复打分）
        if cpu_futures is not None:
            futures = dict(cpu_futures)            # 已由 _prefetch_cpu 提交（DINO 期间并行）
        else:
            futures = {}
            for n, slot in self.slots.items():
                if n not in self._score_slots:     # U75 skip：非融合槽位不打分
                    continue
                if not slot.needs_dino:            # CPU 槽位：线程并行
                    if n == "trad":
                        futures[n] = self._cpu_exec.submit(self._score_trad, slot, item)
                    else:
                        futures[n] = self._cpu_exec.submit(self._slot_image_score, slot, item)
        for n, slot in self.slots.items():         # GPU 槽位：主线程顺序算（与 CPU 并行）
            if n in futures or n not in self._score_slots:
                continue
            s, ts, hms = self._slot_image_score(slot, item)
            self._last_hms[n] = hms
            raw[n] = s
            slot_scores[n] = float(self.calibrators[n].transform([s])[0])
        for n, fut in futures.items():             # 回收 CPU 槽位结果
            s, ts, hms = fut.result()
            self._last_hms[n] = hms
            raw[n] = s
            slot_scores[n] = float(self.calibrators[n].transform([s])[0])
        # 融合分 = self.weights（共识启用时即共识权重，否则等权保底）。
        # 路由仅追溯（U15 实证：门控评估集 init_defect 不可靠，路由 test 侧劣化
        # 0.2021 vs 保底 0.3385），不参与融合分——router_w 落盘供解释。
        if self.router is not None:
            names = sorted(slot_scores)   # U75：skip 模式下只有白名单槽位有分数
            # 路由仅追溯：用 router 所在 device 构造张量，避免持久化加载时 CPU/CUDA 跨设备错配
            try:
                dev = next(self.router.parameters()).device
            except Exception:
                dev = torch.device("cpu")
            svec = torch.tensor([[slot_scores[n] for n in names]], dtype=torch.float32, device=dev)
            with torch.no_grad():
                _, w = self.router.fused(svec)
            self._last_router_w = {n: float(w[0, i].detach().cpu().item()) for i, n in enumerate(names)}
        else:
            self._last_router_w = dict(self.weights)
        fused = fuse(slot_scores, self.weights)
        rec = {"path": item["path"], "fused": float(fused), "slot_scores": slot_scores,
               "raw_scores": raw, "router_w": self._last_router_w,
               "n_tiles": len(item.get("tiles", [])),

               "decision_trace": [{
                   "stage": "fusion",
                   "raw_scores": dict(raw),
                   "calibrated_scores": dict(slot_scores),
                   "effective_weights": {k: float(v) for k, v in self.weights.items()
                                         if k in slot_scores},
                   "fused_before": float(fused),
                   "fused_after": float(fused),
                   "rule_reason": "base_fusion",
               }]}
        # E_open 哨兵（§3.3）：未解释信号分（不进融合）
        if self.open_det is not None and item["feats"] is not None:
            try:
                oraw, _ = self.open_det.score(item["feats"])
                rec["open_raw"] = float(oraw[0])
                rec["open_score"] = float(self.open_cal.transform([oraw[0]])[0])
            except Exception:
                rec["open_score"] = 0.0
        return rec

    def _hier(self, item, slot_scores, hms=None):
        """分层输出 L2/L3（§6.2）：融合热力图 → 掩码 → 检测框 → 类型归因。
        hms 由 _record 缓存提供（避免重复打分）；掩码在低分辨率工作网格构建
        （热力图本为 patch 级，work_side 下语义等价），框坐标放大回原图——
        3000×4096 上 hier 从 711ms（重算）→ 缓存 347ms → 本优化后再降。"""
        from ..decision.hierarchical import (build_heatmap_mask, boxes_from_mask,
                                             attribute_type)
        if hms is None:
            hms = getattr(self, "_last_hms", None)
        if hms is None:
            hms = self._slot_heatmaps(item)
        thresh = self.cfg.get("hierarchical", {}).get("mask_thresh", 0.5)
        H, W = item["img"].shape[:2]
        work_side = self.cfg.get("hierarchical", {}).get("work_side", 512)
        scale = min(work_side / max(H, W), 1.0)
        wh, ww = max(1, int(H * scale)), max(1, int(W * scale))
        _, mask_work, per_slot_maps = build_heatmap_mask(
            hms, (wh, ww), thresh, weight_by_slot=self.weights)
        boxes = []
        mask = None
        if mask_work is not None:
            boxes = boxes_from_mask(mask_work, self.cfg.get("hierarchical", {}).get("min_area", 16))
            inv = 1.0 / scale if scale > 0 else 1.0
            for b in boxes:                    # 框坐标放大回原图
                b["bbox"] = [int(v * inv) for v in b["bbox"]]
            # U56：类型归因在低分辨率工作网格上做（2500² 上 connectedComponents 慢 ~40ms），
            # 分布统计语义等价；L2 像素级掩码仍 resize 回原图交付。
            mask = cv2.resize(mask_work, (W, H), interpolation=cv2.INTER_NEAREST)
        types = self._attribute_types(item, slot_scores, per_slot_maps,
                                      mask_work, (wh, ww))
        return boxes, mask, types

    def _attribute_types(self, item, slot_scores, per_slot_maps, mask_work,
                         image_size):
        """U96：类型归因——有监督类型判别器优先（全局预训练模型跨品类泛化，
        缺件/色彩归因从 U93 的 0% 提升到 0.75/0.80），未 fit 时回退槽位签名
        规则映射（U94：layout 违例类型由槽位内部记录传入）。"""
        if self.defect_clf is not None:
            # U113：图像侧特征优先复用预算 future（predict/predict_frame 读图后提交），
            # 未提交（如 TTA 路径）则 None——predict 内部现场算，数值一致。
            fut = item.get("_clf_imgfeat")
            img_feat = fut.result() if fut is not None else None
            res = self.defect_clf.predict(item["img"], item["feats"], img_feat=img_feat)
            if res is not None:
                t, conf = res
                return [{"type": t, "confidence": round(conf, 3),
                         "evidence": f"有监督类型判别器 top-1 (conf={conf:.3f})"}]
        from ..decision.hierarchical import attribute_type
        return attribute_type(slot_scores, per_slot_maps, mask_work, image_size,
                              layout_violation=self._layout_violation())

    def _layout_violation(self):
        """U94：取 layout 槽位的主导违例类型（缺件少件/尺寸偏差/逻辑错误），
        归因层替代掩码启发式。layout 未启用时返回 None（回退旧启发式）。"""
        slot = self.slots.get("layout")
        return getattr(slot, "_last_violation", None) if slot is not None else None

    def _slot_heatmaps(self, item):
        """收集各槽位热力图（§6.2 L2 定位原料）。U75：skip 模式只收白名单槽位。"""
        hms = {}
        for n, slot in self.slots.items():
            if n not in self._score_slots:
                continue
            if slot.needs_dino:
                _, hm = slot.score_tiles(item["feats"], item["tile_imgs"])
            else:
                _, hm = slot.score_tiles(None, item["tile_imgs"])
            hms[n] = hm
        return hms

    # ---- M3 SSOCL（§7）：在线反馈闭环 ----
    def enable_ssocl(self, cfg, train_normal_paths=None, rng=None):
        """fit 后启用：双库制 + 反馈 handler + 主动选样 + 孵育头。
        train_normal_paths: train/good 路径（锚定库与回归门控基准，红线 3 只用正常图）。"""
        from ..ssocl.feedback import FeedbackHandler
        from ..ssocl.active import ActiveSelector
        from ..ssocl.incubate import HeadManager
        self.handler = FeedbackHandler(self, cfg)
        self.selector = ActiveSelector(
            top_n=cfg.get("active", {}).get("top_n", 10))
        self.head_mgr = HeadManager(cfg.get("incubate", {}))
        paths = train_normal_paths or getattr(self, "_train_normal_paths", [])
        # 锚定集抽样提特征（省内存：只提 gate.anchor_size 张，兼作 coreset 种子）
        if paths:
            n = min(cfg.get("gate", {}).get("anchor_size", 20), len(paths))
            rng = rng or np.random.default_rng(self.cfg.get("seed", 42))
            sample = sorted(rng.choice(paths, n, replace=False).tolist())
            self._anchor_items = self._tile_and_extract(sample)
            self.handler.setup(self, paths, rng=rng, anchor_paths=sample)
        return self

    def _anchor_feats(self):
        """锚定特征（NormalBank.seed_core 用）：锚定集抽样 tile 特征拼接。"""
        import torch as _t
        items = getattr(self, "_anchor_items", []) or []
        if not items:
            return None
        return _t.cat([it["feats"] for it in items], dim=0)

    def feedback(self, path, verdict, label=None, feats=None, box=None):
        """三类反馈统一入口（§7.1）：correct/wrong/none + label 0|1。
        box（U64）：用户在线提供的缺陷框标注（归一化 [x0,y0,x1,y1] 或 boxes 列表），
        转发给 FeedbackHandler（label==1 有框时只入框内 patch）。"""
        assert self.handler is not None, "未 enable_ssocl"
        return self.handler.feedback(path, verdict, label=label, feats=feats, box=box)

    def active_enqueue(self, rec, item):
        """灰区样本入主动选样队列（§7.5）。"""
        if self.selector is not None:
            self.selector.enqueue(rec, item)

    def active_select(self, top_n=None):
        """产出 top-N 询问清单（§7.5 主动选样）。"""
        if self.selector is None:
            return []
        return self.selector.select(self, top_n=top_n)

    def _sem_snapshot(self):
        """保存 sem 在线扩展状态，供回流门控失败时完整恢复。"""
        slot = self.slots.get("sem")
        bank = getattr(slot, "bank", None)
        return None if bank is None else bank.clone()

    def _sem_rollback(self, snapshot):
        """恢复 sem 在线扩展状态。"""
        slot = self.slots.get("sem")
        if slot is not None and snapshot is not None:
            slot.bank = snapshot
            return True
        return False

    def _sem_add(self, feats):
        """回流正常样本并入 sem 扩展（双库制生效路径 1）。"""
        if "sem" in self.slots and hasattr(self.slots["sem"], "update_add"):
            try:
                self.slots["sem"].update_add(feats)
            except Exception as e:
                print(f"[ssocl] sem update_add 失败: {e}")

    def _recalibrate_on_reflow(self, item):
        """回流样本重估 CDF（双库制生效路径 2，§7.7 误检序列 2/4）。
        回流样本是操作员确认的正常图 → 合法并入校准分布（§7.4 合法重建正常分布）。

        U26（2026-08-16）：重估前快照校准器（edges）+ hist 长度，gate 否决可回滚。
        域差适应：即使回流样本与 train 正常库不相似（reflow_pending/rejected），
        操作员确认正常也是合法校准证据——校准层纳入，修复 test 域正常样本的
        CDF 饱和（sem/layout 反向的根因）。"""
        self._cal_snapshot = {n: (cal.edges.copy(), len(self._cal_normal_hist.get(n, [])))
                              for n, cal in self.calibrators.items()}
        for n, cal in self.calibrators.items():
            s, _, _ = self._slot_image_score(self.slots[n], item)
            self._cal_normal_hist[n].append(s)
            cal.fit(self._cal_normal_hist[n])

    def _rollback_calibrators(self):
        """U26：CDF 重估被回归门控否决时恢复快照（edges + hist 长度）。"""
        snap = getattr(self, "_cal_snapshot", None)
        if snap is None:
            return False
        for n, (edges, hist_len) in snap.items():
            self.calibrators[n].edges = edges.copy()
            while len(self._cal_normal_hist.get(n, [])) > hist_len:
                self._cal_normal_hist[n].pop()
        self._cal_snapshot = None
        return True

    def _predict_fused_only(self, path):
        """回归门控用：只算基础 fused（拦截前，反映库/校准更新对正常分的影响）。
        2026-08-30 优化：优先用 _anchor_items 的内存特征直算——原实现走
        _get_item(path) 依赖 128 条 FIFO 缓存，快照装载后/检测流挤占时锚定
        item 被逐出，gate.check 20 张重建特征 ~1.1s（reflow_accepted 1.6s
        超红线的大头）；锚定集仅 20 张，线性查找可忽略，gate.check 降至 ~0.1s。"""
        for it in getattr(self, "_anchor_items", None) or []:
            if it.get("path") == path:
                return self._record(it)["fused"]
        return self._record(self._get_item(path))["fused"]

    def _retrain_router(self, difficult, lr=1e-4, epochs=5):
        """批量级路由微调（§7.6）：难例校准分（正）+ 锚定集正常分（负）MIL 小步。"""
        if self.router is None or not self._anchor_items:
            return False
        from ..fusion.router import train_router_on_matrix
        names = sorted(self.slots)
        pos = np.array([[d["slots"][n] for n in names] for d in difficult],
                       dtype=np.float32)
        neg_paths = [it["path"] for it in self._anchor_items]
        neg = np.array([[self._record(self._get_item(p))["slot_scores"][n]
                         for n in names] for p in neg_paths], dtype=np.float32)
        if len(pos) < 2 or len(neg) < 2:
            return False
        self._router_snapshot = [w.detach().clone() for w in self.router.parameters()]
        _, info = train_router_on_matrix(self.router, pos, neg, epochs=epochs, lr=lr,
                                         margin=0.1, lam_entropy=0.1,
                                         seed=self.cfg.get("seed", 42))
        return True

    def _restore_router(self):
        """回归门控失败回滚路由权重。"""
        if self._router_snapshot is not None:
            for w, snap in zip(self.router.parameters(), self._router_snapshot):
                w.data.copy_(snap)
            self._router_snapshot = None
            return True
        return False

    def evaluate(self, split_items, trace_path=None):
        labels, fused_scores = [], []
        per_slot = None              # U75：从首个 rec 初始化（skip 模式只含白名单槽位）
        paths = []
        logger = TraceLogger(trace_path) if trace_path else None
        t0 = time.time()
        for i, (p, y) in enumerate(split_items):
            r = self.predict(p)
            if per_slot is None:
                per_slot = {n: [] for n in r["slot_scores"]}
            labels.append(y)
            paths.append(p)
            fused_scores.append(r["fused"])
            for n in per_slot:
                per_slot[n].append(r["slot_scores"][n])
            if logger:
                logger.log({k: r[k] for k in ("path", "fused", "slot_scores", "decision",
                                              "boxes", "types")})
            if (i + 1) % 20 == 0:
                print(f"  [eval] {i + 1}/{len(split_items)} ({time.time() - t0:.0f}s)",
                      flush=True)
        if logger:
            logger.close()
        if per_slot is None:                       # 空 split 防御
            per_slot = {}
        ms = (time.time() - t0) / max(len(split_items), 1) * 1000
        out = {"fused": image_metrics(labels, fused_scores), "ms_per_image": ms}
        for n in per_slot:
            out[n] = image_metrics(labels, per_slot[n])
        # U31（2026-08-16）：max 融合对照——每样本取最强槽位校准分。
        # MPDD connector 实证：sem=0.998/blob=0.986/trad=0.917，等权 fused 仅 0.643，
        # 弱槽位稀释强槽位。max 融合是"最强槽位兜底"式对照（报告指标，不改决策）。
        if per_slot and len(per_slot) > 1:
            mat = np.stack([per_slot[n] for n in per_slot], axis=1)   # (N, slots)
            out["fused_max"] = image_metrics(labels, mat.max(axis=1))
            # 顺带登记每样本分数供后续离线分析
            self._per_sample = {n: list(per_slot[n]) for n in per_slot}
        self._last_eval = {"labels": labels, "per_slot": per_slot, "paths": paths}
        return out

    def contribution_report(self, split_items, out_path=None):
        """三层评价（§3.2）：在 eval 侧跑贡献档案，不参与 fit/权重"""
        from .contribution import contribution_profile, defect_type_from_path, save_profile
        if self._last_eval is None or self._last_eval["paths"] != [p for p, _ in split_items]:
            self.evaluate(split_items)
        le = self._last_eval
        types = [defect_type_from_path(p) if y == 1 else None
                 for p, y in zip(le["paths"], le["labels"])]
        profile = contribution_profile(le["per_slot"], le["labels"], types, self.weights)
        if out_path:
            save_profile(profile, out_path)
        return profile


def save_report(report, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
