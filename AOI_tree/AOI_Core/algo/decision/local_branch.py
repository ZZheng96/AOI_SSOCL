"""A 块（AC-002~008）：独立的单级局部分支（整图专家保持不变）。

- 视图：compute_local_tiles 确定性重叠覆盖（与全局分数无关，AC-003）；
  训练/校准/推理同一视图，view_signature 落盘并在加载时校验（AC-005）。
- 模型：独立 SemSlot 记忆库（不与全局 sem 共享），只见正常图（红线 3）。
- 校准：正常图分建库集/holdout；块级 CDF + 图级 max-block，两层分别报告（AC-006）。
- 标签：缺陷图按掩码连通域框做面积占比标签（≥pos 正 / ≤neg 负 / 中间丢弃），
  仅用于报告块级 AUROC 与覆盖（AC-005）；伪异常默认关，开启即报错（未实现即不放行）。
- 推理：full=全量块；budget=ROI 必看 + 全局热度优先 + 轮换抽检（AC-007）。
"""
import hashlib
import json
import time

import numpy as np
import torch

from ..common.tiling import (compute_local_tiles, extract_tiles, coverage_check,
                             tile_labels_by_area, TILE_LABEL_POS_CANDIDATE,
                             TILE_LABEL_NEG_CANDIDATE)
from ..fusion.calibrate import CDFCalibrator
from .branch_budget import BudgetScheduler


def local_cfg(cfg):
    return ((cfg or {}).get("tiling", {}) or {}).get("local", {}) or {}


def resolve_tile_size(lc, bounds=None):
    """AC-004：tile_size 为数值时直接用；为 "auto" 时取 bounds 内最大值（块数最少），
    stride 按 stride_ratio（缺省 448/512，候选值）缩放。bounds 缺失时回落 512 候选。
    → (tile_size, stride, source)。"""
    ts = lc.get("tile_size", 512)
    ratio = float(lc.get("stride_ratio", 448.0 / 512.0))
    if str(ts).lower() == "auto":
        if bounds and bounds.get("status") == "ok":
            t = int(bounds["hi"])
            t = max(int(bounds["lo"]), t)
            return t, max(1, int(round(t * ratio))), f"auto_from_bounds[{bounds['lo']},{bounds['hi']}]"
        return 512, 448, "auto_bounds_missing_candidate_512"
    t = int(ts)
    st = lc.get("stride")
    return t, (int(st) if st is not None else max(1, int(round(t * ratio)))), "cfg"


def view_signature(cfg, tile_size=None, stride=None):
    lc = local_cfg(cfg)
    bb = (cfg or {}).get("backbone", {}) or {}
    pre = ((cfg or {}).get("augment", {}) or {}).get("preprocess", {}) or {}
    if tile_size is None:
        tile_size, stride, _ = resolve_tile_size(lc)
    d = {"tile_size": int(tile_size), "stride": int(stride),
         "roi": lc.get("roi"), "backbone": bb.get("name", "dinov2_vits14"),
         "grid": int(bb.get("grid", 32)), "preprocess": pre}
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16], d


def mask_boxes(mask, min_area=4):
    """0/255 掩码 → 连通域框 xyxy。"""
    import cv2
    if mask is None:
        return []
    n, _, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    return [(int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3]))
            for i in range(1, n) if st[i, 4] >= min_area]


def _auroc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if not len(neg) or not len(pos):
        return None
    allv = np.concatenate([neg, pos])
    r = allv.argsort().argsort().astype(float) + 1
    # 平局取平均秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


class LocalBranch:
    def __init__(self, cfg, backbone, device="cuda", bounds=None):
        from ..slots.sem import SemSlot
        self.cfg = cfg
        lc = local_cfg(cfg)
        if lc.get("pseudo", False):
            raise ValueError("tiling.local.pseudo=true：伪异常未实现，默认关（AC-005）")
        self.tile_size, self.stride, self.tile_size_source = resolve_tile_size(lc, bounds)
        self.hold_idx = []
        self.roi = lc.get("roi")
        self.holdout = float(lc.get("cal_holdout", 0.3))
        bud = lc.get("budget", {}) or {}
        self.mode = bud.get("mode", "full")
        self.max_tiles = int(bud.get("max_tiles", 4))
        self.rotate_n = int(bud.get("rotate_n", 1))
        self.conflict_margin = float(bud.get("conflict_margin", 0.15))
        self.veto = bool(lc.get("veto", False))
        self.veto_ratio = float(lc.get("veto_ratio", 0.5))
        self.pos = float(lc.get("label_pos", TILE_LABEL_POS_CANDIDATE))
        self.neg = float(lc.get("label_neg", TILE_LABEL_NEG_CANDIDATE))
        self.backbone = backbone
        self.device = device
        self.slot = SemSlot(lc.get("slot", {}) or {}, device=device)
        self.cal = CDFCalibrator(int((cfg.get("calibration", {}) or {}).get("cdf_bins", 256)))
        self.sig, self.sig_detail = view_signature(cfg, self.tile_size, self.stride)
        self.sig_detail["tile_size_source"] = self.tile_size_source
        self.thresholds = None      # fit_union_thresholds 结果（Pipeline 写入）
        self.report = {}
        self.scheduler = BudgetScheduler(self.max_tiles, self.rotate_n, self.roi)
        self.enabled = True

    # ---------------- 视图 ----------------
    def tiles_for(self, image_size):
        return compute_local_tiles(image_size, self.tile_size, self.stride, self.roi)

    def _feats(self, img, tiles):
        bb = self.cfg.get("backbone", {}) or {}
        return self.backbone.extract_tiles(extract_tiles(img, tiles),
                                           batch=int(bb.get("extract_batch", 16))).cpu()

    # ---------------- 拟合 ----------------
    def fit(self, normal_imgs, defect_imgs=None, defect_masks=None, log=print):
        t0 = time.time()
        n = len(normal_imgs)
        rng = np.random.RandomState(int(self.cfg.get("seed", 42)))
        perm = rng.permutation(n)
        n_hold = int(round(n * self.holdout))
        same_bank = n_hold < 2 or n - n_hold < 2
        bank_idx = perm if same_bank else perm[n_hold:]
        hold_idx = perm if same_bank else perm[:n_hold]
        self.hold_idx = [int(i) for i in hold_idx]
        # 建库：逐图提取后立即子采样 patch（控内存），再交 SemSlot 贪心 coreset
        per_tile = 256
        g = torch.Generator().manual_seed(int(self.cfg.get("seed", 42)))
        chunks, n_tiles = [], []
        for k, i in enumerate(bank_idx):
            tiles = self.tiles_for(normal_imgs[i].shape[:2])
            f = self._feats(normal_imgs[i], tiles)
            T, D, G, _ = f.shape
            flat = f.flatten(2).permute(0, 2, 1).reshape(-1, D)
            keep = torch.randperm(flat.shape[0], generator=g)[:per_tile * T]
            chunks.append(flat[keep].reshape(-1, D, 1, 1).half())
            n_tiles.append(T)
            if (k + 1) % 10 == 0:
                log(f"    [local] 建库 {k + 1}/{len(bank_idx)} ({time.time() - t0:.0f}s)")
        self.slot.fit({"train_tile_feats": [torch.cat(chunks, 0).float()]})
        del chunks
        # 校准：holdout 正常块分数
        blk, img_max = [], []
        for i in hold_idx:
            tiles = self.tiles_for(normal_imgs[i].shape[:2])
            s, _ = self.slot.score_tiles(self._feats(normal_imgs[i], tiles))
            blk.extend(map(float, s))
            img_max.append(float(np.max(s)))
        self.cal.fit(blk)
        self.normal_block_raw = blk
        self.normal_image_raw = img_max
        rep = {"n_bank": int(len(bank_idx)), "n_holdout": int(len(hold_idx)),
               "same_bank_holdout": bool(same_bank),
               "same_bank_note": ("正常图不足以分出 holdout，校准与建库同集，"
                                  "self-match 使正常分偏低→阈值偏松" if same_bank else ""),
               "n_tiles_per_image": int(np.median(n_tiles)) if n_tiles else 0,
               "view_signature": self.sig, "view": self.sig_detail,
               "label_thresholds": {"pos": self.pos, "neg": self.neg, "source": "candidate_unfrozen"}}
        # 缺陷图：面积标签 → 块级 AUROC + 覆盖
        if defect_imgs:
            pos_s, neg_s, dropped, cov = [], [], 0, []
            for img, m in zip(defect_imgs, defect_masks or [None] * len(defect_imgs)):
                boxes = mask_boxes(m)
                if not boxes:
                    continue
                tiles = self.tiles_for(img.shape[:2])
                s, _ = self.slot.score_tiles(self._feats(img, tiles))
                for lab, v in zip(tile_labels_by_area(tiles, boxes, self.pos, self.neg), s):
                    if lab == 1:
                        pos_s.append(float(v))
                    elif lab == 0:
                        neg_s.append(float(v))
                    else:
                        dropped += 1
                cov.extend(coverage_check(tiles, boxes))
            rep.update({
                "block_auroc_defect_imgs": _auroc(neg_s, pos_s),
                "block_auroc_vs_normal": _auroc(blk, pos_s),
                "n_pos": len(pos_s), "n_neg": len(neg_s), "n_dropped": dropped,
                "coverage_union": (float(np.mean([c["union_ok"] for c in cov])) if cov else None),
                "coverage_whole": (float(np.mean([c["whole_ok"] for c in cov])) if cov else None)})
        rep["fit_seconds"] = round(time.time() - t0, 2)
        self.report = rep
        log(f"  [fit] 局部分支完成 {rep['fit_seconds']}s bank={len(bank_idx)} "
            f"holdout={len(hold_idx)} tiles/img={rep['n_tiles_per_image']}")
        return self

    # ---------------- 推理 ----------------
    def score(self, img, priority_fn=None, force_full=False):
        """→ dict(score raw max-block, block_scores, selected, reasons, tiles, heatmaps, n_total)。
        priority_fn(tiles)->list[float]：budget 模式下全局热度。"""
        tiles = self.tiles_for(img.shape[:2])
        if self.mode == "budget" and not force_full and len(tiles) > self.max_tiles:
            pr = priority_fn(tiles) if priority_fn else [0.0] * len(tiles)
            sel, reasons = self.scheduler.select(tiles, pr)
        else:
            sel = list(range(len(tiles)))
            reasons = {i: "full" for i in sel}
        st = [tiles[i] for i in sel]
        s, hms = self.slot.score_tiles(self._feats(img, st))
        return {"score": float(np.max(s)) if len(s) else 0.0,
                "block_scores": [float(v) for v in s],
                "block_cal": [float(v) for v in self.cal.transform(s)] if len(s) else [],
                "selected": sel, "reasons": {int(k): v for k, v in reasons.items()},
                "tiles": st, "heatmaps": hms, "n_total": len(tiles)}

    # ---------------- 持久化（无 pickle）----------------
    def state_arrays(self):
        return {"bank": self.slot.bank.detach().cpu().numpy().astype(np.float16),
                "edges": np.asarray(self.cal.edges, dtype=np.float64),
                "normal_image_raw": np.asarray(self.normal_image_raw, dtype=np.float64)}

    def state_meta(self):
        return {"view_signature": self.sig, "view": self.sig_detail,
                "tile_size": self.tile_size, "stride": self.stride,
                "thresholds": self.thresholds, "report": self.report,
                "hold_idx": self.hold_idx,
                "mode": self.mode, "scheduler_cursor": int(self.scheduler._cursor)}

    def load_state(self, arrays, meta, log=print):
        # auto 尺寸：加载时沿用快照中按档案解出的尺寸（档案未变则签名一致）
        if self.tile_size_source.startswith("auto") and meta.get("tile_size"):
            self.tile_size, self.stride = int(meta["tile_size"]), int(meta["stride"])
            self.sig, self.sig_detail = view_signature(self.cfg, self.tile_size, self.stride)
            self.sig_detail["tile_size_source"] = (meta.get("view") or {}).get(
                "tile_size_source", self.tile_size_source)
        self.hold_idx = list(meta.get("hold_idx", []))
        self.slot.bank = torch.from_numpy(np.asarray(arrays["bank"])).to(self.device).half()
        self.cal.edges = np.asarray(arrays["edges"], dtype=np.float64)
        self.normal_image_raw = list(np.asarray(arrays.get("normal_image_raw", [])))
        self.thresholds = meta.get("thresholds")
        self.report = meta.get("report", {})
        self.scheduler._cursor = int(meta.get("scheduler_cursor", 0))
        if meta.get("view_signature") != self.sig:
            self.enabled = False
            log(f"[local] 视图签名不一致（快照 {meta.get('view_signature')} vs 当前 {self.sig}），"
                f"局部分支停用（AC-005 训练/推理视图一致）")
        return self
