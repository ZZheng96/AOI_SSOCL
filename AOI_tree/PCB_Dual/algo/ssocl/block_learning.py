"""U80/U81（2026-08-19）：块级在线学习器（用户 U75 方案 + 标注裁切学习）

用户方案：分块检测 + 持续学习——
- 误报块 → 扩大正常库（同类正常模式永久不再误报，块 FP 单调降）
- 漏检缺陷块 → 收缩正常库（移除与缺陷块相似的特征，边界右移，FN 降）
U81 升级（用户明确允许在线阶段输入标注裁切图）：缺陷库改用**框内 crop 特征**
（GT 框+pad 裁切，缺陷占比 >70%，对齐 U71 离线 crop 实验 0.9651 的判别条件），
正常库用正常块特征——在线反馈带框时"标注裁切图学习"，等效离线裁切训练。

- 块特征表示：块 patch 均值 (D,) 归一化
- 正常块库 normal：train/good 块（初始）+ 反馈误报块（扩大）
- 缺陷库 defect：init_defect 框内 crop（初始）+ 反馈框内 crop（标注裁切学习）
  + 漏检块特征（保底）
- 分数修正（score 时）：
    adjusted = raw_head + lambda * (sim_defect - sim_normal)
- 收缩正常库（漏检）：从 normal 移除与漏检缺陷块最近邻相似 > thresh 的特征，
  U81 调优：thresh 提高（默认 0.92）+ 每轮上限 max_shrink（防库崩溃，U80 教训）

诚实边界：初始库只用 train 域（train/good + init_defect 框）；反馈只用
操作员真值（label/box），test 只验不选。
"""
import time
import numpy as np
import torch
import torch.nn.functional as F


class BlockLearner:
    def __init__(self, dim=384, lambda_boost=5.0, topk=3,
                 fp_high_quantile=0.95, fn_low_quantile=0.40,
                 shrink_thresh=0.92, max_shrink=20, max_normal=2048,
                 max_defect=1024):
        self.dim = dim
        self.lambda_boost = lambda_boost      # 库修正缩放（匹配判别头分尺度）
        self.topk = topk
        self.fp_high_quantile = fp_high_quantile  # 正常图高分块（误报候选）判定分位
        self.fn_low_quantile = fn_low_quantile    # 缺陷图含框低分块（漏检候选）判定分位
        self.shrink_thresh = shrink_thresh        # 收缩正常库的相似阈值（U81: 0.92 防过度）
        self.max_shrink = max_shrink              # 每轮收缩上限（U80 教训：480->27 崩溃）
        self.max_normal = max_normal
        self.max_defect = max_defect
        self.normal = []      # list[(D,) fp32 归一化]（正常块特征）
        self.defect = []      # list[(D,) fp32 归一化]（框内 crop 特征 + 漏检块特征）
        self.history = []
        self.stats = {"fp_add": 0, "fn_add": 0, "crop_add": 0, "shrink": 0,
                      "n_learn": 0}

    # ---- 库操作 ----
    def seed(self, normal_feats, defect_feats):
        """初始库：normal_feats/defect_feats: list[(D,)] 或 (N,D) tensor。"""
        self.normal = self._norm_list(normal_feats, self.max_normal)
        self.defect = self._norm_list(defect_feats, self.max_defect)
        return self

    @staticmethod
    def _norm_feat(v):
        v = torch.as_tensor(v, dtype=torch.float32).flatten()
        if v.shape[0] == 0:
            return None
        v = F.normalize(v, dim=0)
        return v

    def _norm_list(self, feats, cap):
        out = []
        for v in feats:
            n = self._norm_feat(v)
            if n is not None:
                out.append(n)
        if len(out) > cap:
            out = out[-cap:]
        return out

    def _lib_tensor(self):
        """正常/缺陷库合并张量 (N,D)（空库返回 None）。"""
        if self.normal:
            return torch.stack(self.normal)
        return None

    def _sim_to(self, feats, lib):
        """feats (B,D) -> 与库最近邻 cos 相似 (B,)。库空返回 zeros。"""
        if lib is None or len(lib) == 0:
            return torch.zeros(feats.shape[0])
        L = torch.stack(lib)
        s = (F.normalize(feats, dim=1) @ L.T).max(dim=1).values
        return s

    # ---- 学习（图级反馈 -> 块级库更新）----
    def learn(self, block_feats, block_scores, label, defect_block_idx=(),
              crop_feats=None):
        """block_feats (B,D) 块特征；block_scores (B,) 判别头分；label 图级真值；
        defect_block_idx: label=1 时含缺陷框的块索引（漏检候选判定用）；
        crop_feats: label=1 时框内 crop 特征列表（标注裁切学习，U81——缺陷库
        用高 SNR 框内特征，等效 U71 离线裁切训练）。
        返回学习记录 dict。"""
        self.stats["n_learn"] += 1
        B = block_feats.shape[0]
        rec = {"t": time.time(), "label": label, "n_blocks": B,
               "fp_add": 0, "fn_add": 0, "crop_add": 0, "shrink": 0}
        if label == 0:
            # 误报块：正常图 + 高分块 → 扩正常库（同类正常模式不再误报）
            th = float(np.quantile(block_scores.cpu().numpy(),
                                   self.fp_high_quantile))
            idx = (block_scores > th).nonzero(as_tuple=False).flatten().tolist()
            for i in idx:
                self.normal.append(self._norm_feat(block_feats[i]))
            self.normal = self.normal[-self.max_normal:]
            rec["fp_add"] = len(idx)
            self.stats["fp_add"] += len(idx)
        else:
            # 标注裁切学习（U81）：框内 crop 特征全量入缺陷库（高 SNR）
            if crop_feats:
                n_crop = 0
                for cf in crop_feats:
                    v = self._norm_feat(cf)
                    if v is not None:
                        self.defect.append(v)
                        n_crop += 1
                rec["crop_add"] = n_crop
                self.stats["crop_add"] += n_crop
            # 漏检缺陷块：含缺陷框 + 低分块 → 入缺陷库（保底）+ 收缩正常库
            db = list(defect_block_idx)
            if db:
                th = float(np.quantile(block_scores.cpu().numpy(),
                                       self.fn_low_quantile))
                n_shrink = 0
                for i in db:
                    if float(block_scores[i]) < th:      # 含缺陷但分数低 = 漏检块
                        v = self._norm_feat(block_feats[i])
                        if v is not None:
                            self.defect.append(v)
                            rec["fn_add"] += 1
                        # 收缩正常库：移除与漏检缺陷块相似 > thresh 的特征，
                        # U81：每轮上限 max_shrink 防库崩溃（U80 教训 480->27）
                        vn = self._norm_feat(block_feats[i])
                        if vn is not None and n_shrink < self.max_shrink:
                            keep = []
                            removed = 0
                            for nf in self.normal:
                                if float(vn @ nf) >= self.shrink_thresh \
                                        and removed < self.max_shrink - n_shrink:
                                    removed += 1
                                    continue
                                keep.append(nf)
                            self.normal = keep[-self.max_normal:]
                            n_shrink += removed
                rec["shrink"] = n_shrink
            self.defect = self.defect[-self.max_defect:]
            self.stats["fn_add"] += rec["fn_add"]
            self.stats["shrink"] += rec["shrink"]
        self.history.append(rec)
        return rec

    # ---- 推理修正 ----
    @torch.no_grad()
    def adjust(self, block_feats, block_scores):
        """块分数修正：adjusted = raw + lambda * (sim_defect - sim_normal)。
        block_feats (B,D) 归一化前特征；block_scores (B,) 判别头原始分。"""
        if not self.normal and not self.defect:
            return block_scores.clone() if torch.is_tensor(block_scores) \
                else torch.as_tensor(block_scores)
        f = torch.as_tensor(block_feats, dtype=torch.float32)
        sim_d = self._sim_to(f, self.defect)
        sim_n = self._sim_to(f, self.normal)
        delta = self.lambda_boost * (sim_d - sim_n)
        s = torch.as_tensor(block_scores, dtype=torch.float32)
        return s + delta

    # ---- 状态/留痕 ----
    def state(self):
        return {"n_normal": len(self.normal), "n_defect": len(self.defect),
                "stats": dict(self.stats)}
