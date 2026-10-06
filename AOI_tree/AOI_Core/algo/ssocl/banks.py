"""M3 双库制（§7.2）+ 缺陷样例库（§7.3）

库资产负债表（§7.2）中的在线可更新部分：
- NormalBank：锚定库（train/good 初始原型，受保护）+ 扩展库（回流样本/新模式，版本化可回滚）
  * 成簇入库：孤立高分样本永不入库（可能是漏检缺陷），连续/自相似/一致性通过才成簇
  * 回流熔断：在线缺陷率估计 > 阈值即熔断正常回流（demo3 1044/1112 缺陷流教训）
- DefectBank：确认缺陷样例库 + 近邻拦截通道（秒级生效，"这批错检下批检对"承载者）
  * 确认缺陷入库后，来料 patch 与样例近邻 → 直接加分（fused += boost）

版本化：snapshot()/rollback() 快照栈（默认保留 3 份），回归门控回滚用。

持久化：save_bank/load_bank 支持把学习记忆整体落盘与恢复（原子写）；
AOI_Core 主干由 algo/persist.py 以同等方式随快照落盘 normal_bank.pkl /
defect_bank.pkl，格式一致、可互换。
"""
import os
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F


def save_bank(bank, path):
    """把 NormalBank/DefectBank 学习态整体落盘（pickle，含张量 fp16）。"""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:                     # 原子写：先 tmp 后替换
        pickle.dump(bank, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load_bank(path, device="cpu"):
    """从 save_bank 落盘文件恢复库实例；张量统一搬到 device。

    2026-10-03 A10：反序列化收口 safe_pickle_load（路径白名单+留痕）；
    白名单外路径（如实验目录）需 AOI_PICKLE_EXTRA_ROOTS 追加。"""
    from ..common.safe_pickle import safe_pickle_load
    bank = safe_pickle_load(path)
    if device and device != "cpu":
        t = getattr(bank, "core", None)
        if torch.is_tensor(t):
            bank.core = t.to(device)
        for seq_attr in ("ext", "samples"):
            for item in (getattr(bank, seq_attr, None) or []):
                if torch.is_tensor(item[0]):
                    item[0] = item[0].to(device)
    return bank


def _norm_patch(feats, device="cpu"):
    """(T,384,G,G) -> (T*G*G,384) 归一化 patch 特征（fp16 CPU 驻留）"""
    f = torch.as_tensor(np.asarray(feats, dtype=np.float32))
    if f.ndim == 3:
        f = f.unsqueeze(0)
    T, D, G, _ = f.shape
    flat = f.flatten(2).permute(0, 2, 1).reshape(-1, D)
    flat = F.normalize(flat, dim=1).half()
    if device == "cuda" and torch.cuda.is_available():
        flat = flat.to(device)
    return flat


class NormalBank:
    """正常模式库：锚定（受保护）+ 扩展（版本化可回滚）+ 成簇入库 + 回流熔断。"""

    def __init__(self, core_max=512, ext_max=256, cluster_k=3, sim_thresh=0.75,
                 defect_ratio_thresh=0.30, snapshots_max=3):
        self.core_max = core_max
        self.ext_max = ext_max
        self.cluster_k = cluster_k          # 成簇所需连续命中次数（§7.2）
        self.sim_thresh = sim_thresh        # 与现有库的相似度阈值（归一化相似）
        self.defect_ratio_thresh = defect_ratio_thresh   # 回流熔断缺陷率阈值（§7.2）
        self.snapshots_max = snapshots_max
        self.core = None                    # (M,384) fp16 锚定库 patch（受保护）
        self.ext = []                       # list[(n,384) fp16, ts, source] 扩展库
        self.cluster_hits = {}              # 相似区域命中计数（成簇证据累积）
        self.recent_decisions = []          # 在线缺陷率滑动窗
        self.defect_ratio = 0.0
        self.version = 0
        self.updated_ts = []
        self._snapshots = []

    # ---- 锚定库（只增不换，受保护）----
    def seed_core(self, tile_feats, rng=None):
        """fit 时用 train/good tile 特征建立锚定库（coreset 抽样 ≤core_max*8 patch）。"""
        if tile_feats is None:
            return self
        rng = rng or np.random.default_rng(0)
        f = _norm_patch(tile_feats)
        n = f.shape[0]
        target = min(self.core_max * 8, n)
        idx = rng.choice(n, target, replace=False)
        self.core = f[idx]
        return self

    def _sim_to(self, feat):
        """feat (n,384) fp16 -> 与库的最近邻相似度 (n,)（库空返回 None）"""
        ref = None
        if self.core is not None and self.core.shape[0]:
            ref = self.core
        if self.ext:
            ext = torch.cat([e[0] for e in self.ext], dim=0)
            ref = torch.cat([ref, ext], dim=0) if ref is not None else ext
        if ref is None:
            return None
        f = feat.float()
        s = (f @ ref.float().T).max(dim=1).values   # 归一化向量点积=cos 相似
        return s.cpu().numpy()

    def _observe_decision(self, is_anomaly):
        """在线缺陷率估计（§7.2 回流熔断）：滑动窗双峰估计的工程代理。"""
        self.recent_decisions.append(1 if is_anomaly else 0)
        if len(self.recent_decisions) > 200:
            self.recent_decisions = self.recent_decisions[-200:]
        self.defect_ratio = float(np.mean(self.recent_decisions))
        return self.defect_ratio

    @property
    def fuse_blocked(self):
        """回流熔断：在线缺陷率估计超阈（§7.2：防'多数派是缺陷'污染正常库）。"""
        return self.defect_ratio > self.defect_ratio_thresh

    def reset_stream_stats(self):
        """清空在线缺陷率滑动窗（2026-08-30）：窗口是当前决策流的实时属性，
        不应随快照 pickle 传播——激活/装载快照后从在线流重新估计，避免
        历史污染窗口（旧模型/L0 阶段的误判流）锁死新会话的正常回流。"""
        self.recent_decisions = []
        self.defect_ratio = 0.0
        return self

    def reflow_candidate(self, tile_feats, min_ext_feat=1):
        """成簇判定（§7.2）：连续出现、自相似、一致性通过才入扩展库。

        返回 "accepted"（成簇入库）/ "pending"（累积证据）/ "rejected"（疑似缺陷，拒绝）。
        """
        f = _norm_patch(tile_feats)
        sim = self._sim_to(f)
        if sim is None:
            return "accepted"          # 库空（理论上 fit 后不空），直接收
        s = float(sim.max())           # 样本与库的最相似 patch
        if s < self.sim_thresh:        # 与现有库差异太大 → 疑为新缺陷而非新正常
            return "rejected"
        key = int(round(s * 100))      # 按相似度桶累积证据
        self.cluster_hits[key] = self.cluster_hits.get(key, 0) + 1
        if self.cluster_hits[key] >= self.cluster_k:
            return "accepted"
        return "pending"

    def add_ext(self, tile_feats, source="feedback", min_feat=64):
        """入扩展库（版本化）；超限淘汰最旧。"""
        f = _norm_patch(tile_feats)
        if f.shape[0] < min_feat:
            return False
        self.ext.append([f, time.time(), source])
        if len(self.ext) > self.ext_max:
            self.ext = self.ext[-self.ext_max:]
        self.version += 1
        self.updated_ts.append({"t": time.time(), "op": "add_ext", "source": source,
                                "v": self.version})
        return True

    def snapshot(self, tag="op"):
        self._snapshots.append({"tag": tag,
                                "ext": [[e[0].clone(), e[1], e[2]] for e in self.ext],
                                "cluster_hits": dict(self.cluster_hits),
                                "updated_ts": list(self.updated_ts),
                                "version": self.version})
        if len(self._snapshots) > self.snapshots_max:
            self._snapshots.pop(0)
        return len(self._snapshots)

    def rollback(self, tag="op"):
        """回滚到最近快照（默认仅回滚扩展库；锚定库永不回滚）。"""
        if not self._snapshots:
            return False
        snap = self._snapshots.pop()
        self.ext = [[e[0], e[1], e[2]] for e in snap["ext"]]
        self.cluster_hits = dict(snap.get("cluster_hits", {}))
        self.updated_ts = list(snap.get("updated_ts", []))
        self.version = snap["version"]
        return True


class DefectBank:
    """确认缺陷样例库 + 近邻拦截（§7.3）：秒级生效，无训练。

    拦截分：来料 patch 与样例库 patch 的最近邻 cos 相似 top-k 均值 → [0, boost]。
    样例少时拦截弱（相似个体才命中），样例多/同类聚集时拦截强——诚实反映
    "1-2 张反馈只能拦截相似个体，泛化拦截需要积累"（§7.3 学习延迟分级）。
    """

    def __init__(self, topk=8, boost=0.3, sim_thresh=0.85, min_hit=1,
                 snapshots_max=3):
        self.topk = topk
        self.boost = boost
        self.sim_thresh = sim_thresh      # 命中样例的相似度门槛
        # U105（2026-08-25）：命中 patch 数下限——易品类学习负增益根因修复。
        # m5 实测（transistor eval 集，学习后）：正常样本 70% 被拦截加分
        # （boost>0），相对差 rel_top8_mean 与缺陷几乎重叠（0.126 vs 0.158）
        # 单纯提高阈值无法区分；但全图 rel≥thresh 的 patch 数是强信号
        # （正常均值 1.3 vs 缺陷均值 31.4）。要求 topk 内命中数 ≥ min_hit
        # 才加分，杜绝"1-3 个噪声 patch"误伤正常样本。
        self.min_hit = min_hit
        self.snapshots_max = snapshots_max
        self.samples = []                 # list[(M,384) fp16, ts, n_patches]
        self.n_hits = []                  # 命中计数（活跃度，供主动学习/淘汰）
        self.version = 0
        self.updated_ts = []
        self._snapshots = []

    def add(self, tile_feats, source="feedback"):
        """确认缺陷入库：特征全量缓存零重算（§7.3 工程前提）。"""
        f = _norm_patch(tile_feats)
        if f.shape[0] < 4:
            return False
        return self._append(f, "add", source)

    def add_patch_feats(self, patch_feats, source="feedback"):
        """U121（2026-08-26，M14 同步）：直接 patch 特征（(K,D) 每行一 patch）
        入库——blob 伪框化拦截库（无 GT 框时用 blob 响应图 top 异常 patch 作
        缺陷区域，demo4 裁切思想：缺陷区域 patch 比整图纯净，信噪比高）。
        mechE 实测（小样本）：solder 整图拦截 0.32 反向 → blob 裁剪 0.52 转正；
        大锚定学习后 +0.007（U121，默认关可选增强）。"""
        f = torch.as_tensor(np.asarray(patch_feats, dtype=np.float32))
        if f.ndim == 2 and f.shape[0] >= 4:
            f = F.normalize(f, dim=1).half()
            return self._append(f, "add_patch", source)
        return False

    def add_box(self, tile_feats, box, source="feedback"):
        """U64（2026-08-18）：框内确认缺陷入库——用户在线提供缺陷框标注
        （归一化 [x0,y0,x1,y1]），只把**框内 patch 特征**入缺陷样例库。

        机理：demo4 用标注裁切图（框内区域）达 0.9——框内特征信噪比高、无背景
        干扰；基线对标 = 反馈时用户给缺陷框 → 只入框内 patch → 拦截更精准
        （比对整图入库存入大量背景 patch 更纯净）。
        信息层次策略（合法界定，U64）：框 = 用户在线反馈时提供的"有标注"信息
        （模拟操作员画框反馈），非 test 信息用于初始化预训练；fit/初始化不触碰。
        tile_feats: (1,D,G,G)（single 模式整图特征）；非 single（T>1）或无有效
        框或框内 patch <4 → 回退 add 整图（保证样例可用）。
        """
        f = torch.as_tensor(np.asarray(tile_feats, dtype=np.float32))
        if f.ndim == 3:
            f = f.unsqueeze(0)
        T, D, G, _ = f.shape
        if T != 1:                                    # 框映射只对整图 single 特征有效
            return self.add(tile_feats, source)
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            return self.add(tile_feats, source)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            return self.add(tile_feats, source)
        x0g = max(0, min(G, int(x0 * G)))
        y0g = max(0, min(G, int(y0 * G)))
        x1g = max(0, min(G, int(np.ceil(x1 * G))))
        y1g = max(0, min(G, int(np.ceil(y1 * G))))
        if x1g - x0g < 2 or y1g - y0g < 2:            # 框太小几乎无 patch
            return self.add(tile_feats, source)
        n_patch = (x1g - x0g) * (y1g - y0g)
        if n_patch < 4:                               # 保底：patch 太少回退整图
            return self.add(tile_feats, source)
        flat = _norm_patch(f[:, :, y0g:y1g, x0g:x1g])
        return self._append(flat, "add_box", source,
                            box=[round(v, 4) for v in (x0, y0, x1, y1)])

    def _append(self, f, op, source, box=None):
        """add/add_box 公共入库：samples/version/updated_ts 留痕。"""
        self.samples.append([f, time.time(), f.shape[0]])
        self.n_hits.append(0)
        self.version += 1
        rec = {"t": time.time(), "op": op, "source": source, "v": self.version}
        if box is not None:
            rec["box"] = box
        self.updated_ts.append(rec)
        return True

    def intercept_score(self, tile_feats, normal_bank=None):
        """-> 近邻拦截加分 ∈ [0, boost]（无样例返回 0）。

        U25（2026-08-16）：相对相似。U24 消融实证：绝对相似加分（sim_thresh=0.85）
        会误伤正常样本（同品类正常 patch 与缺陷样例偶似即被加分）→ 污染 AUROC 排序，
        full 组 gain 为负。修复：加分 ∝（与缺陷库相似 − 与正常库相似）——
        正常样本与两库都相似（rel≈0/负），不加分；缺陷样本缺陷相似高、正常相似低
        （rel 显著正）才加分 → 缺陷侧学习反映到排序，学习后 AUC 提升。
        """
        if not self.samples:
            return 0.0
        f = _norm_patch(tile_feats)
        all_ref = torch.cat([s[0] for s in self.samples], dim=0).float()
        # 分块算相似（内存控制）
        sims_max = torch.empty(f.shape[0], device=f.device, dtype=torch.float32)
        for p0 in range(0, f.shape[0], 1024):
            s = f[p0:p0 + 1024].float() @ all_ref.T        # cos 相似
            sims_max[p0:p0 + 1024] = s.max(dim=1).values
        rel = sims_max
        if normal_bank is not None:
            # 与正常库（锚定+扩展）最近邻相似 → 相对缺陷倾向
            n_sim = torch.as_tensor(normal_bank._sim_to(f), device=f.device,
                                    dtype=torch.float32)
            rel = sims_max - n_sim
        top_rel = rel.topk(min(self.topk, rel.shape[0])).values
        hit = (top_rel >= self.sim_thresh).float()   # sim_thresh 现为相对差阈值
        # U105：命中 patch 数下限——正常样本 topk 内仅 1-3 个噪声命中（rel 略
        # 超阈），缺陷样本 4-8 个（真实缺陷 patch）；< min_hit 直接不加分。
        if int(hit.sum()) < self.min_hit:
            return 0.0
        # 命中 patch 的相对差均值 × boost（未达阈不加分）
        score = float((hit * top_rel).sum() / max(self.topk, 1)) * self.boost
        return round(score, 5)

    def track_hits(self, tile_feats):
        """统计每个样例被命中次数（供缺陷样例活跃度/主动学习参考）。"""
        if not self.samples:
            return
        f = _norm_patch(tile_feats)
        for i, (s, _, _) in enumerate(self.samples):
            if s.shape[0] == 0:
                continue
            sim = (f.float() @ s.float().T).max(dim=0).values
            if float(sim.max()) >= self.sim_thresh:
                self.n_hits[i] += 1

    def snapshot(self, tag="op"):
        self._snapshots.append({"tag": tag,
                                "samples": [[s[0].clone(), s[1], s[2]]
                                            for s in self.samples],
                                "n_hits": list(self.n_hits),
                                "updated_ts": list(self.updated_ts),
                                "version": self.version})
        if len(self._snapshots) > self.snapshots_max:
            self._snapshots.pop(0)
        return len(self._snapshots)

    def rollback(self, tag="op"):
        if not self._snapshots:
            return False
        snap = self._snapshots.pop()
        self.samples = [[e[0], e[1], e[2]] for e in snap["samples"]]
        self.n_hits = list(snap["n_hits"])
        self.updated_ts = list(snap.get("updated_ts", []))
        self.version = snap["version"]
        return True
