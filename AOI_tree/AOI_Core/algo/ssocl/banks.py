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


def _feat_hash(f):
    """AC-028：条目内容 hash（fp16 字节 sha256 前 16 位），血缘/去重/混库核对用。"""
    import hashlib
    t = f.detach().cpu().contiguous().half().numpy()
    return hashlib.sha256(t.tobytes()).hexdigest()[:16]


def _meta(f, op, source, version, ts, image_id=None, **extra):
    """AC-028：记忆条目血缘（来源/时间/版本/hash/独立图标识）。"""
    m = {"op": op, "source": source, "t": ts, "v": version,
         "hash": _feat_hash(f), "n": int(f.shape[0]),
         "image_id": image_id}
    m.update({k: v for k, v in extra.items() if v is not None})
    return m


def _aligned_meta(entries, metas, op):
    """旧快照兼容：meta 缺失/长度不齐时按条目补 legacy 血缘（不改条目本身）。"""
    metas = list(metas or [])
    if len(metas) == len(entries):
        return metas
    # 对不齐时无法可靠逐条对应 → 全部按 legacy 标注（诚实：血缘未知）
    return [{"op": op, "source": "legacy", "t": e[1], "v": None,
             "hash": _feat_hash(e[0]), "n": int(e[0].shape[0]),
             "image_id": None, "legacy": True} for e in entries]


class NormalBank:
    """正常模式库：锚定（受保护）+ 扩展（版本化可回滚）+ 成簇入库 + 回流熔断。"""

    def __init__(self, core_max=512, ext_max=256, cluster_k=3, sim_thresh=0.75,
                 defect_ratio_thresh=0.30, snapshots_max=3):
        self.core_max = core_max
        self.ext_max = ext_max
        self.cluster_k = cluster_k          # 成簇所需独立图命中数（§7.2）
        self.sim_thresh = sim_thresh        # 与现有库的相似度阈值（归一化相似）
        self.defect_ratio_thresh = defect_ratio_thresh   # 回流熔断缺陷率阈值（§7.2）
        self.snapshots_max = snapshots_max
        self.core = None                    # (M,384) fp16 锚定库 patch（受保护）
        self.ext = []                       # list[(n,384) fp16, ts, source] 扩展库
        self.cluster_hits = {}              # 候选簇 id -> 连续命中数（AC-029）
        # AC-029：严格成簇——候选样本间相似（描述子 cos≥cluster_sim）+ 窗口内
        # 连续命中 cluster_k 次才入库；旧实现按"与库相似度"0.01 分桶计数，
        # 互不相似的样本只要库相似度同桶即被累计，不是聚类。
        self.cluster_sim = 0.90
        self.cluster_window = 20            # 超过窗口未再命中 → 簇失效（非连续）
        self._clusters = []                 # [{id, centroid(D,), hits, last_seq}]
        self._seq = 0
        self._cluster_next = 0
        self.ext_meta = []                  # AC-028：与 ext 逐条对齐的血缘
        self.core_meta = None
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
        self.core_meta = {"op": "seed_core", "source": "train_good_anchor",
                          "t": time.time(), "n_src_patches": int(n)}
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

    def reflow_candidate(self, tile_feats, min_ext_feat=1, image_id=None,
                         signature=None):
        """成簇判定（§7.2）：连续出现、自相似、一致性通过才入扩展库。

        返回 "accepted"（成簇入库）/ "pending"（累积证据）/ "rejected"（疑似缺陷，拒绝）。
        """
        f = _norm_patch(tile_feats)
        sim = self._sim_to(f)
        if sim is None:
            return self._cluster_vote(f, image_id=image_id, signature=signature)
        s = float(sim.max())           # 样本与库的最相似 patch
        if s < self.sim_thresh:        # 与现有库差异太大 → 疑为新缺陷而非新正常
            return "rejected"
        return self._cluster_vote(f, image_id=image_id, signature=signature)

    _REP_MAX = 256                     # 簇代表 patch 子样本上限（相似计算成本封顶）

    def _cluster_vote(self, f, image_id=None, signature=None):
        """AC-029：候选间相似 + 连续命中。
        相似 = 新样本 patch 到簇代表 patch 的最近邻 cos 均值（样本对样本，
        而非样本对库）；窗口 cluster_window 内未再命中的簇失效（不连续）。"""
        from ..common.image_identity import near_duplicate
        identity = {"image_id": image_id, "hash": _feat_hash(f), "signature": signature}
        seen = getattr(self, "_cluster_identities", [])
        if any((image_id is not None and m.get("image_id") == image_id)
               or m.get("hash") == identity["hash"]
               or near_duplicate(signature, m.get("signature")) for m in seen):
            return "pending"
        self._cluster_identities = seen + [identity]
        self._seq = getattr(self, "_seq", 0) + 1
        clusters = [c for c in getattr(self, "_clusters", [])
                    if self._seq - c["last_seq"] <= getattr(self, "cluster_window", 20)]
        rep = f
        if rep.shape[0] > self._REP_MAX:
            g = torch.Generator().manual_seed(0)
            rep = rep[torch.randperm(rep.shape[0], generator=g)[:self._REP_MAX]]
        rep = rep.float()
        best, best_s = None, -1.0
        for c in clusters:
            s = float((rep @ c["rep"].T).max(dim=1).values.mean())
            if s > best_s:
                best, best_s = c, s
        if best is not None and best_s >= getattr(self, "cluster_sim", 0.90):
            best["hits"] += 1
            best["last_seq"] = self._seq
        else:
            cid = getattr(self, "_cluster_next", 0)
            self._cluster_next = cid + 1
            best = {"id": cid, "rep": rep, "hits": 1, "last_seq": self._seq}
            clusters.append(best)
        self.cluster_hits = {c["id"]: c["hits"] for c in clusters}
        if best["hits"] >= self.cluster_k:
            clusters.remove(best)                 # 成簇入库后清空该簇证据
            self.cluster_hits.pop(best["id"], None)
            self._clusters = clusters
            return "accepted"
        self._clusters = clusters
        return "pending"

    def add_ext(self, tile_feats, source="feedback", min_feat=64, image_id=None):
        """入扩展库（版本化）；超限淘汰最旧。AC-028：同步记录条目血缘。"""
        f = _norm_patch(tile_feats)
        if f.shape[0] < min_feat:
            return False
        ts = time.time()
        meta = _aligned_meta(self.ext, getattr(self, "ext_meta", None), "add_ext")
        digest = _feat_hash(f)
        if any((image_id is not None and m.get("image_id") == image_id)
               or m.get("hash") == digest for m in meta):
            return False
        self.ext.append([f, ts, source])
        self.version += 1
        meta.append(_meta(f, "add_ext", source, self.version, ts, image_id))
        if len(self.ext) > self.ext_max:
            self.ext = self.ext[-self.ext_max:]
            meta = meta[-self.ext_max:]
        self.ext_meta = meta
        self.updated_ts.append({"t": ts, "op": "add_ext", "source": source,
                                "v": self.version, "hash": meta[-1]["hash"]})
        return True

    def lineage(self):
        """AC-028：库资产血缘清单（记忆资产≠训练权重；逐条来源/时间/版本/hash）。"""
        core = self.core
        return {"kind": "normal_bank", "memory_not_weights": True,
                "version": self.version,
                "core": None if core is None else {
                    "n": int(core.shape[0]), "hash": _feat_hash(core),
                    "protected": True, **(getattr(self, "core_meta", None) or {})},
                "ext": _aligned_meta(self.ext, getattr(self, "ext_meta", None), "add_ext"),
                "pending_clusters": dict(self.cluster_hits)}

    def snapshot(self, tag="op"):
        self._snapshots.append({"tag": tag,
                                "ext": [[e[0].clone(), e[1], e[2]] for e in self.ext],
                                "ext_meta": [dict(m) for m in _aligned_meta(
                                    self.ext, getattr(self, "ext_meta", None), "add_ext")],
                                "cluster_hits": dict(self.cluster_hits),
                                "clusters": [dict(c) for c in getattr(self, "_clusters", [])],
                                "seq": getattr(self, "_seq", 0),
                                "cluster_identities": list(getattr(self, "_cluster_identities", [])),
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
        self.ext_meta = _aligned_meta(self.ext, snap.get("ext_meta"), "add_ext")
        self.cluster_hits = dict(snap.get("cluster_hits", {}))
        self._clusters = [dict(c) for c in snap.get("clusters", [])]
        self._seq = snap.get("seq", getattr(self, "_seq", 0))
        self._cluster_identities = list(snap.get("cluster_identities", []))
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
        self.sample_meta = []             # AC-028：与 samples 对齐的血缘
        self.n_hits = []                  # 命中计数（活跃度，供主动学习/淘汰）
        self.version = 0
        self.updated_ts = []
        self._snapshots = []

    # AC-032：可进孵育的纯 patch 来源（确认框 / 合格伪框）；整图 add 不进孵育
    PURE_OPS = ("add_box", "add_patch")

    def add(self, tile_feats, source="feedback", image_id=None, defect_type=None):
        """确认缺陷入库：特征全量缓存零重算（§7.3 工程前提）。"""
        f = _norm_patch(tile_feats)
        if f.shape[0] < 4:
            return False
        return self._append(f, "add", source, image_id=image_id, defect_type=defect_type)

    def add_patch_feats(self, patch_feats, source="feedback", image_id=None,
                        defect_type=None):
        """U121（2026-08-26，M14 同步）：直接 patch 特征（(K,D) 每行一 patch）
        入库——blob 伪框化拦截库（无 GT 框时用 blob 响应图 top 异常 patch 作
        缺陷区域，demo4 裁切思想：缺陷区域 patch 比整图纯净，信噪比高）。
        mechE 实测（小样本）：solder 整图拦截 0.32 反向 → blob 裁剪 0.52 转正；
        大锚定学习后 +0.007（U121，默认关可选增强）。"""
        f = torch.as_tensor(np.asarray(patch_feats, dtype=np.float32))
        if f.ndim == 2 and f.shape[0] >= 4:
            f = F.normalize(f, dim=1).half()
            return self._append(f, "add_patch", source, image_id=image_id,
                                defect_type=defect_type)
        return False

    def add_box(self, tile_feats, box, source="feedback", image_id=None,
                defect_type=None):
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
            return self.add(tile_feats, source, image_id=image_id, defect_type=defect_type)
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            return self.add(tile_feats, source, image_id=image_id, defect_type=defect_type)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            return self.add(tile_feats, source, image_id=image_id, defect_type=defect_type)
        x0g = max(0, min(G, int(x0 * G)))
        y0g = max(0, min(G, int(y0 * G)))
        x1g = max(0, min(G, int(np.ceil(x1 * G))))
        y1g = max(0, min(G, int(np.ceil(y1 * G))))
        if x1g - x0g < 2 or y1g - y0g < 2:            # 框太小几乎无 patch
            return self.add(tile_feats, source, image_id=image_id, defect_type=defect_type)
        n_patch = (x1g - x0g) * (y1g - y0g)
        if n_patch < 4:                               # 保底：patch 太少回退整图
            return self.add(tile_feats, source, image_id=image_id, defect_type=defect_type)
        flat = _norm_patch(f[:, :, y0g:y1g, x0g:x1g])
        return self._append(flat, "add_box", source, image_id=image_id,
                            box=[round(v, 4) for v in (x0, y0, x1, y1)],
                            defect_type=defect_type)

    def _append(self, f, op, source, box=None, image_id=None, defect_type=None):
        """add/add_box 公共入库：samples/version/updated_ts/sample_meta 留痕。"""
        ts = time.time()
        meta = _aligned_meta(self.samples, getattr(self, "sample_meta", None), "add")
        self.samples.append([f, ts, f.shape[0]])
        self.n_hits.append(0)
        self.version += 1
        meta.append(_meta(f, op, source, self.version, ts, image_id=image_id, box=box,
                          defect_type=(str(defect_type).strip() or None)
                          if defect_type is not None else None))
        self.sample_meta = meta
        rec = {"t": ts, "op": op, "source": source, "v": self.version,
               "hash": meta[-1]["hash"]}
        if box is not None:
            rec["box"] = box
        self.updated_ts.append(rec)
        return True

    def incubation_pool(self):
        """AC-032：孵育正样本池——仅纯 patch（框内/伪框），整图 add 与 legacy 不进。

        -> (samples[list[(M,D) fp16]], image_ids[list[str]])，image_id 缺失时以
        hash 代（同一特征重复入库不重复计数）。"""
        metas = _aligned_meta(self.samples, getattr(self, "sample_meta", None), "add")
        feats, ids = [], []
        for s, m in zip(self.samples, metas):
            if m.get("op") in self.PURE_OPS and not m.get("legacy"):
                feats.append(s[0])
                ids.append(str(m.get("image_id") or m.get("hash")))
        return feats, ids

    ANY_TYPE = "_any"

    def incubation_pool_typed(self):
        """按缺陷类型孵育用：-> (feats, image_ids, defect_types)，与 incubation_pool
        同序同筛选；无类型（含旧快照条目）归 "_any"。"""
        metas = _aligned_meta(self.samples, getattr(self, "sample_meta", None), "add")
        feats, ids, types = [], [], []
        for s, m in zip(self.samples, metas):
            if m.get("op") in self.PURE_OPS and not m.get("legacy"):
                feats.append(s[0])
                ids.append(str(m.get("image_id") or m.get("hash")))
                types.append(str(m.get("defect_type") or self.ANY_TYPE))
        return feats, ids, types

    def lineage(self):
        """AC-028：缺陷样例库血缘（记忆资产≠训练权重）。"""
        metas = _aligned_meta(self.samples, getattr(self, "sample_meta", None), "add")
        _, ids = self.incubation_pool()
        return {"kind": "defect_bank", "memory_not_weights": True,
                "version": self.version,
                "samples": [dict(m, hits=int(h)) for m, h in zip(metas, self.n_hits)],
                "n_pure": len(ids), "n_pure_images": len(set(ids))}

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
                                "sample_meta": [dict(m) for m in _aligned_meta(
                                    self.samples, getattr(self, "sample_meta", None),
                                    "add")],
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
        self.sample_meta = _aligned_meta(self.samples, snap.get("sample_meta"), "add")
        self.n_hits = list(snap["n_hits"])
        self.updated_ts = list(snap.get("updated_ts", []))
        self.version = snap["version"]
        return True
