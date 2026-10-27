"""M3 AHL 孵育头（§7.6）：同类缺陷独立图≥8 时孵育专属头，强化拦截。

- 训练：逻辑回归区分「纯缺陷 patch（正）」vs「正常锚定 patch（负）」
  用 DINO 归一化 patch 特征（384 维），参数量 ~384，符合"独立头不影响主干"
- 输出：孵育头加分 = 正类概率 top-k 均值 → 独立拦截通道（不进槽位表/融合权重，
  规避 §4.4 硬约束一——它不是参与融合的槽位，是缺陷样例拦截的强化版）
- AC-032（2026-10-12）：
  * 正样本只取纯 patch（确认框 add_box / 伪框 add_patch）；无框整图 add 含大量
    背景 patch，不进孵育（DefectBank.incubation_pool）。
  * 触发与重孵育按**独立图**计数（同图多框只算一张），不按样例条数。
  * 独立批准：留出最新若干独立图的纯 patch + 锚定负样本做 holdout，新头
    separation（正类概率均值差）须 ≥ approve_min，且不劣于旧头才替换；否则保持
    旧头并在 log 留痕 rejected+原因。
  * 负样本统一展平为 (N,D)（_anchor_feats 返回 (T,D,G,G)，旧实现直接 concat
    维度不符→被 except 吞掉，头实际从未训练成功）。
"""
import time
import numpy as np

from .banks import _norm_patch
from .candidate_budget import CandidateDeferred


def _flat(x):
    """(T,D,G,G)/(D,G,G)/(N,D) → (N,D) float numpy（归一化）。"""
    if x is None:
        return None
    return _norm_patch(x).float().numpy()


class IncubatedHead:
    """孵育头：sklearn 逻辑回归（无网络依赖，CPU 快）。"""

    def __init__(self, topk=8, boost=0.3, min_samples=8, l2=1.0):
        self.topk = topk
        self.boost = boost
        self.min_samples = min_samples
        self.l2 = l2
        self.clf = None
        self.trained_at = None
        self.n_pos = 0

    def incubate(self, positives, normal_core):
        """训练头。positives: list[(M,D)] 纯缺陷 patch（或带 .samples 的旧库对象）；
        normal_core: 锚定特征（任意 _norm_patch 可接受形状）。正样本抽样上限 2 万，
        负样本等量抽样。"""
        from sklearn.linear_model import LogisticRegression
        if hasattr(positives, "samples"):          # 旧调用：传入整库
            positives = [s[0] for s in positives.samples]
        pos = torch_cat(list(positives)).float().numpy()
        neg = _flat(normal_core)
        if neg is None or len(neg) == 0 or len(pos) == 0:
            raise ValueError("孵育缺正样本或锚定负样本")
        rng = np.random.default_rng(0)
        cap = 20_000
        if len(pos) > cap:
            pos = pos[rng.choice(len(pos), cap, replace=False)]
        n_neg = min(len(neg), len(pos))
        neg = neg[rng.choice(len(neg), n_neg, replace=False)]
        X = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        clf = LogisticRegression(C=1.0 / self.l2, max_iter=500)
        from .candidate_budget import charge
        charge("steps", 500, "incubate_max_iter_reservation")
        clf.fit(X, y)
        charge("steps", 0, "after_incubate")
        self.clf = clf
        self.trained_at = time.time()
        self.n_pos = len(pos)
        return self

    def separation(self, pos, neg):
        """holdout 区分度：正类概率均值(纯缺陷 patch) − 均值(锚定 patch)。"""
        if self.clf is None or pos is None or neg is None or not len(pos) or not len(neg):
            return None
        pp = self.clf.predict_proba(pos)[:, 1].mean()
        pn = self.clf.predict_proba(neg)[:, 1].mean()
        return round(float(pp - pn), 5)

    def score(self, tile_feats):
        """-> 孵育加分 ∈ [0, boost]（头未训练返回 0）。"""
        if self.clf is None:
            return 0.0
        f = _norm_patch(tile_feats).float().numpy()
        if len(f) == 0:
            return 0.0
        p = self.clf.predict_proba(f)[:, 1]
        k = min(self.topk, len(p))
        return round(float(np.sort(p)[-k:].mean()) * self.boost, 5)


def torch_cat(parts):
    import torch
    return torch.cat(parts, dim=0)


ANY_TYPE = "_any"


class HeadManager:
    """孵育头生命周期：纯 patch 独立图≥min_samples 孵育；独立图数≥1.5x 重孵育；
    新头须过 holdout 独立批准才替换。

    按缺陷类型孵育（2026-10-15）：self.head 仍是通用头（全池训练，唯一参与
    boost/拦截的头，语义与接口不变）；self.heads[defect_type] 为类型头——仅对
    带真实类型（非 "_any"）且该类型独立图≥min_samples 的样本孵育，规则与通用头
    相同（holdout 批准/1.5x 重孵育/拒后等新增），只经 type_scores 给结果/报告
    per-type 分数，不进融合与在线拦截。"""

    def __init__(self, cfg=None):
        c = cfg or {}
        self.head = None
        self.last_sample_n = 0          # 上次批准时的独立图数
        self.cfg = c
        self.log = []
        self.last_sep = None            # 当前头 holdout separation
        self.heads = {}                 # defect_type -> IncubatedHead（类型头）
        self.type_state = {}            # defect_type -> {last_sample_n,last_sep,last_reject_n}

    def __setstate__(self, state):
        """旧 pickle（单头格式）兼容：缺失的类型头属性补默认值。"""
        self.__dict__.update(state)
        self.__dict__.setdefault("heads", {})
        self.__dict__.setdefault("type_state", {})
        self.__dict__.setdefault("last_sep", None)

    @staticmethod
    def _pool(defect_bank):
        pool = getattr(defect_bank, "incubation_pool", None)
        if callable(pool):
            return pool()
        # 旧库对象（无血缘）：按条目计数，全部视为正样本（legacy 兼容）
        samples = list(getattr(defect_bank, "samples", []))
        return [s[0] if isinstance(s, (list, tuple)) else s for s in samples], \
            [str(i) for i in range(len(samples))]

    @staticmethod
    def _typed_pool(defect_bank):
        """-> {defect_type: (feats, ids)}，仅真实类型；旧库/无类型 → {}。"""
        pool = getattr(defect_bank, "incubation_pool_typed", None)
        if not callable(pool):
            return {}
        out = {}
        for f, i, t in zip(*pool()):
            if t and t != ANY_TYPE:
                fs, ids = out.setdefault(t, ([], []))
                fs.append(f)
                ids.append(i)
        return out

    def status(self, defect_bank=None):
        """进度/批准状态（engine 展示用）。顶层字段仍描述通用头；per_type 为类型头。"""
        n_img = len(set(self._pool(defect_bank)[1])) if defect_bank is not None else None
        last = next((e for e in reversed(self.log)
                     if "approved" in e and "defect_type" not in e), None)
        typed = self._typed_pool(defect_bank) if defect_bank is not None else {}
        heads = getattr(self, "heads", {}) or {}
        tstate = getattr(self, "type_state", {}) or {}
        per_type = {}
        for t in sorted(set(typed) | set(heads)):
            st = tstate.get(t, {})
            tl = next((e for e in reversed(self.log)
                       if "approved" in e and e.get("defect_type") == t), None)
            per_type[t] = {"trained": heads.get(t) is not None,
                           "n_pure_images": len(set(typed[t][1])) if t in typed else None,
                           "approved_images": st.get("last_sample_n", 0),
                           "separation": st.get("last_sep"),
                           "last_decision": tl}
        return {"trained": self.head is not None,
                "min_samples": int(self.cfg.get("min_samples", 8)),
                "n_pure_images": n_img, "approved_images": self.last_sample_n,
                "separation": getattr(self, "last_sep", None),
                "last_decision": last,
                "per_type": per_type}

    def maybe_incubate(self, defect_bank, normal_core):
        """通用头（返回值/self.head 语义不变）+ 各类型头（只更新 self.heads）。"""
        feats, ids = self._pool(defect_bank)
        st = {"last_sample_n": self.last_sample_n, "last_sep": getattr(self, "last_sep", None),
              "last_reject_n": getattr(self, "last_reject_n", None)}
        self.head = self._incubate_one(feats, ids, normal_core, self.head, st, None)
        self.last_sample_n = st["last_sample_n"]
        self.last_sep = st["last_sep"]
        self.last_reject_n = st["last_reject_n"]
        if self.cfg.get("per_type", True):
            self._incubate_types(defect_bank, normal_core)
        # 原单头语义：通用池独立图不足时返回 None（调用方据此判定未孵育）
        if len(set(ids)) < self.cfg.get("min_samples", 8):
            return None
        return self.head

    def _incubate_types(self, defect_bank, normal_core):
        if not hasattr(self, "heads"):
            self.heads, self.type_state = {}, {}
        typed = self._typed_pool(defect_bank)
        # 预算/耗时封顶：按独立图数降序最多 max_type_heads 个类型
        cap = int(self.cfg.get("max_type_heads", 8))
        order = sorted(typed, key=lambda t: -len(set(typed[t][1])))[:max(0, cap)]
        for t in order:
            feats, ids = typed[t]
            st = self.type_state.setdefault(
                t, {"last_sample_n": 0, "last_sep": None, "last_reject_n": None})
            h = self._incubate_one(feats, ids, normal_core, self.heads.get(t), st, t)
            if h is not None:
                self.heads[t] = h

    def _incubate_one(self, feats, ids, normal_core, cur, st, defect_type):
        """单头孵育/批准（AC-032 规则）。st 为该头的可变状态 dict；返回当前应持有的头。"""
        c = self.cfg.get("min_samples", 8)
        uniq = list(dict.fromkeys(ids))           # 按入库顺序去重的独立图
        n_img = len(uniq)
        if n_img < c:
            return cur
        if cur is not None and n_img < st["last_sample_n"] * 1.5:
            return cur
        n_hold = int(self.cfg.get("holdout_images", 2))
        # 上次未批准：需新增 ≥ max(1,n_hold) 张独立图（刷新 holdout）才重试，防每次反馈重训
        rej = st.get("last_reject_n")
        if rej is not None and n_img < rej + max(1, n_hold):
            return cur
        tag = {} if defect_type is None else {"defect_type": defect_type}
        # holdout：最新 n_hold 张独立图（至少留 c-n_hold 张训练；不足则不留出）
        if n_img - n_hold < max(1, c - n_hold) or n_hold <= 0:
            n_hold = 0
        hold_ids = set(uniq[-n_hold:]) if n_hold else set()
        tr = [f for f, i in zip(feats, ids) if i not in hold_ids]
        ho = [f for f, i in zip(feats, ids) if i in hold_ids]
        h = IncubatedHead(min_samples=c,
                          boost=self.cfg.get("boost", 0.3),
                          topk=self.cfg.get("topk", 8))
        try:
            h.incubate(tr, normal_core)
        except CandidateDeferred:
            raise
        except Exception as e:                # 训练失败（如类别退化）→ 保持旧头
            self.log.append({"t": time.time(), "ok": False, "approved": False,
                             "reason": "train_error", "err": str(e),
                             "n_images": n_img, **tag})
            return cur
        neg = _flat(normal_core)
        sep_new = sep_old = None
        if ho:
            pos_ho = torch_cat(ho).float().numpy()
            rng = np.random.default_rng(1)
            neg_ho = neg[rng.choice(len(neg), min(len(neg), len(pos_ho)), replace=False)]
            sep_new = h.separation(pos_ho, neg_ho)
            if cur is not None:
                sep_old = cur.separation(pos_ho, neg_ho)
        approve_min = float(self.cfg.get("approve_min_sep", 0.2))
        if sep_new is None:
            ok, reason = True, "no_holdout"     # 独立图不足以留出：记录但放行
        elif sep_new < approve_min:
            ok, reason = False, "below_min_sep"
        elif sep_old is not None and sep_new < sep_old:
            ok, reason = False, "worse_than_current"
        else:
            ok, reason = True, "holdout_pass"
        rec = {"t": time.time(), "ok": True, "approved": ok, "reason": reason,
               "n_pos": h.n_pos, "n_images": n_img, "n_holdout_images": n_hold,
               "sep_new": sep_new, "sep_old": sep_old, **tag}
        self.log.append(rec)
        if not ok:
            st["last_reject_n"] = n_img         # 未批准：不替换，保持旧头
            return cur
        st["last_sample_n"] = n_img
        st["last_sep"] = sep_new
        st["last_reject_n"] = None
        return h

    def boost(self, tile_feats):
        return self.head.score(tile_feats) if self.head else 0.0

    def type_scores(self, tile_feats):
        """per-type 孵育分（仅结果/报告用，不进 boost/融合）。无类型头 → {}。"""
        heads = getattr(self, "heads", None)
        if not heads:
            return {}
        return {t: h.score(tile_feats) for t, h in heads.items() if h is not None}
