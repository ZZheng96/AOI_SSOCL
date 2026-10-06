"""M3 主动选样（§7.5）：反馈预算花在最有价值的样本上。

价值 = 不确定度（灰区距离/槽位分歧）× 代表性（与库距离）。
产出 top-N 询问清单请操作员确认；确认结果汇入 §7.1 三类反馈。
"""
import numpy as np


class ActiveSelector:
    def __init__(self, top_n=10, w_unc=0.6, w_rep=0.4):
        self.top_n = top_n
        self.w_unc = w_unc
        self.w_rep = w_rep
        self.queue = []          # [{"path","t","fused","slots","_norm"}]（灰区样本）
        self.asked = set()       # 已询问路径（防重复）

    def enqueue(self, rec, item):
        """灰区样本入队（feedback._unlabeled 调用）。"""
        if rec["path"] in self.asked:
            return
        self.queue.append({"path": rec["path"], "t": rec["t"] if "t" in rec else 0,
                           "fused": rec["fused"], "slots": rec["slot_scores"],
                           "feats": item["feats"]})
        if len(self.queue) > 200:      # 队上限：最旧出队
            self.queue.pop(0)

    def _uncertainty(self, rec, lo, hi):
        """不确定度 = 距灰区中心的距离取反（越近中心越不确定）∈ [0,1]。"""
        span = max(hi - lo, 1e-9)
        center = (lo + hi) / 2
        return 1.0 - min(abs(rec["fused"] - center) / span, 1.0)

    def _disagreement(self, rec):
        """槽位间分歧：校准分方差（分歧大=不同槽位结论冲突，需人工仲裁）。"""
        v = np.array(list(rec["slots"].values()))
        return float(np.clip(np.std(v), 0, 1))

    def _representativeness(self, rec, pipeline):
        """代表性 = 与正常库距离（远=新模式，主动询问价值高）。"""
        from .banks import _norm_patch
        try:
            nb = pipeline.handler.normal_bank if pipeline.handler is not None else None
            if nb is not None:
                sim = nb._sim_to(_norm_patch(rec["feats"]))
                if sim is not None:
                    return 1.0 - float(np.max(sim))      # 相似度低 → 代表性高
        except Exception:
            pass
        return 0.5

    def select(self, pipeline, top_n=None):
        """对队列打分，返回 top-N 询问清单（并标记 asked 防重复）。"""
        n = top_n or self.top_n
        if not self.queue:
            return []
        lo, hi = pipeline.decider.tau_gray, pipeline.decider.tau_high
        scored = []
        for rec in self.queue:
            if rec["path"] in self.asked:
                continue
            u = self._uncertainty(rec, lo, hi)
            d = self._disagreement(rec)
            r = self._representativeness(rec, pipeline)
            value = self.w_unc * max(u, d) + self.w_rep * r
            scored.append({"path": rec["path"], "value": round(value, 4),
                           "uncertainty": round(u, 4), "disagreement": round(d, 4),
                           "representative": round(r, 4)})
        scored.sort(key=lambda x: -x["value"])
        picked = scored[:n]
        for p in picked:
            self.asked.add(p["path"])
        self.queue = [q for q in self.queue if q["path"] not in self.asked]
        return picked
