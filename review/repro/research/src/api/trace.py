"""M4 追溯 API（§9.2 交付物）：上位机查询接口，回溯检测逻辑。

统一暴露：
- 推理记录：逐样本 slot_scores/fused/decision/boxes/types/boost（inferences.jsonl）
- 更新日志：三类反馈、回滚、回归门控历史（handler 记录）
- 贡献档案：槽位贡献分解（contribution_report）
- 系统状态：权重、阈值、双库版本、熔断状态、校准器（供上位机展示与审计）

用法：
  pipe.attach_trace_api("outputs/m0/trace_api")
  api = pipe.trace_api
  api.system_state() / api.recent_inferences() / api.update_log() ...
"""
import os
import json
import numpy as np


def _jsonable(v):
    """递归转 JSON 可序列化（ndarray -> list）。"""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


class TraceAPI:
    def __init__(self, pipeline, log_dir):
        self.p = pipeline
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._f = open(os.path.join(log_dir, "inferences.jsonl"), "a", encoding="utf-8")
        self.n_records = 0

    def record(self, rec):
        """predict 后调用（Pipeline 自动挂钩）：推理记录落盘。"""
        slim = {k: rec[k] for k in ("path", "fused", "slot_scores", "decision",
                                    "boxes", "types", "router_w") if k in rec}
        if "boost" in rec:
            slim["boost"] = rec["boost"]
        self._f.write(json.dumps(_jsonable(slim), ensure_ascii=False) + "\n")
        self._f.flush()
        self.n_records += 1

    def close(self):
        self._f.close()

    # ---- 查询接口 ----
    def recent_inferences(self, n=20):
        """最近 n 条推理记录。"""
        rows = []
        if os.path.exists(os.path.join(self.log_dir, "inferences.jsonl")):
            with open(os.path.join(self.log_dir, "inferences.jsonl"), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        return rows[-n:]

    def update_log(self):
        """更新日志：三类反馈 + 回滚 + 回归门控历史（M3 SSOCL）。"""
        h = self.p.handler
        if h is None:
            return {"feedback": [], "rollback": [], "gate": []}
        return {"feedback": _jsonable(h.feedback_log),
                "rollback": _jsonable(h.rollback_log),
                "gate": _jsonable(h.gate.history)}

    def contribution(self, split_items):
        """槽位贡献档案（§3.2 三层评价）。"""
        return self.p.contribution_report(split_items)

    def system_state(self):
        """系统状态快照：权重/阈值/库版本/熔断/校准器。"""
        st = {"weights": self.p.weights,
              "decider": {"tau_high": float(self.p.decider.tau_high),
                          "tau_gray": float(self.p.decider.tau_gray)},
              "slots": sorted(self.p.slots),
              "calibrator_bins": self.p.cfg.get("calibration", {}).get("cdf_bins", 256),
              "fit_seconds": getattr(self.p, "fit_seconds", None),
              "n_records": self.n_records}
        if self.p.handler is not None:
            nb = self.p.handler.normal_bank
            db = self.p.handler.defect_bank
            st["ssocl"] = {
                "defect_ratio": round(nb.defect_ratio, 4),
                "fuse_blocked": nb.fuse_blocked,
                "normal_bank": {"ext_version": nb.version, "ext_size": len(nb.ext),
                                "core_size": 0 if nb.core is None else nb.core.shape[0]},
                "defect_bank": {"version": db.version, "n_samples": len(db.samples)},
                "difficult_queue": len(self.p.handler.difficult),
                "high_conf_cache": len(self.p.handler.high_conf),
                "active_queue": 0 if self.p.selector is None else len(self.p.selector.queue),
                "stats": self.p.handler.stats}
        if self.p.router_gate is not None:
            st["router_gate"] = self.p.router_gate
        if self.p.consensus_gate is not None:
            st["consensus"] = {k: self.p.consensus_gate[k]
                               for k in ("blend", "weights") if k in self.p.consensus_gate}
        return st

    def export(self, out_path):
        """全量导出（交付审计用）。"""
        doc = {"system_state": self.system_state(),
               "update_log": self.update_log()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        return out_path
