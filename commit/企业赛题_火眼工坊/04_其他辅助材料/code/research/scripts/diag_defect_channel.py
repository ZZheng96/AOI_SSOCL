"""诊断：缺陷原型通道的判别力上限（目标模式第 1 步，2026-08-16）

问题：当前缺陷反馈只走 intercept_score 加分（≤0.25×相对差），学习后 AUROC
最高 0.52。要到 0.9+，缺陷通道必须成为主导信号。

实验（诚实协议）：
  - fit（train/good 100 + init_defect 30）
  - 模拟完美反馈：stream（test[:-30]）中缺陷图的 tile 特征入 DefectBank
    （生产语义 = 操作员确认 NG；eval 集图片绝不入库）
  - eval（test[-30:]，与 stream 不重叠）上算各变体缺陷通道分：
      v_intercept  当前实现（相对相似 top8 × boost0.25）
      v_rel_k      相对相似原始值（缺陷库 max − 正常库 max），k∈{1,4,8,16} mean
      v_rel_sel    原型筛选入库（每缺陷图只留与正常库最不相似的 top20% patch）
      v_head       孵育头（逻辑回归）top-k 概率
  - 输出：各变体单独 AUROC + 与基础 fused 融合的 AUROC

输出：outputs/m0/diag_defect_channel_{category}.json
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline
from src.ssocl.banks import _norm_patch

SSOCL_CFG = {
    "banks": {"ext_max": 64, "cluster_k": 3, "sim_thresh": 0.75,
              "intercept_topk": 8, "intercept_boost": 0.25, "intercept_sim": 0.15},
    "gate": {"tol": 0.05, "anchor_size": 20},
    "active": {"top_n": 10},
    "incubate": {"min_samples": 8},
}


def rel_scores(f, d_ref, nb):
    """相对相似：每 patch 与缺陷库 max 相似 − 与正常库 max 相似。"""
    d_sim = torch.empty(f.shape[0], device=f.device, dtype=torch.float32)
    for p0 in range(0, f.shape[0], 2048):
        s = f[p0:p0 + 2048].float() @ d_ref.T
        d_sim[p0:p0 + 2048] = s.max(dim=1).values
    n_sim = torch.as_tensor(nb._sim_to(f), device=f.device, dtype=torch.float32)
    return d_sim - n_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="solder_smt")
    ap.add_argument("--eval-n", type=int, default=30)
    ap.add_argument("--sel-ratio", type=float, default=0.2,
                    help="原型筛选：每缺陷图保留与正常库最不相似的 top 比例")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(
        os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = datalocal.load_category(
        cfg["datasets"]["data_local"], args.category,
        cfg["protocol"]["n_init_normal"], cfg["protocol"]["n_init_defect"], cfg["seed"])

    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    pipe.enable_ssocl(SSOCL_CFG, train_normal_paths=bundle["init_normal"],
                      rng=np.random.default_rng(42))
    handler = pipe.handler

    test = bundle["test"]
    stream = test[:len(test) - args.eval_n]
    eval_items = test[len(test) - args.eval_n:]
    n_str_def = sum(1 for _, y in stream if y == 1)
    n_eva_def = sum(1 for _, y in eval_items if y == 1)
    print(f"[{args.category}] test={len(test)} stream={len(stream)}"
          f"(缺陷{n_str_def}) eval={len(eval_items)}(缺陷{n_eva_def})", flush=True)

    # ---- 模拟完美反馈：stream 缺陷图入库（全量原型 + 筛选原型两套）----
    sel_patches = []
    for p, y in stream:
        if y != 1:
            continue
        feats = pipe._get_item(p)["feats"]
        handler.defect_bank.add(feats, "perfect_fb")
        f = _norm_patch(feats)
        n_sim = torch.as_tensor(handler.normal_bank._sim_to(f),
                                device=f.device, dtype=torch.float32)
        k = max(4, int(f.shape[0] * args.sel_ratio))
        sel_patches.append(f[n_sim.topk(k, largest=False).indices])
    d_ref = torch.cat([s[0] for s in handler.defect_bank.samples], dim=0).float()
    d_sel = torch.cat(sel_patches, dim=0).float()
    print(f"[bank] 全量原型 {d_ref.shape[0]} patch / 筛选原型 {d_sel.shape[0]} patch",
          flush=True)

    # ---- 孵育头 ----
    from src.ssocl.incubate import IncubatedHead
    head = IncubatedHead(topk=8, boost=1.0)
    head.incubate(handler.defect_bank, handler.normal_bank.core)

    # ---- eval 上算分 ----
    ys, fused0 = [], []
    v_intercept, v_head = [], []
    v_rel, v_rel_sel = {k: [] for k in (1, 4, 8, 16)}, {k: [] for k in (1, 4, 8, 16)}
    for p, y in eval_items:
        item = pipe._get_item(p)
        rec = pipe._record(item)               # 基础融合（拦截前）
        f = _norm_patch(item["feats"])
        ys.append(y)
        fused0.append(rec["fused"])
        v_intercept.append(handler.defect_bank.intercept_score(
            item["feats"], normal_bank=handler.normal_bank))
        v_head.append(head.score(item["feats"]))
        rel = rel_scores(f, d_ref, handler.normal_bank)
        rel_s = rel_scores(f, d_sel, handler.normal_bank)
        for k in (1, 4, 8, 16):
            v_rel[k].append(float(rel.topk(min(k, rel.shape[0])).values.mean()))
            v_rel_sel[k].append(float(rel_s.topk(min(k, rel_s.shape[0])).values.mean()))
        print(f"  {os.path.basename(p)[:40]:<42} y={y} fused={rec['fused']:.3f} "
              f"rel1={v_rel[1][-1]:+.3f} relsel1={v_rel_sel[1][-1]:+.3f}", flush=True)

    ys = np.array(ys)
    fused0 = np.array(fused0)
    def au(x):
        return round(float(roc_auc_score(ys, np.asarray(x, dtype=np.float64))), 4)

    out = {"category": args.category, "n_eval": len(ys), "n_defect": int(ys.sum()),
           "n_bank_samples": len(handler.defect_bank.samples),
           "base_fused_auroc": au(fused0),
           "v_intercept_auroc": au(v_intercept), "v_head_auroc": au(v_head),
           "v_rel_auroc": {k: au(v) for k, v in v_rel.items()},
           "v_rel_sel_auroc": {k: au(v) for k, v in v_rel_sel.items()},
           "fuse_fused_plus_rel8": au(fused0 + np.array(v_rel[8])),
           "fuse_fused_plus_relsel8": au(fused0 + np.array(v_rel_sel[8])),
           "fuse_max_fused_relsel4": au(np.maximum(fused0, np.array(v_rel_sel[4])))}

    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    out_path = os.path.join(cfg["output_dir"],
                            f"diag_defect_channel_{args.category}.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
