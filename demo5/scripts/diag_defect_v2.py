"""诊断 v2：缺陷原型通道（目标模式第 1 步续，2026-08-16）

v1 结论（diag_defect_channel）：全图 patch 入缺陷库 = 把缺陷库变成更大的正常库，
rel 分数反向（0.23）；孵育头 0.41；筛选入库（vs 20 图锚定库）0.22 更差。
根因假说：正常库仅 20 张锚定图 patch，覆盖不足 -> "最不相似 patch"筛出的多是
锚定库没见过的正常背景，不是缺陷。

v2 变体矩阵：
  - 正常库 N2 = 全部 init_normal 图 patch（随机抽样 cap）
  - 缺陷入库筛选：与 N2 相似最低的 top-k patch/图，k ∈ {32,128,256}
  - 聚合：rel top-k mean
  - MIL 逻辑回归头（筛选 patch 为正）
  - eval = te_good 全部 + 未反馈缺陷（50% 反馈率），功效更高
  - 按缺陷类型分层 AUROC（bridge/cold_solder/excess/insufficient）

输出：outputs/m0/diag_defect_v2_{category}.json
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


def defect_type(p):
    parts = p.replace("\\", "/").split("/")
    for i, x in enumerate(parts):
        if x == "test" and i + 1 < len(parts) - 1:
            return parts[i + 1]
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="solder_smt")
    ap.add_argument("--fb-ratio", type=float, default=0.5,
                    help="缺陷图反馈率（入缺陷库比例）")
    ap.add_argument("--normal-cap", type=int, default=16384,
                    help="正常库 patch 上限")
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

    test = bundle["test"]
    goods = [p for p, y in test if y == 0]
    defects = [p for p, y in test if y == 1]
    rng = np.random.default_rng(42)
    rng.shuffle(defects)
    n_fb = int(len(defects) * args.fb_ratio)
    fb_defects, hold_defects = defects[:n_fb], defects[n_fb:]
    print(f"[{args.category}] test: good={len(goods)} defect={len(defects)} "
          f"反馈缺陷={n_fb} 留评缺陷={len(hold_defects)}", flush=True)

    # ---- 正常库 N2：全部 init_normal patch，随机抽样 cap ----
    n_feats = []
    for p in bundle["init_normal"]:
        n_feats.append(_norm_patch(pipe._get_item(p)["feats"]).float())
    n_all = torch.cat(n_feats, dim=0)
    if n_all.shape[0] > args.normal_cap:
        idx = torch.randperm(n_all.shape[0], generator=torch.Generator().manual_seed(0))[:args.normal_cap]
        n_all = n_all[idx]
    n_all = n_all.half().to(device)
    print(f"[bank] 正常库 N2: {n_all.shape[0]} patch（全部 init_normal）", flush=True)

    # ---- 缺陷入库（筛选变体）+ eval 分数 ----
    banks = {k: [] for k in (32, 128, 256)}      # 每图保留与 N2 最不相似 top-k patch
    for p in fb_defects:
        f = _norm_patch(pipe._get_item(p)["feats"]).to(device)
        sims = torch.empty(f.shape[0], device=f.device, dtype=torch.float32)
        for p0 in range(0, f.shape[0], 2048):
            s = f[p0:p0 + 2048].float() @ n_all.float().T
            sims[p0:p0 + 2048] = s.max(dim=1).values
        order = sims.argsort()                    # 升序：最不相似在前
        for k in banks:
            banks[k].append(f[order[:k]])
    for k in banks:
        banks[k] = torch.cat(banks[k], dim=0).float()
        print(f"[bank] 筛选库 k={k}: {banks[k].shape[0]} patch", flush=True)

    # ---- eval：全部 good + 未反馈缺陷 ----
    eval_paths = [(p, 0) for p in goods] + [(p, 1) for p in hold_defects]

    def n2_sim(f):
        out = torch.empty(f.shape[0], device=f.device, dtype=torch.float32)
        for p0 in range(0, f.shape[0], 2048):
            s = f[p0:p0 + 2048].float() @ n_all.float().T
            out[p0:p0 + 2048] = s.max(dim=1).values
        return out

    ys, types, fused0 = [], [], []
    rel = {k: {kk: [] for kk in (1, 8)} for k in banks}
    for p, y in eval_paths:
        item = pipe._get_item(p)
        rec = pipe._record(item)
        f = _norm_patch(item["feats"]).to(device)
        ys.append(y); types.append(defect_type(p)); fused0.append(rec["fused"])
        ns = n2_sim(f)
        for k, d_ref in banks.items():
            ds = torch.empty(f.shape[0], device=f.device, dtype=torch.float32)
            for p0 in range(0, f.shape[0], 2048):
                s = f[p0:p0 + 2048].float() @ d_ref.T
                ds[p0:p0 + 2048] = s.max(dim=1).values
            r = ds - ns
            for kk in (1, 8):
                rel[k][kk].append(float(r.topk(min(kk, r.shape[0])).values.mean()))
    ys = np.array(ys)

    def au(x):
        return round(float(roc_auc_score(ys, np.asarray(x, dtype=np.float64))), 4)

    out = {"category": args.category, "n_eval": len(ys),
           "n_good": int((ys == 0).sum()), "n_defect": int(ys.sum()),
           "fb_ratio": args.fb_ratio, "base_fused_auroc": au(fused0)}
    for k in banks:
        for kk in (1, 8):
            out[f"rel_bank{k}_top{kk}"] = au(rel[k][kk])
    # 分层：主力缺陷类型
    from collections import defaultdict
    for k in (32, 128, 256):
        for tt in sorted(set(types)):
            if tt == "?":
                continue
            m = np.array([t == tt for t in types]) | (ys == 0)
            if m.sum() > 5 and (ys[m] == 1).sum() > 0 and (ys[m] == 0).sum() > 0:
                out.setdefault("by_type", {}).setdefault(f"rel_bank{k}_top1", {})[tt] = \
                    round(float(roc_auc_score(ys[m], np.array(rel[k][1])[m])), 4)

    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    out_path = os.path.join(cfg["output_dir"], f"diag_defect_v2_{args.category}.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
