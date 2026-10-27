"""离线融合方案分析（U31 后续，2026-08-16）

输入：m0 json（含 test._per_sample = {槽位: [每样本校准分]} + labels）
方案对照（全部在 test 域上评估，用于理解融合上界与在线学习收益；不参与训练）：
  equal          等权 mean（= 现 fused 基线）
  auc_oracle     test 侧 AUROC 归一化权重（oracle 上界）
  online_mwu     流式前 k 反馈样本的 Mann-Whitney U → 权重=max(0, U-0.5) 归一化，
                 剩余样本评估（模拟在线学习；k 从 10 到 50）
输出：各方案 AUROC
用法：python scripts/diag_fusion.py outputs/m0/m0_mpdd_connector_tiles36.json [--k 10,20,30]
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from sklearn.metrics import roc_auc_score


def mwu_auc(pos, neg):
    """Mann-Whitney U 归一化（= 一个正例一个负例时判对概率，等价 AUROC）"""
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    m, n = len(pos), len(neg)
    r = np.concatenate([pos, neg])
    order = np.argsort(r, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(r) + 1)
    # 平局取平均秩
    i = 0
    while i < len(r):
        j = i
        while j + 1 < len(r) and r[order[j + 1]] == r[order[i]]:
            j += 1
        if j > i:
            avg = ranks[order[i]]
            for t in range(i, j + 1):
                ranks[order[t]] = avg
        i = j + 1
    return (ranks[len(pos):].sum() - n * (n + 1) / 2) / (m * n) if m * n else 0.5


def run(path, ks):
    d = json.load(open(path, encoding="utf-8"))
    per = d["test"].get("_per_sample")
    if not per:
        print(f"[skip] {path}: 无 _per_sample（需 U31 后重跑生成）")
        return
    labels = np.asarray(d["test"]["_per_sample_labels"])
    slots = list(per.keys())
    X = np.stack([per[n] for n in slots], axis=1)          # (N, S)
    y = labels
    N = len(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    print(f"[{os.path.basename(path)}] N={N} 缺陷={len(pos)} 正常={len(neg)} 槽位={slots}")

    def auroc(scores):
        return float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")

    # equal
    s_equal = X.mean(axis=1)
    print(f"  equal:      {auroc(s_equal):.4f}")

    # oracle（test AUROC 归一化，正权重）
    aucs = np.array([auroc(X[:, j]) for j in range(X.shape[1])])
    w_or = np.clip(aucs - 0.5, 0, None)
    if w_or.sum() > 0:
        w_or = w_or / w_or.sum()
        print(f"  auc_oracle: {auroc(X @ w_or):.4f}  w={np.round(w_or, 3).tolist()}")

    # online_mwu：前 k 反馈估计权重，剩余评估
    for k in ks:
        if k >= N or len(pos) == 0 or len(neg) == 0:
            continue
        fb_idx = np.concatenate([pos[: max(1, k // 2)], neg[: max(1, k // 2)]])
        u = np.array([mwu_auc(X[fb_idx][y[fb_idx] == 1, j],
                              X[fb_idx][y[fb_idx] == 0, j])
                      for j in range(X.shape[1])])
        w = np.clip(u - 0.5, 0, None)
        if w.sum() > 0:
            w = w / w.sum()
            rest = np.setdiff1d(np.arange(N), fb_idx)
            yr = y[rest]
            if len(np.unique(yr)) > 1:
                a = roc_auc_score(yr, X[rest] @ w)
                print(f"  online_mwu k={k}: {a:.4f}  w={np.round(w, 3).tolist()}")
            else:
                print(f"  online_mwu k={k}: 剩余集单类，跳过")
        else:
            print(f"  online_mwu k={k}: 无正权重（U 全 ≤0.5），回退等权")
    # 各槽位单独 AUROC（参考）
    print(f"  slot_auc:   " + " ".join(f"{s}={auroc(X[:, j]):.3f}"
                                       for j, s in enumerate(slots)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="m0 json 路径（可多个，用空格分隔）")
    ap.add_argument("--k", default="10,20,30,50")
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]
    for p in args.json.split():
        run(p, ks)


if __name__ == "__main__":
    main()
