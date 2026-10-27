"""U91 裁剪子集选择过拟合检验（2026-08-22）：data_local test 分层切 A/B，
A 上选最优裁剪子集，B 上验证"选中子集 vs 全槽"——若 B 上选中子集仍 ≥ 全槽，
说明 U90 的裁剪选择未严重过拟合 test（机制可推广）；反之则 U90 的 0.76 是选择偏差。

只做 gold_finger/solder_smt（test 72/82 张，切分统计可用；component 9 / extra 28 太小）。

用法: python scripts/diag_prune_selection.py
"""
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from m0_baseline import build, run_one   # noqa: E402
from src.data import datalocal          # noqa: E402

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0")
CANDIDATES = {
    "gold_finger": [None, "blob", "disc,blob"],        # None=全槽
    "solder_smt": [None, "blob", "disc,blob"],
}


def split_test(test, seed=7, frac=0.5):
    """分层切 A/B（正常/缺陷各一半），返回 (A, B)。"""
    rng = np.random.default_rng(seed)
    n_idx = [i for i, (_, y) in enumerate(test) if y == 0]
    d_idx = [i for i, (_, y) in enumerate(test) if y == 1]
    n_a = set(rng.choice(n_idx, max(1, int(len(n_idx) * frac)), replace=False).tolist())
    d_a = set(rng.choice(d_idx, max(1, int(len(d_idx) * frac)), replace=False).tolist())
    a = set(n_a) | set(d_a)
    A = [test[i] for i in range(len(test)) if i in a]
    B = [test[i] for i in range(len(test)) if i not in a]
    return A, B


def run_slots(cfg, bundle, slots, device, tag):
    """用给定槽位子集 fit+eval 当前 bundle['test']，返回 (fused_auroc, 各槽位)。"""
    if slots:
        keep = set(x.strip() for x in slots.split(",") if x.strip())
        for k in cfg["slots"]:
            cfg["slots"][k]["enabled"] = k in keep
    else:
        for k in cfg["slots"]:
            cfg["slots"][k]["enabled"] = True
    backbone, slots_obj = build(cfg, device)

    from src.eval.offline import Pipeline
    pipe = Pipeline(cfg, backbone, slots_obj)
    pipe.fit(bundle)
    res = pipe.evaluate(bundle["test"])
    fu = res["fused"]["auroc"]
    print(f"  [{tag}] slots={slots or 'all'} test_AUROC={fu:.4f} "
          f"(n={len(bundle['test'])} 缺陷{sum(y for _, y in bundle['test'])})", flush=True)
    return fu


def main():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    print(f"[U91] 裁剪子集选择过拟合检验 device={device}", flush=True)
    for cat, cands in CANDIDATES.items():
        bundle = datalocal.load_category(cfg["datasets"]["data_local"], cat,
                                         cfg["protocol"]["n_init_normal"],
                                         cfg["protocol"]["n_init_defect"],
                                         cfg["seed"], max_test=0)
        A, B = split_test(bundle["test"])
        print(f"\n[U91] {cat}: test {len(bundle['test'])} → A {len(A)}（缺陷{sum(y for _, y in A)}）"
              f" / B {len(B)}（缺陷{sum(y for _, y in B)}）", flush=True)
        # A 上选参（每个候选独立 fit——注意 fit 用完整 init，仅 eval 集不同）
        scores = {}
        for c in cands:
            bA = dict(bundle); bA["test"] = A
            scores[c] = run_slots(cfg, bA, c, device, f"{cat}-A")
        best = max(scores, key=scores.get)
        print(f"  [U91] {cat} A 上最优子集 = {best or 'all'} (AUROC {scores[best]:.4f})", flush=True)
        # B 上验证：选中子集 vs 全槽
        bB = dict(bundle); bB["test"] = B
        s_best = run_slots(cfg, bB, best, device, f"{cat}-B")
        s_all = run_slots(cfg, bB, None, device, f"{cat}-B")
        verdict = "选择未过拟合" if s_best >= s_all else "选择过拟合（B 上全槽更好）"
        print(f"  [U91] {cat} B 验证: 选中子集 {s_best:.4f} vs 全槽 {s_all:.4f} → {verdict}", flush=True)


if __name__ == "__main__":
    main()
