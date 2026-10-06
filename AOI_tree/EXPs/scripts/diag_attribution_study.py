"""追溯正确性研究（U61 雏形，2026-08-16）：归因-学习强相关性的量化评价

核心洞见（用户提出）：既然能正确追溯到错检的逻辑（归因），就应该能推理出
正确检出的逻辑——追溯正确性与学习有效性应强相关。

可操作定义（反事实）：
  系统对样本 i 的归因 = argmax 校准分的槽位 X*（"最强信号来源"）。
  归因正确 = 移除 X* 后 fused 向真值方向移动：
    FP（真正常/高 fused）：fused^-X* < fused    （X* 是虚高来源 → 移除应下降）
    FN（真缺陷/低 fused）：fused^-X* > fused    （X* 抑制了判别 → 移除应上升）
  全样本"归因正确率" = 上述成立的比例。

配套输出：
  1. 逐槽位"虚高次数"（FP 样本上被归因为主因的次数）——看归因是否总指向真值反向槽
  2. 逐槽位 leave-one-out AUC（反事实的批量版）——槽位对融合的（静态）贡献
  3. 归因正确率 × 槽位贡献的关联——追溯正确性与"哪个槽位真正起作用"是否一致

输入：outputs/m0/fullslot_backup/m0_{dataset}_{cat}_single.json（全槽位 _per_sample + labels + weights）
"""
import json
import os
import sys
import glob
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.metrics import image_metrics
from src.fusion.fixed import fuse


def analyze(path):
    d = json.load(open(path, encoding="utf-8"))
    t = d.get("test", {})
    if "_per_sample" not in t:
        return None
    slots = sorted(t["_per_sample"])
    M = np.stack([np.asarray(t["_per_sample"][n], dtype=np.float64) for n in slots])  # (S,N)
    y = np.asarray(t["_per_sample_labels"], dtype=int)
    w = d.get("weights") or {n: 1.0 / len(slots) for n in slots}
    N = len(y)
    # 每样本 fused 与 leave-one-slot-out
    fused = np.array([fuse({n: float(M[i, j]) for i, n in enumerate(slots)}, w)
                      for j in range(N)])
    # 决策阈值近似：fused 在训练正常分位 0.99 → 用 0.5 简化（校准分语义）——保守
    # 更稳：用 fused 中位数附近的错检，避免阈值误差主导
    loo = np.zeros((len(slots), N))
    for i, n in enumerate(slots):
        w_loo = dict(w)
        w_loo.pop(n, None)
        s_ = {m: 1.0 / max(len(w_loo), 1) for m in w_loo}
        loo[i] = np.array([fuse({m: float(M[mi, j]) for mi, m in enumerate(slots) if m != n}, s_)
                           for j in range(N)])
    # 错检定义：用 fused 阈值 0.5（校准分语义：正常集中 0-0.5 占多数，>0.5 视为异常判定）
    fp = (y == 0) & (fused > 0.5)
    fn = (y == 1) & (fused <= 0.5)
    res = {"slots": slots, "n": int(N), "n_fp": int(fp.sum()), "n_fn": int(fn.sum()),
           "fused_auroc": image_metrics(y, fused)["auroc"]}
    # 归因正确率：主导槽位（argmax cal）
    dom = M.argmax(axis=0)                      # (N,) 每样本归因主导槽位
    correct_fp = fp.copy()
    correct_fn = fn.copy()
    for j in range(N):
        i = dom[j]
        if fp[j]:
            correct_fp[j] = bool(loo[i, j] < fused[j])     # 移除主导槽位 → 下降
        if fn[j]:
            correct_fn[j] = bool(loo[i, j] > fused[j])     # 移除 → 上升
    res["fp_correct"] = float(correct_fp.mean()) if int(fp.sum()) > 0 else None
    res["fn_correct"] = float(correct_fn.mean()) if int(fn.sum()) > 0 else None
    # 全错检样本归因正确率
    mis = fp | fn
    corr = np.zeros(N, dtype=bool)
    for j in np.where(mis)[0]:
        i = dom[j]
        corr[j] = (loo[i, j] < fused[j]) if fp[j] else (loo[i, j] > fused[j])
    res["mis_correct"] = float(corr[mis].mean()) if int(mis.sum()) > 0 else None
    # ===== v2（U61 方法修正）：反事实是"数学必然"（argmax 槽位移除必降 fused），
    # 不能证明追溯正确。更严格定义 = 归因指向的槽位是否是真值上反向/虚高的槽位：
    #   FP（正常误检）：归因槽位 X=argmax cal；若 slot_auroc[X] < 0.5（反向槽），
    #     则 X 的高分是假信号 → 归因正确（指向真问题）；若 X 是正常槽（AUROC>0.5），
    #     高分是真信号 → 问题在融合/阈值，归因应升级（不算槽位归因正确）。
    #   FN（缺陷漏检）：归因槽位 X=argmax cal；正常槽信号不足 → 归因指向判别薄弱槽。
    slot_auroc = {n: float(image_metrics(y, M[i])["auroc"]) for i, n in enumerate(slots)}
    fp_idx = list(np.where(fp)[0]); fn_idx = list(np.where(fn)[0])
    fp_correct2 = sum(1 for j in fp_idx if slot_auroc[slots[dom[j]]] < 0.5)
    fn_correct2 = sum(1 for j in fn_idx if slot_auroc[slots[dom[j]]] >= 0.5)
    res["fp_correct_v2"] = fp_correct2 / len(fp_idx) if fp_idx else None
    res["fn_correct_v2"] = fn_correct2 / len(fn_idx) if fn_idx else None
    res["fp_slots_n"] = {}
    res["fn_slots_n"] = {}
    for j in fp_idx:
        res["fp_slots_n"][slots[dom[j]]] = res["fp_slots_n"].get(slots[dom[j]], 0) + 1
    for j in fn_idx:
        res["fn_slots_n"][slots[dom[j]]] = res["fn_slots_n"].get(slots[dom[j]], 0) + 1
    res["slot_auroc"] = {n: round(v, 4) for n, v in slot_auroc.items()}
    # 逐槽位：虚高归因次数（FP 上被 argmax 的次数）、FN 上被 argmax 次数
    res["fp_dom_count"] = {slots[i]: int((fp & (dom == i)).sum()) for i in range(len(slots))}
    res["fn_dom_count"] = {slots[i]: int((fn & (dom == i)).sum()) for i in range(len(slots))}
    # 逐槽位静态贡献（leave-one-out AUC 差，反向槽=负贡献）
    res["loo_auroc"] = {}
    for i, n in enumerate(slots):
        res["loo_auroc"][n] = round(float(image_metrics(y, loo[i])["auroc"]), 4)
    res["slot_auroc"] = {n: round(float(image_metrics(y, M[i])["auroc"]), 4)
                         for i, n in enumerate(slots)}
    return res


def main():
    pats = sys.argv[1:] or [
        r"outputs\m0\fullslot_backup\m0_mvtec_screw_single.json",
        r"outputs\m0\fullslot_backup\m0_mvtec_transistor_single.json",
        r"outputs\m0\fullslot_backup\m0_mvtec_cable_single.json",
        r"outputs\m0\fullslot_backup\m0_btad_01_single.json",
        r"outputs\m0\fullslot_backup\m0_mpdd_bracket_white_single.json",
    ]
    for p in pats:
        if not os.path.exists(p):
            print(f"skip {p}")
            continue
        r = analyze(p)
        if r is None:
            print(f"no _per_sample: {p}")
            continue
        print(f"\n=== {os.path.basename(p)} (n={r['n']}, fused AUROC={r['fused_auroc']:.4f}) ===")
        print(f"  错检: FP={r['n_fp']} FN={r['n_fn']}")
        print(f"  [dbg] r 类型={type(r)} fp_correct repr={r.get('fp_correct')!r} "
              f"fn_correct repr={r.get('fn_correct')!r} mis_correct repr={r.get('mis_correct')!r}")
        fc, fnc, mc = r.get('fp_correct'), r.get('fn_correct'), r.get('mis_correct')
        print(f"  归因正确率(反事实, 数学必然=不可信): FP={fc if fc is None else f'{fc:.3f}'} "
              f"合计={mc if mc is None else f'{mc:.3f}'}")
        v2 = (r.get('fp_correct_v2'), r.get('fn_correct_v2'))
        print(f"  [v2 严格定义] 归因指向反向/薄弱槽位正确率: "
              f"FP={v2[0] if v2[0] is None else f'{v2[0]:.3f}'} "
              f"FN={v2[1] if v2[1] is None else f'{v2[1]:.3f}'}")
        print(f"  FP 样本主导槽位分布(被 argmax): {r.get('fp_slots_n')}")
        print(f"  逐槽位 AUROC: " + " ".join(f"{n}={r['slot_auroc'][n]:.3f}"
                                             for n in r["slots"]))
        print(f"  逐槽位: AUROC / LOO-AUROC / FP归因次数 / FN归因次数")
        for n in r["slots"]:
            print(f"    {n:<8} {r['slot_auroc'][n]:.3f} / {r['loo_auroc'][n]:.3f} "
                  f"/ {r['fp_dom_count'][n]} / {r['fn_dom_count'][n]}")


if __name__ == "__main__":
    main()
