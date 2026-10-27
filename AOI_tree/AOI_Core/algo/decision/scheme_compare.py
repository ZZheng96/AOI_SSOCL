"""AC-008/009：三方案对照（整图 / 全局+全量局部 / 全局+预算细判）与副作用分报。

纯函数，输入为逐图分数（离线评价或合成数据），不跑模型、不写存储。
显存不在此测量：调用方传入实测峰值（peak_vram_mb），缺失时如实记 None。
"""
import numpy as np

from .branch_budget import side_effects

SCHEMES = ("global_only", "global_local_full", "global_local_budget")


def _rates(pred, labels):
    pred, labels = np.asarray(pred, bool), np.asarray(labels, bool)
    pos, neg = labels.sum(), (~labels).sum()
    return {"recall": float(pred[labels].mean()) if pos else None,
            "fpr": float(pred[~labels].mean()) if neg else None}


def compare_schemes(images, tau_g, tau_l, peak_vram_mb=None):
    """images: list[dict]，每图：
        label(0/1)、s_global、blocks(全量块局部分 list)、budget_idx(预算细判选中的块索引)、
        defect_tiles(可选，缺陷所在块索引集合，用于覆盖率)、veto(可选，全局报警被局部否决的开关)。
    → {schemes: {name: 指标}, side_effects: {name: 四分类计数}, deltas_vs_global: …}"""
    labels = np.array([bool(im["label"]) for im in images])
    vram = peak_vram_mb or {}
    out, fx = {}, {}
    g_alarm = np.array([im["s_global"] >= tau_g for im in images])
    for name in SCHEMES:
        pred, n_blocks, cov_hit, cov_tot = [], 0, 0, 0
        cnt = {"added_by_local": 0, "confirmed": 0, "global_only": 0, "vetoed": 0, "none": 0}
        for im, ga in zip(images, g_alarm):
            blocks = np.asarray(im.get("blocks") or [], float)
            if name == "global_only":
                idx = []
            elif name == "global_local_full":
                idx = list(range(len(blocks)))
            else:
                idx = [int(i) for i in im.get("budget_idx") or []]
            n_blocks += len(idx)
            la = bool(len(idx)) and float(blocks[idx].max()) >= tau_l
            veto = bool(im.get("veto")) and bool(len(idx))
            local_veto = veto and not la
            k = side_effects(ga, la, veto=veto, local_veto=local_veto)
            cnt[k] += 1
            pred.append(k in ("added_by_local", "confirmed", "global_only"))
            dt = set(im.get("defect_tiles") or [])
            if dt:
                cov_tot += len(dt)
                cov_hit += len(dt & set(idx))
        pred = np.array(pred)
        r = _rates(pred, labels)
        out[name] = {"n_blocks_total": int(n_blocks),
                     "n_blocks_per_image": round(n_blocks / max(len(images), 1), 3),
                     "defect_tile_coverage": (round(cov_hit / cov_tot, 4) if cov_tot else None),
                     **r,
                     "rescued": int((pred & ~g_alarm & labels).sum()),          # 全局漏、方案检出
                     "new_false_alarms": int((pred & ~g_alarm & ~labels).sum()),
                     "veto_new_misses": int((~pred & g_alarm & labels).sum()),   # 全局检出被否决
                     "peak_vram_mb": vram.get(name),
                     "vram_source": "measured" if vram.get(name) is not None else "not_measured"}
        fx[name] = cnt
    base = out["global_only"]
    deltas = {n: {"recall": (None if out[n]["recall"] is None else round(out[n]["recall"] - base["recall"], 4)),
                  "fpr": (None if out[n]["fpr"] is None else round(out[n]["fpr"] - base["fpr"], 4))}
              for n in SCHEMES[1:]}
    return {"tau_global": float(tau_g), "tau_local": float(tau_l), "n_images": len(images),
            "schemes": out, "side_effects": fx, "deltas_vs_global": deltas}
