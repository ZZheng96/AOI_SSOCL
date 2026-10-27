"""C/D/E 块（AC-018~027）：校准契约 / 融合基线 / 路由准入 / 域差 / 槽位能力 / 权重语义。

全部为纯函数 + 静态表：只消费 fit 已算好的分数（零额外前向），供 fit 末尾
生成 pipe.cde_report、persist 落盘 cde_report.json、API/UI 展示。

口径约定（写进报告字段，防止误读）：
- 校准分 = 正常分布分位（"比多少比例的正常训练图更异常"），**不是缺陷概率**；
- 所有 AUC 均为协议内开发集（train/good + init_defect）统计，test 只验不选；
- router 默认 trace_only：只有分层 val/test 上同时胜过固定权重与硬规则 margin
  才 permitted，且 permitted 只表示"允许人工切换"，不自动接管。
"""
import numpy as np

# ---------------------------------------------------------------- 静态表
# AC-025 槽位能力清单：参照 / fit 类型 / 是否用缺陷 / 定位方式 / 域敏感性
SLOT_CAPABILITIES = {
    "sem": {"reference": "train/good tile 特征记忆库（PatchCore 式 coreset）",
            "fit_type": "nonparam_memory", "uses_defect": False,
            "localization": "patch 最近邻距离热图", "domain_sensitivity": "high"},
    "disc": {"reference": "伪异常合成（cutpaste/移植/色斑）+ 可选 init_defect 块",
             "fit_type": "discriminative_head", "uses_defect": True,
             "localization": "tile 判别分热图", "domain_sensitivity": "medium"},
    "inp": {"reference": "图内自相似（内在原型），不依赖训练库",
            "fit_type": "zero_shot", "uses_defect": False,
            "localization": "patch 与图内原型偏离", "domain_sensitivity": "low"},
    "open": {"reference": "train/good 特征 PCA 子空间",
             "fit_type": "pca_reconstruction", "uses_defect": False,
             "localization": "重构误差 topk（不进融合，仅不完备哨兵）",
             "domain_sensitivity": "high"},
    "shead": {"reference": "图像级判别头：伪异常 + init_defect 真实正样本",
              "fit_type": "discriminative_head", "uses_defect": True,
              "localization": "滑窗判别分", "domain_sensitivity": "medium"},
    "blob": {"reference": "DoG 斑点响应 vs train/good 统计",
             "fit_type": "handcrafted_stat", "uses_defect": False,
             "localization": "斑点位置", "domain_sensitivity": "low"},
    "trad": {"reference": "传统特征组（纹理/边缘）vs train/good 统计",
             "fit_type": "handcrafted_stat", "uses_defect": "group_select_only",
             "localization": "块级统计偏离", "domain_sensitivity": "medium"},
    "layout": {"reference": "train/good 布局统计（尺寸/数量/顺序）",
               "fit_type": "statistical_layout", "uses_defect": False,
               "localization": "部件框", "domain_sensitivity": "medium"},
    "tpl": {"reference": "用户显式金样板（非隐式配对）",
            "fit_type": "template_diff", "uses_defect": False,
            "localization": "对位差分图", "domain_sensitivity": "low"},
    "color": {"reference": "train/good 色彩统计",
              "fit_type": "handcrafted_stat", "uses_defect": False,
              "localization": "色差区域", "domain_sensitivity": "medium"},
}

# AC-033 核心槽与融合后拦截通道的职责边界：训练 / 更新 / 定位 / 域敏感 / 不承诺项
SLOT_ROLES = {
    "disc": {"role": "共享冻结 DINO 特征上的正常/异常判别边界",
             "train": "fit 时伪异常 + 可选 init_defect 块训练轻量头",
             "update": "disc_ft：带框反馈的框内 patch 在线微调头参数（gate 否决回滚，DINO 不动）",
             "not": "不是缺陷类型分类器"},
    "sem": {"role": "正常外观记忆（最近邻距离）",
            "train": "fit 时 train/good coreset，无梯度",
            "update": "NormalBank 锚定 core 不变，扩展库经门控增删",
            "not": "向量库不是训练权重"},
    "inp": {"role": "图内自相似，零样本兜底",
            "train": "无训练、无库",
            "update": "不随反馈更新",
            "not": "对全图一致的系统性缺陷不敏感"},
}
POST_FUSION_ROLES = {
    "defect_knn": {"role": "确认缺陷样例的近邻后验拦截（加分 ∝ 缺陷库相似 − 正常库相似）",
                   "train": "无训练；人工确认框/掩码 patch 入 DefectBank",
                   "update": "入库即生效，经锚定 FPR 门控，失败回滚",
                   "localization": "命中 patch（topk 相对相似）",
                   "domain_sensitivity": "high",
                   "not": "1～2 张图只承诺检出相似缺陷；只加分不减分"},
    "incubate": {"role": "确认样例孵育的轻量头，后验拦截加分",
                 "train": "纯 patch（add_box/add_patch）vs 锚定 patch，按独立图计数",
                 "update": "达 min_samples 后孵育，holdout 批准且不劣于旧头才替换",
                 "localization": "patch 判别分 topk",
                 "domain_sensitivity": "high",
                 "not": "不替代 disc，不进融合权重，不做类型分类"},
}

# AC-024 冷启动数据角色：正常图 vs 缺陷图各自喂给谁
COLD_START_ROLES = {
    "normal_images": ["sem 记忆库", "槽位正常统计（blob/trad/layout/color）",
                      "CDF 校准", "判定阈值 τ/灰区", "open PCA", "局部分支正常库"],
    "defect_images": ["槽位体检 slot_sanity", "判别头 disc/shead 正样本",
                      "sanity 融合权重", "router 训练/准入评估"],
    "backbone": "DINO 冻结（不随品类微调）",
    "never": "test 图不进任何 fit/选择（红线 3）",
}

# AC-027 权重语义：五个概念分开，gate ≠ weights
WEIGHTS_SEMANTICS = {
    "raw_scores": "槽位原始分（尺度各异，不可直接比较）",
    "calibrated_scores": "CDF 校准分：正常分布分位，非概率；>1 为超训练最大值外推",
    "effective_weights": "本次判定实际生效的融合权重（fixed/sanity/consensus/domain_prior 之一）",
    "router_candidate_weights": "router softmax 候选权重：仅追溯，applied=false，不参与 fused",
    "gate": "router 准入结论（permitted/trace_only），是门控判断不是权重",
}


def _auc(neg, pos):
    from sklearn.metrics import roc_auc_score
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if neg.size == 0 or pos.size == 0:
        return None
    return round(float(roc_auc_score([0] * neg.size + [1] * pos.size,
                                     np.concatenate([neg, pos]))), 4)


def _fuse_mat(cal, names, w):
    """cal: dict[name -> array]；w: dict。返回逐样本加权和（与 fixed.fuse 同口径）。"""
    if not names:
        return np.zeros(0)
    return sum(float(w.get(n, 0.0)) * np.asarray(cal[n], float) for n in names)


# ---------------------------------------------------------------- AC-018/019
def calibration_contract(fit_scores, calibrators, cdf_bins, isolation="self",
                         holdout_scores=None, min_n=20, n_boot=200, seed=0):
    """每槽位校准契约：方法/样本量/查表/外推/裁剪 + 尾部稳定性 + 自匹配偏差。

    fit_scores: dict[name -> list]  拟合校准器所用正常分；
    holdout_scores: dict[name -> list] holdout 模式下"自匹配侧"正常分（槽位 fit 用过的图）。
    尾部稳定性：bootstrap 的 p95/p99 标准差（原始分尺度）。
    self_match_bias：自匹配侧在独立校准器下的 p95 − 0.95。<0 表示自匹配压低正常分——
    若像 self 模式那样用这些分拟合 CDF/τ，τ 会被低估（偏离"真实正常"分位，新正常图 FPR 膨胀）。"""
    rng = np.random.default_rng(seed)
    out = {}
    for n, cal in calibrators.items():
        s = np.asarray(fit_scores.get(n, []), float)
        is_tpl = type(cal).__name__ == "TplCalibrator"
        row = {
            "method": "tpl_linear" if is_tpl else "cdf",
            "n": int(s.size),
            "n_bins": None if is_tpl else int(cdf_bins),
            "lookup": ("s / ref（ref=train p95，退化时 0.05）" if is_tpl
                       else "分位边界 searchsorted（side=right）→ idx/n_bins"),
            "extrapolation": ("线性（无上界外推区分）" if is_tpl
                              else "超上界按末段斜率线性外推 >1；超下界线性外推 <0"),
            "clip": [-1.0, 2.0],
            "semantics": "异常程度分位（比多少比例的正常图更异常），非缺陷概率",
            "small_sample": bool(s.size < min_n),
            "separate_contract": bool(is_tpl),
        }
        if s.size >= 3:
            boots = [np.quantile(rng.choice(s, s.size, replace=True), [0.95, 0.99])
                     for _ in range(n_boot)]
            b = np.asarray(boots)
            row["tail"] = {"p95": round(float(np.quantile(s, 0.95)), 6),
                           "p99": round(float(np.quantile(s, 0.99)), 6),
                           "p95_boot_std": round(float(b[:, 0].std()), 6),
                           "p99_boot_std": round(float(b[:, 1].std()), 6),
                           "n_boot": n_boot}
            # 尾部可分辨精度：p99 以上只有 n*(1-0.99) 个样本支撑
            row["tail"]["support_above_p99"] = round(float(s.size * 0.01), 2)
        h = None if holdout_scores is None else holdout_scores.get(n)
        if h is not None and len(h):
            # 自匹配侧（槽位 fit 用过的正常图）在独立校准器下的分布
            hc = np.asarray(cal.transform(h), float)
            row["self_matched_side"] = {"n": int(len(h)),
                                        "cal_p95": round(float(np.quantile(hc, 0.95)), 4),
                                        "cal_median": round(float(np.median(hc)), 4)}
            row["self_match_bias"] = round(float(np.quantile(hc, 0.95)) - 0.95, 4)
        out[n] = row
    return {"isolation": isolation,
            "self_match_risk": isolation == "self",
            "note": ("self：校准器与阈值在槽位 fit 同批正常图上统计，存在自匹配"
                     "（sem 记忆库含自身→正常分偏低→阈值偏松）风险"
                     if isolation == "self" else
                     "holdout：槽位只在 fit 子集拟合，CDF/阈值/open 用独立留出正常图"),
            "slots": out}


# ---------------------------------------------------------------- AC-020
def fusion_baselines(cal_n, cal_d, names, weights_by_mode, prod_mode, slot_sanity=None,
                     exclude_unverified=True):
    """固定融合基线分开报告。weights_by_mode: {equal/sanity/consensus/domain_prior: w}。
    hard_rule = 槽位校准分取 max（无学习的硬规则基线）。
    unverified：sanity 缺失或 ≤0.5 的槽位（exclude_unverified 时不进基线）。"""
    sanity = slot_sanity or {}
    unverified = [n for n in names if sanity.get(n) is None or sanity.get(n, 0) <= 0.5]
    used = [n for n in names if not (exclude_unverified and n in unverified)] or list(names)
    rows = {}
    for mode, w in weights_by_mode.items():
        if not w:
            continue
        ws = {n: float(w.get(n, 0.0)) for n in used}
        s = sum(ws.values())
        ws = {n: v / s for n, v in ws.items()} if s > 0 else {n: 1.0 / len(used) for n in used}
        rows[mode] = {"auc": _auc(_fuse_mat(cal_n, used, ws), _fuse_mat(cal_d, used, ws)),
                      "weights": {n: round(v, 4) for n, v in ws.items()}}
    if used:
        hn = np.max(np.stack([np.asarray(cal_n[n], float) for n in used]), axis=0)
        hd = np.max(np.stack([np.asarray(cal_d[n], float) for n in used]), axis=0)
        rows["hard_rule_max"] = {"auc": _auc(hn, hd), "weights": None}
    return {"production_mode": prod_mode, "slots_used": used,
            "unverified_slots": unverified, "exclude_unverified": bool(exclude_unverified),
            "evaluation": "in_sample（协议内 train/good + init_defect，非 test）",
            "baselines": rows}


# ---------------------------------------------------------------- AC-021/023
def stratified_split(n_neg, n_pos, ratios=(0.6, 0.2, 0.2), seed=42):
    """正负样本各自按比例切 train/val/test（分层），返回 {split: (neg_idx, pos_idx)}。"""
    rng = np.random.default_rng(seed)

    def _cut(n):
        idx = rng.permutation(n)
        a = int(round(n * ratios[0]))
        b = a + int(round(n * ratios[1]))
        return idx[:a], idx[a:b], idx[b:]
    nn_, pp_ = _cut(n_neg), _cut(n_pos)
    return {k: (nn_[i], pp_[i]) for i, k in enumerate(("train", "val", "test"))}


def router_admission(neg_feats, pos_feats, names, base_weights, rcfg, seed=42,
                     min_per_split=3):
    """分层三集准入：router 只在 train 拟合，val/test 同时胜固定权重与硬规则 margin
    才 permitted。返回报告（含过拟合差、归一化熵、维度签名）。"""
    import torch
    from .router import LinearRouter, train_router_on_matrix
    margin = float(rcfg.get("admission_margin", rcfg.get("gate_margin", 0.005)))
    neg_feats, pos_feats = np.asarray(neg_feats, float), np.asarray(pos_feats, float)
    sp = stratified_split(len(neg_feats), len(pos_feats), seed=seed)
    rep = {"slot_names": list(names), "n_slots": len(names),
           "n_params": len(names) * len(names) + len(names),
           "split_sizes": {k: {"neg": int(len(v[0])), "pos": int(len(v[1]))}
                           for k, v in sp.items()},
           "margin": margin, "decision": "trace_only", "permitted": False}
    if any(len(v[0]) < min_per_split or len(v[1]) < min_per_split for v in sp.values()):
        rep["reason"] = f"样本不足（每集正/负 <{min_per_split}），只做追溯"
        rep["insufficient"] = True
        return rep
    r = LinearRouter(len(names))
    tr_n, tr_p = sp["train"]
    r, info = train_router_on_matrix(
        r, pos_feats[tr_p], neg_feats[tr_n], epochs=rcfg.get("epochs", 50),
        lr=rcfg.get("lr", 1e-3), margin=rcfg.get("margin", 0.1),
        lam_entropy=rcfg.get("lam_entropy", 0.1), seed=seed)
    bw = np.array([float(base_weights.get(n, 0.0)) for n in names])
    aucs = {}
    ent = []
    for k, (ni, pi) in sp.items():
        with torch.no_grad():
            fn, wn = r.fused(torch.tensor(neg_feats[ni], dtype=torch.float32))
            fp, wp = r.fused(torch.tensor(pos_feats[pi], dtype=torch.float32))
        w_all = torch.cat([wn, wp]).numpy()
        ent.append(float((-(w_all * np.log(np.clip(w_all, 1e-12, 1))).sum(-1)).mean()))
        aucs[k] = {"router": _auc(fn.numpy(), fp.numpy()),
                   "fixed": _auc(neg_feats[ni] @ bw, pos_feats[pi] @ bw),
                   "hard_rule": _auc(neg_feats[ni].max(1), pos_feats[pi].max(1))}
    h_max = np.log(max(len(names), 2))
    rep.update({"auc": aucs, "train_info": info,
                "overfit_gap": round(aucs["train"]["router"] - aucs["val"]["router"], 4),
                "norm_entropy": round(float(np.mean(ent)) / h_max, 4),
                "collapsed": bool(float(np.mean(ent)) / h_max < 0.1)})
    ok = all(aucs[k]["router"] >= max(aucs[k]["fixed"], aucs[k]["hard_rule"]) + margin
             for k in ("val", "test"))
    rep["permitted"] = bool(ok and not rep["collapsed"])
    rep["decision"] = "permitted" if rep["permitted"] else "trace_only"
    rep["reason"] = ("val/test 均胜固定权重与硬规则 margin" if rep["permitted"] else
                     "未在 val/test 同时胜固定权重与硬规则（或权重坍缩），只做追溯")
    return rep


# ---------------------------------------------------------------- AC-022
def domain_shift(ref_fused, new_fused, tau, alpha, slot_sanity=None,
                 drift_iqr_thresh=0.5, fpr_ratio_thresh=2.0, min_n=5):
    """域差指标：fused 中位数漂移/IQR、正常 FPR 相对 α 膨胀、槽位可靠性。
    ref_fused 为训练正常 fused；new_fused 为新域正常 fused（None=仅给定义）。"""
    ref = np.asarray(ref_fused, float)
    rep = {"definition": {
        "median_shift_iqr": "(median(new) − median(ref)) / IQR(ref)",
        "fpr_inflation": "新域正常图 FPR(τ) / α",
        "thresholds": {"median_shift_iqr": drift_iqr_thresh, "fpr_inflation": fpr_ratio_thresh},
        "normal_drift": f"两项均未超阈值视为正常漂移；样本 <{min_n} 不判"},
        "slot_reliability": {n: ("reliable" if v is not None and v >= 0.7 else
                                 "weak" if v is not None and v > 0.5 else "unreliable")
                             for n, v in (slot_sanity or {}).items()},
        "mechanisms": {
            "hard_prior": "fusion.domain_weights：train_auroc<阈值时按先验倾斜（静态）",
            "regress_gate": "在线回流 RegressGate：正常均值漂移超限即回滚（守门）",
            "calibration_drift": "归因结论：建议回流重建+重估 CDF（诊断，不改权重）",
            "dynamic_routing": "not_integrated（未接入前不称为动态路由）"}}
    if new_fused is None or len(new_fused) < min_n or ref.size < min_n:
        rep["status"] = "insufficient"
        return rep
    new = np.asarray(new_fused, float)
    iqr = float(np.subtract(*np.quantile(ref, [0.75, 0.25]))) or 1e-12
    shift = (float(np.median(new)) - float(np.median(ref))) / iqr
    fpr = float((new >= tau).mean())
    infl = fpr / max(alpha, 1e-12)
    rep.update({"n_ref": int(ref.size), "n_new": int(new.size),
                "median_shift_iqr": round(shift, 4), "fpr_new": round(fpr, 4),
                "alpha": alpha, "fpr_inflation": round(infl, 4),
                "status": ("shift" if abs(shift) > drift_iqr_thresh or infl > fpr_ratio_thresh
                           else "normal_drift")})
    return rep


# ---------------------------------------------------------------- AC-025
def slot_health(pipe):
    """健康页：能力清单 + 运行时状态（fitted/校准样本量/体检/权重/小样本）。"""
    contract = ((getattr(pipe, "cde_report", None) or {}).get("calibration") or {}).get("slots", {})
    rows = {}
    names = list(getattr(pipe, "slots", {}).keys())
    if getattr(pipe, "open_det", None) is not None and "open" not in names:
        names.append("open")
    for n in names:
        cal = (getattr(pipe, "calibrators", {}) or {}).get(n)
        san = (getattr(pipe, "slot_sanity", {}) or {}).get(n)
        hist = (getattr(pipe, "_cal_normal_hist", {}) or {}).get(n) or []
        rows[n] = {**SLOT_CAPABILITIES.get(n, {"reference": "unknown"}),
                   "fitted": cal is not None or (n == "open" and pipe.open_det is not None),
                   "in_fusion": n != "open",
                   "cal_n": len(hist),
                   "sanity_auc": san,
                   "weight": round(float((pipe.weights or {}).get(n, 0.0)), 4)
                   if getattr(pipe, "weights", None) else None,
                   "small_sample": contract.get(n, {}).get("small_sample"),
                   "status": ("unverified" if san is None else
                              "reverse" if san < 0.5 else "weak" if san < 0.7 else "ok")}
    for n, r in SLOT_ROLES.items():
        if n in rows:
            rows[n].update(r)
    return {"slots": rows, "post_fusion": post_fusion_health(pipe),
            "cold_start_roles": COLD_START_ROLES,
            "weights_semantics": WEIGHTS_SEMANTICS}


def post_fusion_health(pipe):
    """AC-033：融合后拦截通道（缺陷近邻 / 孵育头）职责 + 运行时状态；不进融合权重。"""
    h = getattr(pipe, "handler", None)
    db = getattr(h, "defect_bank", None) if h is not None else None
    hm = getattr(pipe, "head_mgr", None)
    n_img = len(set(db.incubation_pool()[1])) if db is not None else 0
    knn = {**POST_FUSION_ROLES["defect_knn"], "in_fusion": False,
           "active": bool(db is not None and db.samples
                          and not getattr(pipe, "_ablate_intercept", False)),
           "n_samples": len(db.samples) if db is not None else 0,
           "n_pure_images": n_img}
    st = hm.status(db) if hm is not None and db is not None else {}
    inc = {**POST_FUSION_ROLES["incubate"], "in_fusion": False,
           "enabled": hm is not None, "trained": bool(st.get("trained")),
           "min_samples": st.get("min_samples"), "n_pure_images": n_img,
           "separation": st.get("separation"),
           "last_decision": (st.get("last_decision") or {}).get("reason"),
           "per_type": st.get("per_type") or {}}
    return {"defect_knn": knn, "incubate": inc}


# ---------------------------------------------------------------- AC-037
# 机制事实标记：按代码实际行为填写，不夸大（评审要求"按事实标记"）
MECHANISM_FACTS = {
    "auto_reflow": {"status": "conditional",
                    "fact": "仅 reflow_candidate 成簇 accepted 且未熔断才回流；真值正常走 labeled_reflow；全程 RegressGate 回滚"},
    "clustering": {"status": "implemented_limited",
                   "fact": "NormalBank 余弦成簇判定（cluster_k/sim_thresh），非无监督聚类发现新类"},
    "multi_head": {"status": "implemented_limited",
                   "fact": "HeadManager 除通用头外按 defect_type 维护类型头 dict（类型独立图≥min_samples 才孵育，"
                           "同走 holdout 批准且 separation 不降）；仅通用头进拦截/boost，类型头只出 per-type 分数"},
    "auto_feature_registration": {"status": "candidate_only",
                                  "fact": "E_open new_defect 处置自动建候选 draft（按品类/缺陷类型归并计数，auto_online=False），"
                                          "不自动评估/激活/上线，需人工发起 AC-017 准入+门禁"},
}
# 对照组口径；current 必需，其余缺失则在报告中标 missing
MECHANISM_VARIANTS = ("patchcore", "fixed_fusion", "current", "new_head")


def _rates_at(neg, pos, tau):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    fpr = round(float((neg >= tau).mean()), 4) if neg.size else None
    rec = round(float((pos >= tau).mean()), 4) if pos.size else None
    return fpr, rec


def _boot_auc_delta(neg_a, pos_a, neg_b, pos_b, n_boot, seed):
    """同一评测集配对 bootstrap：ΔAUROC(a-b) 的 95% CI。"""
    rng = np.random.default_rng(seed)
    na, pa = np.asarray(neg_a, float), np.asarray(pos_a, float)
    nb, pb = np.asarray(neg_b, float), np.asarray(pos_b, float)
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, na.size, na.size)
        j = rng.integers(0, pa.size, pa.size)
        a, b = _auc(na[i], pa[j]), _auc(nb[i], pb[j])
        if a is not None and b is not None:
            ds.append(a - b)
    if not ds:
        return None
    return [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)]


def mechanism_gain_report(variants, tau, ablations=None, groups=None, n_defect_used=0,
                          n_boot=200, seed=0):
    """AC-037 分机制收益对照（纯函数，调用方负责在同一评测集上取分）。
    variants: {name: {"neg": [...], "pos": [...]}}，至少含 current；
    ablations: {mechanism: {"on": {...}, "off": {...}}}，on/off 同评测集（记忆/近邻/孵育/open/回滚）；
    groups: {group: {"neg_idx": [...], "pos_idx": [...]}} 按来源分报副作用；
    n_defect_used: 当前策略训练/学习消耗的缺陷样本数（披露用）。"""
    if "current" not in variants:
        raise ValueError("variants 必须含 current")
    rows = {}
    for name, v in variants.items():
        fpr, rec = _rates_at(v["neg"], v["pos"], tau)
        rows[name] = {"auroc": _auc(v["neg"], v["pos"]), "fpr_at_tau": fpr,
                      "recall_at_tau": rec, "n_neg": len(v["neg"]), "n_pos": len(v["pos"])}
    cur = variants["current"]
    for name, v in variants.items():
        if name == "current" or len(v["neg"]) != len(cur["neg"]) or len(v["pos"]) != len(cur["pos"]):
            continue
        a, b = rows["current"]["auroc"], rows[name]["auroc"]
        rows[name]["current_minus_this"] = None if a is None or b is None else round(a - b, 4)
        rows[name]["delta_ci95"] = _boot_auc_delta(cur["neg"], cur["pos"], v["neg"], v["pos"],
                                                   n_boot, seed)

    mech = {}
    for m, ab in (ablations or {}).items():
        on, off = ab["on"], ab["off"]
        a_on, a_off = _auc(on["neg"], on["pos"]), _auc(off["neg"], off["pos"])
        f_on, r_on = _rates_at(on["neg"], on["pos"], tau)
        f_off, r_off = _rates_at(off["neg"], off["pos"], tau)
        e = {"gain_auroc": None if a_on is None or a_off is None else round(a_on - a_off, 4),
             "gain_ci95": _boot_auc_delta(on["neg"], on["pos"], off["neg"], off["pos"], n_boot, seed),
             "side_effect": {
                 "fpr_delta": None if f_on is None or f_off is None else round(f_on - f_off, 4),
                 "recall_delta": None if r_on is None or r_off is None else round(r_on - r_off, 4)}}
        if groups:
            per = {}
            for g, ix in groups.items():
                ni, pi = list(ix.get("neg_idx", [])), list(ix.get("pos_idx", []))
                fo, ro = _rates_at(np.asarray(on["neg"])[ni], np.asarray(on["pos"])[pi], tau)
                ff, rf = _rates_at(np.asarray(off["neg"])[ni], np.asarray(off["pos"])[pi], tau)
                per[g] = {"fpr_delta": None if fo is None or ff is None else round(fo - ff, 4),
                          "recall_delta": None if ro is None or rf is None else round(ro - rf, 4)}
            e["side_effect"]["by_group"] = per
        ci = e["gain_ci95"]
        e["verdict"] = ("gain" if ci and ci[0] > 0 else "harm" if ci and ci[1] < 0
                        else "inconclusive")
        mech[m] = e

    pc, cu = rows.get("patchcore", {}).get("auroc"), rows["current"]["auroc"]
    disclosure = {
        "below_patchcore": None if pc is None or cu is None else bool(cu < pc),
        "patchcore_uses_defects": False,
        "current_defects_used": int(n_defect_used),
        "note": ("当前策略低于 PatchCore 且额外消耗缺陷样本" if pc is not None and cu is not None
                 and cu < pc and n_defect_used > 0 else None)}
    return {"tau": float(tau), "variants": rows,
            "missing_variants": [v for v in MECHANISM_VARIANTS if v not in variants],
            "mechanisms": mech, "disclosure": disclosure, "facts": MECHANISM_FACTS}
