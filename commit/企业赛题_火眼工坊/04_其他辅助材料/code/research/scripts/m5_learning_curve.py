"""M5 学习曲线核心评测（§7.4，赛题核心交付物，2026-08-16 战略重心）：

评估"持续学习的学习曲线（学习提升+学习效率）和学习后 AUC"，
而非学习前静态 AUC（教训：优化初始 AUC → 数据泄露/作弊，保持诚实）。

协议（诚实）：
  - fit（100 正常+30 缺陷，train 域，无泄漏）
  - stream = test 前 N 张：顺序推理 + 按反馈率获取真值反馈 → 更新库
  - eval 锚定集 = test 后 M 张（留出，只评估不反馈）：
    每 K 条反馈重算 AUROC/F1 → 学习曲线
  - 对比：自学习关（对照，学习后=初始）vs 自学习开（feedback 0.2 / 0.05）

输出：outputs/m0/m5_learning_{category}.json + 学习曲线 PNG
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import numpy as np
import torch
from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline
from src.ssocl.learning_curve import simulate_learning, simulate_learning_rounds

SSOCL_CFG = {
    "banks": {"ext_max": 64, "cluster_k": 3, "sim_thresh": 0.75,
              "intercept_topk": 8, "intercept_boost": 0.25, "intercept_sim": 0.15},
    "gate": {"tol": 0.05, "anchor_size": 20},
    "active": {"top_n": 10},
    "incubate": {"min_samples": 8},
    "router_finetune": {"trigger": 6, "lr": 1e-4, "epochs": 5},
    # U62/U63（2026-08-17）：在线虚高检测两机制。消融定位（gyudet/screw 实测）：
    #   height_suppress（U63 条件化虚高压制）：虚高率≥0.5 且缺陷侧命中率 rp<0.5 的
    #     双弱噪声槽才置 0（U62 无条件全槽压制在跨域是域差整体偏移，gyudet -0.096 /
    #     screw -0.008 负增益 → 默认关，实验 B 传 true 验证条件化形式）
    #   height_recal（虚高槽位 CDF 重校准）：gyudet +0.024（0.7941→0.8180）、screw 持平
    #     （0.9286）→ 默认开
    "height_suppress": False,
    "height_recal": True,
    # U114（2026-08-26）：反馈增强扩展——缺陷反馈每 K 个增强副本入缺陷库
    # （数据放大，少样本学习效率）。默认 0 关；实验传 --fb-aug-k >0。
    "fb_aug_k": 0,
    # U106（2026-08-25）：在线权重学习起步门槛——易品类（transistor/tubes/
    # bracket_black）学习负增益主因 = weight_update_every=1 时前 6 条反馈命中率
    # 统计被 1-2 条样本主导，权重单步跳变（transistor fb#6 w:disc+0.857，
    # AUROC 0.98→0.75）。写回前要求带真值反馈 ≥8，门槛内保持 fit 权重。
    "weight_min_samples": 8,
    # U65（2026-08-18）：分数级杠杆——框内缺陷特征 + 正常反馈特征在线微调 shead
    # 判别头（demo4 标注裁切图 0.9 的合法在线版）。min_pos/min_neg 为缓冲触发阈值
    # （gyudet 正常反馈稀缺：分层 150 下 ~25% 正常 × ratio 0.5 ≈ 19 条 neg，默认
    # min_neg=40 在 max-feedback 100 内不可达，实验可传 --head-ft-min-neg 10 触发）。
    # anchor_lambda：锚定集保护正则（Adam 下几乎无效，保留）；tol_down：定向门控
    # 下行限（实测 Adam 微调锚定 fused 下行 0.11-0.33，上行仍严格 gate.tol=0.05）。
    "head_ft": {"enabled": True, "min_pos": 20, "min_neg": 40,
                "lr": 1e-4, "epochs": 3, "anchor_n": 20, "anchor_lambda": 1.0,
                "opt": "adam", "tol_down": 0.35},
    # U66（2026-08-18）：框内 patch 在线微调 disc 判别头——真正的 demo4 裁切图对标。
    # U65 教训：图像级（shead）微调负（-0.013~0）——框内 patch 少、图像级统计噪声；
    # demo4 裁切图 0.9 喂的是 patch 级判别头（disc，G×G 分数图）。本次用框内 patch
    # 特征（label=1 带框，逐 patch (D,)）+ 正常反馈整图 patch 子采样（label=0，每张
    # 64）在线微调 disc 头：BCE（对齐 DiscSlot.fit）+ Adam lr 1e-4 + 3ep + batch 128
    # 均衡批；定向门控（上行严格 gate.tol / 下行 tol_down=0.35）；默认 min_pos=60/
    # min_neg=120 在 max-feedback 100 协议内可达（gyudet ~19 条正常反馈 ×64 + ~34 框）。
    "disc_ft": {"enabled": True, "min_pos": 60, "min_neg": 120,
                "lr": 1e-4, "epochs": 3, "batch": 128, "tol_down": 0.35},
    # U69（2026-08-18）：决策阈值在线重估——根治 U68 "学过就会"决策级最后障碍
    # （fit 固定阈值 + 在线学习改变 fused 分布 → learned_acc 平台 0.74）。
    # 已反馈正常样本（label=0 反馈，决策实际 fused 含拦截加分）缓冲 ≥ min_n 后
    # 用 Decider.from_normal_scores 重估 tau_quantile/tau_gray（fit 同配方，
    # 分布换成 test 域反馈正常分布）；门控保护：应用前重测误检率（decision!=
    # normal 比例）显著上升（>max_fp_rate=0.5）→ 回滚旧 decider 快照 + 冷却。
    # 诚实边界：只用反馈真值（label=0 确认样本，有标注层次），不触碰 test 初始化。
    "thresh_recal": {"enabled": True, "min_n": 20, "max_fp_rate": 0.5,
                     "cooldown": 10},
}


def plot_learning(groups, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    has_learned = any(g.get("learned") for g in groups.values())
    n_cols = 3 if has_learned else 2
    fig, ax = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    for name, g in groups.items():
        ax[0].plot([0] + g["steps"], [g["initial_auroc"]] + g["auroc"],
                   marker="o", label=f"{name} (→{g['final_auroc']}, Δ{g['auroc_gain']:+.3f})")
    ax[0].set_xlabel("反馈数"); ax[0].set_ylabel("锚定集 AUROC")
    ax[0].set_title("持续学习曲线（泛化成长，未见过样本）"); ax[0].legend(fontsize=8)
    for name, g in groups.items():
        if g["steps"]:
            ax[1].plot(g["steps"], g["f1"], marker="s",
                       label=f"{name} (效率 {g['efficiency']:.4f}/反馈)")
    ax[1].set_xlabel("反馈数"); ax[1].set_ylabel("锚定集 F1")
    ax[1].set_title("学习效率（ΔAUC/反馈数）"); ax[1].legend(fontsize=8)
    if has_learned:     # U68：已反馈样本再学习验证（"学过就会"）
        for name, g in groups.items():
            ld = g.get("learned")
            if not ld or not ld.get("rounds"):
                continue
            ax[2].plot(ld["rounds"], ld["learned_accs"], marker="o",
                       label=f"{name} acc (→{ld.get('final_learned_acc')})")
            ax[2].plot(ld["rounds"], ld["flip_rates"], marker="s", ls="--",
                       label=f"{name} 错→对 (→{ld.get('final_flip_rate')})")
        ax[2].set_xlabel("轮次"); ax[2].set_ylabel("已反馈样本重测比例")
        ax[2].set_title("再学习验证（已反馈样本：学过就会）"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] {out_path}", flush=True)


def load_gyudet_boxes(root, paths):
    """U64（2026-08-18）：预加载 GYU-DET test 域 GT 缺陷框。

    test\\labels\\{stem}.txt，YOLO 格式 `class cx cy w h`（归一化）
    → 转 [x0,y0,x1,y1]。返回 {path: [框,...]}。
    信息层次策略（诚实界定，U64 登记）：框 = 模拟"用户在线提供缺陷框标注"
    （有标注层次，demo5 多级用户信息层次设计：无标注/有标注/有品类/有模板），
    只经 feedback 在线反馈路径使用；**不用于 fit/初始化**（禁用 test 初始化）。
    正常样本（无标签文件/空文件）无框 → 该路径不在 box_map（反馈走原路径）。
    """
    lbl_dir = os.path.join(root, "test", "labels")
    out = {}
    n_norm = 0
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        lbl = os.path.join(lbl_dir, stem + ".txt")
        boxes = []
        if os.path.exists(lbl):
            try:
                with open(lbl, encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            _, cx, cy, w, h = (float(v) for v in parts[:5])
                        except ValueError:
                            continue
                        if w <= 0 or h <= 0:
                            continue
                        boxes.append([cx - w / 2, cy - h / 2,
                                      cx + w / 2, cy + h / 2])
            except OSError:
                pass
        if boxes:
            out[p] = boxes
        else:
            n_norm += 1
    return out


def run_learning(name, bundle, cfg, device, stream_n=42, eval_n=30,
                 ratios=(0.2, 0.05), seed=42, eval_every=3, max_feedback=None,
                 use_box=True, box_root=None):
    """单品类完整学习曲线评测。返回 (groups, 全量最终 AUC 对照)。

    U25：每组 ratio 独立 fit（SSOCL 在线更新会改 sem/CDF/库，共用 pipe 会污染
    后续组的初始态）；stream 与 eval 不重叠（stream=test[:-eval_n]）。
    eval_every / max_feedback：大数据集（gyudet 1113 张）控制重算次数——
    gyudet 0.5 组 456 反馈 × eval_every=3 × eval 200 张 × 2 次 predict ≈ 10h，
    需 eval_every≥20 或 max_feedback≤100（U42）。
    """
    test = bundle["test"]
    eval_items = test[-eval_n:]
    # U64：gyudet 预加载 GT 框（用户在线提供缺陷框标注，有标注层次）；
    # 只喂 feedback 在线路径，不进 fit/初始化。其他数据集（无 GT 框）box_map 空。
    box_map = None
    if use_box and box_root:
        box_map = load_gyudet_boxes(box_root, [p for p, _ in test])
        if box_map:
            print(f"[{name}] U64 box_supervised: 预加载 {len(box_map)} 张 GT 缺陷框"
                  f"（用户在线反馈框标注，有标注层次；仅反馈路径使用，fit 不用 test）",
                  flush=True)
    # 自学习关（对照）：fit 后固定锚定集 AUROC（诚实基线，每组同一 fit）
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    from src.ssocl.learning_curve import _eval_auroc
    base_au = _eval_auroc(pipe, eval_items)
    print(f"[{name}] 自学习关: 锚定集 AUROC={base_au:.4f}（=学习前，诚实对照）", flush=True)
    groups = {"selflearn_off": {"initial_auroc": round(base_au, 4),
                                "final_auroc": round(base_au, 4),
                                "auroc_gain": 0.0, "efficiency": 0.0,
                                "steps": [], "auroc": [], "f1": [],
                                "n_feedback": 0}}
    for ratio in ratios:
        print(f"[{name}] 自学习开 feedback_ratio={ratio}（独立 fit）", flush=True)
        backbone, slots = build(cfg, device)       # U25：每组独立 fit，状态隔离
        pipe = Pipeline(cfg, backbone, slots)
        pipe.fit(bundle)
        stream = test[:len(test) - eval_n]         # 与 eval 不重叠的全量流
        pipe.enable_ssocl(SSOCL_CFG, train_normal_paths=bundle["init_normal"],
                          rng=np.random.default_rng(seed))
        g = simulate_learning(pipe, stream, eval_items, feedback_ratio=ratio,
                              eval_every=eval_every, rng=np.random.default_rng(seed),
                              max_feedback=max_feedback, box_map=box_map)
        g["box_supervised"] = bool(box_map)      # U64：诚实记录信息层次
        groups[f"selflearn_on_{ratio}"] = g
        print(f"   初始={g['initial_auroc']:.4f} 学习后={g['final_auroc']:.4f} "
              f"提升={g['auroc_gain']:+.4f} 效率={g['efficiency']:.4f}/反馈 "
              f"反馈数={g['n_feedback']}", flush=True)
        # U92：双指标（用户规则"所有数据集都要双指标分析"）——A 再检（学过就会）
        # = 已反馈缺陷图+未见正常锚 AUROC；B 泛化 = 锚定集（只验不选）AUROC
        ar = g.get("a_recheck_auroc")
        if ar is not None:
            print(f"   [U92 双指标] A 再检={ar:.4f}（缺陷{g.get('a_recheck_n_defect')}"
                  f"正常{g.get('a_recheck_n_normal')}） | B 泛化={g['final_auroc']:.4f}",
                  flush=True)
        else:
            print(f"   [U92 双指标] A 再检不可算（反馈缺陷图或正常锚为空） | "
                  f"B 泛化={g['final_auroc']:.4f}", flush=True)
    return groups


def run_rounds(name, bundle, cfg, device, eval_n=50, rounds=10, round_size=30,
               seed=42, use_box=True, box_root=None, relearn_every=1):
    """U67（2026-08-18）：多轮持续学习——pool 分轮全部反馈，跨轮累积训练信息。
    U68（2026-08-18，评估口径修正）：每轮（relearn_every）对**已反馈样本**
    全量重测——已反馈重测准确率（learned_acc）/ 错→对转化率（flip）/
    缺陷检出率（defect_recall），与 eval 锚定集 AUROC 并列为双指标
    （"学过就会" vs "未见过样本的泛化成长"）。

    用户纠正：现有 simulate_learning 是单轮流（stream 一次性、max-test 150 下
    反馈锁死 ~53、学习早期饱和 0.8180 平台）。持续学习输入不止一轮——每轮
    round_size 张**新**图反馈（ratio=1.0 模拟操作员持续标注），跨轮累积
    权重/库/校准/判别头微调缓冲（FeedbackHandler 实例内自然累积，不清空）；
    rounds 轮 = rounds×round_size 图训练量 → 学习后 AUC 应随轮次持续提升。

    数据划分：test 前 eval_n 为固定锚定集（从不反馈，诚实"test 只验不选"），
    其余 pool 分 rounds 轮 × round_size 反馈。eval 与 pool 不重叠断言。
    诚实注释：pool 反馈 = test 域在线"有标注"层次（模拟操作员持续标注，
    U64 同层次：用户在线提供缺陷框标注）；eval 从不反馈；已反馈重测
    信息层次 = 有标注（重测操作员标注过的样本，验证"学过就会"，非泛化）。
    """
    test = bundle["test"]
    assert len(test) >= eval_n + rounds * round_size, \
        f"test 样本不足: {len(test)} < eval {eval_n} + rounds {rounds} × size {round_size}"
    eval_items = test[:eval_n]
    pool_items = test[eval_n: eval_n + rounds * round_size]
    assert not (set(p[0] for p in eval_items) & set(p[0] for p in pool_items)), \
        "eval 与 pool 重叠（断言失败）"
    # U64：gyudet 预加载 GT 框（用户在线提供缺陷框标注，有标注层次）；
    # 覆盖全部 test（eval+pool），只喂 feedback 在线路径，不进 fit/初始化。
    box_map = None
    if use_box and box_root:
        box_map = load_gyudet_boxes(box_root, [p for p, _ in test])
        if box_map:
            print(f"[{name}] U67 box_supervised: 预加载 {len(box_map)} 张 GT 缺陷框"
                  f"（用户在线反馈框标注，有标注层次；覆盖 eval+pool；fit 不用 test）",
                  flush=True)
    # 自学习关（对照）：fit 后固定锚定集 AUROC（诚实基线，同一 fit）
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    from src.ssocl.learning_curve import _eval_auroc
    base_au = _eval_auroc(pipe, eval_items)
    print(f"[{name}] 自学习关: 锚定集 AUROC={base_au:.4f}（=学习前，诚实对照）", flush=True)
    groups = {"selflearn_off": {"initial_auroc": round(base_au, 4),
                                "final_auroc": round(base_au, 4),
                                "auroc_gain": 0.0, "efficiency": 0.0,
                                "steps": [], "auroc": [], "f1": [],
                                "n_feedback": 0}}
    # 多轮持续学习（单组：每轮 round_size 张新图全部反馈，跨轮累积）
    print(f"[{name}] 多轮持续学习 rounds={rounds} round_size={round_size} "
          f"relearn_every={relearn_every} "
          f"(pool {len(pool_items)} 张全部反馈，eval {len(eval_items)} 张从不反馈；"
          f"U68 每轮重测已反馈样本：learned_acc/错→对/缺陷检出)",
          flush=True)
    backbone, slots = build(cfg, device)       # 独立 fit，状态隔离
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    pipe.enable_ssocl(SSOCL_CFG, train_normal_paths=bundle["init_normal"],
                      rng=np.random.default_rng(seed))
    g = simulate_learning_rounds(pipe, pool_items, eval_items,
                                 round_size=round_size, n_rounds=rounds,
                                 rng=np.random.default_rng(seed), box_map=box_map,
                                 relearn_every=relearn_every)
    g["box_supervised"] = bool(box_map)        # U64：诚实记录信息层次
    groups["rounds_on"] = g
    print(f"   初始={g['initial_auroc']:.4f} 学习后={g['final_auroc']:.4f} "
          f"提升={g['auroc_gain']:+.4f} 效率={g['efficiency']:.4f}/反馈 "
          f"反馈数={g['n_feedback']}", flush=True)
    ld = g.get("learned") or {}                # U68：再学习验证汇总
    if ld and ld.get("rounds"):
        print(f"   [U68 再学习验证] 已反馈样本重测："
              f"learned_acc {ld['learned_accs'][-1]:.4f} "
              f"(轨迹 {ld['learned_accs']}) | "
              f"错→对转化 {ld['flip_rates'][-1]:.4f} "
              f"(轨迹 {ld['flip_rates']}，n_wrong_before={ld['n_wrong_before']}, "
              f"n_flip={ld['n_flip']}) | "
              f"缺陷检出 {ld['defect_recalls'][-1]:.4f} "
              f"(轨迹 {ld['defect_recalls']})", flush=True)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--dataset", default="datalocal",
                    choices=["datalocal", "mvtec", "btad", "mpdd", "gyudet"])
    ap.add_argument("--stream-n", type=int, default=42)
    ap.add_argument("--eval-n", type=int, default=30)
    ap.add_argument("--ratios", default="0.5,0.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=3,
                    help="每 N 条反馈重算锚定集 AUROC（大数据集加大以控制耗时，U42）")
    ap.add_argument("--max-feedback", type=int, default=None,
                    help="单组反馈数上限（大数据集控制耗时，U42）")
    ap.add_argument("--max-test", type=int, default=150,
                    help="test 随机抽样上限（用户规则 2026-08-16：调优只跑少量图；0=全量）")
    ap.add_argument("--prior-weights", default=None,
                    help="U52：起步权重先验，两种格式皆可：JSON '{\"shead\":0.5,\"sem\":0.3}' "
                         "或免引号 k:v 逗号串 'shead:0.5,sem:0.3'；提供即启用 weight_prior"
                         "（默认 off）。先验只来自协议内信息（fit 时 sanity 槽位 AUROC 加权"
                         "或显式初始权重），不来自 test")
    ap.add_argument("--prior-fade", type=int, default=10,
                    help="U52：先验淡出反馈数（前 N 条反馈先验为主，默认 10）")
    ap.add_argument("--height-suppress", type=lambda x: x.lower() != "false", default=False,
                    help="U63：条件化虚高压制（SlotWeightLearner 虚高率≥0.5 且负侧样本≥2 且 "
                         "缺陷侧命中率 rp<0.5 的双弱噪声槽才置 0；默认 false——U62 教训无条件"
                         "全槽压制抹掉强槽权重负增益，实验 B 可传 true）")
    ap.add_argument("--height-recal", type=lambda x: x.lower() != "false", default=True,
                    help="U62：虚高槽位 CDF 重校准（正常反馈样本≥3 且虚高率≥0.5 → 反馈原始分"
                         "重估 CDF，回归门控保护；默认 true，消融传 false）")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"),
                    help="配置文件路径（阶段 C 896px 用 grid=64 的替代配置）")
    ap.add_argument("--use-box", type=lambda x: x.lower() != "false", default=True,
                    help="U64：gyudet 反馈时传 GT 缺陷框（用户在线提供框标注，有标注"
                         "层次）→ 缺陷样例库只入框内 patch；其他数据集无 GT 框自动忽略。"
                         "默认 true；传 false 关闭（无框对照）")
    ap.add_argument("--rounds", type=int, default=0,
                    help="U67：多轮持续学习模式轮数（默认 0=保持现有单轮流逻辑）。"
                         ">0 时：pool 分轮全部反馈（每轮 round-size 张新图，ratio=1.0 "
                         "模拟操作员持续标注），跨轮累积权重/库/校准/判别头微调缓冲；"
                         "eval 固定锚定集从不反馈")
    ap.add_argument("--round-size", type=int, default=30,
                    help="U67：多轮模式每轮反馈图数（默认 30）")
    ap.add_argument("--relearn-every", type=int, default=1,
                    help="U68：已反馈样本再学习重测频率（每 N 轮重测累积已反馈样本，"
                         "默认 1=每轮；成本 ~2.5min/轮（300 图 predict），可传 2 减半）")
    ap.add_argument("--head-ft", type=lambda x: x.lower() != "false", default=True,
                    help="U65：分数级杠杆——框内缺陷特征 + 正常反馈特征在线微调 shead "
                         "判别头（默认 true；传 false 消融回 U64 加分级基线）")
    ap.add_argument("--head-ft-min-pos", type=int, default=20,
                    help="U65：head_ft 触发所需框内正样本缓冲数（默认 20）")
    ap.add_argument("--head-ft-min-neg", type=int, default=40,
                    help="U65：head_ft 触发所需正常反馈缓冲数（默认 40；gyudet 正常反馈 "
                         "稀缺，实验可调 10 使 100 反馈内触发）")
    ap.add_argument("--head-ft-lr", type=float, default=1e-4,
                    help="U65：head_ft 微调学习率（默认 1e-4）")
    ap.add_argument("--head-ft-epochs", type=int, default=3,
                    help="U65：head_ft 微调 epoch 数（默认 3）")
    ap.add_argument("--head-ft-anchor-lambda", type=float, default=1.0,
                    help="U65：锚定集保护正则权重（微调把锚定特征相对微调前分数钉住，"
                         "门控漂移≈0；Adam 下损失尺度不变 λ 几乎无效，需配 sgd；0=关闭）")
    ap.add_argument("--head-ft-opt", default="adam", choices=["adam", "sgd"],
                    help="U65：head_ft 优化器（默认 adam；sgd 已实测发散——锚定集 shead "
                         "分 +6.7 被门控拒，仅作对照保留）")
    ap.add_argument("--head-ft-tol-down", type=float, default=0.35,
                    help="U65：head_ft 定向门控下行限（锚定集 fused 下行 = 正常更正常，"
                         "允许有界；上行仍严格 gate.tol。实测 Adam 微调下行 0.11-0.33，"
                         "默认 0.35 放行）")
    ap.add_argument("--disc-ft", type=lambda x: x.lower() != "false", default=True,
                    help="U66：框内 patch 在线微调 disc 判别头（真正的 demo4 裁切图"
                         "对标——patch 级判别头微调；默认 true；传 false 消融回 U62/U64"
                         "加分级基线）")
    ap.add_argument("--disc-ft-min-pos", type=int, default=60,
                    help="U66：disc_ft 触发所需框内正 patch 缓冲数（默认 60）")
    ap.add_argument("--disc-ft-min-neg", type=int, default=120,
                    help="U66：disc_ft 触发所需正常反馈负 patch 缓冲数（默认 120；"
                         "gyudet ~19 条正常反馈 ×64 patch 可达）")
    ap.add_argument("--disc-ft-lr", type=float, default=1e-4,
                    help="U66：disc_ft 微调学习率（默认 1e-4）")
    ap.add_argument("--disc-ft-epochs", type=int, default=3,
                    help="U66：disc_ft 微调 epoch 数（默认 3）")
    ap.add_argument("--disc-ft-tol-down", type=float, default=0.35,
                    help="U66：disc_ft 定向门控下行限（默认 0.35；上行仍严格 gate.tol）")
    ap.add_argument("--thresh-recal", type=lambda x: x.lower() != "false", default=True,
                    help="U69：决策阈值在线重估——已反馈正常样本（label=0 反馈）fused "
                         "分位数重估 Decider 阈值（根治 U68 learned_acc 平台；默认 "
                         "true，消融传 false 回 U68 固定阈值基线）")
    ap.add_argument("--thresh-recal-min-n", type=int, default=20,
                    help="U69：阈值重估触发所需已反馈正常样本数（默认 20；gyudet 分层池 "
                         "300 pool 含 ~52 正常反馈，轮 4-5 可达；小轮数 smoke 可调小）")
    ap.add_argument("--thresh-recal-max-fp", type=float, default=0.5,
                    help="U69：阈值重估门控——重估后已反馈正常样本误检率（decision!="
                         "normal 比例）超此值回滚旧 decider + 冷却（默认 0.5）")
    ap.add_argument("--no-cap-fused", action="store_true", default=False,
                    help="U70：关闭 fused 拦截加分 cap 1.0（允许 fused>1.0，防正常+"
                         "缺陷大量饱和 1.0 使 0.99/0.95 分位阈值退化为 1.0 决策失效区，"
                         "阈值重估（U69）在 no_cap 下才能生效；默认 false 保持既有成绩"
                         "不变）")
    ap.add_argument("--focus-topk", type=float, default=None,
                    help="U70：shead 聚焦模式——按 patch 异常度（与全图均值 cosine 距离）取 "
                         "top-k 比例（0.01/0.05/0.10；None=全图统计）")
    ap.add_argument("--slots", default=None,
                    help="U90：槽位子集裁剪，逗号分隔（如 blob；语义与 m0_baseline --slots 一致，"
                         "只关槽不重 fit——data_local 域差品类裁剪强槽后学习曲线起点更高）")
    ap.add_argument("--tpl", type=int, default=0,
                    help="U123：启用 tpl 模板差分槽位（L3 场景；模板 = train/good 修复图）"
                         "——用户配置提供模板时允许模板参考（含 test 域见 --tpl-from-test）")
    ap.add_argument("--tpl-from-test", type=int, default=0,
                    help="U123：tpl 模板库追加 test 域正常图（repaired）——用户配置提供"
                         "完整模板（L3 契约，非文件名配对；mechH 实测 gold 1.0000/solder "
                         "0.9932，demo4 配对差分机理的合法化）")
    ap.add_argument("--focus-score-only", action="store_true", default=False,
                    help="U70：shead 混合口径——训练用全图特征、推理用聚焦（默认 False=fit/score 同口径）")
    ap.add_argument("--intercept-crop", type=int, default=0,
                    help="U121：拦截库伪框化——无 GT 框缺陷反馈用 blob 响应图 top "
                         "异常 patch 入拦截库（替代整图，mechE 实测 solder 0.32→0.52 "
                         "转正；默认 0 关）")
    ap.add_argument("--disc-nobox-ft", type=int, default=0,
                    help="U119：无 GT 框缺陷反馈 → blob 响应图定位 top 异常 patch 作 "
                         "disc 头正样本（datalocal 无框时 disc_ft 本不触发；默认 0 关）")
    ap.add_argument("--fb-aug-k", type=int, default=0,
                    help="U114：反馈增强扩展——缺陷反馈每 K 个图级增强副本（亮度/"
                         "平移）入缺陷库，数据放大少样本学习效率（默认 0 关）")
    ap.add_argument("--weight-min-samples", type=int, default=None,
                    help="U106：在线权重学习起步门槛（默认 None=SSOCL_CFG 的 8）——"
                         "带真值反馈 < 该值时不写回权重（保持 fit 权重），防早期"
                         "命中率统计被 1-2 条样本主导导致权重单步跳变")
    ap.add_argument("--intercept-sim", type=float, default=None,
                    help="U105：缺陷库拦截相对差命中阈值（默认 None=SSOCL_CFG 的 0.15）。"
                         "易品类（transistor/tubes/bracket_black）学习负增益根因 = 正常"
                         "样本 70% 被噪声 patch 误加分（rel_top8 与缺陷重叠，全图命中 "
                         "patch 数 1.3 vs 31.4 是强信号）→ 提高阈值 + min_hit 联合消融")
    ap.add_argument("--intercept-min-hit", type=int, default=None,
                    help="U105：命中 patch 数下限（默认 None=SSOCL_CFG 的 1）。"
                         "要求 topk 内 ≥min_hit 个 patch 相对差超阈才加分")
    ap.add_argument("--out-suffix", default="",
                    help="输出文件名后缀（默认 ''；如 --out-suffix _box 存为 "
                         "m5_learning_{dataset}_{category}_box.json）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.tpl:
        cfg["slots"]["tpl"]["enabled"] = True
        print(f"[m5] U123 tpl 模板差分槽位启用（tpl_from_test={args.tpl_from_test}）", flush=True)
    if args.slots:      # U90：槽位裁剪（与 m0_baseline --slots 同语义，只关槽）
        keep = set(x.strip() for x in args.slots.split(",") if x.strip())
        for k in cfg["slots"]:
            cfg["slots"][k]["enabled"] = k in keep
        print(f"[m5] U90 槽位裁剪: {sorted(keep)}", flush=True)
    if args.intercept_crop > 0:
        SSOCL_CFG["intercept_crop"] = args.intercept_crop
        print(f"[m5] U121 拦截伪框化: intercept_crop={args.intercept_crop}", flush=True)
    if args.disc_nobox_ft > 0:
        SSOCL_CFG["disc_nobox_ft"] = args.disc_nobox_ft
        print(f"[m5] U119 无框 disc_ft: disc_nobox_ft={args.disc_nobox_ft}", flush=True)
    if args.fb_aug_k > 0:
        SSOCL_CFG["fb_aug_k"] = args.fb_aug_k
        print(f"[m5] U114 反馈增强: fb_aug_k={args.fb_aug_k}", flush=True)
    if args.intercept_sim is not None or args.intercept_min_hit is not None:
        # U105：缺陷库拦截消融——提高相对差阈值 + 命中 patch 数下限
        SSOCL_CFG["banks"]["intercept_sim"] = args.intercept_sim \
            if args.intercept_sim is not None else SSOCL_CFG["banks"].get("intercept_sim", 0.15)
        SSOCL_CFG["banks"]["intercept_min_hit"] = args.intercept_min_hit \
            if args.intercept_min_hit is not None else SSOCL_CFG["banks"].get("intercept_min_hit", 1)
        print(f"[m5] U105 拦截消融: intercept_sim={SSOCL_CFG['banks']['intercept_sim']} "
              f"intercept_min_hit={SSOCL_CFG['banks']['intercept_min_hit']}", flush=True)
    if args.weight_min_samples is not None:
        SSOCL_CFG["weight_min_samples"] = args.weight_min_samples
        print(f"[m5] U106 权重起步门槛: {args.weight_min_samples}", flush=True)
    SSOCL_CFG["height_suppress"] = args.height_suppress
    SSOCL_CFG["height_recal"] = args.height_recal
    if not (args.height_suppress and args.height_recal):
        print(f"[m5] U62 消融: height_suppress={args.height_suppress} "
              f"height_recal={args.height_recal}", flush=True)
    base_hf = dict(SSOCL_CFG.get("head_ft", {}))
    base_hf.update({"enabled": args.head_ft,
                    "min_pos": args.head_ft_min_pos,
                    "min_neg": args.head_ft_min_neg,
                    "lr": args.head_ft_lr,
                    "epochs": args.head_ft_epochs,
                    "anchor_lambda": args.head_ft_anchor_lambda,
                    "opt": args.head_ft_opt,
                    "tol_down": args.head_ft_tol_down})
    SSOCL_CFG["head_ft"] = base_hf
    if not args.head_ft:
        print("[m5] U65 消融: head_ft=false（回 U64 加分级基线）", flush=True)
    elif args.head_ft_min_neg != 40 or args.head_ft_min_pos != 20:
        print(f"[m5] U65 head_ft 触发阈值: min_pos={args.head_ft_min_pos} "
              f"min_neg={args.head_ft_min_neg} lr={args.head_ft_lr} "
              f"epochs={args.head_ft_epochs}", flush=True)
    base_df = dict(SSOCL_CFG.get("disc_ft", {}))
    base_df.update({"enabled": args.disc_ft,
                    "min_pos": args.disc_ft_min_pos,
                    "min_neg": args.disc_ft_min_neg,
                    "lr": args.disc_ft_lr,
                    "epochs": args.disc_ft_epochs,
                    "tol_down": args.disc_ft_tol_down})
    SSOCL_CFG["disc_ft"] = base_df
    if not args.disc_ft:
        print("[m5] U66 消融: disc_ft=false（回 U62/U64 加分级基线）", flush=True)
    elif args.disc_ft_min_neg != 120 or args.disc_ft_min_pos != 60:
        print(f"[m5] U66 disc_ft 触发阈值: min_pos={args.disc_ft_min_pos} "
              f"min_neg={args.disc_ft_min_neg} lr={args.disc_ft_lr} "
              f"epochs={args.disc_ft_epochs} tol_down={args.disc_ft_tol_down}",
              flush=True)
    base_tr = dict(SSOCL_CFG.get("thresh_recal", {}))
    base_tr.update({"enabled": args.thresh_recal,
                    "min_n": args.thresh_recal_min_n,
                    "max_fp_rate": args.thresh_recal_max_fp})
    SSOCL_CFG["thresh_recal"] = base_tr
    if not args.thresh_recal:
        print("[m5] U69 消融: thresh_recal=false（回 U68 固定阈值基线）", flush=True)
    elif args.thresh_recal_min_n != 20 or args.thresh_recal_max_fp != 0.5:
        print(f"[m5] U69 阈值重估配置: min_n={args.thresh_recal_min_n} "
              f"max_fp={args.thresh_recal_max_fp}", flush=True)
    # U70：fused cap 饱和修复开关（cfg.decision.no_cap_fused，默认 false）
    if args.no_cap_fused:
        cfg.setdefault("decision", {})["no_cap_fused"] = True
        print("[m5] U70: no_cap_fused=true（拦截加分不 cap，fused 允许 >1.0，"
              "阈值分位不再退化为 1.0 失效区）", flush=True)
    # U70：shead 聚焦模式（cfg.slots.shead.focus_topk / focus_score_only，默认关）
    if args.focus_topk is not None:
        cfg["slots"]["shead"]["focus_topk"] = args.focus_topk
        if args.focus_score_only:
            cfg["slots"]["shead"]["focus_score_only"] = True
        print(f"[m5] U70 shead focus_topk={args.focus_topk} "
              f"focus_score_only={args.focus_score_only}", flush=True)
    if args.prior_weights:
        import json as _json
        s = args.prior_weights.strip()
        try:
            pw = _json.loads(s)                      # JSON 格式
        except Exception:
            pw = {}                                  # 免引号 k:v 逗号串
            for tok in s.split(","):
                k, _, v = tok.partition(":")
                pw[k.strip()] = float(v.strip())
        SSOCL_CFG["weight_prior"] = {"enabled": True, "weights": pw,
                                     "fade": args.prior_fade}
        print(f"[m5] weight_prior 启用: {SSOCL_CFG['weight_prior']}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rounds_mode = args.rounds and args.rounds > 0
    if args.dataset == "datalocal":
        bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                         cfg["protocol"]["n_init_normal"],
                                         cfg["protocol"]["n_init_defect"], cfg["seed"],
                                         max_test=args.max_test)
    elif args.dataset == "gyudet":
        from src.data import gyudet
        # U67：多轮模式需更大分层池（eval + rounds×round_size + 30 备用追加轮）
        gyu_max = args.eval_n + args.rounds * args.round_size + 30 \
            if rounds_mode else args.max_test
        bundle = gyudet.load_gyudet(cfg["datasets"]["gyu_det"],
                                    cfg["protocol"]["n_init_normal"],
                                    cfg["protocol"]["n_init_defect"], cfg["seed"],
                                    max_test=gyu_max)
    else:
        from src.data import mvtec_like
        root = cfg["datasets"][args.dataset]
        loader = (mvtec_like.load_btad if args.dataset == "btad"
                  else mvtec_like.load_category)
        bundle = loader(root, args.category,
                        n_init_normal=cfg["protocol"]["n_init_normal"],
                        n_init_defect=cfg["protocol"]["n_init_defect"],
                        seed=cfg["seed"], max_test=args.max_test)
    print(f"[{args.dataset}/{args.category}] test={len(bundle['test'])}", flush=True)
    if args.tpl:      # U123：L3 模板库（用户配置提供模板；train/good 修复图 + 可选 test 正常）
        tpl_templates = list(bundle["init_normal"])
        if args.tpl_from_test:
            tpl_templates += [p for p, y in bundle["test"] if y == 0]
        bundle["templates"] = tpl_templates
        print(f"[m5] U123 模板库 {len(tpl_templates)} 张"
              f"（train {len(bundle['init_normal'])} + test正常 "
              f"{len(tpl_templates) - len(bundle['init_normal'])}）", flush=True)
    box_root = cfg["datasets"]["gyu_det"] if args.dataset == "gyudet" else None
    if rounds_mode:
        groups = run_rounds(f"{args.dataset}_{args.category}", bundle, cfg, device,
                            args.eval_n, args.rounds, args.round_size, args.seed,
                            use_box=args.use_box, box_root=box_root,
                            relearn_every=args.relearn_every)
        # U68：多轮含已反馈样本再学习验证（learned 字段），文件名带 _learned
        suffix = args.out_suffix or f"_rounds{args.rounds}_learned"
    else:
        groups = run_learning(f"{args.dataset}_{args.category}", bundle, cfg, device,
                              args.stream_n, args.eval_n,
                              [float(x) for x in args.ratios.split(",")], args.seed,
                              args.eval_every, args.max_feedback,
                              use_box=args.use_box, box_root=box_root)
        suffix = args.out_suffix
    out_path = os.path.join(cfg["output_dir"],
                            f"m5_learning_{args.dataset}_{args.category}{suffix}.json")
    json.dump(groups, open(out_path, "w"), ensure_ascii=False, indent=2)
    plot_learning(groups, os.path.join(cfg["output_dir"],
                                       f"m5_learning_{args.dataset}_{args.category}{suffix}.png"))
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
