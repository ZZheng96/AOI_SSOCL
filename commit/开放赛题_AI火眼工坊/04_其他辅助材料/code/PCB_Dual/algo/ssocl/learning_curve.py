"""M3 学习曲线模拟器（§7.4）：自学习开/关对照，量化"这批错检，下批检对"。

在线模拟：在评测流上顺序推理；每张后按反馈率（5%/20%）模拟操作员反馈真值，
走 §7.1 三类反馈（正确正常→回流 / 错误缺陷→样例库+难例 / 错误正常→回流+归因）。
无反馈组 = 自学习关闭对照。输出累计混淆矩阵 → 累计 F1 / 学习延迟曲线。

诚实边界：模拟中的"反馈真值"来自评测集标签（实验用），生产上即操作员标注。
"""
import time
import numpy as np


def simulate(pipeline, split_items, feedback_ratio=0.2, with_feedback=True,
             cfg=None, rng=None, progress_every=50, log_every=25):
    """split_items: [(path, label)] 评测流。返回学习曲线数据 dict。

    返回:
      steps: list[int]        报告步数（log_every 采样）
      cum_f1 / cum_precision / cum_recall: list[float]
      learned: 累计已拦截/已反馈缺陷数
      feedback_log: 每次反馈的动作摘要（trace）
    """
    rng = rng or np.random.default_rng(0)
    c = cfg or {}
    t0 = time.time()
    tp = fp = tn = fn = 0
    steps, f1s, precs, recs = [], [], [], []
    n_learned = 0
    trace = []
    for i, (path, y) in enumerate(split_items):
        r = pipeline.predict(path)
        pred = 1 if r["decision"] != "normal" else 0
        if pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 0: tn += 1
        else: fn += 1
        # 模拟操作员反馈（§7.1）：以反馈率获得真值
        if with_feedback and rng.random() < feedback_ratio:
            # 反馈的"判对/判错"取决于预测是否正确
            correct = (pred == y)
            entry = pipeline.feedback(path, "correct" if correct else "wrong",
                                      label=y)
            action = entry.get("action", "")
            if action.startswith(("defect", "difficult", "reflow")):
                n_learned += 1
            trace.append({"step": i, "path": path, "y": y, "pred": pred,
                          "action": action,
                          "attribution": entry.get("attribution")})
        if i % log_every == 0 or i == len(split_items) - 1:
            d = tp + fn or 1
            n = tp + fp or 1
            prec = tp / n
            rec = tp / d
            steps.append(i + 1)
            precs.append(round(prec, 4))
            recs.append(round(rec, 4))
            f1s.append(round(2 * prec * rec / (prec + rec + 1e-9), 4))
        if progress_every and (i % progress_every == 0):
            print(f"  [lc] step {i + 1}/{len(split_items)} "
                  f"tp={tp} fp={fp} fn={fn} tn={tn} "
                  f"F1={2 * tp / (2 * tp + fp + fn + 1e-9):.4f} "
                  f"elapsed={time.time() - t0:.0f}s")
    return {"steps": steps, "cum_f1": f1s, "cum_precision": precs, "cum_recall": recs,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_feedback": len(trace), "n_learned": n_learned,
            "feedback_ratio": feedback_ratio,
            "trace": trace, "elapsed": round(time.time() - t0, 1),
            "final_f1": round(2 * tp / (2 * tp + fp + fn + 1e-9), 4)}


def curve_gap(learned, baseline):
    """自学习 vs 关闭的最终 F1 差与曲线下面积差（量化"学习收益"）。"""
    la = learned["final_f1"]
    ba = baseline["final_f1"]
    return {"final_f1_gap": round(la - ba, 4),
            "learned_f1": la, "baseline_f1": ba,
            "learned_n_feedback": learned["n_feedback"]}


def _eval_auroc(pipeline, eval_items):
    """固定锚定集上重算 fused AUROC（学习后 AUC，核心交付物 §7.4）。"""
    from sklearn.metrics import roc_auc_score
    ys, scores = [], []
    for p, y in eval_items:
        r = pipeline.predict(p)
        ys.append(y)
        scores.append(r["fused"])
    return float(roc_auc_score(ys, scores))


def _eval_full(pipeline, eval_items):
    """锚定集单趟推理（2026-08-29 走查修复：原实现 AUROC/F1 分两趟 predict，
    评估点成本翻倍）。返回 (labels, fused_scores, preds, boosts)。"""
    ys, scores, preds, boosts = [], [], [], []
    for p, y in eval_items:
        r = pipeline.predict(p)
        ys.append(y)
        scores.append(float(r["fused"]))
        preds.append(1 if r["decision"] != "normal" else 0)
        boosts.append(float(r.get("boost", 0.0)))
    return ys, scores, preds, boosts


def simulate_learning(pipeline, stream_items, eval_items, feedback_ratio=0.2,
                      eval_every=5, rng=None, progress_every=100,
                      max_feedback=None, box_map=None, progress_cb=None):
    """在线学习模拟（核心交付物 §7.4，2026-08-16 升级为 AUROC 追踪）：

    顺序推理流（stream，可反馈）→ 按反馈率获取真值 → 更新库 →
    每 eval_every 条反馈在**固定锚定集**（留出，只评估不反馈）上重算 AUROC/F1
    → 学习曲线（学习提升）+ 学习后 AUC + 学习效率（ΔAUC/反馈数）。

    box_map（U64，2026-08-18）：{path: [归一化框,...]}——用户在线提供的缺陷框
    标注（有标注层次，模拟操作员画框反馈）；label==1 的反馈把框传给
    pipeline.feedback → 缺陷库只入框内 patch。box_map 为空则走无框原路径。

    战略重心（用户明确，2026-08-16）：评估学习曲线与学习后 AUC，
    而非学习前静态 AUC——初始 AUC 如实报告，合法在线学习证明提升。

    返回 dict:
      steps / auroc / f1: 反馈数 → 锚定集 AUROC/F1 曲线
      initial_auroc / final_auroc: 学习前/后 AUC
      auroc_gain: 学习提升（final - initial）
      efficiency: 每反馈 AUROC 增益（ΔAUC/反馈数）
      n_feedback: 总反馈数
      box_supervised: 是否使用框标注反馈（U64 诚实记录信息层次）
    """
    from sklearn.metrics import f1_score, roc_auc_score
    rng = rng or np.random.default_rng(0)
    # 进度估算（2026-08-29）：总前向 ≈ 流推理 + 初始/末次/周期锚定集评估
    # + A 再检。仅预估用于进度条，不影响计算。
    exp_fb = min(max_feedback or len(stream_items),
                 int(len(stream_items) * feedback_ratio) + 1)
    n_eval_pts = 2 + exp_fb // max(eval_every, 1)
    total_fw = (len(stream_items) + n_eval_pts * len(eval_items)
                + exp_fb + len(eval_items))
    done_fw = 0

    def _report(n_eval_done=0):
        nonlocal done_fw
        done_fw += n_eval_done
        if progress_cb:
            progress_cb(min(done_fw, total_fw), total_fw)

    ys_i, sc_i, _, _ = _eval_full(pipeline, eval_items)
    initial = float(roc_auc_score(ys_i, sc_i))
    _report(len(eval_items))
    n_fb = 0
    steps, aurocs, f1s = [], [], []
    box_supervised = bool(box_map)
    n_box_fb = 0
    fed_defect_paths = []     # U92：已反馈缺陷图（A 再检样本，学习闭环"学过就会"）
    for i, (path, y) in enumerate(stream_items):
        r = pipeline.predict(path)
        pred = 1 if r["decision"] != "normal" else 0
        _report(1)
        if rng.random() < feedback_ratio:
            correct = (pred == y)
            box = box_map.get(path) if box_map else None
            pipeline.feedback(path, "correct" if correct else "wrong", label=y,
                              box=box)
            if y == 1:
                fed_defect_paths.append(path)     # U92：记录已反馈缺陷图
            if box:
                n_box_fb += 1
            n_fb += 1
            if n_fb % eval_every == 0:
                ys2, sc2, preds2, _ = _eval_full(pipeline, eval_items)
                au = float(roc_auc_score(ys2, sc2))
                f1s.append(round(float(f1_score(ys2, preds2)), 4))
                steps.append(n_fb)
                aurocs.append(round(au, 4))
                _report(len(eval_items))
                print(f"  [lc] 反馈={n_fb} 锚定集AUROC={au:.4f} "
                      f"F1={f1s[-1]:.4f}", flush=True)
            if max_feedback and n_fb >= max_feedback:
                break
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [stream] {i + 1}/{len(stream_items)} 反馈={n_fb}", flush=True)
    ys_f, sc_f, _, boost_f = _eval_full(pipeline, eval_items)
    final = float(roc_auc_score(ys_f, sc_f))
    _report(len(eval_items))
    gain = final - initial
    # U92：A 再检（学习闭环"学过就会"）= 已反馈缺陷图 + 未见正常锚的 AUROC；
    # B 泛化 = eval_items（只验不选）AUROC（即 final）。正常锚 = eval 集 y=0 样本。
    a_recheck = None
    normal_anchor = [(p, 0) for p, y in eval_items if y == 0]
    if fed_defect_paths and normal_anchor:
        a_items = [(p, 1) for p in fed_defect_paths] + normal_anchor
        a_recheck = round(_eval_auroc(pipeline, a_items), 4)
        _report(len(a_items))
    # U64：拦截命中统计（样例数/命中/评估集 boost 覆盖），诚实报告框精化效果；
    # 复用末次锚定集评估的 boost（2026-08-29：省一趟 n_eval 前向）
    intercept = _intercept_stats(pipeline, eval_items, eval_boosts=boost_f)
    # U65/U66：分数级杠杆留痕（head_ft/disc_ft 触发统计，公共汇总函数）
    ft = _ft_summary(pipeline)
    return {"steps": steps, "auroc": aurocs, "f1": f1s,
            "initial_auroc": round(initial, 4), "final_auroc": round(final, 4),
            "auroc_gain": round(gain, 4),
            "efficiency": round(gain / max(n_fb, 1), 4),
            "a_recheck_auroc": a_recheck,
            "a_recheck_n_defect": len(fed_defect_paths),
            "a_recheck_n_normal": len(normal_anchor),
            "n_feedback": n_fb,
            "box_supervised": box_supervised, "n_box_feedback": n_box_fb,
            "intercept": intercept, "head_ft": ft["head_ft"],
            "disc_ft": ft["disc_ft"], "thresh_recal": ft["thresh_recal"],
            "weight_learn": ft["weight_learn"]}


def simulate_learning_rounds(pipeline, pool_items, eval_items, round_size=30,
                             n_rounds=10, rng=None, box_map=None,
                             relearn_every=1, eval_per_sample=0,
                             progress_cb=None):
    """多轮持续学习模拟（U67，2026-08-18，用户纠正单轮流限制；
    U68，2026-08-18，评估口径修正——已反馈样本再学习验证）：

    现有 simulate_learning 是单轮流：stream_items 一次性（max-test 150 → 反馈
    锁死 ~53，学习早期饱和 0.8180 平台）。用户设想：持续学习输入不止一轮——
    每轮 30 张**新**图反馈，跨轮累积训练信息（权重/库/校准/判别头微调缓冲），
    10 轮=300 图训练量，逼近 demo4 离线全量裁切训练（~1000 图）→ 学习后 AUC
    随轮次持续提升。

    U68（用户纠正评估口径，2026-08-18）：学习曲线必须同时测
      ①已反馈样本的**再学习准确性**（"学过就会"）——第一次判错 → 监督学习后
        判对：每轮（或 relearn_every 轮）把**已反馈样本**（fed 累积，含本轮）
        重新 predict，统计
        - learned_accuracy（learned_accs）：重测 decision==label 比例
        - flip_to_correct（flip_rates）：学习前判错、学习后判对的样本数 /
          学习前判错样本数（"错→对转化率"，核心指标）
        - fed_defect_recall（defect_recalls）：已反馈缺陷样本中重测
          predict=缺陷 的比例（学习后应高）
      ②未见过样本的**泛化成长**（eval 锚定集 AUROC，先低后高、整体提高）。
      信息层次诚实记录：已反馈样本重测 = "有标注"层次（同 U64 box 层次，
      样本本身就是操作员标注过的）；eval 集从不反馈（test 只验不选）。
      成本控制：已反馈样本全量重测（300 图 × predict ~0.5s ≈ 2.5min/轮，
      10 轮累计 ~25min 可接受；relearn_every=2 可减半）。

    实施：
      pool_items: 可反馈图池（不放回，按序分轮）；eval_items: 固定锚定集（从不反馈）
      每轮：取 pool[r*round_size:(r+1)*round_size] → 全部反馈（ratio=1.0，
      模拟操作员持续标注；带框 box_map）→ 锚定集 _eval_auroc + F1 重算 →
      已反馈样本全量重测（U68，relearn_every 控制频率）
      跨轮累积：SlotWeightLearner 权重 / 缺陷样例库 / 正常回流库 / 虚高槽位
      CDF 重校准缓冲 / head_ft/disc_ft 微调缓冲（_disc_ft_counts/_box_pos_feats
      等）全部在 pipeline.handler 实例内，跨轮不清空即自然累积——300 图反馈下
      disc_ft（min_pos=60/min_neg=120 patch）应真正触发（单轮 53 反馈下信息量
      不足，U65/U66 中性/负的核心原因，U67 验证点）。

    诚实注释：pool 反馈 = test 域在线"有标注"层次（模拟操作员持续标注，
    同 U64 box 层次）；eval 从不反馈（test 只验不选）；已反馈重测信息层次
    = 有标注（重测的是操作员标注过的样本，验证"学过就会"，非泛化指标）。

    返回 dict（rounds 为轮次轴，steps 为累计反馈数轴——与 simulate_learning
    的 steps/auroc/f1 结构兼容，plot 可直接复用）:
      rounds / steps / auroc / f1: 轮次 → 锚定集 AUROC/F1 曲线（steps=累计反馈数）
      initial_auroc / final_auroc / auroc_gain / efficiency / n_feedback
      box_supervised / n_box_feedback / intercept / head_ft / disc_ft
      learned（U68 新增）: {
        rounds: 重测轮次；n_fed: 累计已反馈样本数；n_wrong_before: 学习前判错数
                （flip 分母）；n_flip: 错→对转化数；
        learned_accs / flip_rates / defect_recalls: 每重测轮 已反馈准确率/
                错→对转化率/缺陷检出率；
        final_learned_acc / final_flip_rate / final_defect_recall: 末轮汇总;
        relearn_every: 重测频率
      }
    """
    from sklearn.metrics import f1_score
    rng = rng or np.random.default_rng(0)
    # 轮数自适应（2026-08-29，系统集成）：原 assert pool ≥ round_size×n_rounds
    # 在小数据集品类直接崩；改为按可用轮池收紧轮数，不足一轮才报错。
    n_rounds = min(int(n_rounds), len(pool_items) // max(int(round_size), 1))
    if n_rounds < 1:
        raise ValueError(
            f"pool 样本不足: {len(pool_items)} < round_size {round_size}（不足一轮）")
    # eval_per_sample>0：每轮锚定评估用每类抽样子集（控每轮评估成本），
    # 初始/末次 AUROC 与 eval_per_sample 留痕仍用完整锚定集（口径不变）。
    eval_full = list(eval_items)
    eval_sub = eval_full
    if eval_per_sample and eval_per_sample > 0:
        _an = [it for it in eval_full if it[1] == 1]
        _no = [it for it in eval_full if it[1] == 0]
        eval_sub = _an[:int(eval_per_sample)] + _no[:int(eval_per_sample)]
        if len(eval_sub) < 4:
            eval_sub = eval_full
    initial = _eval_auroc(pipeline, eval_full)
    n_fb = 0
    rounds, steps, aurocs, f1s = [], [], [], []
    ys2 = [y for _, y in eval_sub]
    box_supervised = bool(box_map)
    n_box_fb = 0
    fed = []                       # U68：已反馈样本累积 (path, y, pred_before)
    l_rounds, l_accs, l_flips, l_defrec = [], [], [], []
    l_nfed, l_nwrong, l_nflip = [], [], []
    # 进度估算（仅进度条用）：反馈前向 + 每轮锚定评估/F1/重测 + 初始/末次/留痕评估
    total_fw = (len(eval_full) * 3
                + n_rounds * (round_size + 2 * len(eval_sub))
                + round_size * n_rounds * (n_rounds + 1) // 2)
    done_fw = 0

    def _report(n=1):
        nonlocal done_fw
        done_fw += n
        if progress_cb:
            progress_cb(min(done_fw, total_fw), total_fw)

    _report(len(eval_full))
    for r in range(n_rounds):
        chunk = pool_items[r * round_size:(r + 1) * round_size]
        for path, y in chunk:
            rr = pipeline.predict(path)
            pred = 1 if rr["decision"] != "normal" else 0
            correct = (pred == y)
            box = box_map.get(path) if box_map else None
            pipeline.feedback(path, "correct" if correct else "wrong", label=y,
                              box=box)
            if box:
                n_box_fb += 1
            n_fb += 1
            fed.append((path, y, pred))          # U68：记录学习前预测
            _report(1)
        au = _eval_auroc(pipeline, eval_sub)
        _report(len(eval_sub))
        preds = [1 if pipeline.predict(p2)["decision"] != "normal" else 0
                 for p2, _ in eval_sub]
        _report(len(eval_sub))
        f1 = round(float(f1_score(ys2, preds)), 4)
        rounds.append(r + 1)
        steps.append(n_fb)
        aurocs.append(round(au, 4))
        f1s.append(f1)
        # ---- U68 再学习验证（评估口径修正）：已反馈样本重测 ----
        learned_txt = ""
        if (r + 1) % relearn_every == 0 or r == n_rounds - 1:
            acc_ok = flip_ok = 0
            n_wrong = 0
            def_ok = def_n = 0
            for path, y, pb in fed:
                pa = 1 if pipeline.predict(path)["decision"] != "normal" else 0
                _report(1)
                if pa == y:
                    acc_ok += 1
                if pb != y:                        # 学习前判错（flip 分母）
                    n_wrong += 1
                    if pa == y:
                        flip_ok += 1               # 错→对
                if y == 1:
                    def_n += 1
                    if pa == 1:
                        def_ok += 1
            l_rounds.append(r + 1)
            l_nfed.append(len(fed))
            l_nwrong.append(n_wrong)
            l_nflip.append(flip_ok)
            l_accs.append(round(acc_ok / max(len(fed), 1), 4))
            l_flips.append(round(flip_ok / max(n_wrong, 1), 4))
            l_defrec.append(round(def_ok / max(def_n, 1), 4))
            learned_txt = (f" 已反馈重测acc={l_accs[-1]:.4f} "
                           f"错→对={l_flips[-1]:.4f} 缺陷检出={l_defrec[-1]:.4f}")
        print(f"  [lc-rounds] 轮 {r + 1}/{n_rounds} 反馈={n_fb}{learned_txt} "
              f"AUROC={au:.4f} F1={f1:.4f}", flush=True)
    final = _eval_auroc(pipeline, eval_full)
    _report(len(eval_full))
    gain = final - initial
    intercept = _intercept_stats(pipeline, eval_full)
    ft = _ft_summary(pipeline)
    # U71（2026-08-18）：记录 eval 锚定集 per_sample（fused scores + labels +
    # decision），供学习后 FP/FN 主因诊断（误报 vs 漏检）
    eval_ps = {"scores": [], "labels": [], "decisions": []}
    for p, y in eval_full:
        r = pipeline.predict(p)
        _report(1)
        eval_ps["scores"].append(round(float(r["fused"]), 6))
        eval_ps["labels"].append(int(y))
        eval_ps["decisions"].append(r["decision"])
    learned = {"rounds": l_rounds, "n_fed": l_nfed, "n_wrong_before": l_nwrong,
               "n_flip": l_nflip, "learned_accs": l_accs, "flip_rates": l_flips,
               "defect_recalls": l_defrec, "relearn_every": relearn_every}
    if l_accs:
        learned["final_learned_acc"] = l_accs[-1]
        learned["final_flip_rate"] = l_flips[-1]
        learned["final_defect_recall"] = l_defrec[-1]
    return {"rounds": rounds, "steps": steps, "auroc": aurocs, "f1": f1s,
            "initial_auroc": round(initial, 4), "final_auroc": round(final, 4),
            "auroc_gain": round(gain, 4),
            "efficiency": round(gain / max(n_fb, 1), 4),
            "n_feedback": n_fb, "round_size": round_size, "n_rounds": n_rounds,
            "box_supervised": box_supervised, "n_box_feedback": n_box_fb,
            "intercept": intercept, "head_ft": ft["head_ft"],
            "disc_ft": ft["disc_ft"], "thresh_recal": ft["thresh_recal"],
            "weight_learn": ft["weight_learn"],
            "learned": learned, "eval_per_sample": eval_ps}


def _ft_summary(pipeline):
    """U65/U66：分数级杠杆留痕汇总（head_ft/disc_ft 触发次数/正负样本数/gate 结果）+
    U69：决策阈值重估留痕（重估/否决次数/事件），simulate_learning 与
    simulate_learning_rounds 共用。"""
    head_ft_sum = {}
    if pipeline.handler is not None:
        h = pipeline.handler
        head_ft_sum = {"n_ft": int(h.stats.get("head_ft", 0)),
                       "n_reject": int(h.stats.get("head_ft_reject", 0)),
                       "n_pos_total": int(h._head_ft_counts["pos"]),
                       "n_neg_total": int(h._head_ft_counts["neg"]),
                       "events": list(h.head_ft_log)}
    disc_ft_sum = {}
    if pipeline.handler is not None:
        h = pipeline.handler
        disc_ft_sum = {"n_ft": int(h.stats.get("disc_ft", 0)),
                       "n_reject": int(h.stats.get("disc_ft_reject", 0)),
                       "n_pos_total": int(h._disc_ft_counts["pos"]),
                       "n_neg_total": int(h._disc_ft_counts["neg"]),
                       "events": list(h.disc_ft_log)}
    # U69：决策阈值重估留痕（n_recal/n_reject/events：n_normal/fp_before/fp_after/
    # tau_high/tau_gray/status），json 落盘供报告重估次数与门控结果
    thresh_sum = {}
    if pipeline.handler is not None:
        h = pipeline.handler
        thresh_sum = {"n_recal": int(h.stats.get("thresh_recal", 0)),
                      "n_reject": int(h.stats.get("thresh_recal_reject", 0)),
                      "events": list(h.thresh_recal_log)}
    # U107：在线权重写回留痕（n_apply/n_reject——门控通过/否决次数）
    weight_sum = {}
    if pipeline.handler is not None:
        h = pipeline.handler
        weight_sum = {"n_apply": int(h.stats.get("weight_apply", 0)),
                      "n_reject": int(h.stats.get("weight_reject", 0))}
    return {"head_ft": head_ft_sum, "disc_ft": disc_ft_sum,
            "thresh_recal": thresh_sum, "weight_learn": weight_sum}


def _intercept_stats(pipeline, eval_items, eval_boosts=None):
    """U64：缺陷样例库拦截统计——样例数、命中计数、评估集上 boost>0 覆盖
    （缺陷侧命中 = 拦截真正生效的证据）。eval_boosts 可复用已算好的
    锚定集 boost（避免重复前向），None 则现场推理。"""
    if pipeline.handler is None:
        return {}
    db = pipeline.handler.defect_bank
    n_boost = n_boost_def = 0
    if eval_boosts is None:
        boosts = []
        for p, y in eval_items:
            r = pipeline.predict(p)
            boosts.append(float(r.get("boost", 0.0)))
    else:
        boosts = [float(b) for b in eval_boosts]
    for b, (_, y) in zip(boosts, eval_items):
        if b > 0:
            n_boost += 1
            if y == 1:
                n_boost_def += 1
    return {"n_samples": len(db.samples),
            "n_box_samples": sum(1 for u in db.updated_ts if u.get("op") == "add_box"),
            "n_hits_total": int(sum(db.n_hits)),
            "eval_n": len(eval_items), "eval_boost_gt0": n_boost,
            "eval_boost_gt0_defect": n_boost_def,
            "eval_defect_n": sum(1 for _, y in eval_items if y == 1)}
