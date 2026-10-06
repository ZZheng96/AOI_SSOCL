"""M3 三类反馈 handler（§7.1）：在线 SSOCL 闭环编排层

- 判对：正常 → 回流评估（成簇入库 + CDF 重估）；缺陷 → 入缺陷样例库
- 判错：难例入样例库 + 难例队列（≥10 触发批量级路由微调）+ 归因决策树
- 无反馈：高置信/灰区分类 → 主动选样队列（§7.5）

每次更新过回归门控（锚定集 fused 分布漂移超限 → 回滚，§7.2/§7.6），全程留痕。
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

from .banks import NormalBank, DefectBank
from .attribution import attribute_failure
from ..slots.shead import DeviationLoss, box_patch_feats
from ..decide import Decider
from ..fusion.fixed import fuse
from ..common.io import load_image


class RegressGate:
    """锚定集回归门控：更新后正常锚定集 fused 分布漂移超限 → 拒绝更新（回滚）。"""

    def __init__(self, tol=0.05, anchor_size=20):
        self.tol = tol
        self.anchor_size = anchor_size
        self.anchor_paths = []     # 锚定集 = train/good 抽样
        self.base_mean = None
        self.base_std = None
        self.history = []

    def setup(self, pipeline, anchor_paths):
        """锚定集（train/good 抽样，已由 enable_ssocl 提取特征）算基准分布。
        红线 3：锚定集只用正常图。"""
        self.anchor_paths = list(anchor_paths)
        scores = [pipeline._predict_fused_only(p) for p in self.anchor_paths]
        self.base_mean = float(np.mean(scores))
        self.base_std = float(np.std(scores)) + 1e-9
        return self

    def check(self, pipeline, tag="update"):
        """更新后漂移检查：|new_mean - base_mean| > tol 判定退化。
        返回 (ok, drift, new_mean)；ok=False 时调用方回滚。"""
        scores = [pipeline._predict_fused_only(p) for p in self.anchor_paths]
        new_mean = float(np.mean(scores))
        drift = abs(new_mean - self.base_mean) / self.base_std
        ok = drift <= self.tol
        self.history.append({"t": time.time(), "tag": tag, "drift": round(drift, 4),
                             "new_mean": round(new_mean, 4), "ok": ok})
        return ok, drift, new_mean


class SlotWeightLearner:
    """在线槽位权重学习（U46，2026-08-16）：反馈真值驱动融合权重持续更新。

    动机：等权融合被反向槽位稀释（U31），sanity 加权是离线的（fit 时一次）。
    本学习器在反馈流上逐样本更新：每个带真值反馈样本，检查各槽位校准分
    与真值的一致性（缺陷=分>0.5 命中；正常=分<0.5 命中），分别累计正/负侧
    命中率，均衡后得每槽位"判别命中率"，权重=max(0, rate-0.5) 归一化。

    类别不平衡保护（gyudet 缺陷率 95%，U43）：正/负侧分开计数后取均衡，
    负侧样本不足时（<min_neg）该槽位维持中性（0.5），防偏置。
    诚实边界：只用反馈真值（操作员标注/实验标签），不触碰 test 选参。"""

    def __init__(self, slots, min_neg=3, initial_weights=None, prior_fade=10,
                 height_suppress=True):
        self.slots = list(slots)
        self.min_neg = min_neg
        self.n = {s: [0, 0] for s in self.slots}        # [总, 命中]
        self.pos = {s: [0, 0] for s in self.slots}      # [总, 命中]（缺陷侧）
        self.neg = {s: [0, 0] for s in self.slots}      # [总, 命中]（正常侧）
        self.n_updates = 0
        # U62/U63（2026-08-17）：条件化显式虚高压制——不改命中率逻辑，只加显式压制。
        # 虚高率 height(s) = 1 - neg_hit/neg_total（正常反馈样本上该槽位 cal>0.5 比例）。
        # U63 修正（U62 教训：跨域全槽虚高是域差整体偏移，无条件置 0 抹掉强槽权重负增益）：
        # 压制条件收紧为 neg_total >= 2 且 height >= 0.5 且缺陷侧命中率 rp<0.5——
        # 只压制"双弱噪声槽"（trad 型，缺陷侧也判别不好），shead/sem 强槽 rp 高不压制。
        # 触发记录进 height_log（供 handler 打印/报告），_height_flagged 去重。
        self.height_suppress = height_suppress
        self.height_log = []
        self._height_flagged = set()
        # U52（2026-08-16）：起步权重先验（weight_prior 实验）。
        # initial_weights: 协议内先验（fit 时 sanity 槽位 AUROC 加权或显式初始权重），
        # 不来自 test；前 prior_fade 条反馈以先验为主线性淡出到在线学习权重，
        # 压缩 0-10 反馈"探索浪费"段（U51 机理：权重学习约 10-20 反馈才见效）。
        self.initial_weights = initial_weights
        self.prior_fade = int(prior_fade or 0)

    def update(self, slot_scores, label):
        """slot_scores: {槽位: 校准分}；label: 0 正常 / 1 缺陷。"""
        self.n_updates += 1
        for s in self.slots:
            sc = slot_scores.get(s, 0.5)
            hit = (sc > 0.5) == (label == 1)
            self.n[s][0] += 1
            self.n[s][1] += int(hit)
            (self.pos if label == 1 else self.neg)[s][0] += 1
            (self.pos if label == 1 else self.neg)[s][1] += int(hit)
        return self

    def _learned(self):
        """纯在线学习权重（U46 原逻辑）：均衡命中率-0.5，置负为 0，归一化。"""
        w = {}
        for s in self.slots:
            tp, th = self.pos[s]
            np_, nh = self.neg[s]
            rp = th / max(tp, 1)
            rn = nh / max(np_, 1) if np_ >= self.min_neg else 0.5   # 负侧不足→中性
            w[s] = max(0.0, 0.5 * rp + 0.5 * rn - 0.5)
        tot = sum(w.values())
        if tot <= 0:
            return {s: 1.0 / len(self.slots) for s in self.slots}
        return {s: v / tot for s, v in w.items()}

    def weights(self):
        """当前权重（每更新后可调）。有先验时：前 prior_fade 条反馈先验→学习权重线性插值。
        U62（实施 1）：计算后强制虚高压制——正常侧虚高率（cal>0.5 比例）≥0.5 且
        负侧样本 ≥2 的槽位置 0（不改命中率逻辑，只加显式压制；全置零回退等权）。"""
        learned = self._learned()
        if self.initial_weights is None or self.prior_fade <= 0:
            w = learned
        else:
            t = min(self.n_updates / self.prior_fade, 1.0)
            if t >= 1.0:
                w = learned
            else:
                prior = {}
                k = len(self.slots)
                for s in self.slots:                   # 先验未覆盖槽位用等权补
                    prior[s] = float(self.initial_weights.get(s, 1.0 / k))
                ptot = sum(prior.values())
                if ptot <= 0:
                    w = learned
                else:
                    prior = {s: v / ptot for s, v in prior.items()}
                    w = {s: (1.0 - t) * prior[s] + t * learned[s] for s in self.slots}
                    tot = sum(w.values())
                    if tot <= 0:
                        w = learned
        # U62/U63：条件化显式虚高压制（只在最终权重上强制，保持命中率逻辑原样）
        if self.height_suppress:
            for s in self.slots:
                tp, th = self.pos[s]
                np_, nh = self.neg[s]
                rp = th / max(tp, 1)
                # U63：虚高率 ≥0.5 且缺陷侧命中率也低（rp<0.5，双弱噪声槽）→ 置 0
                if np_ >= 2 and (1.0 - nh / np_) >= 0.5 and rp < 0.5:
                    w[s] = 0.0
                    if s not in self._height_flagged:
                        self._height_flagged.add(s)
                        self.height_log.append({"n_updates": self.n_updates, "slot": s,
                                                "neg_total": np_, "neg_hit": nh,
                                                "pos_hit_rate": round(rp, 3),
                                                "height": round(1.0 - nh / np_, 3)})
            tot = sum(w.values())
            if tot <= 0:
                return {s: 1.0 / len(self.slots) for s in self.slots}
            return {s: v / tot for s, v in w.items()}
        return w


class FeedbackHandler:
    """三类反馈统一入口（§7.1）；主动选样经灰区队列汇入。"""

    def __init__(self, pipeline, cfg):
        self.p = pipeline
        b = cfg.get("banks", {})
        self.normal_bank = NormalBank(
            core_max=b.get("core_max", 512), ext_max=b.get("ext_max", 256),
            cluster_k=b.get("cluster_k", 3), sim_thresh=b.get("sim_thresh", 0.75),
            defect_ratio_thresh=b.get("defect_ratio_thresh", 0.30))
        self.defect_bank = DefectBank(
            topk=b.get("intercept_topk", 8), boost=b.get("intercept_boost", 0.3),
            sim_thresh=b.get("intercept_sim", 0.85),
            min_hit=b.get("intercept_min_hit", 1))
        g = cfg.get("gate", {})
        self.gate = RegressGate(tol=g.get("tol", 0.05),
                                anchor_size=g.get("anchor_size", 20))
        r = cfg.get("router_finetune", {})
        self.rt_trigger = r.get("trigger", 10)
        self.rt_lr = r.get("lr", 1e-4)
        self.rt_epochs = r.get("epochs", 5)
        self.difficult = []            # 难例队列（判错缺陷）
        self.high_conf = []            # 无反馈高置信缺陷缓存（待确认）
        self.feedback_log = []         # 全部反馈留痕（§7.1/红线 5）
        self.rollback_log = []
        self.fuse_blocked = False
        self.stats = {"reflow": 0, "pending": 0, "defect_add": 0, "router_ft": 0,
                      "rollback": 0, "gate_reject": 0, "height_recal": 0,
                      "thresh_recal": 0, "thresh_recal_reject": 0,
                      "weight_apply": 0, "weight_reject": 0}
        # U46：在线槽位权重学习（反馈真值驱动融合权重持续更新）
        # U52：起步权重先验（weight_prior，默认 off；实验 on）——cfg.weight_prior =
        # {"enabled": true, "weights": {槽位: 权重}, "fade": N}。
        # 先验只来自协议内信息（fit 时 sanity 槽位 AUROC 加权或显式初始权重），不来自 test；
        # 首次 feedback 前直接应用到 pipeline.weights（初始 AUROC 即带先验），
        # 之后由 SlotWeightLearner 前 fade 条反馈线性淡出到在线学习权重。
        wp = cfg.get("weight_prior") or {}
        init_w = None
        if wp.get("enabled") and wp.get("weights"):
            init_w = {s: float(v) for s, v in wp["weights"].items()
                      if s in getattr(pipeline, "slots", {})}
            if init_w:
                tot = sum(init_w.values())
                if tot > 0:
                    init_w = {s: v / tot for s, v in init_w.items()}
                    self.p.weights = dict(init_w)     # 首次 feedback 前直接应用
                    print(f"[ssocl] weight_prior 应用: "
                          + " ".join(f"{k}={v:.3f}" for k, v in init_w.items()), flush=True)
        self.weight_learner = SlotWeightLearner(
            list(pipeline.slots), initial_weights=init_w,
            prior_fade=wp.get("fade", 10),
            # U62/U63 结论（2026-08-17）：显式虚高压制默认关闭——跨域品类（gyudet/screw）
            # 所有槽位对 test 正常样本普遍虚高（域差整体偏移），无条件置 0 回退等权会抹掉
            # SlotWeightLearner 学到的强槽权重（gyudet -0.096、screw -0.008 负增益）；
            # U63 改为条件化（虚高率≥0.5 且缺陷侧命中率 rp<0.5 才压制），实验 B 验证。
            height_suppress=bool(cfg.get("height_suppress", False))) \
            if hasattr(pipeline, "slots") else None
        self.weight_update_every = int(cfg.get("weight_update_every", 1))
        # U106（2026-08-25）：在线权重学习起步门槛——易品类学习负增益主因定位
        # （transistor fb#6 AUROC 0.9814→0.7453 暴跌，同刻 w:disc+0.857 单步
        # 跳变）：weight_update_every=1 时每条反馈都写回权重，前 6 条反馈的
        # 命中率统计被 1-2 条异常样本主导 → 融合权重剧烈偏转破坏排序。
        # 修复：写回前要求带真值反馈数 ≥ weight_min_samples（默认 8），
        # 让命中率统计有足够样本；门槛内保持 fit 时的 sanity/等权权重。
        self.weight_min_samples = int(cfg.get("weight_min_samples", 8))
        self._w_fb_since_update = 0
        # U107（2026-08-25）：权重写回回归门控——U106 门槛只消除早期暴跌、
        # 负增益未修复（transistor -0.037/tubes -0.062/bracket_black -0.062），
        # 根因 = 在线权重学习写回的权重本身在易品类上劣于 fit 权重（初始已优）。
        # 修复：写回前快照旧权重 → 应用新权重 → gate.check（锚定集= train/good
        # 正常锚，红线 3 协议内）→ 漂移超 tol 回滚旧权重 + 冷却（防反复尝试）。
        # 诚实边界：锚定集只用 train/good，不用 eval/test 选参（红线 B 泛化
        # "test 只验不选"）。
        self._w_snapshot = None            # 写回前权重快照（gate 否决回滚）
        self._w_cooldown = -99             # 上次权重 gate 失败的带真值反馈计数
        self.weight_gate_cooldown = int(cfg.get("weight_gate_cooldown", 10))
        # U107/U108（2026-08-25）：权重写回门控 tol——难品类（train_auroc<0.95）
        # 需要权重学习（screw 正收益 +0.061），U107 用 gate.tol=0.05 过紧误伤
        # （screw 写回 drift 0.078-0.18 全被拦，+0.061→+0.008）；放宽到
        # weight_gate_tol（默认 0.3）只拦极端跳变。易品类由 U108 冻结兜底。
        self.weight_gate_tol = float(cfg.get("weight_gate_tol", 0.3))
        # U108（2026-08-25）：权重学习自适应启停——按 fit 时 train 域 fused AUROC
        # （协议内 init_normal+init_defect）判定初始权重质量。U108 首版阈值 0.95
        # 实测不可靠（screw train 0.9853 被误冻丢失 +0.05 正收益；bracket_black
        # train 0.8913 未冻但学习有害）→ 阈值收紧到 0.99：仅 train 域几乎完美
        # （≥0.99，transistor 1.0/tubes 0.987/bottle 1.0）才冻结；其余走 U109
        # 反馈样本 margin 判据（协议内有标注层次）。
        self.weight_learn_max_train_auroc = float(cfg.get("weight_learn_max_train_auroc", 0.99))
        # U109（2026-08-25）：反馈样本 fused 分离度（margin）判据——U107 正常锚
        # 均值漂移测不到"缺陷侧排序破坏"（transistor 劣质写回 drift 0.018 通过
        # 但 AUROC 掉 0.046）。本判据用**已反馈样本**（有标注层次，协议内）在
        # 新旧权重下的 fused 分离度（缺陷均值 − 正常均值）对比：写回使 margin
        # 劣化超 tol → 回滚。零额外前向（复用 feedback 时缓存的 slot_scores）。
        self.weight_margin_tol = float(cfg.get("weight_margin_tol", 0.02))
        self._recent_fb = []             # 滚动窗口：最近带真值反馈 (slot_scores, label)
        # U110（2026-08-25）：权重学习最少反馈样本门槛——U108+U109 组合实测：
        # screw（53 反馈）学习正收益 +0.061；tubes（21）/bracket_black（10）/
        # bottle（9）反馈少，权重学习学出的权重劣于 fit 权重（负增益或依赖
        # 拦截加分），且 margin 判据在少样本窗口失效（缺陷样本个位数）。
        # 在线权重学习（7 槽位命中率统计）需要足够反馈：n_labeled_fb < 30
        # 直接冻结（保持 fit 权重最安全）。协议内（只用反馈数量，不触碰 test）。
        # U110 修订（U110v2，2026-08-25）：gold_finger（train_auroc=0.7355，21
        # 反馈）正增益 +0.486 主要由权重学习驱动，被 30 门槛误冻 → 判据分级：
        # 初始权重**明显欠优**（train_auroc < weight_min_train_auroc=0.85，域差
        # 场景）→ 反馈少也放行（学习是主增益通道）；初始**已可用**（≥0.85，
        # bracket_black 0.8913 型）且反馈少 → 冻结（学习是噪声扰动）。
        self.weight_min_feedback = int(cfg.get("weight_min_feedback", 30))
        self.weight_min_train_auroc = float(cfg.get("weight_min_train_auroc", 0.85))
        # U114（2026-08-26）：反馈增强扩展——缺陷反馈每 K 个增强副本入缺陷库
        # （数据放大，少样本学习效率）。默认 0 关（不改变既有成绩）；实验传 >0。
        self.fb_aug_k = int(cfg.get("fb_aug_k", 0))
        # U119（2026-08-26）：无框缺陷反馈 → blob 响应图定位 top 异常 patch 作
        # disc 头正样本（demo4 裁切图的合法近似）。datalocal 无 GT 框 → disc_ft
        # 本不触发（U118 空转发现）；blob 是域不变特征（跨域强正），响应图 top
        # patch ≈ 缺陷区域。默认 0 关；实验传 1。
        self.disc_nobox_ft = int(cfg.get("disc_nobox_ft", 0))
        # U121（2026-08-26）：拦截库伪框化——无 GT 框缺陷反馈用 blob 响应图 top
        # 异常 patch 入拦截库（替代整图，demo4 裁切思想：缺陷区域 patch 信噪比高）。
        # mechE 实测：solder 整图拦截 0.32 反向 → blob 裁剪 0.52 转正。默认 0 关。
        self.intercept_crop = int(cfg.get("intercept_crop", 0))
        # U62（2026-08-17）：在线虚高检测——虚高槽位 CDF 重校准（实施 2）。
        # 机理（U61）：trad 类槽位正常样本 cal 顶格（train/good 拟合 CDF 在 test 正常
        # 域差下失效饱和）是 FP 误检主因；修复 = 用确认正常（label=0）反馈样本的
        # 校准前原始分重估虚高槽位 CDF（纳入正常分布），而非只降权。
        # 触发：某槽位正常反馈样本 ≥3 且虚高率（cal>0.5 比例）≥0.5；
        # 门控保护：锚定集 fused 漂移超 tol → 回滚该槽位（edges + hist 长度快照）。
        # 诚实边界：只用反馈真值（label=0 确认样本），不触碰 test。
        self.height_recal = bool(cfg.get("height_recal", True))
        slot_names = list(pipeline.slots) if hasattr(pipeline, "slots") else []
        self._normal_fb_raw = {n: [] for n in slot_names}   # 正常反馈样本各槽位原始分
        self._normal_fb_cal = {n: [] for n in slot_names}   # 同批校准分（虚高率判定用）
        self.height_recal_log = []
        self._hr_snapshot = None                             # 重估快照（gate 否决回滚）
        self._hr_cooldown = {}                               # 槽位 -> 失败反馈计数（冷却 10）
        self._n_labeled_fb = 0                               # 带真值反馈计数（冷却用）
        # U65（2026-08-18）：分数级杠杆——框内缺陷特征 + 正常反馈特征在线微调
        # shead 判别头（demo4 用标注裁切图训练判别头达 0.9 的合法在线版）。
        # 机理（U64 结论）：拦截加分是"加分级"杠杆，后期被 normal_bank 相对相似
        # 抵消，学习后锁死 0.8180；本机制把反馈真值（含缺陷框）转成 shead 输入
        # 格式（label=1 且带框 → 框内 patch 特征；label=0 → 整图特征），缓冲达标
        # 后在线微调 head（DeviationLoss + Adam，backbone 冻结）→ "分数级"杠杆。
        # 诚实边界：只用反馈路径真值（操作员标注/实验标签 + 在线框），不触碰
        # test 初始化/fit；回归门控保护（锚定集 fused 漂移超 tol 回滚 head 权重）。
        hf = cfg.get("head_ft", {}) or {}
        self.head_ft_on = bool(hf.get("enabled", True))
        self.hf_min_pos = int(hf.get("min_pos", 20))
        self.hf_min_neg = int(hf.get("min_neg", 40))
        self.hf_lr = float(hf.get("lr", 1e-4))
        self.hf_epochs = int(hf.get("epochs", 3))
        # U65 调优：锚定集保护正则——锚定特征（train/good，门控测量对象）以"相对
        # 微调前分数"的 L2 钉住，微调只移动 test 域打分面（框内缺陷↑、反馈正常↓）。
        # 注意（实测）：Adam 对损失尺度不变，λ 放大对锚定约束几乎无效（λ=1→20 只
        # 把锚定 shead 移动 -0.248→-0.125）；sgd 又会发散（锚定分 +6.7）。故最终
        # 走定向门控（见 _maybe_head_ft）：上行（正常被异常化=回归）严格 tol_up，
        # 下行（正常更正常=head_ft 意图方向）允许有界 tol_down。
        self.hf_anchor_n = int(hf.get("anchor_n", 20))
        self.hf_anchor_lambda = float(hf.get("anchor_lambda", 1.0))
        self.hf_opt = str(hf.get("opt", "adam"))
        self.hf_tol_up = float(hf.get("tol_up", g.get("tol", 0.05)))
        self.hf_tol_down = float(hf.get("tol_down", 0.35))
        self.box_pos_feats = []            # label=1 且带框：框内 patch 特征（shead 输入格式）
        self.normal_fb_feats = []          # label=0 反馈：整图特征（shead 输入格式）
        self.head_ft_log = []              # 每次微调留痕（n_pos/n_neg/drift/epochs/lr）
        self._head_ft_counts = {"pos": 0, "neg": 0}
        self._head_ft_snapshot = None      # 微调前 head 权重快照（gate 否决回滚）
        self._head_ft_cooldown = -99       # gate 失败冷却（按带真值反馈计数）
        # U66（2026-08-18）：框内 patch 在线微调 disc 判别头——真正的 demo4 裁切图对标。
        # U65 教训：图像级（shead）微调负（-0.013~0）——框内 patch 少、图像级统计噪声、
        # 门控预算中毒；demo4 用标注裁切图达 0.9 时喂的是 **patch 级判别头**（disc，
        # G×G 分数图），不是图像级头——本次用框内 patch 特征在线微调 disc 头。
        # 缓冲：label=1 且带框 → 框内 patch 特征（item["feats"][:,:,y0g:y1g,x0g:x1g]
        # 展平 (n,D) 每行一 patch，n≈5-30）；label=0 → 整图 patch 特征子采样（每张 64）。
        # 触发：pos≥min_pos 且 neg≥min_neg → BCE（对齐 DiscSlot.fit 配方：BCEWithLogits
        # + Adam）lr 1e-4 + 2-3 epoch + batch 128 + 均衡批采样（U65 教训：正负各半）。
        # 只改 disc 头参数（disc 槽无 backbone 参数；m0.yaml disc.freeze=true 仅让 fit
        # 跳过训练，头参数从未 requires_grad=False → 在线微调直接可用，无需临时 unfreeze）。
        # 门控保护：复用 U65 定向门控——锚定集 fused 上行（正常被异常化=回归）严格
        # tol_up=gate.tol，下行（正常更正常=微调意图方向）允许有界 tol_down。
        # 微调后 disc 槽打分立即用新头（pipeline.slots["disc"].head 权重更新，
        # score_tiles 同一 forward 路径生效）。诚实留痕：disc_ft_log + stats。
        df = cfg.get("disc_ft", {}) or {}
        self.disc_ft_on = bool(df.get("enabled", True))
        self.df_min_pos = int(df.get("min_pos", 60))
        self.df_min_neg = int(df.get("min_neg", 120))
        self.df_lr = float(df.get("lr", 1e-4))
        self.df_epochs = int(df.get("epochs", 3))
        self.df_batch = int(df.get("batch", 128))
        self.df_tol_down = float(df.get("tol_down", 0.35))
        self.disc_pos_patch_feats = []     # 框内 patch（list[tensor (n,D)]，label=1 带框）
        self.disc_neg_patch_feats = []     # 整图 patch 子采样（list[tensor (64,D)]，label=0）
        self.disc_ft_log = []              # 每次微调留痕（n_pos/n_neg/signed/drift/epochs/lr）
        self._disc_ft_counts = {"pos": 0, "neg": 0}
        self._disc_ft_snapshot = None      # 微调前 disc 头权重快照（gate 否决回滚）
        self._disc_ft_cooldown = -99       # gate 失败冷却（按带真值反馈计数）
        # U69（2026-08-18）：决策阈值在线重估——根治 U68 "学过就会"决策级最后障碍。
        # 机理（U68 根因）：Decider 阈值（tau_quantile=0.99/tau_gray=0.95）在 fit 时
        # 用 train/good fused 分布固定；在线学习（权重学习/虚高重校准/缺陷库拦截
        # 加分）改变 fused 分布后阈值未重估——已反馈正常样本落在旧阈值以上（误检）、
        # 缺陷样本相对正常分布的分离度未兑现为决策正确率（learned_acc 平台 0.74）。
        # 本机制收集**决策实际 fused 分**（label=0 反馈样本，含拦截加分，与
        # Decider.decide/predict 同口径，见 feedback() 决策对齐）缓冲 ≥ min_n 后重估：
        #   pipeline.decider = Decider.from_normal_scores(normal_fused_fb,
        #       tau_quantile, tau_gray)（与 fit 同配方，仅分布换成 test 域反馈正常分布）
        # 触发：每次 label=0 反馈后检查（与虚高重校准同频；min_n/cooldown 门槛防高频）。
        # 门控保护（sanity）：应用前用新 decider 对已反馈正常样本重测误检率
        # （decision!=normal 比例）——显著上升（>max_fp_rate=0.5，正常侧过半崩）→
        # 回滚旧 decider 快照（self._decider_snapshot）+ 冷却 cooldown 条带真值反馈。
        # 留痕：thresh_recal_log（n_normal/fp_before/fp_after/tau_high/tau_gray/status）
        # + stats.thresh_recal / thresh_recal_reject（报告重估次数/门控结果）。
        # 诚实边界：只用反馈真值（label=0 确认样本，有标注层次），不触碰 test 初始化/fit。
        tr = cfg.get("thresh_recal", {}) or {}
        self.thresh_recal_on = bool(tr.get("enabled", True))
        self.tr_min_n = int(tr.get("min_n", 20))
        self.tr_max_fp_rate = float(tr.get("max_fp_rate", 0.5))
        self.tr_cooldown = int(tr.get("cooldown", 10))
        self.normal_fused_fb = []            # 已反馈正常样本的决策实际 fused（含拦截加分）
        self.defect_fused_fb = []            # 已反馈缺陷样本的决策实际 fused（门控缺陷侧保护）
        self.defect_fused_max = int(tr.get("defect_max", 500))   # 缺陷缓冲滚动上限
        self._decider_snapshot = None        # 重估前 decider 快照（门控否决回滚用）
        self._tr_last_recal = -99            # 上次重估（成功/否决）的带真值反馈计数（冷却）
        self.thresh_recal_log = []           # 每次重估留痕（n_normal/fp/tau/status）

    # ---- 初始化 ----
    def setup(self, pipeline, train_normal_paths, rng=None, anchor_paths=None):
        """fit 后调用：锚定库种子 + 回归门控基准。
        anchor_paths: 已提特征的锚定集（enable_ssocl 抽样，兼作 seed_core 与门控）。"""
        rng = rng or np.random.default_rng(0)
        if anchor_paths is None:
            anchor_paths = self.gate.anchor_paths
        self.gate.setup(self.p, anchor_paths or train_normal_paths)
        self.normal_bank.seed_core(self.p._anchor_feats(), rng=rng)
        return self

    # ---- 三类反馈（§7.1）----
    def feedback(self, path, verdict, label=None, feats=None, box=None):
        """verdict: "correct"|"wrong"|"none"；label: 0 正常 / 1 缺陷（判对/判错时已知）。
        box（U64，2026-08-18）：用户在线提供的缺陷框标注——归一化 [x0,y0,x1,y1]
        或 [框1, 框2, ...]（list of boxes，一图多框）。label==1 且 box 提供时只把
        框内 patch 特征入缺陷样例库（add_box，信噪比更高）；否则走原 add 整图。
        信息层次策略（U64 登记）：框 = 在线反馈路径的"有标注"层次信息（模拟操作员
        画框反馈），诚实记录 entry["box"]；fit/初始化不用 test，合法。
        返回处理记录（供学习曲线/追溯）。"""
        item = self.p._get_item(path, feats)
        rec = self.p._record(item)                     # 打分/校准/基础融合（拦截前）
        # U69：决策口径对齐 predict——拦截加分后 fused 才是决策实际分数
        #（_finalize 同款：defect_bank 相对相似加分 + head_mgr boost，min cap 1.0；
        # 修复原 feedback 用拦截前 fused 决策与 predict 决策脱节，阈值重估/learned_acc
        # 均须对齐决策实际口径）
        # U70：no_cap_fused 时同样不 cap（与 _finalize 同口径，防反馈侧与 predict
        # 侧 fused 分布不一致导致阈值重估失真）
        if not self.p._ablate_intercept:
            boost = self.defect_bank.intercept_score(
                item["feats"], normal_bank=self.normal_bank)
            if self.p.head_mgr is not None:
                boost += self.p.head_mgr.boost(item["feats"])
            rec["boost"] = round(boost, 5)
            fused_boosted = rec["fused"] + boost
            if not self.p.cfg.get("decision", {}).get("no_cap_fused", False):
                fused_boosted = min(fused_boosted, 1.0)
            rec["fused"] = float(fused_boosted)
        rec["decision"] = self.p.decider.decide(rec["fused"])
        lab = "defect" if (label or 0) else "normal"
        entry = {"t": time.time(), "path": path, "verdict": verdict,
                 "label": lab, "fused": round(rec["fused"], 4),
                 "decision": rec["decision"], "slots": rec["slot_scores"],
                 "box": box is not None}
        # U65：分数级杠杆原料收集——label=0 反馈整图特征（neg）/ label=1 且带框的
        # 框内 patch 特征（pos），转 shead 输入格式入缓冲（微调触发见 _maybe_head_ft）
        if self.head_ft_on and item["feats"] is not None:
            if label == 0:
                self._collect_head_ft_neg(item)
            elif label == 1 and box is not None:
                self._collect_head_ft_pos(item, box)
        # U66：disc 头微调原料收集——label=0 反馈整图 patch 子采样（neg）/ label=1
        # 且带框的框内 patch 特征（pos，逐 patch (D,) 为 disc 输入格式，网格映射同
        # DefectBank.add_box），缓冲达标触发见 _maybe_disc_ft
        if self.disc_ft_on and item["feats"] is not None:
            if label == 0:
                self._collect_disc_ft_neg(item)
            elif label == 1 and box is not None:
                self._collect_disc_ft_pos(item, box)
            elif label == 1 and self.disc_nobox_ft:
                # U119：无 GT 框缺陷反馈 → blob 响应图定位 top 异常 patch 作正样本
                self._collect_disc_ft_pos_nobox(item)
        if verdict == "correct":
            if label == 0:                             # 正常判对 → 回流评估
                if self.p._ablate_reflow:              # U24 消融：关回流
                    entry["action"] = "reflow_ablated"
                else:
                    entry.update(self._reflow(item, "correct_normal"))
            else:                                      # 缺陷判对 → 入样例库
                self._add_defect_sample(item, box, "correct_defect")
                self.stats["defect_add"] += 1
                entry["action"] = "defect_sample_add"
        elif verdict == "wrong":
            if label == 0:                             # 误检：正常→异常
                entry.update(self._reflow(item, "false_positive"))
                entry["attribution"] = attribute_failure(
                    rec, self.p, mode="fp", item=item)
            else:                                      # 漏检：缺陷→正常
                self._add_defect_sample(item, box, "false_negative")
                self.difficult.append({"path": path, "slots": rec["slot_scores"]})
                self.stats["defect_add"] += 1
                entry["action"] = "difficult_queued"
                entry["attribution"] = attribute_failure(
                    rec, self.p, mode="fn", item=item)
                self._maybe_finetune_router()
        else:                                          # 无反馈：置信门槛分流
            entry.update(self._unlabeled(item, rec))
        self.feedback_log.append(entry)
        # U46：在线权重学习——带真值反馈才更新（label 已知），每 weight_update_every 条
        # 反馈把学习权重写入 pipeline（融合层持续学习）。回归门控保护：漂移则下轮恢复。
        if self.weight_learner is not None and label is not None:
            self.weight_learner.update(rec["slot_scores"], label)
            self._w_fb_since_update += 1
            self._n_labeled_fb += 1
            # U109：缓存最近带真值反馈样本的槽位校准分（权重写回 margin 判据用，
            # 零额外前向——只用融合权重重新加权缓存分）
            self._recent_fb.append((dict(rec["slot_scores"]), label))
            if len(self._recent_fb) > 12:
                self._recent_fb = self._recent_fb[-12:]
            # U62（实施 2 原料）：收集确认正常（label=0）反馈样本的各槽位
            # 校准前原始分 + 校准分（虚高率判定与 CDF 重估用，只来自反馈真值）
            if self.height_recal and label == 0:
                for n in self._normal_fb_raw:
                    if n in rec["raw_scores"] and n in rec["slot_scores"]:
                        self._normal_fb_raw[n].append(float(rec["raw_scores"][n]))
                        self._normal_fb_cal[n].append(float(rec["slot_scores"][n]))
            # U69：收集确认正常（label=0）反馈样本的决策实际 fused 分（含拦截
            # 加分，与 Decider.decide 同口径）→ 阈值重估缓冲；每次 label=0 反馈
            # 后触发检查（min_n/cooldown 门槛见 _maybe_thresh_recal）
            if label == 0:
                self.normal_fused_fb.append(float(rec["fused"]))
                self._maybe_thresh_recal()
            elif label == 1:
                # U69：缺陷侧门控原料——label=1 反馈样本决策实际 fused（含拦截
                # 加分），供 _maybe_thresh_recal 检查新阈值下缺陷漏检率（防 tau
                # 顶格使缺陷全判 normal；滚动上限防缓冲爆炸）
                self.defect_fused_fb.append(float(rec["fused"]))
                if len(self.defect_fused_fb) > self.defect_fused_max:
                    self.defect_fused_fb = self.defect_fused_fb[-self.defect_fused_max:]
            if self._w_fb_since_update >= self.weight_update_every \
                    and self._n_labeled_fb >= self.weight_min_samples:
                self._w_fb_since_update = 0
                self._maybe_apply_weights()
                # U62（实施 1）：打印虚高压制触发（SlotWeightLearner 内已强制置 0）
                for h in self.weight_learner.height_log:
                    print(f"[ssocl] U62 虚高压制: 槽位 {h['slot']} 负侧 {h['neg_total']} "
                          f"命中 {h['neg_hit']} 虚高率 {h['height']:.2f} → 权重置 0",
                          flush=True)
                self.weight_learner.height_log = []
                self._maybe_height_recal()
        # 回流熔断状态（§7.2）：由预测决策流更新，反馈侧仅记录
        self.fuse_blocked = self.normal_bank.fuse_blocked
        # U65：分数级杠杆触发检查（缓冲达标 → 在线微调 shead 头 + 门控保护）
        self._maybe_head_ft()
        # U66：disc 头微调触发检查（缓冲达标 → 在线微调 disc 头 + 定向门控保护）
        self._maybe_disc_ft()
        return entry

    # ---- 内部动作 ----
    def _add_defect_sample(self, item, box, source):
        """U64：缺陷样本入缺陷样例库——有框走 add_box（框内 patch，更纯净），
        无框走原 add 整图。box 支持单框 [x0,y0,x1,y1] 或 list of boxes（逐框
        add_box，一图多框全利用）。entry 留痕 box/n_boxes。"""
        if box is None:
            if self.intercept_crop and "blob" in self.p.slots \
                    and item.get("tile_imgs") is not None:
                patch = self._blob_top_patches(item)   # U121：伪框化（缺陷区域 patch）
                if patch is not None:
                    self.defect_bank.add_patch_feats(patch, source + "_crop")
                    self.defect_bank.track_hits(item["feats"])
                    self._aug_add(item, source)        # U114：增强副本入库（保留）
                    return
            self.defect_bank.add(item["feats"], source)
            self.defect_bank.track_hits(item["feats"])
            self._aug_add(item, source)              # U114：增强副本入库（无框也执行）
            return
        single = (len(box) == 4 and isinstance(box[0], (int, float)))
        boxes = [box] if single else list(box)
        added = 0
        for b in boxes:
            if self.defect_bank.add_box(item["feats"], b, source):
                added += 1
        if added == 0:                                 # 框全部无效 → 整图兜底
            self.defect_bank.add(item["feats"], source)
            added = 1
        self.defect_bank.track_hits(item["feats"])
        self.stats["defect_box"] = self.stats.get("defect_box", 0) + 1
        self._aug_add(item, source)                  # U114：增强副本入库

    def _aug_add(self, item, source):
        """U114（2026-08-26）：反馈增强扩展——缺陷反馈生成 K 个图级增强副本
        （亮度±15 / 平移±4px 交替，温和防破坏缺陷特征），副本特征入缺陷库。
        目的：一条反馈覆盖光照/位置变体，少样本（gold_finger 反馈仅 21 条）下
        拦截库覆盖更广、学习效率更高（数据放大）。成本：每反馈 K 次前向。
        诚实边界：增强副本来自已反馈缺陷样本（有标注层次），非 test 选参。"""
        if self.fb_aug_k > 0 and item.get("path"):
            try:
                img = item.get("img")
                if img is None:
                    img = load_image(item["path"])
                for i in range(self.fb_aug_k):
                    aimg = self._aug_lite(img, i)
                    ait = self.p._get_item(f"{item['path']}__aug{i}", img=aimg)
                    if self.intercept_crop:
                        # U121b（2026-08-26）：增强副本也 blob 裁剪后入库（纯度优先）
                        # ——整图增强副本入拦截库会泛化过宽、正常样本全被加分（U121+U114
                        # 组合实验 gold 0.7227→0.7148 实证）；增强视角下 blob 定位取
                        # top patch，多样性↑且纯度↑。
                        patch = self._blob_top_patches(ait)
                        if patch is not None:
                            self.defect_bank.add_patch_feats(patch, source + "_aug_crop")
                            continue
                    self.defect_bank.add(ait["feats"], source + "_aug")
                self.stats["defect_aug"] = self.stats.get("defect_aug", 0) + self.fb_aug_k
            except Exception as e:                     # noqa: BLE001 增强失败不阻断主路径
                print(f"[ssocl] 反馈增强失败: {e}")

    def _aug_lite(self, img, i):
        """U114：温和图级增强（交替亮度/平移）。"""
        import cv2
        if i % 2 == 0:
            b = 15 if i % 4 == 0 else -15
            return np.clip(img.astype(np.float32) + b, 0, 255).astype(np.uint8)
        dx = 4 if i % 4 == 1 else -4
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                              borderMode=cv2.BORDER_REPLICATE)

    def _reflow(self, item, source):
        """正常回流（§7.2）：成簇判定 → 入扩展库 → sem 并入 → CDF 重估 → 门控。

        U26 结论（2026-08-16）：曾试"pending/rejected/熔断 也 CDF 重估 + 校准放行"，
        在 gold_finger（基线反向品类）上无益反害（学习后 0.1989→0.1761），solder 无变化
        ——域差品类缺陷样本 fused 本低于正常（反向），校准纳入 test 正常样本使缺陷
        相对分位更低，排序更差。故回退：仅成簇 accepted 才走完整回流（§7.2），
        校准重估与扩展库同步、回归门控完整保护（漂移即回滚全部）。"""
        cand = self.normal_bank.reflow_candidate(item["feats"])
        if cand != "accepted":
            self.stats["pending"] += 1
            return {"action": f"reflow_{cand}"}
        # 熔断：多数派是缺陷的流不回流（§7.2）
        if self.normal_bank.fuse_blocked:
            return {"action": "reflow_blocked_by_fuse_ratio"}
        self.normal_bank.snapshot(f"reflow_{source}")
        ok = self.normal_bank.add_ext(item["feats"], source=source)
        if not ok:
            self.normal_bank.rollback()
            return {"action": "reflow_rejected_small"}
        if not self.p._ablate_sem:                     # U24 消融：关 sem 扩展
            self.p._sem_add(item["feats"])             # 扩展库并入 sem（双库制生效）
        if not self.p._ablate_cdf:                     # U24 消融：关 CDF 重估
            self.p._recalibrate_on_reflow(item)        # CDF 重估（§7.7 误检序列 2/4）
        ok, drift, _ = self.gate.check(self.p, f"reflow_{source}")
        if not ok:                                     # 锚定集漂移 → 回滚全部
            self.normal_bank.rollback()
            if not self.p._ablate_cdf:
                self.p._rollback_calibrators()
            self.stats["gate_reject"] += 1
            self.stats["rollback"] += 1
            self.rollback_log.append({"t": time.time(), "op": "reflow",
                                      "reason": "regress_gate", "drift": round(drift, 4)})
            return {"action": "reflow_rolled_back", "drift": round(drift, 4)}
        self.stats["reflow"] += 1
        return {"action": "reflow_accepted", "drift": round(drift, 4)}

    def _maybe_apply_weights(self):
        """U107（2026-08-25）：在线权重写回 + 回归门控（锚定集 fused 漂移超
        tol 回滚 + 冷却）。

        U106 门槛（weight_min_samples=8）只消除了早期暴跌（fb#6），但易品类
        负增益未修复（transistor -0.037/tubes -0.062/bracket_black -0.062）——
        根因 = 在线权重学习写回的权重本身劣于 fit 权重（易品类初始权重已优，
        学习权重是扰动）。本门控以 **train/good 锚定集** fused 漂移为判据
        （RegressGate，红线 3 协议内）：新权重让正常锚整体漂移超 tol → 回滚
        旧权重 + 冷却，拦大跳变、放行渐进正收益（screw 型难品类）。
        诚实边界：锚定集只用 train/good，不触碰 eval/test 选参（红线 B
        "test 只验不选"）。"""
        if self.weight_learner is None:
            return
        # U108：初始权重已优（train 域 fused AUROC ≥ 门槛，协议内）→ 冻结权重学习
        train_au = getattr(self.p, "train_auroc", None)
        if train_au is not None and train_au >= self.weight_learn_max_train_auroc:
            return
        # U110（修订）：初始已可用（train_auroc ≥ 0.85）且反馈不足（<30）→ 冻结
        # 权重学习（少样本学出的权重劣于 fit 权重）；初始明显欠优（<0.85，域差
        # 场景 gold_finger 型）→ 反馈少也放行（学习是主增益通道）。
        if train_au is not None and train_au >= self.weight_min_train_auroc \
                and self._n_labeled_fb < self.weight_min_feedback:
            return
        if self._n_labeled_fb - self._w_cooldown < self.weight_gate_cooldown:
            return                               # 上次 gate 失败冷却中
        old_w = dict(self.p.weights)
        new_w = self.weight_learner.weights()
        if all(abs(new_w[k] - old_w[k]) < 1e-6 for k in old_w):
            return                               # 权重无变化，跳过
        # ---- U109：反馈样本 fused 分离度（margin）判据（协议内有标注层次，
        # 零额外前向）——写回使已反馈缺陷/正常样本的分离度劣化超 tol → 回滚。
        # U107 正常锚均值漂移测不到缺陷侧排序破坏（transistor drift 0.018 通过
        # 但 AUROC 掉 0.046），margin 判据直接对准"缺陷 vs 正常排序分离"。----
        m_old = self._fb_margin(self._recent_fb, old_w)
        if m_old is not None:                    # 已反馈缺陷+正常样本齐备才判
            m_new = self._fb_margin(self._recent_fb, new_w)
            if m_new < m_old - self.weight_margin_tol:
                self._w_cooldown = self._n_labeled_fb
                self.stats["gate_reject"] += 1
                self.stats["weight_reject"] += 1
                self.rollback_log.append({"t": time.time(), "op": "weight_learn",
                                          "reason": "margin_deg",
                                          "drift": round(m_old - m_new, 4)})
                print(f"[ssocl] U109 margin 否决: 权重写回使已反馈样本 fused 分离度 "
                      f"{m_old:.4f}→{m_new:.4f}（劣化 {m_old - m_new:.4f} > "
                      f"{self.weight_margin_tol}），回滚（冷却 "
                      f"{self.weight_gate_cooldown} 反馈）", flush=True)
                return
        self._w_snapshot = old_w
        self.p.weights = dict(new_w)
        # U107：正常锚定集 fused 漂移判据（放宽 tol=weight_gate_tol，防难品类
        # 正收益被 gate.tol=0.05 误拦；易品类已由 U108 冻结不会走到这里）
        scores = [self.p._predict_fused_only(p) for p in self.gate.anchor_paths]
        new_mean = float(np.mean(scores))
        drift = abs(new_mean - self.gate.base_mean) / self.gate.base_std
        if drift > self.weight_gate_tol:   # 极端跳变 → 回滚旧权重
            self.p.weights = old_w
            self._w_cooldown = self._n_labeled_fb
            self.stats["gate_reject"] += 1
            self.stats["weight_reject"] += 1
            self.rollback_log.append({"t": time.time(), "op": "weight_learn",
                                      "reason": "regress_gate",
                                      "drift": round(drift, 4)})
            print(f"[ssocl] U107 门控否决: 权重写回锚定集 fused 漂移 {drift:.3f} "
                  f"> {self.weight_gate_tol}，回滚旧权重（冷却 "
                  f"{self.weight_gate_cooldown} 反馈）", flush=True)
            return
        self.stats["weight_apply"] += 1
        # 修复（2026-08-25）：m_old=None（最近反馈仅单侧样本，margin 判据不可用）
        # 时原打印 `{m_old:.4f}` 崩溃（TypeError: unsupported format string ...）
        m_str = (f"{m_old:.4f}→{m_new:.4f}" if m_old is not None else "N/A")
        print(f"[ssocl] U107 权重写回通过门控: drift={drift:.4f} ≤ "
              f"{self.weight_gate_tol} margin={m_str}"
              f" 累计应用 {self.stats['weight_apply']} 次", flush=True)

    def _fb_margin(self, recent, weights):
        """U109：已反馈样本在给定权重下的 fused 分离度（缺陷 fused 均值 − 正常
        fused 均值）。任一侧样本缺失返回 None（判据不可用）。复用 feedback 时
        缓存的槽位校准分（slot_scores），零额外前向。"""
        ds, ns = [], []
        for ss, lab in recent:
            f = float(fuse(ss, weights))
            (ds if lab == 1 else ns).append(f)
        if not ds or not ns:
            return None
        return float(np.mean(ds) - np.mean(ns))

    def _maybe_height_recal(self):
        """U62（实施 2）：虚高槽位 CDF 重校准（核心增量，根治 saturate）。

        触发条件：某槽位收集到的正常反馈样本（label=0）≥3 且虚高率
        （反馈时 cal>0.5 比例）≥0.5 → 用这些样本的校准前原始分重估该槽位 CDF
        （extend 到 _cal_normal_hist 后 fit，参照 _recalibrate_on_reflow 单槽位版），
        把 test 域正常分布纳入校准——修复 train/good 拟合 CDF 在域差下失效导致的
        cal 顶格（U61 机理）。门控保护：锚定集 fused 漂移超 tol → 回滚该槽位
        （edges + hist 长度快照）；失败冷却 10 反馈防高频 gate.check。
        诚实边界：只用反馈真值（label=0 确认样本），不触碰 test。"""
        if not self.height_recal or not self._normal_fb_raw:
            return
        targets = []                                   # (槽位, raw, cal, 虚高率)
        for n in self._normal_fb_raw:
            raw = self._normal_fb_raw[n]
            cal = self._normal_fb_cal[n]
            if len(raw) < 3:
                continue
            if self._n_labeled_fb - self._hr_cooldown.get(n, -99) < 10:
                continue                               # 上次 gate 失败冷却中
            height = float(np.mean([c > 0.5 for c in cal]))
            if height >= 0.5:
                targets.append((n, raw, cal, height))
        if not targets:
            return
        # 快照触发槽位（edges + hist 长度），gate 否决可回滚
        self._hr_snapshot = {n: (self.p.calibrators[n].edges.copy(),
                                 len(self.p._cal_normal_hist.get(n, [])))
                             for n, _, _, _ in targets}
        for n, raw, _, height in targets:
            self.p._cal_normal_hist[n].extend(raw)
            self.p.calibrators[n].fit(self.p._cal_normal_hist[n])
            print(f"[ssocl] U62 虚高检测: 槽位 {n} 正常反馈 {len(raw)} 条 虚高率 "
                  f"{height:.2f} → CDF 重估（反馈正常原始分并入校准分布）", flush=True)
        ok, drift, _ = self.gate.check(self.p, "height_recal")
        if not ok:
            self._rollback_height_recal()
            for n, _, _, _ in targets:
                self._hr_cooldown[n] = self._n_labeled_fb
            self.stats["gate_reject"] += 1
            self.stats["rollback"] += 1
            self.rollback_log.append({"t": time.time(), "op": "height_recal",
                                      "reason": "regress_gate", "drift": round(drift, 4)})
            print(f"[ssocl] U62 门控否决: 锚定集漂移 {drift:.3f} > tol，"
                  f"回滚虚高重估（冷却 10 反馈）", flush=True)
            return
        self.stats["height_recal"] += 1
        for n, raw, _, height in targets:
            self.height_recal_log.append({"t": time.time(), "slot": n,
                                          "n_fb": len(raw), "height": round(height, 3),
                                          "drift": round(drift, 4)})
            self._normal_fb_raw[n] = []                # 已并入校准分布，清空防重复触发
            self._normal_fb_cal[n] = []
        self._hr_snapshot = None
        print(f"[ssocl] U62 虚高重估通过门控: drift={drift:.4f} ≤ tol，"
              f"累计触发 {self.stats['height_recal']} 次", flush=True)

    def _rollback_height_recal(self):
        """U62：虚高重估被回归门控否决时恢复触发槽位快照（edges + hist 长度）。"""
        snap = self._hr_snapshot
        if not snap:
            return False
        for n, (edges, hist_len) in snap.items():
            self.p.calibrators[n].edges = edges.copy()
            while len(self.p._cal_normal_hist.get(n, [])) > hist_len:
                self.p._cal_normal_hist[n].pop()
        self._hr_snapshot = None
        return True

    # ---- U69：决策阈值在线重估（根治 U68 "学过就会"决策级最后障碍）----
    def _maybe_thresh_recal(self):
        """U69：已反馈正常样本 fused 分位数重估 Decider 阈值（决策级杠杆）。

        触发：normal_fused_fb（label=0 反馈样本的**决策实际 fused**，含拦截加分，
        与 predict/_finalize 同口径，见 feedback() 决策对齐）缓冲 ≥ tr_min_n（默认
        20）→ 用这批分布重估（与 fit 同配方，仅分布换成 test 域反馈正常分布）：
          pipeline.decider = Decider.from_normal_scores(normal_fused_fb,
              tau_quantile, tau_gray)
        门控保护（sanity）：应用前用**新 decider** 对已反馈正常样本重测误检率
        （decision!=normal 比例）——显著上升（>tr_max_fp_rate=0.5，正常侧过半崩，
        学会的正常样本被大规模误检）→ 回滚旧 decider 快照（_restore_decider，
        本就没应用，防御性恢复）+ 冷却 tr_cooldown 条带真值反馈（防 disc_ft 等
        瞬时扰动下反复重试）；缓冲保留（正常反馈稀缺，后续重估用更大样本集）。
        成功：快照旧 decider（self._decider_snapshot = pipeline.decider）→ 应用新
        阈值 → 清空缓冲（已并入新分布，防重复触发）。
        留痕：thresh_recal_log（n_normal/fp_before/fp_after/tau_high/tau_gray/status）
        + stats.thresh_recal / thresh_recal_reject。
        诚实边界：只用反馈真值（label=0 确认样本，有标注层次），不触碰 test 初始化/fit。"""
        if not self.thresh_recal_on:
            return
        if len(self.normal_fused_fb) < self.tr_min_n:
            return
        if self._n_labeled_fb - self._tr_last_recal < self.tr_cooldown:
            return                               # 上次重估（成功/否决）后冷却中
        fused = np.asarray(self.normal_fused_fb, dtype=np.float64)
        old = self.p.decider
        # 门控基线：旧阈值下这批已反馈正常样本的误检率（决策与学习机制失配信号）
        fp_before = float(np.mean([old.decide(float(f)) != "normal" for f in fused]))
        qh = self.p.cfg["decision"]["tau_quantile"]
        qg = self.p.cfg["decision"]["tau_gray"]
        new = Decider.from_normal_scores(fused, qh, qg)
        # 预演：新阈值下正常侧误检率——显著上升 → 回滚（不应用）
        fp_after = float(np.mean([new.decide(float(f)) != "normal" for f in fused]))
        # 缺陷侧保护（U69 修复，首跑实测 2026-08-18）：tau 被正常 fused 高分位
        # 推到顶格（1.0，拦截加分 cap 饱和）时缺陷几乎全判 normal——首跑轮 4
        # tau_high 0.8065→1.0000 后缺陷检出 0.75→0.27、learned_acc 0.72→0.39 崩。
        # 用 label=1 反馈样本（决策实际 fused）在旧/新阈值下分别算漏检率
        # （decision==normal 比例），恶化 >0.05 → 否决（保持旧阈值，等效回滚）。
        leak_before = leak_after = None
        if len(self.defect_fused_fb) >= 10:
            df = np.asarray(self.defect_fused_fb, dtype=np.float64)
            leak_before = float(np.mean([old.decide(float(f)) == "normal" for f in df]))
            leak_after = float(np.mean([new.decide(float(f)) == "normal" for f in df]))
        self._decider_snapshot = old            # 快照（门控否决回滚用）
        if fp_after > self.tr_max_fp_rate:      # sanity：正常侧误检过半 → 否决
            self._restore_decider()             # 回滚旧 decider（防御性，保持状态一致）
            self._tr_last_recal = self._n_labeled_fb
            self.stats["thresh_recal_reject"] = self.stats.get("thresh_recal_reject", 0) + 1
            self.thresh_recal_log.append({"t": time.time(), "n_normal": int(len(fused)),
                                          "fp_before": round(fp_before, 4),
                                          "fp_after": round(fp_after, 4),
                                          "tau_high": round(new.tau_high, 4),
                                          "tau_gray": round(new.tau_gray, 4),
                                          "status": "rejected", "reason": "fp_rise"})
            print(f"[ssocl] U69 门控否决(正常侧): 阈值重估后已反馈正常样本误检率 "
                  f"{fp_before:.3f}→{fp_after:.3f} > {self.tr_max_fp_rate}，"
                  f"回滚旧 decider（tau_high {old.tau_high:.4f} 保持），"
                  f"冷却 {self.tr_cooldown} 反馈（缓冲保留）", flush=True)
            return
        if leak_after is not None and leak_after > leak_before + 0.05:
            # 缺陷侧恶化（tau 顶格型）：否决，保持旧阈值
            self._restore_decider()
            self._tr_last_recal = self._n_labeled_fb
            self.stats["thresh_recal_reject"] = self.stats.get("thresh_recal_reject", 0) + 1
            self.thresh_recal_log.append({"t": time.time(), "n_normal": int(len(fused)),
                                          "fp_before": round(fp_before, 4),
                                          "fp_after": round(fp_after, 4),
                                          "leak_before": round(leak_before, 4),
                                          "leak_after": round(leak_after, 4),
                                          "tau_high": round(new.tau_high, 4),
                                          "tau_gray": round(new.tau_gray, 4),
                                          "status": "rejected", "reason": "defect_leak"})
            print(f"[ssocl] U69 门控否决(缺陷侧): 新阈值缺陷漏检率 "
                  f"{leak_before:.3f}→{leak_after:.3f} 恶化（tau_high "
                  f"{old.tau_high:.4f}→{new.tau_high:.4f} 顶格），回滚旧 decider，"
                  f"冷却 {self.tr_cooldown} 反馈（缓冲保留）", flush=True)
            return
        self.p.decider = new                    # 应用新阈值（在线生效，predict 立即使用）
        self._tr_last_recal = self._n_labeled_fb
        self.stats["thresh_recal"] = self.stats.get("thresh_recal", 0) + 1
        self.thresh_recal_log.append({"t": time.time(), "n_normal": int(len(fused)),
                                      "fp_before": round(fp_before, 4),
                                      "fp_after": round(fp_after, 4),
                                      "tau_high": round(new.tau_high, 4),
                                      "tau_gray": round(new.tau_gray, 4),
                                      "status": "applied"})
        print(f"[ssocl] U69 阈值重估: 已反馈正常样本 {len(fused)} 条 "
              f"误检率 {fp_before:.3f}→{fp_after:.3f} "
              f"tau_high {old.tau_high:.4f}→{new.tau_high:.4f} "
              f"tau_gray {old.tau_gray:.4f}→{new.tau_gray:.4f} "
              f"累计触发 {self.stats['thresh_recal']} 次", flush=True)
        self.normal_fused_fb = []               # 已并入新分布，清空防重复触发

    def _restore_decider(self):
        """U69：阈值重估被门控否决时恢复快照 decider（防御性——sanity 失败时本就没
        应用新阈值，恢复旧引用保持状态一致；快照同时供未来"应用后回滚"语义复用）。"""
        if self._decider_snapshot is not None:
            self.p.decider = self._decider_snapshot
            self._decider_snapshot = None
            return True
        return False

    # ---- U65：分数级杠杆（shead 判别头在线微调）----
    def _collect_head_ft_neg(self, item):
        """label=0 反馈整图特征 → shead 输入格式入缓冲（correct_normal 与
        wrong+label0 共用；多 tile 每 tile 一行，均合法正常样本）。"""
        s = self.p.slots.get("shead")
        if s is None:
            return
        with torch.no_grad():
            x = box_patch_feats(item["feats"].float(), s.chan_stats, s.patch_stats)
        self.normal_fb_feats.append(x)
        self._head_ft_counts["neg"] += x.shape[0]
        if len(self.normal_fb_feats) > 128:          # 内存护栏：滚动窗口
            self.normal_fb_feats = self.normal_fb_feats[-128:]

    def _collect_head_ft_pos(self, item, box):
        """label=1 且带框：逐框把框内 patch 特征转 shead 输入格式入缓冲。
        框无效/T≠1 跳过（保底由 _add_defect_sample 的 add_box 整图兜底，互不干扰）。"""
        single = (len(box) == 4 and isinstance(box[0], (int, float)))
        boxes = [box] if single else list(box)
        n = 0
        for b in boxes:
            x = self._box_patch_feats(item, b)
            if x is not None:
                self.box_pos_feats.append(x)
                n += 1
        self._head_ft_counts["pos"] += n
        if len(self.box_pos_feats) > 128:            # 内存护栏：滚动窗口
            self.box_pos_feats = self.box_pos_feats[-128:]

    def _box_patch_feats(self, item, box):
        """归一化框 [x0,y0,x1,y1] → 框内 patch 特征 (1, Din)（shead 输入格式：
        patch 均值 + 6 全局统计，box_patch_feats 对任意 (B,D,H,W) 成立）。
        网格映射与 DefectBank.add_box 一致（int/ceil + clip [0,G]）；无效返回 None。"""
        s = self.p.slots.get("shead")
        f = item["feats"]
        if s is None or f is None:
            return None
        f = torch.as_tensor(f, dtype=torch.float32)
        if f.ndim == 3:
            f = f.unsqueeze(0)
        T, D, G, _ = f.shape
        if T != 1:                                     # 框映射只对整图 single 特征有效
            return None
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            return None
        x0g = max(0, min(G, int(x0 * G)))
        y0g = max(0, min(G, int(y0 * G)))
        x1g = max(0, min(G, int(np.ceil(x1 * G))))
        y1g = max(0, min(G, int(np.ceil(y1 * G))))
        if x1g - x0g < 2 or y1g - y0g < 2:            # 框太小几乎无 patch
            return None
        with torch.no_grad():
            return box_patch_feats(f[:, :, y0g:y1g, x0g:x1g],
                                   s.chan_stats, s.patch_stats)

    def _maybe_head_ft(self):
        """U65：缓冲达标（pos≥min_pos 且 neg≥min_neg）→ 在线微调 shead 判别头。

        配方：DeviationLoss(margin=5) + Adam(lr 1e-4) + epochs + batch 32（demo4
        _train_discriminative_head 同损失；负样本=正常反馈整图、正样本=框内 patch）。
        只改 head 参数（backbone 冻结），槽位打分路径不变。门控保护：微调前快照
        head 权重，锚定集 fused 漂移超 tol → 回滚（失败冷却 10 带真值反馈）。
        诚实留痕：head_ft_log（n_pos/n_neg/drift）+ stats.head_ft/head_ft_reject。"""
        if not self.head_ft_on or "shead" not in self.p.slots:
            return
        if len(self.box_pos_feats) < self.hf_min_pos or \
           len(self.normal_fb_feats) < self.hf_min_neg:
            return
        if self._n_labeled_fb - self._head_ft_cooldown < 10:
            return                                   # 上次 gate 失败冷却中
        slot = self.p.slots["shead"]
        pos = torch.cat(self.box_pos_feats, dim=0).float()      # (Np, Din)
        neg = torch.cat(self.normal_fb_feats, dim=0).float()    # (Nn, Din)
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            return
        s_before = self._anchor_shead_mean()
        margin = float(self.p.cfg.get("slots", {}).get("shead", {}).get("margin", 5.0))
        self._head_ft_snapshot = [w.detach().clone() for w in slot.head.parameters()]
        slot.head.train()
        loss_fn = DeviationLoss(margin=margin)
        if self.hf_opt == "sgd":                 # U65 调优：sgd 使锚定正则权重线性生效
            opt = torch.optim.SGD(slot.head.parameters(), lr=self.hf_lr)
        else:
            opt = torch.optim.Adam(slot.head.parameters(), lr=self.hf_lr)
        # 锚定集保护正则（U65 实测调优：仅反馈负样本时微调把锚定集 shead 原始分
        # 整体下移 -0.34 → 锚定 fused 漂移 0.33 被门控否决）。锚定特征以"相对微调前
        # 分数"的 L2 正则并入损失——锚定区域（门控测量对象）显式钉住，微调只移动
        # test 域打分面（框内缺陷↑、反馈正常↓），门控漂移应≈0。
        anchor_feats, anchor_ref = [], []
        a_items = getattr(self.p, "_anchor_items", None)
        if a_items:
            with torch.no_grad():
                for it in a_items[:self.hf_anchor_n]:
                    if it["feats"] is None:
                        continue
                    ax = box_patch_feats(it["feats"].float(),
                                         slot.chan_stats, slot.patch_stats)
                    anchor_feats.append(ax)
                    anchor_ref.append(float(slot.head(ax.to(slot.device))[0]))
        # 均衡批采样（U65 实测调优：pos 主导的混合批会把正常侧也推高 → 锚定集
        # fused 漂移 0.28 > tol 被门控否决；均衡批让正常侧梯度每批都在场，微调局部化）
        k = min(pos.shape[0], neg.shape[0])
        bs = min(32, 2 * k)
        for ep in range(self.hf_epochs):
            pi = torch.randperm(pos.shape[0])[:k]
            ni = torch.randperm(neg.shape[0])[:k]
            Xb = torch.cat([pos[pi], neg[ni]]).to(slot.device)   # (2k, Din)
            yb = torch.cat([torch.ones(k), torch.zeros(k)]).to(slot.device)
            perm = torch.randperm(2 * k)
            for i in range(0, 2 * k, bs):
                idx = perm[i:i + bs]
                scores = slot.head(Xb[idx])
                loss = loss_fn(scores, yb[idx])
                if anchor_feats:
                    A = torch.cat(anchor_feats).to(slot.device)
                    ref = torch.tensor(anchor_ref, device=slot.device)
                    loss = loss + self.hf_anchor_lambda * \
                        (slot.head(A) - ref).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        slot.head.eval()                             # 恢复打分模式（score_tiles 无梯度）
        s_after = self._anchor_shead_mean()
        shift = (round(s_after - s_before, 4)
                 if s_before is not None and s_after is not None else None)
        # 定向回归门控（U65 调优，锚定集 fused 基准复用 gate.base_mean/std）：
        # 门控保护对象（红线 3）= 正常锚定集被异常化 → fused **上升**是回归，严格
        # tol_up 拒绝；head_ft 的意图方向 = 正常分下降（更正常）→ fused 下行允许
        # 有界 tol_down。实测：Adam 微调锚定 fused 全为下行（-0.023~-0.068），
        # sgd 发散为上行 +0.296（signed +1.43 > tol_up 被拒）——定向门控语义正确。
        scores = [self.p._predict_fused_only(p) for p in self.gate.anchor_paths]
        new_mean = float(np.mean(scores))
        signed = (new_mean - self.gate.base_mean) / self.gate.base_std
        drift = abs(signed)
        ok = (-self.hf_tol_down <= signed <= self.hf_tol_up)
        if not ok:                                   # 门控否决：回滚 head 权重
            self._restore_head_ft()
            self._head_ft_cooldown = self._n_labeled_fb
            self.stats["gate_reject"] += 1
            self.stats["head_ft_reject"] = self.stats.get("head_ft_reject", 0) + 1
            self.rollback_log.append({"t": time.time(), "op": "head_ft",
                                      "reason": "regress_gate",
                                      "drift": round(drift, 4)})
            print(f"[ssocl] U65 门控否决: head_ft 锚定集 fused "
                  f"{self.gate.base_mean:.4f}→{new_mean:.4f} signed={signed:.3f} "
                  f"(上行限 {self.hf_tol_up} / 下行限 {self.hf_tol_down})，"
                  f"锚定集 shead 原始分移动 {shift}，回滚 head "
                  f"（pos={pos.shape[0]} neg={neg.shape[0]}，冷却 10 反馈）", flush=True)
            return
        self.stats["head_ft"] = self.stats.get("head_ft", 0) + 1
        self.head_ft_log.append({"t": time.time(), "n_pos": int(pos.shape[0]),
                                 "n_neg": int(neg.shape[0]), "drift": round(drift, 4),
                                 "signed": round(signed, 4),
                                 "epochs": self.hf_epochs, "lr": self.hf_lr,
                                 "anchor_shead_shift": shift})
        self.box_pos_feats = []                      # 成功后清空缓冲，重新积累
        self.normal_fb_feats = []
        print(f"[ssocl] U65 head_ft 完成: pos={pos.shape[0]} neg={neg.shape[0]} "
              f"signed={signed:.4f} 锚定集 shead 移动 {shift} "
              f"累计触发 {self.stats['head_ft']} 次", flush=True)

    def _anchor_shead_mean(self):
        """U65 诊断：锚定集（train/good）shead 原始分均值——量化 head 微调对正常域
        打分的影响（门控漂移的槽位级归因）。"""
        s = self.p.slots.get("shead")
        items = getattr(self.p, "_anchor_items", None)
        if s is None or not items:
            return None
        with torch.no_grad():
            vals = [float(s.score_tiles(it["feats"].float().to(s.device))[0][0])
                    for it in items]
        return float(np.mean(vals))

    def _restore_head_ft(self):
        """U65：head_ft 被门控否决时恢复快照权重。"""
        snap = self._head_ft_snapshot
        slot = self.p.slots.get("shead")
        if not snap or slot is None:
            self._head_ft_snapshot = None
            return False
        for w, s in zip(slot.head.parameters(), snap):
            w.data.copy_(s)
        self._head_ft_snapshot = None
        return True

    # ---- U66：框内 patch 在线微调 disc 判别头（真正的 demo4 裁切图对标）----
    def _collect_disc_ft_neg(self, item):
        """label=0 反馈 → 整图 patch 特征子采样（每张 64，控制内存与正负均衡）。
        disc 头输入 = 单 patch 特征 (D,)，故缓冲为 list[tensor (n,D)]，逐行一 patch。"""
        f = item["feats"]
        if f is None:
            return
        f = torch.as_tensor(f, dtype=torch.float32)
        if f.ndim == 3:
            f = f.unsqueeze(0)
        patches = f.reshape(-1, f.shape[1])          # (T*G*G, D)（single T=1）
        n = min(64, patches.shape[0])
        sel = torch.randperm(patches.shape[0])[:n]
        self.disc_neg_patch_feats.append(patches[sel].detach().cpu())
        self._disc_ft_counts["neg"] += n
        if len(self.disc_neg_patch_feats) > 128:     # 内存护栏：滚动窗口
            self.disc_neg_patch_feats = self.disc_neg_patch_feats[-128:]

    def _collect_disc_ft_pos(self, item, box):
        """label=1 且带框：逐框把框内 patch 特征（item["feats"][:,:,y0g:y1g,x0g:x1g]
        展平 (n,D) 每行一 patch）入正缓冲。网格映射与 DefectBank.add_box 一致
        （int/ceil + clip [0,G]）；非 single/无效框跳过（保底由 _add_defect_sample 兜底）。"""
        f = item["feats"]
        if f is None:
            return
        f = torch.as_tensor(f, dtype=torch.float32)
        if f.ndim == 3:
            f = f.unsqueeze(0)
        T, D, G, _ = f.shape
        if T != 1:                                   # 框映射只对整图 single 特征有效
            return
        single = (len(box) == 4 and isinstance(box[0], (int, float)))
        boxes = [box] if single else list(box)
        n = 0
        for b in boxes:
            try:
                x0, y0, x1, y1 = (float(v) for v in b)
            except (TypeError, ValueError):
                continue
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                continue
            x0g = max(0, min(G, int(x0 * G)))
            y0g = max(0, min(G, int(y0 * G)))
            x1g = max(0, min(G, int(np.ceil(x1 * G))))
            y1g = max(0, min(G, int(np.ceil(y1 * G))))
            if x1g - x0g < 2 or y1g - y0g < 2:       # 框太小几乎无 patch
                continue
            box_feats = f[:, :, y0g:y1g, x0g:x1g].reshape(-1, D).detach().cpu()
            self.disc_pos_patch_feats.append(box_feats)
            n += box_feats.shape[0]
        self._disc_ft_counts["pos"] += n
        if len(self.disc_pos_patch_feats) > 128:     # 内存护栏：滚动窗口
            self.disc_pos_patch_feats = self.disc_pos_patch_feats[-128:]

    def _blob_top_patches(self, item, ratio=0.25):
        """U119/U121：blob 响应图（域不变特征）降采样到 patch 网格，返回 top
        ratio 异常 patch 特征 (K, D) numpy；失败返回 None。"""
        if item.get("tile_imgs") is None or "blob" not in self.p.slots:
            return None
        f = item["feats"]
        if f is None:
            return None
        f = torch.as_tensor(f, dtype=torch.float32)
        if f.ndim == 3:
            f = f.unsqueeze(0)
        T, D, G, _ = f.shape
        if T != 1:
            return None
        import cv2
        blob = self.p.slots["blob"]
        t = item["tile_imgs"][0]                     # single 模式整图 RGB
        g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        r = blob._dog_response(g)                    # 像素级 DoG 响应图
        h, w = r.shape
        r32 = np.zeros((G, G), dtype=np.float32)
        for i in range(G):
            y0_, y1_ = i * h // G, (i + 1) * h // G
            for j in range(G):
                r32[i, j] = r[y0_:y1_, j * w // G:(j + 1) * w // G].max()
        K = max(1, int(G * G * ratio))
        top_idx = np.argpartition(r32.ravel(), -K)[-K:]
        return f[0].permute(1, 2, 0).reshape(-1, D)[top_idx].numpy()

    def _collect_disc_ft_pos_nobox(self, item):
        """U119（2026-08-26）：无 GT 框缺陷反馈 → 用 blob 响应图（域不变特征，
        U97/U113 实证跨域强正）定位 top 异常 patch 作 disc 头正样本——demo4 标注
        裁切图的合法近似（blob 响应图 top patch ≈ 缺陷区域）。
        诚实边界：blob 定位是无监督近似（非 GT 框），含噪声 patch；由 disc_ft
        门控（margin 判据 + 锚定漂移回滚）保护，若污染则回滚不生效。"""
        f = item["feats"]
        if f is None or "blob" not in self.p.slots:
            return
        f = torch.as_tensor(f, dtype=torch.float32)
        if f.ndim == 3:
            f = f.unsqueeze(0)
        T, D, G, _ = f.shape
        if T != 1:
            return
        patch = self._blob_top_patches(item)
        if patch is None:
            return
        sel = torch.as_tensor(patch, dtype=torch.float32).detach().cpu()
        self.disc_pos_patch_feats.append(sel)
        self._disc_ft_counts["pos"] += sel.shape[0]
        if len(self.disc_pos_patch_feats) > 128:     # 内存护栏：滚动窗口
            self.disc_pos_patch_feats = self.disc_pos_patch_feats[-128:]

    def _maybe_disc_ft(self):
        """U66：缓冲达标（pos≥min_pos 且 neg≥min_neg）→ 在线微调 disc 判别头。

        配方：BCEWithLogits（对齐 DiscSlot.fit：patch 级判别，sigmoid 打分语义一致）
        + Adam(lr 1e-4) + epochs + batch 128 + 均衡批采样（U65 教训：正负各半，
        防 pos 主导批把正常侧也推高）。只改 disc 头参数（disc 槽无 backbone 参数；
        m0.yaml disc.freeze=true 仅 fit 跳过训练，头参数从未 requires_grad=False →
        直接可训练）。门控保护：快照 head 权重，锚定集 fused 漂移超限回滚——
        定向门控（上行=正常被异常化严格 tol_up=gate.tol；下行=正常更正常（微调
        意图方向）允许有界 tol_down），失败冷却 10 带真值反馈。诚实留痕：
        disc_ft_log（n_pos/n_neg/signed/drift/epochs/lr）+ stats.disc_ft/disc_ft_reject。
        微调后 disc 槽打分立即用新头（score_tiles 同一 forward 路径生效）。"""
        if not self.disc_ft_on or "disc" not in self.p.slots:
            return
        # 触发判断按**缓冲内 patch 总数**（disc 是 patch 级判别头，样本=单个 patch；
        # 每条缓冲 = 一张图/一个框的 (n,D) tensor，len(list) 是图/框数非 patch 数）
        n_pos_buf = int(sum(x.shape[0] for x in self.disc_pos_patch_feats))
        n_neg_buf = int(sum(x.shape[0] for x in self.disc_neg_patch_feats))
        if n_pos_buf < self.df_min_pos or n_neg_buf < self.df_min_neg:
            return
        if self._n_labeled_fb - self._disc_ft_cooldown < 10:
            return                                   # 上次 gate 失败冷却中
        slot = self.p.slots["disc"]
        pos = torch.cat(self.disc_pos_patch_feats, dim=0).float()      # (Np, D)
        neg = torch.cat(self.disc_neg_patch_feats, dim=0).float()      # (Nn, D)
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            return
        self._disc_ft_snapshot = [w.detach().clone() for w in slot.head.parameters()]
        slot.head.train()
        opt = torch.optim.Adam(slot.head.parameters(), lr=self.df_lr)
        # 均衡批采样（U65 实测调优）：正负各半，正常侧梯度每批在场，微调局部化
        k = min(pos.shape[0], neg.shape[0])
        bs = min(self.df_batch, 2 * k)
        for ep in range(self.df_epochs):
            pi = torch.randperm(pos.shape[0])[:k]
            ni = torch.randperm(neg.shape[0])[:k]
            Xb = torch.cat([pos[pi], neg[ni]]).to(slot.device)        # (2k, D)
            yb = torch.cat([torch.ones(k), torch.zeros(k)]).to(slot.device)
            perm = torch.randperm(2 * k)
            for i in range(0, 2 * k, bs):
                idx = perm[i:i + bs]
                logit = slot.head(Xb[idx])
                loss = F.binary_cross_entropy_with_logits(logit, yb[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        slot.head.eval()                             # 恢复打分模式（score_tiles 无梯度）
        # 定向回归门控（复用 U65 设计，锚定集 fused 基准复用 gate.base_mean/std）：
        # 门控保护对象（红线 3）= 正常锚定集被异常化 → fused **上升**是回归，严格
        # tol_up 拒绝；disc 微调意图方向 = 框内缺陷 patch 分数↑/正常 patch 分数↓ →
        # 锚定集 fused 下行（更正常）允许有界 tol_down。
        scores = [self.p._predict_fused_only(p) for p in self.gate.anchor_paths]
        new_mean = float(np.mean(scores))
        signed = (new_mean - self.gate.base_mean) / self.gate.base_std
        drift = abs(signed)
        ok = (-self.df_tol_down <= signed <= self.hf_tol_up)
        if not ok:                                   # 门控否决：回滚 disc 头权重
            self._restore_disc_ft()
            self._disc_ft_cooldown = self._n_labeled_fb
            self.stats["gate_reject"] += 1
            self.stats["disc_ft_reject"] = self.stats.get("disc_ft_reject", 0) + 1
            self.rollback_log.append({"t": time.time(), "op": "disc_ft",
                                      "reason": "regress_gate",
                                      "drift": round(drift, 4)})
            print(f"[ssocl] U66 门控否决: disc_ft 锚定集 fused "
                  f"{self.gate.base_mean:.4f}→{new_mean:.4f} signed={signed:.3f} "
                  f"(上行限 {self.hf_tol_up} / 下行限 {self.df_tol_down})，回滚 disc 头 "
                  f"（pos={pos.shape[0]} neg={neg.shape[0]}，冷却 10 反馈）", flush=True)
            return
        self.stats["disc_ft"] = self.stats.get("disc_ft", 0) + 1
        self.disc_ft_log.append({"t": time.time(), "n_pos": int(pos.shape[0]),
                                 "n_neg": int(neg.shape[0]), "drift": round(drift, 4),
                                 "signed": round(signed, 4),
                                 "epochs": self.df_epochs, "lr": self.df_lr})
        self.disc_pos_patch_feats = []               # 成功后清空缓冲，重新积累
        self.disc_neg_patch_feats = []
        print(f"[ssocl] U66 disc_ft 完成: pos={pos.shape[0]} neg={neg.shape[0]} "
              f"signed={signed:.4f} 累计触发 {self.stats['disc_ft']} 次", flush=True)

    def _restore_disc_ft(self):
        """U66：disc_ft 被门控否决时恢复快照权重。"""
        snap = self._disc_ft_snapshot
        slot = self.p.slots.get("disc")
        if not snap or slot is None:
            self._disc_ft_snapshot = None
            return False
        for w, s in zip(slot.head.parameters(), snap):
            w.data.copy_(s)
        self._disc_ft_snapshot = None
        return True

    def _unlabeled(self, item, rec):
        """无反馈分流（§7.1）：高置信缺陷缓存待确认；灰区进主动选样队列。"""
        hi = self.p.decider.tau_high
        lo = self.p.decider.tau_gray
        if rec["fused"] >= hi:
            self.high_conf.append({"path": rec["path"], "t": time.time(),
                                   "fused": rec["fused"]})
            return {"action": "high_conf_cache"}
        if lo <= rec["fused"] < hi:
            self.p.active_enqueue(rec, item)           # 灰区 → 主动选样（§7.5）
            return {"action": "active_queue"}
        return {"action": "low_value_skip"}

    def _maybe_finetune_router(self):
        """批量级路由微调（§7.6）：难例≥trigger 才触发；lr≤1e-4、≤5ep、仅难例+锚定集。"""
        if len(self.difficult) < self.rt_trigger or self.p.router is None:
            return
        if not self.p._retrain_router(self.difficult, lr=self.rt_lr,
                                      epochs=self.rt_epochs):
            return
        ok, drift, _ = self.gate.check(self.p, "router_finetune")
        if not ok:
            self.p._restore_router()
            self.stats["rollback"] += 1
            self.rollback_log.append({"t": time.time(), "op": "router_finetune",
                                      "reason": "regress_gate", "drift": round(drift, 4)})
            return
        self.stats["router_ft"] += 1
        self.difficult = []
